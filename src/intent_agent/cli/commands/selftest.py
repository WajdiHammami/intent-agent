"""`intent-agent selftest` — the whole pipeline offline, with a scripted model and user."""

from __future__ import annotations

import argparse
from typing import Sequence

from intent_agent.adapters.llm.scripted import ScriptedLanguageModel
from intent_agent.adapters.users.scripted import ScriptedUser
from intent_agent.cli.common import add_filter_args, filtered
from intent_agent.cli.theme import console
from intent_agent.core.pipeline import IntentPipeline
from intent_agent.eval.dataset import EvalSample, get_sample


def run_selftest(samples: Sequence[EvalSample]) -> int:
    console.print(f"\n[bold]Offline self-test — {len(samples)} intents, scripted model (no Ollama)[/]")
    failures = 0
    for s in samples:
        try:
            r = IntentPipeline(ScriptedLanguageModel(s.expected_class), ScriptedUser()).run(s.intent)
            assert r.service_intent.service_class == s.expected_class, "wrong class"
            assert r.state.is_complete, "dialogue not complete"
            assert r.state.turn_count <= r.state.max_turns, "turn limit exceeded"
            assert r.service_intent.class_slots.get("deployment_mode"), "no deployment mode"
        except Exception as exc:
            failures += 1
            console.print(f"  [red]✗[/] {s.id}: {exc}")
    if samples and not failures:
        console.print(f"  [green]✓[/] {len(samples)} intents ran through the graph and produced a profile")

    sample = get_sample("CF-01")
    llm = ScriptedLanguageModel(sample.expected_class,
                                {"isolation_level": "Dedicated", "cost_profile": "LowCost"})
    state = IntentPipeline(llm, ScriptedUser()).run(sample.intent).state
    if state.conflicts and all(c.resolved for c in state.conflicts):
        console.print(f"  [green]✓[/] conflict path: {len(state.conflicts)} conflict(s) "
                      "flagged by the KG and resolved")
    else:
        failures += 1
        console.print(f"  [red]✗[/] conflict path: expected resolved conflicts, got {state.conflicts}")

    console.print("\n" + ("[green]Self-test passed.[/]" if not failures else f"[red]{failures} failure(s).[/]"))
    return 1 if failures else 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("selftest", help="offline pipeline test with a scripted model (no Ollama)")
    add_filter_args(p)
    p.set_defaults(func=lambda args, settings: run_selftest(filtered(args)))
