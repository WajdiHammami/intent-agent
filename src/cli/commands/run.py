"""`intent-agent run` — run benchmark intents with the simulated user and grade them."""

from __future__ import annotations

import argparse
import logging
import random

from rich.markup import escape
from rich.rule import Rule

from bootstrap import simulated_pipeline
from cli.common import add_filter_args, connect, filtered
from cli.render import checks_panel, conflicts_lines, dialogue_lines, profile_panel
from cli.theme import console
from config import Settings
from core.knowledge import KnowledgeGraph
from eval.checks import run_and_check
from eval.dataset import get_sample


def run(args: argparse.Namespace, settings: Settings) -> int:
    try:
        samples = [get_sample(i) for i in args.ids]
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/]")
        return 2
    if not samples:
        pool = filtered(args)
        samples = random.Random(args.seed).sample(pool, min(args.random, len(pool)))
    if not samples:
        console.print("[red]No samples match those filters.[/]")
        return 2
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="    %(message)s")

    llm, kg = connect(settings), KnowledgeGraph()
    passed = 0
    for s in samples:
        console.print(Rule(f"[bold]{s.id}[/] · {s.expected_class} · {s.request_type}", align="left"))
        console.print(f"[bold]›[/] {escape(s.intent)}\n")
        try:
            with console.status("[dim]Running the pipeline…[/]"):
                result, _, checks = run_and_check(simulated_pipeline(llm, s.scenario, kg=kg), s, kg)
        except Exception as exc:
            console.print(f"[red]pipeline error: {exc}[/]")
            continue
        console.print(dialogue_lines(result.state))
        if result.state.conflicts:
            console.print(conflicts_lines(result.state))
        console.print(profile_panel(result.service_intent, as_json=args.json))
        console.print(checks_panel(checks))
        passed += all(c.passed for c in checks)

    console.print(f"\n[bold]{passed}/{len(samples)}[/] sample(s) passed every check.")
    return 0 if passed == len(samples) else 1


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="run benchmark intents end-to-end and check them against labels")
    p.add_argument("ids", nargs="*", metavar="ID", help="sample ids (default: a random pick)")
    add_filter_args(p)
    p.add_argument("--random", type=int, default=3, metavar="N", help="random sample size (default 3)")
    p.add_argument("--seed", type=int, help="random seed")
    p.add_argument("--json", action="store_true", help="print the full profile JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="print the pipeline trace")
    p.set_defaults(func=run)
