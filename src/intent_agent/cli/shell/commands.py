"""Slash commands of the interactive shell."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from intent_agent.cli.commands.check import run_checks
from intent_agent.cli.commands.graph import diagrams
from intent_agent.cli.commands.selftest import run_selftest
from intent_agent.cli.render import sample_panel, samples_table
from intent_agent.cli.theme import ACCENT, console
from intent_agent.eval.dataset import ALL_SAMPLES, REQUEST_TYPES, get_sample, resolve_class, select
from intent_agent.eval.experiments import EXPERIMENTS

if TYPE_CHECKING:
    from intent_agent.cli.shell.app import Shell

EXIT = object()


@dataclass(frozen=True)
class SlashCommand:
    handler: Callable[["Shell", list[str]], object]
    usage: str
    help: str


def _filters(tokens: list[str]) -> dict:
    opts: dict = {}
    for tok in (t.lower() for t in tokens):
        if tok in ("tc", "mc", "rc", "mac", "dc"):
            opts["cls"] = resolve_class(tok)
        elif tok in REQUEST_TYPES:
            opts["rtype"] = tok
        elif tok == "adversarial":
            opts["adversarial"] = True
        elif tok in ("well-formed", "wellformed"):
            opts["adversarial"] = False
    return opts


def _samples_by_id(ids: list[str]):
    try:
        return [get_sample(i) for i in ids]
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/] — see /samples")
        return []


def c_help(sh, args):
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column(style=f"bold {ACCENT}", no_wrap=True)
    t.add_column(style="dim", no_wrap=True)
    t.add_column()
    for name, cmd in COMMANDS.items():
        t.add_row(name, escape(cmd.usage), cmd.help)
    console.print(Panel(t, title="Commands", title_align="left", border_style="dim"))
    console.print("[dim]Anything not starting with / is a service intent. Tab completes, "
                  "Ctrl+C cancels, Ctrl+D exits. During questions, /cancel stops.[/]\n")


def c_run(sh, args):
    if not args:
        console.print("[dim]Usage: /run <ID> [ID …]   e.g. /run CF-43 SEED-TC-01[/]")
        return
    sh.run_samples(_samples_by_id(args))


def c_random(sh, args):
    pool = select(ALL_SAMPLES, **_filters(args))
    n = next((int(a) for a in args if a.isdigit()), 1)
    if not pool:
        console.print("[red]No samples match those filters.[/]")
        return
    sh.run_samples(random.sample(pool, min(n, len(pool))))


def c_samples(sh, args):
    pool = select(ALL_SAMPLES, **_filters(args))
    console.print(samples_table(pool, console.width))
    console.print(f"[dim]{len(pool)} sample(s) · /show <ID> · /run <ID>[/]\n")


def c_show(sh, args):
    for s in _samples_by_id(args[:1]) if args else []:
        console.print(sample_panel(s))
    if not args:
        console.print("[dim]Usage: /show <ID>[/]")


def c_model(sh, args):
    if args:
        sh.switch_model(args[0])
        return
    models = sh.llm.list_models()
    if not models:
        console.print(f"[red]Cannot list models at {sh.llm.base_url}[/] — is `ollama serve` running?")
    for m in models:
        console.print(f" {f'[{ACCENT}]●[/]' if m == sh.llm.model else ' '} {m}")
    console.print("[dim]Switch with /model <tag>[/]")


def c_scenario(sh, args):
    if args == ["off"]:
        sh.scenario = None
        console.print("[green]●[/] You answer the questions again.")
    elif args:
        sh.scenario = " ".join(args)
        console.print("[green]●[/] The LLM will now answer questions as this user.")
    else:
        console.print(f"Simulated user: [dim]{escape(sh.scenario)}[/]" if sh.scenario else
                      '[dim]You answer. Set a persona: /scenario "You are the port IT manager. …"[/]')


def toggle(attr: str, label: str):
    def handler(sh, args):
        setattr(sh, attr, not getattr(sh, attr))
        console.print(f"[green]●[/] {label} {'on' if getattr(sh, attr) else 'off'}")
    return handler


def c_export(sh, args):
    if not sh.last_profile:
        console.print("[dim]No profile yet — describe a service first.[/]")
        return
    path = Path(args[0] if args else "profile.json")
    path.write_text(json.dumps(sh.last_profile, indent=2, ensure_ascii=False, default=str))
    console.print(f"[green]●[/] Saved to {path}")


def c_bench(sh, args):
    if not args or args[0] not in EXPERIMENTS:
        console.print(f"[dim]Usage: /bench <{'|'.join(EXPERIMENTS)}> [options][/]")
        return
    cmd = [sys.executable, "-m", "intent_agent", "--model", sh.llm.model,
           "--ollama-url", sh.llm.base_url, "--results-dir", str(sh.settings.results_dir),
           "bench", *args]
    console.print(f"[dim]$ intent-agent bench {escape(' '.join(args))}[/]")
    subprocess.run(cmd)


def c_graph(sh, args):
    for title, mermaid in diagrams().items():
        console.print(Panel(escape(mermaid), title=title, title_align="left", border_style="dim"))


COMMANDS: dict[str, SlashCommand] = {
    "/help": SlashCommand(c_help, "", "show commands"),
    "/run": SlashCommand(c_run, "<ID…>", "replay benchmark intents with the simulated user"),
    "/random": SlashCommand(c_random, "[N] [filters]", "run N random benchmark intents"),
    "/samples": SlashCommand(c_samples, "[filters]", "list intents (TC, ambiguous, adversarial…)"),
    "/show": SlashCommand(c_show, "<ID>", "show one benchmark intent"),
    "/scenario": SlashCommand(c_scenario, "[text|off]", "let the LLM answer as this user"),
    "/model": SlashCommand(c_model, "[tag]", "list or switch Ollama models"),
    "/verbose": SlashCommand(toggle("verbose", "Verbose trace"), "", "toggle the pipeline trace"),
    "/json": SlashCommand(toggle("json_mode", "Raw JSON output"), "", "toggle raw JSON profiles"),
    "/export": SlashCommand(c_export, "[file]", "save the last profile as JSON"),
    "/check": SlashCommand(lambda sh, a: run_checks(sh.settings), "", "check setup and model"),
    "/selftest": SlashCommand(lambda sh, a: run_selftest(select(ALL_SAMPLES, **_filters(a))),
                              "[filters]", "offline pipeline test (no model)"),
    "/bench": SlashCommand(c_bench, "<exp> [opts]", "run " + " | ".join(EXPERIMENTS)),
    "/graph": SlashCommand(c_graph, "", "show the LangGraph diagrams"),
    "/clear": SlashCommand(lambda sh, a: console.clear(), "", "clear the screen"),
    "/exit": SlashCommand(lambda sh, a: EXIT, "", "quit"),
}
