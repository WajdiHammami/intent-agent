"""Rich renderables shared by the subcommands and the interactive shell."""

from __future__ import annotations

import json
from typing import Iterable, Sequence

from rich import box
from rich.console import Group, RenderableType
from rich.json import JSON
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.theme import ACCENT
from eval.checks import Check
from eval.dataset import CLASS_ABBREVIATIONS, EvalSample
from models.dialogue import DialogueState
from models.service import ServiceIntent

PROFILE_FIELDS = ("vertical_domain", "geographic_scope", "isolation_level",
                  "priority_level", "cost_profile", "traffic_pattern")


def profile_panel(si: ServiceIntent, as_json: bool = False) -> Panel:
    d = si.to_dict()
    title = f"[bold]Service profile · {si.service_class}[/]"
    if as_json:
        return Panel(JSON(json.dumps(d, default=str)), title=title, title_align="left",
                     border_style=ACCENT)
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", no_wrap=True)
    t.add_column()
    for k in PROFILE_FIELDS:
        t.add_row(k, escape(str(d.get(k, ""))))
    t.add_row("", "")
    for k, v in d.get("class_slots", {}).items():
        t.add_row(k, Text(str(v), style=f"bold {ACCENT}" if k == "deployment_mode" else ""))
    kpi = d.get("kpi_profile") or {}
    if kpi:
        t.add_row("", "")
        t.add_row("kpi targets", escape(
            f"latency ≤ {kpi.get('latency_max_ms')} ms · reliability {kpi.get('reliability')} · "
            f"availability {kpi.get('availability')}"))
    return Panel(t, title=title, title_align="left", border_style=ACCENT, padding=(0, 1))


def checks_panel(checks: Sequence[Check]) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    for c in checks:
        t.add_row("[green]✓[/]" if c.passed else "[red]✗[/]", c.name, f"[dim]{escape(c.detail)}[/]")
    ok = all(c.passed for c in checks)
    return Panel(t, title="Checks against labels", title_align="left",
                 border_style="green" if ok else "red")


def dialogue_lines(state: DialogueState) -> RenderableType:
    if not state.turn_history:
        return Text("(no clarification questions — everything came from the intent)", style="dim")
    lines = [
        Text.assemble(("? " if t.role == "system" else "› ", "bold cyan" if t.role == "system" else "green"),
                      t.content)
        for t in state.turn_history
    ]
    return Group(*lines)


def conflicts_lines(state: DialogueState) -> RenderableType:
    return Group(*[
        Text.assemble(("⚠ ", "yellow"), c.description,
                      ("  [resolved]" if c.resolved else "  [open]", "dim"))
        for c in state.conflicts
    ])


def samples_table(samples: Iterable[EvalSample], width: int) -> Table:
    t = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False)
    t.add_column("ID", no_wrap=True, style=ACCENT, min_width=10)
    t.add_column("class", no_wrap=True, min_width=5)
    t.add_column("type", no_wrap=True, min_width=13)
    t.add_column("scenario", no_wrap=True, style="dim", min_width=11)
    t.add_column("intent", overflow="ellipsis", no_wrap=True, max_width=max(20, width - 52))
    for s in samples:
        t.add_row(s.id, CLASS_ABBREVIATIONS[s.expected_class], s.request_type,
                  s.scenario_label, escape(s.intent))
    return t


def sample_panel(s: EvalSample) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    for k, v in [("class", s.expected_class), ("type", s.request_type),
                 ("scenario", s.scenario_label), ("gold mode", s.gold_deployment_mode or "—"),
                 ("expected slots", json.dumps(s.expected_slots) if s.expected_slots else "—"),
                 ("expected conflict", " + ".join(s.expected_conflicts) or "—")]:
        t.add_row(k, escape(v))
    body = Group(t, Text(""), Text("Intent", style="bold"), Text(s.intent), Text(""),
                 Text("Simulated user", style="bold"), Text(s.scenario, style="dim"))
    return Panel(body, title=f"[bold]{s.id}[/]", title_align="left", border_style=ACCENT)
