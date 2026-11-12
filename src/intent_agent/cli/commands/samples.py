"""`intent-agent samples` — browse the benchmark."""

from __future__ import annotations

import argparse
import json

from intent_agent.cli.common import add_filter_args, filtered
from intent_agent.cli.render import sample_panel, samples_table
from intent_agent.cli.theme import console
from intent_agent.config import Settings
from intent_agent.eval.dataset import get_sample


def run(args: argparse.Namespace, settings: Settings) -> int:
    if args.show:
        try:
            console.print(sample_panel(get_sample(args.show)))
        except KeyError as exc:
            console.print(f"[red]{exc.args[0]}[/]")
            return 1
        return 0
    samples = filtered(args)
    if args.json:
        print(json.dumps([s.to_dict() for s in samples], indent=2, ensure_ascii=False))
        return 0
    console.print(samples_table(samples, console.width))
    console.print(f"[dim]{len(samples)} sample(s) · --show <ID> for details[/]")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("samples", help="list benchmark intents, or --show one")
    add_filter_args(p)
    p.add_argument("--show", metavar="ID", help="every field of one sample")
    p.add_argument("--json", action="store_true", help="print the filtered samples as JSON")
    p.set_defaults(func=run)
