"""`intent-agent ask` — one free-text intent; you (or a simulated persona) answer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intent_agent.bootstrap import simulated_pipeline
from intent_agent.cli.common import connect
from intent_agent.cli.console_user import ConsoleUser
from intent_agent.cli.render import dialogue_lines, profile_panel
from intent_agent.cli.theme import console
from intent_agent.config import Settings
from intent_agent.core.pipeline import IntentPipeline


def run(args: argparse.Namespace, settings: Settings) -> int:
    llm = connect(settings)
    if args.scenario:
        with console.status("[dim]Running with the simulated user…[/]"):
            result = simulated_pipeline(llm, args.scenario).run(args.intent)
        console.print(dialogue_lines(result.state))
    else:
        result = IntentPipeline(llm, ConsoleUser()).run(args.intent)
    console.print(profile_panel(result.service_intent, as_json=args.json))
    if args.output:
        Path(args.output).write_text(json.dumps(result.service_intent.to_dict(), indent=2, default=str))
        console.print(f"[dim]Saved to {args.output}[/]")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("ask", help="run one free-text intent")
    p.add_argument("intent", help="the service request, in quotes")
    p.add_argument("--scenario", metavar="TEXT", help="let the LLM answer as this persona")
    p.add_argument("--json", action="store_true", help="print the profile as JSON")
    p.add_argument("--output", metavar="FILE", help="save the profile JSON to FILE")
    p.set_defaults(func=run)
