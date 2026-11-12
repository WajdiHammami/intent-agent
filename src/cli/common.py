"""Argument helpers shared by the subcommands."""

from __future__ import annotations

import argparse
import sys

from adapters.llm.ollama import ModelUnavailableError, OllamaLanguageModel
from bootstrap import create_language_model
from cli.theme import console
from config import Settings
from eval.dataset import ALL_SAMPLES, REQUEST_TYPES, EvalSample, select


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--class", dest="cls", metavar="CLASS", help="TC, MC, RC or MaC")
    parser.add_argument("--type", dest="rtype", choices=REQUEST_TYPES)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adversarial", dest="adversarial", action="store_const", const=True,
                       help="ambiguous + contradictory only")
    group.add_argument("--well-formed", dest="adversarial", action="store_const", const=False,
                       help="seed + paraphrased only")


def filtered(args: argparse.Namespace) -> list[EvalSample]:
    return select(ALL_SAMPLES, cls=args.cls, rtype=args.rtype, adversarial=args.adversarial)


def connect(settings: Settings) -> OllamaLanguageModel:
    with console.status(f"[dim]Connecting to {settings.model} @ {settings.ollama_url}…[/]"):
        try:
            return create_language_model(settings)
        except ModelUnavailableError as exc:
            console.print(f"[red]{exc}[/]")
            sys.exit(1)
