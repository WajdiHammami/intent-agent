"""`intent-agent chat` (and no command) — the interactive shell."""

from __future__ import annotations

import argparse

from config import Settings


def run(args: argparse.Namespace, settings: Settings) -> int:
    from cli.shell.app import Shell  # prompt_toolkit only when needed
    Shell(settings).loop()
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("chat", help="interactive shell (also the default)").set_defaults(func=run)
