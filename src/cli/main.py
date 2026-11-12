"""
intent-agent — command-line entry point.

    intent-agent                       interactive shell
    intent-agent check | selftest      setup check / offline self-test
    intent-agent samples | run | ask   browse, replay and try intents
    intent-agent bench <experiment>    eval, pure-llm, ablation, sweep, generate
    intent-agent graph                 LangGraph diagrams
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from cli.commands import COMMANDS
from cli.commands import shell as shell_command
from config import Settings


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("intent-agent")
    except Exception:
        return "dev"


def build_parser() -> argparse.ArgumentParser:
    env = Settings.from_env()
    p = argparse.ArgumentParser(
        prog="intent-agent",
        description="Knowledge-guided LLM: 6G service intents → validated profiles.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    p.add_argument("--model", default=env.model, help=f"Ollama model tag (default {env.model})")
    p.add_argument("--ollama-url", default=env.ollama_url, help="Ollama server URL")
    p.add_argument("--results-dir", default=str(env.results_dir),
                   help=f"experiment output folder (default {env.results_dir})")
    sub = p.add_subparsers(dest="command", metavar="<command>")
    for module in COMMANDS:
        module.register(sub)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env().with_overrides(
        model=args.model, ollama_url=args.ollama_url, results_dir=args.results_dir
    )
    func = getattr(args, "func", shell_command.run)
    try:
        return func(args, settings) or 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
