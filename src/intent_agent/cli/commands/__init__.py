"""Subcommands — each module exposes register(subparsers)."""

from intent_agent.cli.commands import ask, bench, check, graph, run, samples, selftest, shell

COMMANDS = [shell, check, selftest, samples, run, ask, bench, graph]

__all__ = ["COMMANDS"]
