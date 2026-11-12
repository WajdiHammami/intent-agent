"""`intent-agent bench <experiment>` — the paper experiments (eval, pure-llm, ablation, …)."""

from __future__ import annotations

import argparse

from eval.experiments import EXPERIMENTS


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("bench", help="run an experiment: " + ", ".join(EXPERIMENTS))
    experiments = p.add_subparsers(dest="experiment", metavar="<experiment>", required=True)
    for name, module in EXPERIMENTS.items():
        ep = experiments.add_parser(name, help=module.HELP)
        module.configure(ep)
        ep.set_defaults(func=module.run)
