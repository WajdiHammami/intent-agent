"""
Experiments — each module exposes NAME, HELP, configure(parser) and
run(args, settings) and is registered here for the CLI's ``bench`` command.
"""

from eval.experiments import ablation, evaluate, generate, kg_sweep, pure_llm

EXPERIMENTS = {m.NAME: m for m in (evaluate, pure_llm, ablation, kg_sweep, generate)}

__all__ = ["EXPERIMENTS"]
