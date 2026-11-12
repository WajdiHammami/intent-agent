"""
KG coverage sweep — vary the fraction of active conflict rules (0–100 %) and
measure how much the conflict metrics depend on knowledge-graph completeness.

At coverage f only the first ceil(N·f) rules of a class are active, universal
rules first (see KGConflictDetector).
"""

from __future__ import annotations

import argparse

from intent_agent.bootstrap import simulated_pipeline
from intent_agent.config import Settings
from intent_agent.core.knowledge.conflicts import KGConflictDetector
from intent_agent.core.knowledge.graph import KnowledgeGraph
from intent_agent.eval.dataset import ALL_SAMPLES, CONTRADICTORY_SUBSET
from intent_agent.eval.experiments.common import add_sample_filters, configure_logging, connect, pick
from intent_agent.eval.harness import run_benchmark, write_json
from intent_agent.eval.metrics import aggregate, print_metric_matrix

NAME = "sweep"
HELP = "conflict-rule coverage sweep over the contradictory subset"

DEFAULT_PERCENTAGES = [0, 25, 50, 75, 100]
SWEEP_ROWS = [
    ("M3_avg_turns", "M3  Avg Turns", False),
    ("M5_conflict_detection_rate", "M5  Conflict Detection Rate", True),
    ("M7_policy_validity_rate", "M7  Policy Validity Rate", True),
    ("M8_contradiction_detection_accuracy", "M8  Contradiction Det. Acc.", True),
]


def configure(parser: argparse.ArgumentParser) -> None:
    add_sample_filters(parser, with_type=False)
    parser.add_argument("--fractions", nargs="+", type=int, default=DEFAULT_PERCENTAGES,
                        metavar="PCT", help="coverage percentages, e.g. 0 25 50 75 100")
    parser.add_argument("--all-samples", action="store_true",
                        help="use all 200 intents instead of the contradictory subset")


def run(args: argparse.Namespace, settings: Settings, llm=None) -> int:
    configure_logging(args.verbose)
    samples = pick(args, ALL_SAMPLES if args.all_samples else CONTRADICTORY_SUBSET)
    percentages = sorted(set(args.fractions))
    kg = KnowledgeGraph()

    print(f"\nIntent Agent — KG Coverage Sweep: {len(samples)} samples × {percentages} %")
    for cid in sorted({s.expected_class for s in samples}):
        print(f"  {cid:<22} {len(kg.conflict_rules(cid))} rules")
    llm = llm or connect(settings)

    output: dict[str, dict] = {}
    for pct in percentages:
        print(f"\n── KG coverage: {pct}% ──")
        detector = KGConflictDetector(kg, coverage=pct / 100)
        outcomes = run_benchmark(
            samples,
            lambda s: simulated_pipeline(llm, s.scenario, kg=kg, conflict_detector=detector),
            kg, verbose=args.verbose,
        )
        if outcomes:
            results = [o.metrics for o in outcomes]
            output[str(pct)] = {"kg_fraction": pct / 100,
                                "aggregate": aggregate(results).to_dict(),
                                "samples": [r.summary() for r in results]}
    if not output:
        print("All fractions failed.")
        return 1

    print_metric_matrix("KG CONFLICT-RULE COVERAGE SWEEP",
                        {f"{p}%": d["aggregate"] for p, d in output.items()}, rows=SWEEP_ROWS)
    path = write_json(settings.results_dir / "kg_sweep_results.json", output)
    print(f"\nResults saved → {path}")
    return 0
