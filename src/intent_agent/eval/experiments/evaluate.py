"""Main evaluation — the full pipeline over the benchmark (paper results table)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from intent_agent.bootstrap import simulated_pipeline
from intent_agent.config import Settings
from intent_agent.core.knowledge.graph import KnowledgeGraph
from intent_agent.eval.dataset import ALL_SAMPLES
from intent_agent.eval.experiments.common import add_sample_filters, configure_logging, connect, pick
from intent_agent.eval.harness import run_benchmark, write_json
from intent_agent.eval.metrics import aggregate, print_metric_matrix, print_results_table

NAME = "eval"
HELP = "full pipeline over the benchmark (writes transcripts + results)"


def configure(parser: argparse.ArgumentParser) -> None:
    add_sample_filters(parser)
    parser.add_argument("--compare", action="store_true",
                        help="compare with the previous results_full.json")


def run(args: argparse.Namespace, settings: Settings, llm=None) -> int:
    configure_logging(args.verbose)
    samples = pick(args, ALL_SAMPLES)
    print(f"\nIntent Agent — Batch Evaluation — {len(samples)} sample(s)")
    llm = llm or connect(settings)
    kg = KnowledgeGraph()

    outcomes = run_benchmark(
        samples, lambda s: simulated_pipeline(llm, s.scenario, kg=kg), kg, verbose=args.verbose
    )
    if not outcomes:
        print("\nAll samples failed — nothing to save.")
        return 1

    results = [o.metrics for o in outcomes]
    agg = aggregate(results)
    print_results_table(results, agg)

    out = settings.results_dir
    results_path, previous_path = out / "results_full.json", out / "results_full_old.json"
    if results_path.exists():
        shutil.copy(results_path, previous_path)
    write_json(out / "transcripts.json", [o.transcript() for o in outcomes])
    write_json(results_path, {"aggregate": agg.to_dict(), "samples": [r.summary() for r in results]})
    print(f"\nTranscripts → {out / 'transcripts.json'}\nResults     → {results_path}")

    if args.compare:
        if previous_path.exists():
            old = json.loads(previous_path.read_text())["aggregate"]
            print_metric_matrix("PREVIOUS vs NEW", {"previous": old, "new": agg.to_dict()},
                                baseline="previous")
        else:
            print("\n[compare] No previous results — skipping comparison.", file=sys.stderr)
    return 0
