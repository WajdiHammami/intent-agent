"""Pure-LLM baseline — the KG's reasoning replaced by the LLM (paper comparison)."""

from __future__ import annotations

import argparse
import json

from intent_agent.bootstrap import simulated_pipeline
from intent_agent.config import Settings
from intent_agent.core.knowledge.graph import KnowledgeGraph
from intent_agent.eval.dataset import ALL_SAMPLES, CONTRADICTORY_SUBSET
from intent_agent.eval.experiments.common import add_sample_filters, configure_logging, connect, pick
from intent_agent.eval.experiments.pure_llm.nodes import NODE_OVERRIDES
from intent_agent.eval.harness import run_benchmark, write_json
from intent_agent.eval.metrics import aggregate, print_metric_matrix, print_results_table

NAME = "pure-llm"
HELP = "Pure-LLM baseline: no KG reasoning (compare with the ablation baseline)"


def configure(parser: argparse.ArgumentParser) -> None:
    add_sample_filters(parser, with_type=False)
    parser.add_argument("--contradictory-only", action="store_true",
                        help="only the 60 contradictory intents (paper comparison)")


def run(args: argparse.Namespace, settings: Settings, llm=None) -> int:
    configure_logging(args.verbose)
    pool = CONTRADICTORY_SUBSET if args.contradictory_only else ALL_SAMPLES
    samples = pick(args, pool)
    dataset = "contradictory" if args.contradictory_only else "all"
    print(f"\nIntent Agent — Pure-LLM baseline: {len(samples)} samples ({dataset})")
    llm = llm or connect(settings)
    kg = KnowledgeGraph()  # metrics only — the pipeline does not consult it

    outcomes = run_benchmark(
        samples,
        lambda s: simulated_pipeline(llm, s.scenario, kg=kg, node_overrides=NODE_OVERRIDES),
        kg, verbose=args.verbose,
    )
    if not outcomes:
        print("All samples failed.")
        return 1
    results = [o.metrics for o in outcomes]
    agg = aggregate(results)
    print_results_table(results, agg)

    columns = {"Pure-LLM": agg.to_dict()}
    ablation = settings.results_dir / "ablation_results.json"
    if ablation.exists():
        baseline = json.loads(ablation.read_text()).get("baseline", {}).get("aggregate")
        if baseline:
            columns["KG baseline"] = baseline
    print_metric_matrix("PURE-LLM vs KG BASELINE", columns, baseline=columns.get("KG baseline"))

    path = write_json(settings.results_dir / "pure_llm_results.json", {
        "variant": "pure_llm", "dataset": dataset, "n_samples": len(results),
        "aggregate": agg.to_dict(), "samples": [r.summary() for r in results],
    })
    print(f"\nResults saved → {path}")
    return 0
