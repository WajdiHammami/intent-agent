"""Component ablation — turn extraction and KG conflict checking on and off."""

from __future__ import annotations

import argparse
import json

from bootstrap import simulated_pipeline
from config import Settings
from core.knowledge.graph import KnowledgeGraph
from eval.dataset import ALL_SAMPLES
from eval.experiments.ablation.variants import VARIANTS
from eval.experiments.common import add_sample_filters, configure_logging, connect, pick
from eval.harness import run_benchmark, write_json
from eval.metrics import aggregate, print_metric_matrix

NAME = "ablation"
HELP = "component ablation (extraction / KG conflicts / LLM conflicts)"


def configure(parser: argparse.ArgumentParser) -> None:
    add_sample_filters(parser)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                        metavar="VARIANT", help=f"default: all of {list(VARIANTS)}")


def run(args: argparse.Namespace, settings: Settings, llm=None) -> int:
    configure_logging(args.verbose)
    samples = pick(args, ALL_SAMPLES)
    print(f"\nIntent Agent — Ablation Study: {len(samples)} samples × {len(args.variants)} variants")
    llm = llm or connect(settings)
    kg = KnowledgeGraph()

    output: dict[str, dict] = {}
    for name in args.variants:
        variant = VARIANTS[name]
        print(f"\n── Variant: {variant.label} ──")
        options = variant.pipeline_options(llm, kg)
        outcomes = run_benchmark(
            samples, lambda s: simulated_pipeline(llm, s.scenario, **options), kg,
            verbose=args.verbose,
        )
        if outcomes:
            results = [o.metrics for o in outcomes]
            output[name] = {"aggregate": aggregate(results).to_dict(),
                            "samples": [r.summary() for r in results]}
    if not output:
        print("All variants failed.")
        return 1

    path = settings.results_dir / "ablation_results.json"
    if path.exists():  # keep the file cumulative across partial runs
        output = {**json.loads(path.read_text()), **output}
    write_json(path, output)
    print(f"\nResults saved → {path}")

    columns = {VARIANTS[v].label: output[v]["aggregate"] for v in args.variants if v in output}
    baseline = output.get("baseline", {}).get("aggregate")
    print_metric_matrix("ABLATION STUDY RESULTS", columns, baseline=baseline)
    return 0
