#!/usr/bin/env python3
"""
run_eval.py — Batch Evaluation Runner

Runs the full pipeline over a labeled dataset and writes:
  eval/transcripts.json   — per-sample dialogue + all metrics
  eval/results_full.json  — aggregate + per-sample metric summary

Usage
-----
    python run_eval.py                          # run all 72 samples
    python run_eval.py --class TC               # only TimeCritical samples
    python run_eval.py --type contradictory     # only contradictory samples
    python run_eval.py --ids TC-01 MC-03        # specific sample IDs
    python run_eval.py --verbose                # print dialogue turns
    python run_eval.py --compare                # diff vs previous results_full.json

Filters can be combined, e.g.:
    python run_eval.py --class RC --type seed

Output files
------------
  eval/transcripts.json
      List of records, one per sample:
        sample_id, request_type, expected_class, predicted_class,
        intent, scenario, dialogue, output, metrics (M1–M8)

  eval/results_full.json
      {
        "aggregate": { M1..M8, per_class, per_request_type },
        "samples":   [ per-sample metric summary ]
      }
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval.dataset import ALL_SAMPLES, EvalSample
from eval.metrics import aggregate, compute_sample_metrics, print_results_table
from ontology.kg import KGOntology
from ontology.ontology_v3 import ALL_CLASSES
from pipeline.dialogue_runner import DialogueRunner
from pipeline.llm_client import LLMClient

EVAL_DIR          = Path(__file__).parent / "eval"
TRANSCRIPTS_PATH  = EVAL_DIR / "transcripts.json"
RESULTS_FULL_PATH = EVAL_DIR / "results_full.json"
OLD_RESULTS_PATH  = EVAL_DIR / "results_full_old.json"


# ── Per-sample helpers ─────────────────────────────────────────────────────────

def run_sample(
    sample: EvalSample,
    llm: LLMClient,
    kg: KGOntology,
    verbose: bool = False,
):
    """Run the pipeline for one sample. Returns (ServiceIntent, DialogueState)."""
    runner = DialogueRunner(
        scenario=sample.scenario,
        llm=llm,
        kg=kg,
        verbose=verbose,
    )
    return runner.run(sample.intent)


def build_transcript(sample: EvalSample, si, state, result) -> dict:
    """Build one record for transcripts.json."""
    dialogue = [
        {"role": t.role, "content": t.content}
        for t in state.turn_history
    ]
    return {
        "sample_id":      sample.id,
        "request_type":   sample.request_type,
        "scenario_label": sample.scenario_label,
        "expected_class": sample.expected_class,
        "predicted_class": result.predicted_class,
        "intent":   sample.intent,
        "scenario": sample.scenario,
        "dialogue": dialogue,
        "output":   si.to_dict(),
        "metrics": {
            "M1_classification_correct":    result.classification_correct,
            "M2_extraction_coverage":       result.extraction_coverage,
            "M3_turns":                     result.turns_to_termination,
            "M4_profile_completeness":      result.profile_completeness,
            "M5_conflict_detection_rate":   result.conflict_detection_rate,
            "M6_proper_termination":        result.proper_termination,
            "M7_policy_valid":              result.policy_valid,
            "M8_contradiction_detected": (
                result.detected_conflicts > 0
                if result.request_type == "contradictory" else None
            ),
        },
    }


def build_results_full(results, agg) -> dict:
    """Build the payload for results_full.json."""
    return {
        "aggregate": {
            "n_samples":                          agg.n_samples,
            "M1_classification_accuracy":         agg.classification_accuracy,
            "M2_avg_extraction_coverage":         agg.avg_extraction_coverage,
            "M2_std_extraction_coverage":         agg.std_extraction_coverage,
            "M3_avg_turns":                       agg.avg_turns,
            "M3_std_turns":                       agg.std_turns,
            "M4_avg_profile_completeness":        agg.avg_profile_completeness,
            "M5_conflict_detection_rate":         agg.overall_conflict_detection_rate,
            "M6_proper_termination_rate":         agg.proper_termination_rate,
            "M7_policy_validity_rate":            agg.policy_validity_rate,
            "M8_contradiction_detection_accuracy": agg.contradiction_detection_accuracy,
            "per_class":        agg.per_class,
            "per_request_type": agg.per_request_type,
        },
        "samples": [
            {
                "id":                    r.sample_id,
                "expected_class":        r.expected_class,
                "predicted_class":       r.predicted_class,
                "request_type":          r.request_type,
                "classification_correct": r.classification_correct,
                "extraction_coverage":   r.extraction_coverage,
                "turns":                 r.turns_to_termination,
                "profile_completeness":  r.profile_completeness,
                "conflict_detection_rate": r.conflict_detection_rate,
                "proper_termination":    r.proper_termination,
                "policy_valid":          r.policy_valid,
                "detected_conflicts":    r.detected_conflicts,
            }
            for r in results
        ],
    }


# ── Comparison helper ──────────────────────────────────────────────────────────

def print_comparison(new_agg, old_path: Path) -> None:
    """Print side-by-side comparison of new vs previous aggregate metrics."""
    if not old_path.exists():
        print("\n[compare] No previous results found — skipping comparison.")
        return

    old = json.loads(old_path.read_text())["aggregate"]
    new = {
        "n_samples":                          new_agg.n_samples,
        "M1_classification_accuracy":         new_agg.classification_accuracy,
        "M5_conflict_detection_rate":         new_agg.overall_conflict_detection_rate,
        "M7_policy_validity_rate":            new_agg.policy_validity_rate,
        "M8_contradiction_detection_accuracy": new_agg.contradiction_detection_accuracy,
        "M3_avg_turns":                       new_agg.avg_turns,
    }

    print("\n" + "=" * 70)
    print("COMPARISON: Previous results vs New results")
    print("=" * 70)
    print(f"{'Metric':<45} {'Old':>8} {'New':>8} {'Δ':>8}")
    print("-" * 70)

    keys = [
        ("n_samples",                           "N samples",                       False),
        ("M1_classification_accuracy",          "M1  Classification Accuracy",     True),
        ("M5_conflict_detection_rate",          "M5  Conflict Detection Rate",     True),
        ("M7_policy_validity_rate",             "M7  Policy Validity Rate",        True),
        ("M8_contradiction_detection_accuracy", "M8  Contradiction Det. Accuracy", True),
        ("M3_avg_turns",                        "M3  Avg Turns",                   False),
    ]
    for key, label, is_rate in keys:
        o = old.get(key, "—")
        n = new.get(key, "—")
        if isinstance(o, float) and isinstance(n, float):
            fmt   = ".1%" if is_rate else ".2f"
            dfmt  = "+.1%" if is_rate else "+.2f"
            fo, fn = f"{o:{fmt}}", f"{n:{fmt}}"
            delta  = f"{(n - o):{dfmt}}"
        elif isinstance(o, int) and isinstance(n, int):
            fo, fn = str(o), str(n)
            delta  = f"{n - o:+d}"
        else:
            fo = fn = delta = "—"
        print(f"  {label:<43} {fo:>8} {fn:>8} {delta:>8}")
    print("=" * 70)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Intent Agent — batch evaluation")
    parser.add_argument(
        "--model",
        default="models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--class", dest="filter_class", default=None,
        metavar="PREFIX",
        help="Filter by sample ID prefix: TC, MC, RC, DC, CF, PAR, AMB",
    )
    parser.add_argument(
        "--type", dest="filter_type", default=None,
        choices=["seed", "paraphrased", "ambiguous", "contradictory"],
        help="Filter by request type",
    )
    parser.add_argument(
        "--ids", nargs="+", default=None,
        metavar="ID",
        help="Run specific sample IDs only (e.g. TC-01 MC-03)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Print comparison against the previous results_full.json",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print dialogue turns for each sample",
    )
    args = parser.parse_args()

    # Main 72-sample evaluation set
    samples = list(ALL_SAMPLES)
    if args.filter_class:
        samples = [s for s in samples if s.id.startswith(args.filter_class)]
    if args.filter_type:
        samples = [s for s in samples if s.request_type == args.filter_type]
    if args.ids:
        samples = [s for s in samples if s.id in args.ids]

    if not samples:
        print("No samples matched the filter — nothing to run.")
        sys.exit(1)

    print(f"\nIntent Agent — Batch Evaluation — {len(samples)} sample(s)")

    # Check model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Pass the correct path with:  --model /path/to/model.gguf")
        sys.exit(1)

    print("Loading LLM model ...")
    llm = LLMClient(model_path=str(model_path))
    kg  = KGOntology(ALL_CLASSES)

    results:     list = []
    transcripts: list = []

    for i, sample in enumerate(samples, 1):
        print(
            f"\n[{i}/{len(samples)}] {sample.id}"
            f"  ({sample.expected_class}, {sample.request_type})"
        )
        print(f"  Intent: {sample.intent[:80]}...")

        try:
            si, state = run_sample(sample, llm, kg, verbose=args.verbose)
            result = compute_sample_metrics(
                sample_id=sample.id,
                expected_class=sample.expected_class,
                expected_conflict_keywords=sample.expected_conflicts,
                request_type=sample.request_type,
                si=si,
                state=state,
                kg=kg,
            )

            cls_sym = "Y" if result.classification_correct else "N"
            pt_sym  = "Y" if result.proper_termination else "N"
            pv_sym  = "Y" if result.policy_valid else "N"
            cf_sym  = (
                f"{result.detected_conflicts}/{result.expected_conflicts}"
                if result.expected_conflicts > 0 else "-"
            )
            print(
                f"  Class:{cls_sym} {result.predicted_class:<22} "
                f"Cov:{result.extraction_coverage:.0%}  "
                f"Turns:{result.turns_to_termination}  "
                f"Compl:{result.profile_completeness:.0%}  "
                f"Conflicts:{cf_sym}  ProperT:{pt_sym}  PolV:{pv_sym}"
            )
            results.append(result)
            transcripts.append(build_transcript(sample, si, state, result))

        except Exception as exc:
            print(f"  ERROR: {exc}")
            if args.verbose:
                traceback.print_exc()

    if not results:
        print("\nAll samples failed — nothing to save.")
        sys.exit(1)

    agg = aggregate(results)
    print_results_table(results, agg)

    # Save outputs — back up previous results first if they exist
    if RESULTS_FULL_PATH.exists():
        import shutil
        shutil.copy(RESULTS_FULL_PATH, OLD_RESULTS_PATH)

    TRANSCRIPTS_PATH.write_text(json.dumps(transcripts, indent=2))
    print(f"\nTranscripts  → {TRANSCRIPTS_PATH}")

    RESULTS_FULL_PATH.write_text(json.dumps(build_results_full(results, agg), indent=2))
    print(f"Results      → {RESULTS_FULL_PATH}")

    if args.compare:
        print_comparison(agg, OLD_RESULTS_PATH)


if __name__ == "__main__":
    main()
