#!/usr/bin/env python3
"""
run_kg_sweep.py — KG Conflict-Rule Depth Sweep

Varies the fraction of active conflict rules from 0% to 100% and measures
how KG completeness affects conflict detection (M5) and contradiction
detection accuracy (M8).

The sweep runs exclusively on contradictory samples — those are the ones
that contain embedded conflicting requirements and are the natural test bed
for conflict-rule coverage.

Rule ordering
-------------
Rules are ranked by priority (universal rules first, then class-specific)
and enabled incrementally: at fraction f, the first ceil(N_total * f) rules
are active.  This models a KG that is being progressively built out from
the most fundamental constraints.

Output
------
  eval/kg_sweep_results.json  — per-fraction, per-sample results
  (table + summary printed to stdout)

Usage
-----
    python run_kg_sweep.py
    python run_kg_sweep.py --fractions 0 25 50 75 100
    python run_kg_sweep.py --class CF
    python run_kg_sweep.py --all-samples     # run on all samples, not just contradictory
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval.dataset import ALL_SAMPLES, CONTRADICTORY_SUBSET, EvalSample
from eval.metrics import aggregate, compute_sample_metrics
from ontology.kg import KGOntology, _UNIVERSAL_CONFLICTS
from ontology.ontology_v3 import ALL_CLASSES
from ontology.types import ConflictRule
from pipeline.dialogue_runner import DialogueRunner
from pipeline.llm_client import LLMClient
from pipeline.models import ConflictFlag, DialogueState, SlotValue

EVAL_DIR = Path(__file__).parent / "eval"
OUT_PATH = EVAL_DIR / "kg_sweep_results.json"

DEFAULT_FRACTIONS = [0, 25, 50, 75, 100]   # percentages


# ── Partial KG: limits active conflict rules ───────────────────────────────────

class PartialKGOntology(KGOntology):
    """
    KGOntology subclass that caps the number of active conflict rules.

    Rules are ordered: universal rules first (most fundamental), then
    class-specific rules.  At fraction f, only the first ceil(N*f) rules
    are evaluated in check_conflicts().

    This models incremental KG development — a partially-specified ontology
    that is missing some domain-specific conflict knowledge.
    """

    def __init__(self, classes, *, max_rules: int | None = None):
        super().__init__(classes)
        self._max_rules = max_rules   # None = no limit (full system)

    def check_conflicts(self, class_id: str, satisfied: dict[str, SlotValue]) -> list[ConflictFlag]:
        node = self.get_class(class_id)
        # Universal rules first, then class-specific — priority order
        all_rules: list[ConflictRule] = _UNIVERSAL_CONFLICTS + node.conflicts

        if self._max_rules is not None:
            all_rules = all_rules[: self._max_rules]

        flags: list[ConflictFlag] = []
        for rule in all_rules:
            subject_present = (
                rule.subject_field in satisfied
                and str(satisfied[rule.subject_field].value) == rule.subject_value
            )
            object_present = (
                rule.object_field in satisfied
                and str(satisfied[rule.object_field].value) == rule.object_value
            )
            if subject_present and object_present:
                flags.append(ConflictFlag(
                    conflict_type="kpi_conflict",
                    description=rule.message or (
                        f"{rule.subject_field}={rule.subject_value} conflicts with "
                        f"{rule.object_field}={rule.object_value}"
                    ),
                    subject_field=rule.subject_field,
                    object_field=rule.object_field,
                ))
        return flags


def total_rules_for_class(kg: KGOntology, class_id: str) -> int:
    node = kg.get_class(class_id)
    return len(_UNIVERSAL_CONFLICTS) + len(node.conflicts)


def max_rules_from_fraction(kg: KGOntology, class_id: str, fraction: float) -> int:
    """Return the number of rules to keep given a 0..1 fraction."""
    n = total_rules_for_class(kg, class_id)
    return math.ceil(n * fraction)


# ── Per-fraction eval ──────────────────────────────────────────────────────────

def run_fraction(
    samples: list[EvalSample],
    fraction: float,
    llm: LLMClient,
    base_kg: KGOntology,
    verbose: bool,
) -> list:
    """Run all samples with a KG limited to `fraction` of its conflict rules."""
    results = []

    for i, sample in enumerate(samples, 1):
        print(f"    [{i}/{len(samples)}] {sample.id}", end="", flush=True)
        try:
            # Cap rules for this sample's class
            n_rules = max_rules_from_fraction(base_kg, sample.expected_class, fraction)
            partial_kg = PartialKGOntology(ALL_CLASSES, max_rules=n_rules)

            runner = DialogueRunner(
                scenario=sample.scenario,
                llm=llm,
                kg=partial_kg,
                verbose=verbose,
            )
            si, state = runner.run(sample.intent)
            result = compute_sample_metrics(
                sample_id=sample.id,
                expected_class=sample.expected_class,
                expected_conflict_keywords=sample.expected_conflicts,
                request_type=sample.request_type,
                si=si,
                state=state,
                kg=partial_kg,
            )
            print(
                f"  M5:{result.conflict_detection_rate:.0%}"
                f"  M8:{'Y' if result.detected_conflicts > 0 else 'N'}"
                f"  turns:{result.turns_to_termination}"
            )
            results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if verbose:
                traceback.print_exc()

    return results


# ── Output helpers ─────────────────────────────────────────────────────────────

def _agg_dict(agg) -> dict:
    return {
        "n_samples":                          agg.n_samples,
        "M1_classification_accuracy":         agg.classification_accuracy,
        "M2_avg_extraction_coverage":         agg.avg_extraction_coverage,
        "M3_avg_turns":                       agg.avg_turns,
        "M5_conflict_detection_rate":         agg.overall_conflict_detection_rate,
        "M7_policy_validity_rate":            agg.policy_validity_rate,
        "M8_contradiction_detection_accuracy": agg.contradiction_detection_accuracy,
    }


def print_sweep_table(sweep_aggs: dict[int, object]) -> None:
    fracs = sorted(sweep_aggs.keys())
    col_w = 10

    print("\n" + "=" * (44 + col_w * len(fracs)))
    print("KG CONFLICT-RULE DEPTH SWEEP")
    print("=" * (44 + col_w * len(fracs)))
    print(f"{'KG coverage →':<44}" + "".join(f"{f:>9}%" for f in fracs))
    print("-" * (44 + col_w * len(fracs)))

    metrics = [
        ("M3_avg_turns",                         "M3  Avg Turns",                False),
        ("M5_conflict_detection_rate",           "M5  Conflict Detection Rate",  True),
        ("M7_policy_validity_rate",              "M7  Policy Validity Rate",     True),
        ("M8_contradiction_detection_accuracy",  "M8  Contradiction Det. Acc.",  True),
    ]
    agg_dicts = {f: _agg_dict(agg) for f, agg in sweep_aggs.items()}

    for key, label, is_rate in metrics:
        row = f"  {label:<42}"
        for f in fracs:
            val = agg_dicts[f].get(key)
            if isinstance(val, float):
                fmt = ".1%" if is_rate else ".2f"
                row += f"{val:{fmt}}".rjust(col_w)
            elif isinstance(val, int):
                row += str(val).rjust(col_w)
            else:
                row += "—".rjust(col_w)
        print(row)

    print("=" * (44 + col_w * len(fracs)))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Intent Agent — KG conflict-rule depth sweep")
    parser.add_argument("--model", default="models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")
    parser.add_argument(
        "--fractions", nargs="+", type=int, default=DEFAULT_FRACTIONS,
        metavar="PCT",
        help="KG coverage percentages to test, e.g. --fractions 0 25 50 75 100",
    )
    parser.add_argument("--class", dest="filter_class", default=None, metavar="PREFIX")
    parser.add_argument("--ids", nargs="+", default=None, metavar="ID")
    parser.add_argument(
        "--all-samples", action="store_true",
        help="Run on all samples instead of only contradictory ones",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # The sweep runs on the 40-sample contradictory subset by default;
    # --all-samples widens it to the full 72-sample set.
    all_pool = list(ALL_SAMPLES)
    contradictory_pool = list(CONTRADICTORY_SUBSET)

    # By default, run on contradictory samples only
    if args.all_samples:
        samples = all_pool
    else:
        samples = contradictory_pool

    if args.filter_class:
        samples = [s for s in samples if s.id.startswith(args.filter_class)]
    if args.ids:
        samples = [s for s in samples if s.id in args.ids]

    if not samples:
        print("No samples matched — nothing to run.")
        sys.exit(1)

    fractions_float = [f / 100.0 for f in sorted(set(args.fractions))]
    n_runs = len(samples) * len(fractions_float)

    sample_label = "all" if args.all_samples else "contradictory only"
    print(f"\nIntent Agent — KG Conflict-Rule Depth Sweep")
    print(f"  Dataset  : {len(samples)} samples ({sample_label})")
    print(f"  Fractions: {[f'{int(f*100)}%' for f in fractions_float]}")
    print(f"  Total    : {n_runs} inference runs\n")

    # Show rule counts per class
    base_kg = KGOntology(ALL_CLASSES)
    class_ids = list({s.expected_class for s in samples})
    print("  Conflict rules per class:")
    for cid in sorted(class_ids):
        n = total_rules_for_class(base_kg, cid)
        print(f"    {cid:<25} {n} rules")
    print()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print("Loading LLM model ...")
    llm = LLMClient(model_path=str(model_path))

    sweep_aggs:    dict[int, object] = {}
    sweep_results: dict[int, list]   = {}

    for frac in fractions_float:
        pct = int(frac * 100)
        print(f"\n── KG coverage: {pct}% ──")
        results = run_fraction(samples, frac, llm, base_kg, args.verbose)
        if results:
            agg = aggregate(results)
            sweep_aggs[pct]    = agg
            sweep_results[pct] = results

    if not sweep_aggs:
        print("All fractions failed.")
        sys.exit(1)

    print_sweep_table(sweep_aggs)

    # Save JSON
    output = {
        str(pct): {
            "kg_fraction": pct / 100.0,
            "aggregate": _agg_dict(agg),
            "samples": [
                {
                    "id":                      r.sample_id,
                    "expected_class":          r.expected_class,
                    "request_type":            r.request_type,
                    "conflict_detection_rate": r.conflict_detection_rate,
                    "detected_conflicts":      r.detected_conflicts,
                    "expected_conflicts":      r.expected_conflicts,
                    "turns":                   r.turns_to_termination,
                    "policy_valid":            r.policy_valid,
                }
                for r in sweep_results[pct]
            ],
        }
        for pct, agg in sweep_aggs.items()
    }
    EVAL_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
