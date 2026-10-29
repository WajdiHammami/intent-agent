"""
Evaluation Metrics

Computes per-sample and aggregate metrics for the paper evaluation.

Paper metrics (§IV):
  M1 — Intent Classification Accuracy
  M2 — Slot Extraction Coverage     (no-question fraction)
  M3 — Clarification Efficiency     (avg turns to valid profile)
  M4 — Profile Completeness         (mandatory slots filled at output)
  M5 — Conflict Detection Rate      (expected conflicts actually triggered)
  M6 — Proper Termination Rate      (Algorithm 1 reached, not turn-limit)

Step 3 additions:
  M7 — Policy Validity Rate         (proper_termination AND profile_completeness == 1.0)
  M8 — Contradiction Detection Accuracy  (over contradictory samples only)
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from ontology.kg import KGOntology
from pipeline.models import DialogueState, ServiceIntent


@dataclass
class SampleResult:
    """All metrics for a single dialogue run."""
    sample_id: str
    expected_class: str
    predicted_class: str

    # M1
    classification_correct: bool

    # M2
    extraction_coverage: float        # 0–1, higher = fewer questions needed

    # M3
    turns_to_termination: int

    # M4
    profile_completeness: float       # mandatory_filled / mandatory_total

    # M5
    expected_conflicts: int           # number of conflict keywords expected
    detected_conflicts: int           # number actually triggered
    resolved_conflicts: int
    conflict_detection_rate: float    # detected / expected (1.0 if expected=0)

    # M6
    proper_termination: bool          # True = Algorithm 1, False = turn limit

    # M7
    policy_valid: bool                # proper_termination AND profile_completeness == 1.0

    # Extras
    request_type: str                 # seed | paraphrased | ambiguous | contradictory
    mandatory_filled: int
    mandatory_total: int
    output_json: dict


@dataclass
class AggregateMetrics:
    """Aggregate metrics over all samples."""
    n_samples: int

    # M1
    classification_accuracy: float

    # M2
    avg_extraction_coverage: float
    std_extraction_coverage: float

    # M3
    avg_turns: float
    std_turns: float

    # M4
    avg_profile_completeness: float

    # M5
    overall_conflict_detection_rate: float

    # M6
    proper_termination_rate: float

    # M7
    policy_validity_rate: float

    # M8 — contradictory samples only; 1.0 if none present
    contradiction_detection_accuracy: float

    # Per-class breakdown
    per_class: dict[str, dict]

    # Per-request-type breakdown: seed / paraphrased / ambiguous / contradictory
    per_request_type: dict[str, dict]


def _get_slot_value(state: DialogueState, field_name: str) -> str:
    """Safely retrieve a satisfied slot's string value."""
    sv = state.satisfied.get(field_name)
    if sv is None:
        return ""
    return str(sv.value) if sv.value is not None else ""


def compute_sample_metrics(
    sample_id: str,
    expected_class: str,
    expected_conflict_keywords: list[str],
    request_type: str,
    si: ServiceIntent,
    state: DialogueState,
    kg: KGOntology,
) -> SampleResult:
    """Compute all metrics for a single completed dialogue."""
    predicted_class = state.current_class or ""
    class_id = predicted_class

    # M1 — classification accuracy
    classification_correct = (predicted_class == expected_class)

    # M2 — extraction coverage
    extraction_cov = state.extraction_coverage()

    # M3 — turns
    turns = state.turn_count

    # M4 — profile completeness
    try:
        mandatory = kg.get_mandatory_slot_names(class_id)
        mandatory_total = len(mandatory)
        mandatory_filled = len(mandatory & set(state.satisfied.keys()))
        # vertical_domain is always required
        if "vertical_domain" in state.satisfied:
            mandatory_filled = min(mandatory_filled + 1, mandatory_total + 1)
            mandatory_total += 1
        profile_completeness = mandatory_filled / mandatory_total if mandatory_total > 0 else 1.0
    except KeyError:
        mandatory_filled = 0
        mandatory_total = 0
        profile_completeness = 0.0

    # M5 — conflict detection
    n_expected = len(expected_conflict_keywords)
    n_detected = len(state.conflicts)
    n_resolved = sum(1 for c in state.conflicts if c.resolved)

    # Check which expected keywords were actually found in conflict descriptions
    detected_expected = 0
    for kw in expected_conflict_keywords:
        if any(kw.lower() in c.description.lower() for c in state.conflicts):
            detected_expected += 1
    conflict_detection_rate = (
        detected_expected / n_expected if n_expected > 0 else 1.0
    )

    # M6 — proper termination (Algorithm 1 vs turn limit)
    try:
        proper_termination = kg.check_termination(class_id, state)
    except Exception:
        proper_termination = False

    # M7 — policy validity
    policy_valid = proper_termination and profile_completeness >= 1.0

    return SampleResult(
        sample_id=sample_id,
        expected_class=expected_class,
        predicted_class=predicted_class,
        classification_correct=classification_correct,
        extraction_coverage=round(extraction_cov, 3),
        turns_to_termination=turns,
        profile_completeness=round(profile_completeness, 3),
        expected_conflicts=n_expected,
        detected_conflicts=n_detected,
        resolved_conflicts=n_resolved,
        conflict_detection_rate=round(conflict_detection_rate, 3),
        proper_termination=proper_termination,
        policy_valid=policy_valid,
        request_type=request_type,
        mandatory_filled=mandatory_filled,
        mandatory_total=mandatory_total,
        output_json=si.to_dict(),
    )


def aggregate(results: list[SampleResult]) -> AggregateMetrics:
    """Compute aggregate metrics from a list of sample results."""
    n = len(results)
    if n == 0:
        raise ValueError("No results to aggregate")

    # M1
    classification_accuracy = sum(r.classification_correct for r in results) / n

    # M2
    coverages = [r.extraction_coverage for r in results]
    avg_cov = statistics.mean(coverages)
    std_cov = statistics.stdev(coverages) if n > 1 else 0.0

    # M3
    turns = [r.turns_to_termination for r in results]
    avg_turns = statistics.mean(turns)
    std_turns = statistics.stdev(turns) if n > 1 else 0.0

    # M4
    avg_completeness = statistics.mean(r.profile_completeness for r in results)

    # M5
    total_expected = sum(r.expected_conflicts for r in results)
    total_detected_expected = sum(
        r.detected_conflicts if r.expected_conflicts == 0
        else int(r.conflict_detection_rate * r.expected_conflicts)
        for r in results
    )
    overall_conflict_rate = (
        total_detected_expected / total_expected if total_expected > 0 else 1.0
    )

    # M6
    proper_term_rate = sum(r.proper_termination for r in results) / n

    # M7 — policy validity rate
    policy_validity_rate = sum(r.policy_valid for r in results) / n

    # M8 — contradiction detection accuracy (contradictory samples only)
    contradictory = [r for r in results if r.request_type == "contradictory"]
    if contradictory:
        contradiction_detection_accuracy = sum(
            1 for r in contradictory if r.detected_conflicts > 0
        ) / len(contradictory)
    else:
        contradiction_detection_accuracy = 1.0

    # Per-class breakdown
    classes = sorted(set(r.expected_class for r in results))
    per_class: dict[str, dict] = {}
    for cls in classes:
        cls_results = [r for r in results if r.expected_class == cls]
        nc = len(cls_results)
        per_class[cls] = {
            "n": nc,
            "classification_accuracy": round(
                sum(r.classification_correct for r in cls_results) / nc, 3
            ),
            "avg_extraction_coverage": round(
                statistics.mean(r.extraction_coverage for r in cls_results), 3
            ),
            "avg_turns": round(
                statistics.mean(r.turns_to_termination for r in cls_results), 2
            ),
            "avg_profile_completeness": round(
                statistics.mean(r.profile_completeness for r in cls_results), 3
            ),
            "proper_termination_rate": round(
                sum(r.proper_termination for r in cls_results) / nc, 3
            ),
            "policy_validity_rate": round(
                sum(r.policy_valid for r in cls_results) / nc, 3
            ),
        }

    # Per-request-type breakdown
    request_types = ["seed", "paraphrased", "ambiguous", "contradictory"]
    per_request_type: dict[str, dict] = {}
    for rtype in request_types:
        rtype_results = [r for r in results if r.request_type == rtype]
        if not rtype_results:
            continue
        nr = len(rtype_results)
        per_request_type[rtype] = {
            "n": nr,
            "classification_accuracy": round(
                sum(r.classification_correct for r in rtype_results) / nr, 3
            ),
            "policy_validity_rate": round(
                sum(r.policy_valid for r in rtype_results) / nr, 3
            ),
            "avg_turns": round(
                statistics.mean(r.turns_to_termination for r in rtype_results), 2
            ),
        }

    return AggregateMetrics(
        n_samples=n,
        classification_accuracy=round(classification_accuracy, 3),
        avg_extraction_coverage=round(avg_cov, 3),
        std_extraction_coverage=round(std_cov, 3),
        avg_turns=round(avg_turns, 2),
        std_turns=round(std_turns, 2),
        avg_profile_completeness=round(avg_completeness, 3),
        overall_conflict_detection_rate=round(overall_conflict_rate, 3),
        proper_termination_rate=round(proper_term_rate, 3),
        policy_validity_rate=round(policy_validity_rate, 3),
        contradiction_detection_accuracy=round(contradiction_detection_accuracy, 3),
        per_class=per_class,
        per_request_type=per_request_type,
    )


def print_results_table(results: list[SampleResult], agg: AggregateMetrics) -> None:
    """Print a formatted results table to stdout."""

    # ── Per-sample table ──────────────────────────────────────────────────────
    W = 120
    print("\n" + "=" * W)
    print(
        f"{'ID':<12} {'Expected':<22} {'Predicted':<22} {'Cls':>4} "
        f"{'Cov':>6} {'Turns':>6} {'Compl':>6} {'ConflDR':>8} "
        f"{'ProperT':>8} {'PolV':>5} {'Type':<14}"
    )
    print("-" * W)
    for r in results:
        cls_ok  = "Y" if r.classification_correct else "N"
        pt_ok   = "Y" if r.proper_termination else "N"
        pv_ok   = "Y" if r.policy_valid else "N"
        print(
            f"{r.sample_id:<12} {r.expected_class:<22} {r.predicted_class:<22} "
            f"{cls_ok:>4} {r.extraction_coverage:>6.2f} {r.turns_to_termination:>6} "
            f"{r.profile_completeness:>6.2f} {r.conflict_detection_rate:>8.2f} "
            f"{pt_ok:>8} {pv_ok:>5} {r.request_type:<14}"
        )

    # ── Step 1–2 aggregate (original metrics) ─────────────────────────────────
    print("=" * W)
    print(f"\n{'AGGREGATE — Steps 1 & 2 (Core Metrics)'}")
    print(f"  Samples                         : {agg.n_samples}")
    print(f"  M1  Classification Accuracy     : {agg.classification_accuracy:.1%}")
    print(f"  M2  Extraction Coverage         : {agg.avg_extraction_coverage:.1%}  (±{agg.std_extraction_coverage:.2f})")
    print(f"  M3  Avg Turns to Termination    : {agg.avg_turns:.2f}  (±{agg.std_turns:.2f})")
    print(f"  M4  Avg Profile Completeness    : {agg.avg_profile_completeness:.1%}")
    print(f"  M5  Conflict Detection Rate     : {agg.overall_conflict_detection_rate:.1%}")
    print(f"  M6  Proper Termination Rate     : {agg.proper_termination_rate:.1%}")

    # ── Step 3 aggregate (new metrics) ────────────────────────────────────────
    print(f"\n{'AGGREGATE — Step 3 (Policy & Contradiction Metrics)'}")
    print(f"  M7  Policy Validity Rate        : {agg.policy_validity_rate:.1%}")
    print(f"  M8  Contradiction Det. Accuracy : {agg.contradiction_detection_accuracy:.1%}  (contradictory samples only)")

    # ── Per-request-type breakdown ────────────────────────────────────────────
    print(f"\n{'Per-request-type breakdown':}")
    rt_header = (
        f"  {'Type':<16} {'n':>3} {'ClsAcc':>7} {'PolValid':>9} "
        f"{'AvgTurns':>9}"
    )
    print(rt_header)
    print("  " + "-" * (len(rt_header) - 2))
    for rtype, m in agg.per_request_type.items():
        print(
            f"  {rtype:<16} {m['n']:>3} "
            f"{m['classification_accuracy']:>7.1%} "
            f"{m['policy_validity_rate']:>9.1%} "
            f"{m['avg_turns']:>9.2f}"
        )

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print(f"\n{'Per-class breakdown':}")
    cls_header = (
        f"  {'Class':<24} {'n':>3} {'ClsAcc':>7} {'Coverage':>9} "
        f"{'Turns':>6} {'Compl':>6} {'ProperT':>8} {'PolValid':>9}"
    )
    print(cls_header)
    print("  " + "-" * (len(cls_header) - 2))
    for cls, m in agg.per_class.items():
        print(
            f"  {cls:<24} {m['n']:>3} "
            f"{m['classification_accuracy']:>7.1%} "
            f"{m['avg_extraction_coverage']:>9.2f} "
            f"{m['avg_turns']:>6.2f} "
            f"{m['avg_profile_completeness']:>6.2f} "
            f"{m['proper_termination_rate']:>8.1%} "
            f"{m['policy_validity_rate']:>9.1%}"
        )
    print()
