"""Aggregate metrics over a set of per-sample results."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from eval.metrics.sample import SampleResult


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

    # Share of non-contradictory samples that raised any conflict (lower is better)
    false_alarm_rate: float

    # Per-class breakdown
    per_class: dict[str, dict]

    # Per-request-type breakdown: seed / paraphrased / ambiguous / contradictory
    per_request_type: dict[str, dict]

    def to_dict(self) -> dict[str, Any]:
        """Aggregate metrics keyed by paper metric id (used in every result file)."""
        return {
            "n_samples": self.n_samples,
            "M1_classification_accuracy": self.classification_accuracy,
            "M2_avg_extraction_coverage": self.avg_extraction_coverage,
            "M2_std_extraction_coverage": self.std_extraction_coverage,
            "M3_avg_turns": self.avg_turns,
            "M3_std_turns": self.std_turns,
            "M4_avg_profile_completeness": self.avg_profile_completeness,
            "M5_conflict_detection_rate": self.overall_conflict_detection_rate,
            "M6_proper_termination_rate": self.proper_termination_rate,
            "M7_policy_validity_rate": self.policy_validity_rate,
            "M8_contradiction_detection_accuracy": self.contradiction_detection_accuracy,
            "false_alarm_rate": self.false_alarm_rate,
            "per_class": self.per_class,
            "per_request_type": self.per_request_type,
        }


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

    # M5 — expected conflicts that were found. Conflicts raised on samples
    # that expect none are false alarms, reported separately, never credited.
    total_expected = sum(r.expected_conflicts for r in results)
    total_detected_expected = sum(
        round(r.conflict_detection_rate * r.expected_conflicts)
        for r in results if r.expected_conflicts > 0
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

    clean = [r for r in results if r.request_type != "contradictory"]
    false_alarm_rate = (
        sum(1 for r in clean if r.detected_conflicts > 0) / len(clean) if clean else 0.0
    )

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
        false_alarm_rate=round(false_alarm_rate, 3),
        per_class=per_class,
        per_request_type=per_request_type,
    )
