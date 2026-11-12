"""
Per-sample metrics (paper §IV).

  M1 classification accuracy      M5 conflict detection rate
  M2 extraction coverage          M6 proper termination (Algorithm 1)
  M3 turns to termination         M7 policy validity (M6 and full completeness)
  M4 profile completeness         M8 contradiction detection (aggregate only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.knowledge.graph import KnowledgeGraph
from core.pipeline.runner import PipelineResult
from eval.dataset.schema import EvalSample
from models.dialogue import DialogueState
from models.service import ServiceIntent


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

    def summary(self) -> dict[str, Any]:
        """Compact per-sample record used in result files."""
        return {
            "id": self.sample_id,
            "expected_class": self.expected_class,
            "predicted_class": self.predicted_class,
            "request_type": self.request_type,
            "classification_correct": self.classification_correct,
            "extraction_coverage": self.extraction_coverage,
            "turns": self.turns_to_termination,
            "profile_completeness": self.profile_completeness,
            "conflict_detection_rate": self.conflict_detection_rate,
            "proper_termination": self.proper_termination,
            "policy_valid": self.policy_valid,
            "detected_conflicts": self.detected_conflicts,
            "expected_conflicts": self.expected_conflicts,
        }


def compute_sample_metrics(
    sample_id: str,
    expected_class: str,
    expected_conflict_keywords: list[str],
    request_type: str,
    si: ServiceIntent,
    state: DialogueState,
    kg: KnowledgeGraph,
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


def evaluate_sample(sample: EvalSample, result: PipelineResult, kg: KnowledgeGraph) -> SampleResult:
    """Metrics for one pipeline run against its labeled sample."""
    return compute_sample_metrics(
        sample_id=sample.id,
        expected_class=sample.expected_class,
        expected_conflict_keywords=sample.expected_conflicts,
        request_type=sample.request_type,
        si=result.service_intent,
        state=result.state,
        kg=kg,
    )
