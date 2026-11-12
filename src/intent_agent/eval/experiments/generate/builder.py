"""Turn an LLM-proposed variant into a labeled EvalSample (labels need human review)."""

from __future__ import annotations

from intent_agent.eval.dataset.schema import EvalSample

ID_PREFIX = {"paraphrased": "PAR", "ambiguous": "AMB", "contradictory": "CF"}


def next_id(seed_id: str, variant_type: str, index: int, taken: set[str]) -> str:
    """e.g. TC-01 → TC-PAR-01, with a numeric suffix if that id is taken."""
    class_prefix = seed_id.split("-")[0] if "-" in seed_id else "XX"
    base = candidate = f"{class_prefix}-{ID_PREFIX.get(variant_type, 'GEN')}-{index:02d}"
    n = 1
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


def build_sample(seed: EvalSample, variant: dict, variant_type: str, new_id: str) -> EvalSample:
    conflicts = []
    if variant_type == "contradictory" and variant.get("conflict_description"):
        conflicts = [variant["conflict_description"]]
    return EvalSample(
        id=new_id,
        intent=variant["intent"],
        scenario=variant.get("scenario", seed.scenario),
        expected_class=seed.expected_class,
        expected_slots={},                      # needs human review
        expected_conflicts=conflicts,           # needs human review
        request_type=variant_type,
        scenario_label=seed.scenario_label,
        gold_deployment_mode=seed.gold_deployment_mode,
    )
