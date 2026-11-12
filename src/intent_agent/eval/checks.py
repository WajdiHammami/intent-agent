"""Compare one pipeline run with its labels — the ✓/✗ checks shown by the CLI."""

from __future__ import annotations

from dataclasses import dataclass

from intent_agent.core.pipeline.runner import PipelineResult
from intent_agent.eval.dataset.schema import EvalSample
from intent_agent.eval.metrics.sample import SampleResult


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _same(got: object, want: object) -> bool:
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return abs(float(got) - float(want)) < 1e-6
    return str(got).lower() == str(want).lower()


def label_checks(sample: EvalSample, run: PipelineResult, metrics: SampleResult) -> list[Check]:
    values = {k: sv.value for k, sv in run.state.satisfied.items()}
    values.update(run.service_intent.class_slots)

    checks = [Check("class", metrics.classification_correct,
                    f"expected {sample.expected_class}, got {metrics.predicted_class}")]
    for field, want in sample.expected_slots.items():
        got = values.get(field)
        checks.append(Check(f"slot {field}", _same(got, want), f"expected {want}, got {got}"))
    if sample.request_type == "contradictory":
        checks.append(Check(
            "conflict detected", metrics.detected_conflicts > 0,
            f"expected {' + '.join(sample.expected_conflicts)}, detected {metrics.detected_conflicts}",
        ))
    if sample.gold_deployment_mode:
        got = run.service_intent.class_slots.get("deployment_mode")
        checks.append(Check("deployment mode", got == sample.gold_deployment_mode,
                            f"expected {sample.gold_deployment_mode}, got {got}"))
    checks.append(Check(
        "policy valid", metrics.policy_valid,
        f"completeness {metrics.profile_completeness:.0%}, "
        f"proper termination {metrics.proper_termination}",
    ))
    return checks


def run_and_check(pipeline, sample: EvalSample, kg) -> tuple[PipelineResult, SampleResult, list[Check]]:
    """Run one sample through ``pipeline`` and grade it against its labels."""
    from intent_agent.eval.metrics.sample import evaluate_sample

    run = pipeline.run(sample.intent)
    metrics = evaluate_sample(sample, run, kg)
    return run, metrics, label_checks(sample, run, metrics)
