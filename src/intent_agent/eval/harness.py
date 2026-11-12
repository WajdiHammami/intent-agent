"""Run a pipeline over benchmark samples and collect metrics — shared by all experiments."""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from intent_agent.core.knowledge.graph import KnowledgeGraph
from intent_agent.core.pipeline.runner import IntentPipeline, PipelineResult
from intent_agent.eval.dataset.schema import EvalSample
from intent_agent.eval.metrics.sample import SampleResult, evaluate_sample

log = logging.getLogger("intent_agent.eval")

PipelineFactory = Callable[[EvalSample], IntentPipeline]


@dataclass
class Outcome:
    sample: EvalSample
    run: PipelineResult
    metrics: SampleResult

    def transcript(self) -> dict[str, Any]:
        m = self.metrics
        return {
            "sample_id": self.sample.id,
            "request_type": self.sample.request_type,
            "scenario_label": self.sample.scenario_label,
            "expected_class": self.sample.expected_class,
            "predicted_class": m.predicted_class,
            "intent": self.sample.intent,
            "scenario": self.sample.scenario,
            "dialogue": [{"role": t.role, "content": t.content} for t in self.run.state.turn_history],
            "output": self.run.service_intent.to_dict(),
            "metrics": {
                "M1_classification_correct": m.classification_correct,
                "M2_extraction_coverage": m.extraction_coverage,
                "M3_turns": m.turns_to_termination,
                "M4_profile_completeness": m.profile_completeness,
                "M5_conflict_detection_rate": m.conflict_detection_rate,
                "M6_proper_termination": m.proper_termination,
                "M7_policy_valid": m.policy_valid,
                "M8_contradiction_detected": (
                    m.detected_conflicts > 0 if m.request_type == "contradictory" else None
                ),
            },
        }


def progress_line(o: Outcome) -> str:
    m = o.metrics
    m8 = ("Y" if m.detected_conflicts else "N") if m.request_type == "contradictory" else "-"
    return (f"cls:{'Y' if m.classification_correct else 'N'}  cov:{m.extraction_coverage:.0%}  "
            f"turns:{m.turns_to_termination}  M7:{'Y' if m.policy_valid else 'N'}  M8:{m8}")


def run_benchmark(
    samples: Iterable[EvalSample],
    make_pipeline: PipelineFactory,
    kg: KnowledgeGraph,
    *,
    verbose: bool = False,
    echo: Callable[[str], None] = print,
) -> list[Outcome]:
    """Run every sample; a failing sample is reported and skipped."""
    samples = list(samples)
    outcomes: list[Outcome] = []
    for i, sample in enumerate(samples, 1):
        prefix = f"  [{i:>3}/{len(samples)}] {sample.id:<12}"
        try:
            run = make_pipeline(sample).run(sample.intent)
            outcome = Outcome(sample, run, evaluate_sample(sample, run, kg))
        except Exception as exc:  # keep going: one bad sample must not sink a long run
            echo(f"{prefix} ERROR: {exc}")
            if verbose:
                traceback.print_exc()
            continue
        echo(f"{prefix} {progress_line(outcome)}")
        outcomes.append(outcome)
    return outcomes


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return path
