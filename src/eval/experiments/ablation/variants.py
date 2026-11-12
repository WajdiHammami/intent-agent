"""Ablation variants — each one is just a pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.knowledge.conflicts import KGConflictDetector, NullConflictDetector
from core.knowledge.graph import KnowledgeGraph
from eval.experiments.ablation.llm_conflicts import LLMConflictDetector


@dataclass(frozen=True)
class Variant:
    label: str
    extraction: bool
    detector: Callable[[Any, KnowledgeGraph], Any]

    def pipeline_options(self, llm: Any, kg: KnowledgeGraph) -> dict[str, Any]:
        return {"kg": kg, "extraction": self.extraction, "conflict_detector": self.detector(llm, kg)}


def _kg(llm, kg):
    return KGConflictDetector(kg)


def _none(llm, kg):
    return NullConflictDetector()


def _llm(llm, kg):
    return LLMConflictDetector(llm)


VARIANTS: dict[str, Variant] = {
    "baseline": Variant("Baseline (full system)", True, _kg),
    "no_s2": Variant("No S2 (no extraction)", False, _kg),
    "no_kg": Variant("No KG conflicts", True, _none),
    "no_s2_no_kg": Variant("No S2 + No KG", False, _none),
    "llm_conflicts": Variant("LLM conflict det.", True, _llm),
}
