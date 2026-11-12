"""
Conflict detectors — implementations of the ConflictDetector port.

The pipeline asks a detector, not the knowledge graph directly, so the
experiments can swap the strategy: full KG rules, a partial KG (rule
coverage sweep), no detection at all (ablation), or an LLM (ablation).
"""

from __future__ import annotations

import math
from typing import Mapping

from intent_agent.core.knowledge.graph import KnowledgeGraph
from intent_agent.models.dialogue import ConflictFlag, SlotValue


class KGConflictDetector:
    """
    Detect conflicts with the knowledge graph's rules.

    ``coverage`` (0..1) keeps only the first ceil(N * coverage) rules of a
    class, universal rules first — modelling a partially built ontology.
    """

    def __init__(self, kg: KnowledgeGraph, coverage: float = 1.0) -> None:
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"coverage must be in [0, 1], got {coverage}")
        self.kg = kg
        self.coverage = coverage

    def active_rule_count(self, class_id: str) -> int:
        return math.ceil(len(self.kg.conflict_rules(class_id)) * self.coverage)

    def detect(self, class_id: str, satisfied: Mapping[str, SlotValue]) -> list[ConflictFlag]:
        rules = self.kg.conflict_rules(class_id)[: self.active_rule_count(class_id)]
        return self.kg.check_conflicts(class_id, satisfied, rules=rules)


class NullConflictDetector:
    """Never reports a conflict (ablation: no conflict-checking layer)."""

    def detect(self, class_id: str, satisfied: Mapping[str, SlotValue]) -> list[ConflictFlag]:
        return []
