"""
Pipeline Data Models

Central data structures shared across all pipeline stages.
Two design points worth noting:
  1. ServiceIntent adds class_slots dict for class-specific elicited values.
  2. KPIProfile is now numeric (see ontology/types.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from ontology.types import KPIProfile


# ── Final Output ──────────────────────────────────────────────────────────────

@dataclass
class ServiceIntent:
    """
    The final structured output of the pipeline.

    Universal fields are always present. class_slots holds every
    class-specific slot elicited during the dialogue (application_type,
    cycle_time_ms, fallback_mode, ntn_tier, device_density_per_km2, etc.).
    """
    service_class: str
    vertical_domain: str
    kpi_profile: KPIProfile
    traffic_pattern: str
    geographic_scope: str
    isolation_level: str
    priority_level: str
    cost_profile: str
    class_slots: dict[str, Any] = field(default_factory=dict)

    def reliability_class(self) -> str:
        return {
            "TimeCritical": "URLLC",
            "MissionCritical": "Critical",
            "RemoteConnectivity": "Standard",
            "MassiveConnectivity": "MassiveIoT",
        }.get(self.service_class, "Unknown")

    def coverage_scope(self) -> str:
        return {
            "Campus": "Local", "Urban": "Urban", "CityWide": "Urban",
            "Regional": "Regional", "National": "Regional",
            "Maritime": "Wide-Area", "Rural": "Wide-Area",
            "Aerial": "Wide-Area", "Polar": "Wide-Area", "CityFringe": "Wide-Area",
        }.get(self.geographic_scope, self.geographic_scope)

    def to_dict(self) -> dict:
        return {
            "service_class": self.service_class,
            "vertical_domain": self.vertical_domain,
            "kpi_profile": self.kpi_profile.to_dict(),
            "traffic_pattern": self.traffic_pattern,
            "geographic_scope": self.geographic_scope,
            "isolation_level": self.isolation_level,
            "priority_level": self.priority_level,
            "cost_profile": self.cost_profile,
            "reliability_class": self.reliability_class(),
            "coverage_scope": self.coverage_scope(),
            "deployment_scenario": self.class_slots.get("deployment_scenario", ""),
            "deployment_mode": self.class_slots.get("deployment_mode", ""),
            "fallback_requirement": self.class_slots.get("deployment_mode", ""),
            "class_slots": self.class_slots,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Classification State ──────────────────────────────────────────────────────

@dataclass
class SingleClass:
    """
    One class identified with sufficient signal.

    confidence_proxy is NOT an LLM self-reported number.
    It is derived from objective signals:
        - token logprob entropy over the class distribution
        - top-2 margin (score[1] - score[2])
        - slot extraction coverage from Stage 2
    """
    class_id: str
    group: Literal["A", "B", "C", "D"]
    confidence_proxy: float  # 0.0–1.0


@dataclass
class AmbiguousClass:
    """
    Multiple candidate classes with insufficient margin to commit.
    The dialogue loop opens with a group-level disambiguation question.
    """
    candidates: list[tuple[str, float]]  # [(class_id, score), ...] sorted desc
    reason: Literal["low_margin", "low_coverage", "high_entropy"]


@dataclass
class MultiClass:
    """
    Intent spans two service classes connected by an ACCOMPANIES edge.
    User must confirm decomposition before elicitation begins.
    Output: one ServiceIntent per class.
    """
    classes: list[str]  # [primary_class, secondary_class]
    accompanies_confirmed: bool = False


@dataclass
class OutOfScope:
    """
    No class matches the intent above the minimum confidence threshold.
    closest_class is offered as a fallback suggestion.
    """
    reason: str
    closest_class: Optional[str] = None


ClassificationState = Union[SingleClass, AmbiguousClass, MultiClass, OutOfScope]


# ── Dialogue Primitives ───────────────────────────────────────────────────────

@dataclass
class SlotValue:
    """
    A filled slot with provenance metadata.

    source tracks HOW the value was obtained:
        "extracted"     — LLM extracted from initial intent text
        "implied"       — LLM inferred from context
        "user_provided" — user answered a direct question
        "default"       — class-level default applied

    confidence tracks HOW CERTAIN the extraction/inference was:
        "high"   — explicit mention or clear implication
        "medium" — reasonable inference, should confirm
        "low"    — uncertain, ask user to confirm

    Paper metric: extraction_coverage() counts source in
    {"extracted", "implied"} as "no question needed".
    """
    field: str
    value: Any
    source: Literal["extracted", "implied", "user_provided", "default"]
    confidence: Literal["high", "medium", "low"]


@dataclass
class ConflictFlag:
    """
    A triggered conflict between two satisfied slot values.
    Detected by KGOntology.check_conflicts() after each slot update.
    Dialogue cannot terminate while any ConflictFlag is unresolved.
    """
    conflict_type: Literal["kpi_conflict", "resource_conflict", "class_conflict"]
    description: str
    subject_field: str = ""
    object_field: str = ""
    resolved: bool = False
    resolution: Optional[str] = None


@dataclass
class DialogueTurn:
    """A single turn in the dialogue history."""
    role: Literal["system", "user"]
    content: str
    extracted_slots: dict[str, SlotValue] = field(default_factory=dict)


# ── Dialogue State ────────────────────────────────────────────────────────────

@dataclass
class DialogueState:
    """
    The single source of truth for the pipeline.

    Injected into every LLM prompt via to_prompt_context().
    Updated by each pipeline stage and each dialogue turn.
    """

    intent: str                          # original user utterance, never modified

    classification: ClassificationState
    current_class: Optional[str]         # active class hypothesis (None = OutOfScope)

    satisfied: dict[str, SlotValue]      # field_name → SlotValue
    missing: list[str]                   # mandatory slot names not yet filled
    conditionally_activated: list[str]   # slot names activated by prior answers

    conflicts: list[ConflictFlag] = field(default_factory=list)

    turn_count: int = 0
    max_turns: int = 12
    turn_history: list[DialogueTurn] = field(default_factory=list)
    revision_proposed: bool = False
    escalation_proposed: bool = False
    decomposition_confirmed: bool = False

    shared_context: dict[str, SlotValue] = field(default_factory=dict)

    is_complete: bool = False
    skip_dialogue: bool = False

    service_intent: Optional[ServiceIntent] = None

    def to_prompt_context(self) -> dict:
        return {
            "current_class": self.current_class,
            "satisfied": {
                k: {"value": v.value, "source": v.source, "confidence": v.confidence}
                for k, v in self.satisfied.items()
            },
            "missing": self.missing,
            "activated_conditionals": self.conditionally_activated,
            "unresolved_conflicts": [
                {"description": c.description, "type": c.conflict_type}
                for c in self.conflicts
                if not c.resolved
            ],
            "turn": self.turn_count,
            "max_turns": self.max_turns,
            "shared_context": {
                k: {"value": v.value, "source": v.source}
                for k, v in self.shared_context.items()
            },
        }

    def extraction_coverage(self) -> float:
        """
        Paper metric: fraction of required slots filled without asking.
        Formula: non_asked / (satisfied + missing)
        """
        total_required = len(self.satisfied) + len(self.missing)
        if total_required == 0:
            return 0.0
        non_asked = sum(
            1 for v in self.satisfied.values()
            if v.source in {"extracted", "implied", "default"}
        )
        return non_asked / total_required

    def is_at_turn_limit(self) -> bool:
        return self.turn_count >= self.max_turns

    def add_turn(self, role: Literal["system", "user"], content: str,
                 extracted: dict[str, SlotValue] | None = None) -> None:
        self.turn_history.append(
            DialogueTurn(role=role, content=content, extracted_slots=extracted or {})
        )
        if role == "user":
            self.turn_count += 1

    def satisfy_slot(self, slot: SlotValue) -> None:
        self.satisfied[slot.field] = slot
        if slot.field in self.missing:
            self.missing.remove(slot.field)
        if slot.field in self.conditionally_activated:
            self.conditionally_activated.remove(slot.field)

    def add_conflict(self, conflict: ConflictFlag) -> None:
        existing = {c.description for c in self.conflicts}
        if conflict.description not in existing:
            self.conflicts.append(conflict)

    def resolve_conflict(self, description: str, resolution: str) -> None:
        for c in self.conflicts:
            if c.description == description:
                c.resolved = True
                c.resolution = resolution
                return
