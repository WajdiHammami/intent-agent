"""
ScriptedLanguageModel — a deterministic, offline LanguageModel.

Classifies to a fixed class, extracts a fixed dict, and parses every answer
to the first valid value of the slot's type. Lets the whole pipeline run
without an LLM (self-test, unit tests, demos).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.ports import History
from models.ontology import SlotDefinition


def first_valid_value(slot: SlotDefinition) -> Any:
    if slot.dtype == "enum" and slot.enum_values:
        return slot.enum_values[0]
    return {"float": 10.0, "int": 10, "bool": False}.get(slot.dtype, "Balanced")


class ScriptedLanguageModel:
    def __init__(
        self,
        class_id: str = "MissionCritical",
        extracted: Mapping[str, Any] | None = None,
    ) -> None:
        self.class_id = class_id
        self.extracted = dict(extracted or {})
        self.calls: list[str] = []

    def classify_intent(self, intent: str) -> dict[str, Any]:
        self.calls.append("classify")
        return {"class_id": self.class_id, "confidence": 1.0, "reason": "scripted"}

    def extract_slots(self, intent: str, slots: Sequence[SlotDefinition]) -> dict[str, Any]:
        self.calls.append("extract")
        return dict(self.extracted)

    def parse_answer(self, slot: SlotDefinition, answer: str) -> dict[str, Any]:
        self.calls.append("parse")
        return {"value": first_valid_value(slot), "confidence": "high"}

    def parse_resolution(self, conflict: str, options: Mapping[str, Sequence[Any] | None],
                         answer: str) -> dict[str, Any]:
        """Change the first changeable field to its first allowed value."""
        self.calls.append("resolve")
        for field, values in options.items():
            return {"field": field, "value": values[0] if values else "Balanced"}
        return {"field": None, "value": None}

    def chat_json(self, system: str, user: str, max_tokens: int = 256,
                  temperature: float = 0.0) -> dict[str, Any]:
        self.calls.append("chat_json")
        return {}

    def simulate_user(self, scenario: str, history: History, question: str) -> str:
        self.calls.append("simulate")
        return "(scripted answer)"
