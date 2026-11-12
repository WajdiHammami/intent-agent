"""
FieldVocabulary — the values the ontology accepts for each field.

Built from every slot definition in the ontology (all classes, universal and
conditional slots), plus the values conflict rules mention and the class-
implied priority levels. A value named by a conflict rule is accepted even if
a class's own options exclude it (e.g. GEO for a TimeCritical ``ntn_tier``):
dropping it would erase the very contradiction the rule exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from core.knowledge.values import InvalidValue, coerce_value
from models.dialogue import SlotValue
from models.ontology import RequiresParameter, SlotDefinition

if TYPE_CHECKING:
    from core.knowledge.graph import KnowledgeGraph

PRIORITY_LEVELS = ("Critical", "High", "Normal", "Low")


@dataclass
class _Entry:
    dtype: str
    options: list[str] = field(default_factory=list)   # enum only; ordered, deduplicated

    def add(self, values) -> None:
        for v in values:
            if v not in self.options:
                self.options.append(v)


class FieldVocabulary:
    def __init__(self, kg: "KnowledgeGraph") -> None:
        self._entries: dict[str, _Entry] = {}
        for class_id in kg.all_class_ids():
            slots: list[SlotDefinition] = [*kg.get_required_slots(class_id),
                                           *kg.get_class(class_id).requires_conditional]
            vertical = kg.get_slot_definition(class_id, "vertical_domain")
            for slot in [*slots, *([vertical] if vertical else [])]:
                entry = self._entries.setdefault(slot.field, _Entry(slot.dtype))
                if slot.dtype == "enum":
                    entry.add(slot.enum_values)
            for rule in kg.conflict_rules(class_id):
                for f, v in ((rule.subject_field, rule.subject_value),
                             (rule.object_field, rule.object_value)):
                    entry = self._entries.setdefault(f, _Entry("enum"))
                    if entry.dtype == "enum":
                        entry.add([v])
        self._entries.setdefault("priority_level", _Entry("enum")).add(PRIORITY_LEVELS)

    def knows(self, field_name: str) -> bool:
        return field_name in self._entries

    def options(self, field_name: str) -> list[str] | None:
        entry = self._entries.get(field_name)
        return list(entry.options) if entry and entry.dtype == "enum" else None

    def coerce(self, field_name: str, raw: Any) -> Any:
        """``raw`` as an accepted value of ``field_name``, or raise InvalidValue."""
        entry = self._entries.get(field_name)
        if entry is None:
            raise InvalidValue(f"{field_name!r} is not a field of the ontology")
        slot = RequiresParameter(field=field_name, dtype=entry.dtype, question="",
                                 enum_values=entry.options)
        return coerce_value(slot, raw)

    def invalid_fields(self, satisfied: Mapping[str, SlotValue]) -> list[str]:
        """Known fields whose current value the ontology does not accept."""
        bad = []
        for name, sv in satisfied.items():
            if not self.knows(name):
                continue
            try:
                self.coerce(name, sv.value)
            except InvalidValue:
                bad.append(name)
        return bad
