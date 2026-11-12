"""
Clarification loop — ask the user how to resolve the first open conflict.

The question lists the fields that may change (never class-implied ones) and
their allowed values. The answer is parsed into a (field, value) choice and
validated against the field's definition; only a valid change resolves the
conflict. Otherwise the conflict stays open and kg_check routes back here
(bounded by the turn limit). Whether the change actually removes the conflict
is re-checked by kg_check, which reopens it if not.
"""

from __future__ import annotations

import logging
from typing import Any

from core.knowledge.values import InvalidValue, allowed_values, coerce_value
from core.pipeline.context import PipelineContext
from core.pipeline.nodes._dialogue import ask_user
from core.pipeline.state import ElicitState
from models.dialogue import ConflictFlag, DialogueState, Question, SlotValue
from models.ontology import SlotDefinition

log = logging.getLogger("intent_agent.pipeline")


class ResolveConflictNode:
    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx

    def __call__(self, s: ElicitState) -> dict:
        state = s["dialogue"]
        conflict = next(c for c in state.conflicts if not c.resolved)
        changeable = self.changeable_fields(state, conflict)
        if not changeable:                       # nothing the user may change — cannot loop
            log.info("[S3] Conflict has no changeable field: %s", conflict.description)
            conflict.resolvable = False
            state.resolve_conflict(conflict.description, "unresolvable: no changeable field")
            return {"dialogue": state}

        options = {f: self._options(slot, state, f) for f, slot in changeable.items()}
        question = Question(text=_question_text(conflict, options), kind="conflict",
                            field=next(iter(changeable)))
        answer, history = ask_user(self.ctx, question, s["history"], state)

        choice = self.ctx.llm.parse_resolution(conflict.description, options, answer)
        field, raw = choice.get("field"), choice.get("value")
        try:
            if field not in changeable:
                raise InvalidValue(f"{field!r} is not one of {list(changeable)}")
            value = coerce_value(changeable[field], raw)
            if options[field] is not None and value not in options[field]:
                raise InvalidValue(f"{value!r} would not change {field}")
        except InvalidValue as exc:
            log.info("[S3] Conflict answer not usable (%s) — asking again", exc)
            return {"dialogue": state, "history": history}

        state.satisfy_slot(SlotValue(field=field, value=value, source="user_provided",
                                     confidence="high"))
        state.resolve_conflict(conflict.description, f"{field} → {value} ({answer})")
        log.info("[S3] Conflict resolved: %s → %s", field, value)
        return {"dialogue": state, "history": history}

    def changeable_fields(self, state: DialogueState, conflict: ConflictFlag) -> dict[str, SlotDefinition]:
        """Fields of the conflict the user may change: defined slots, not implied by the class."""
        kg, class_id = self.ctx.kg, state.current_class
        implied = kg.get_implied_fields(class_id)
        fields: dict[str, SlotDefinition] = {}
        for field in (conflict.object_field, conflict.subject_field):
            slot = kg.get_slot_definition(class_id, field) if field else None
            if slot is not None and field not in implied:
                fields[field] = slot
        return fields

    @staticmethod
    def _options(slot: SlotDefinition, state: DialogueState, field: str) -> list[Any] | None:
        values = allowed_values(slot)
        if values is None:
            return None
        current = state.satisfied.get(field)
        return [v for v in values if current is None or str(v) != str(current.value)]


def _question_text(conflict: ConflictFlag, options: dict[str, list[Any] | None]) -> str:
    choices = " or ".join(
        f"{field} ({', '.join(map(str, values))})" if values else field
        for field, values in options.items()
    )
    return (f"There is a conflict: {conflict.description} "
            f"To resolve it, change {choices}. What would you like to change?")
