"""
Clarification loop — ask for the next missing slot and record the parsed value.

The parsed value must be one the ontology accepts for the field. An unusable
answer is asked again once; after MAX_ATTEMPTS the slot is given up on (the
profile then stays incomplete, which the metrics report honestly).
"""

from __future__ import annotations

import logging

from core.knowledge.values import InvalidValue
from core.pipeline.context import PipelineContext
from core.pipeline.nodes._dialogue import ask_user
from core.pipeline.state import ElicitState
from models.dialogue import Question, SlotValue

log = logging.getLogger("intent_agent.pipeline")

MAX_ATTEMPTS = 2   # unusable answers before a slot is given up on


class AskSlotNode:
    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx

    def __call__(self, s: ElicitState) -> dict:
        state = s["dialogue"]
        field = state.missing[0]
        slot = self.ctx.kg.get_slot_definition(state.current_class, field)
        if slot is None:                     # unknown slot — skip it
            state.missing.remove(field)
            return {"dialogue": state}

        answer, history = ask_user(
            self.ctx, Question(text=slot.question, kind="slot", field=field), s["history"], state
        )
        parsed = self.ctx.llm.parse_answer(slot, answer)
        try:
            value = self.ctx.kg.vocabulary().coerce(field, parsed.get("value"))
        except InvalidValue as exc:
            state.attempts[field] = state.attempts.get(field, 0) + 1
            if state.attempts[field] >= MAX_ATTEMPTS:   # give up rather than loop
                log.info("[S3] No usable answer for %r (%s) — skipping", field, exc)
                state.missing.remove(field)
            else:
                log.info("[S3] Unusable answer for %r (%s) — asking again", field, exc)
            return {"dialogue": state, "history": history}

        state.satisfy_slot(SlotValue(
            field=field, value=value, source="user_provided",
            confidence=parsed.get("confidence", "medium"),
        ))
        return {"dialogue": state, "history": history}
