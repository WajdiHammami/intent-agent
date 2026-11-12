"""Clarification loop — ask for the next missing slot and record the parsed value."""

from __future__ import annotations

import logging

from intent_agent.core.pipeline.context import PipelineContext
from intent_agent.core.pipeline.nodes._dialogue import ask_user
from intent_agent.core.pipeline.state import ElicitState
from intent_agent.models.dialogue import Question, SlotValue

log = logging.getLogger("intent_agent.pipeline")


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
        if parsed.get("value") is not None:
            state.satisfy_slot(SlotValue(
                field=field,
                value=parsed["value"],
                source="user_provided",
                confidence=parsed.get("confidence", "medium"),
            ))
        else:                                # unparseable — drop it to avoid looping
            log.info("[S3] Could not parse answer for %r — skipping", field)
            state.missing.remove(field)
        return {"dialogue": state, "history": history}
