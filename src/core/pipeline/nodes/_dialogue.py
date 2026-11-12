"""Helpers shared by the nodes that talk to the user."""

from __future__ import annotations

import logging
from typing import Sequence

from core.pipeline.context import PipelineContext
from models.dialogue import DialogueState, Question

log = logging.getLogger("intent_agent.pipeline")


def ask_user(
    ctx: PipelineContext,
    question: Question,
    history: Sequence[dict[str, str]],
    state: DialogueState,
) -> tuple[str, list[dict[str, str]]]:
    """Put a question to the user, record the turn, and return the new history."""
    answer = ctx.user.ask(question, history)
    state.add_turn("system", question.text)
    state.add_turn("user", answer)
    log.info("[Q] %s", question.text)
    log.info("[A] %s", answer)
    return answer, [
        *history,
        {"role": "assistant", "content": f"Question: {question.text}"},
        {"role": "user", "content": answer},
    ]

