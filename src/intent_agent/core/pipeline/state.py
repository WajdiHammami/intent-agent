"""LangGraph state schemas for the pipeline graph and the clarification subgraph."""

from __future__ import annotations

from typing import Literal, TypedDict

from intent_agent.models.dialogue import DialogueState
from intent_agent.models.service import ServiceIntent

#: Each clarification turn is two graph steps; 12 turns plus skipped slots
#: stays far below this.
RECURSION_LIMIT = 500

ElicitAction = Literal["resolve_conflict", "ask_slot", "done"]


class PipelineState(TypedDict, total=False):
    """Carried through classify → extract → elicit → assemble."""
    intent: str
    class_id: str
    dialogue: DialogueState
    service_intent: ServiceIntent


class ElicitState(TypedDict):
    """Carried through the clarification loop."""
    dialogue: DialogueState
    history: list[dict[str, str]]
    action: ElicitAction
