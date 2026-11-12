"""Shared fixtures."""

from __future__ import annotations

import pytest

from adapters.llm.scripted import ScriptedLanguageModel
from adapters.users.scripted import ScriptedUser
from core.knowledge import KGConflictDetector, KnowledgeGraph
from core.pipeline.context import PipelineContext
from models.dialogue import DialogueState, SingleClass, SlotValue


@pytest.fixture(scope="session")
def kg() -> KnowledgeGraph:
    return KnowledgeGraph()


@pytest.fixture
def make_state(kg):
    """Build a DialogueState for a class with the given satisfied values."""
    def _make(class_id: str = "MissionCritical", missing: list[str] | None = None, **values) -> DialogueState:
        return DialogueState(
            intent="test intent",
            classification=SingleClass(class_id=class_id, group=kg.get_group(class_id), confidence_proxy=0.9),
            current_class=class_id,
            satisfied={k: SlotValue(k, v, "extracted", "high") for k, v in values.items()},
            missing=list(missing or []),
            conditionally_activated=[],
        )
    return _make


@pytest.fixture
def make_ctx(kg):
    def _make(llm=None, user=None, detector=None) -> PipelineContext:
        return PipelineContext(kg=kg, llm=llm or ScriptedLanguageModel(), user=user or ScriptedUser(),
                               conflicts=detector or KGConflictDetector(kg))
    return _make
