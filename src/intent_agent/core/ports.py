"""
Ports — the interfaces the core depends on.

The pipeline only talks to these protocols. Concrete implementations live in
``intent_agent.adapters`` (LangChain/Ollama model, simulated user, …) or in
the CLI (interactive user), and are injected at construction time.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from intent_agent.models.dialogue import ConflictFlag, Question, SlotValue
from intent_agent.models.ontology import SlotDefinition

#: Chat history as role/content pairs ("assistant" asked, "user" answered).
History = Sequence[Mapping[str, str]]


@runtime_checkable
class LanguageModel(Protocol):
    """The NLP capabilities the pipeline needs. It never makes decisions."""

    def classify_intent(self, intent: str) -> dict[str, Any]:
        """Return {"class_id": str, "confidence": float, "reason": str}."""

    def extract_slots(self, intent: str, slots: Sequence[SlotDefinition]) -> dict[str, Any]:
        """Return {field: value} for slots stated or clearly implied by the intent."""

    def parse_answer(self, slot: SlotDefinition, answer: str) -> dict[str, Any]:
        """Return {"value": typed value or None, "confidence": "high|medium|low"}."""

    def parse_resolution(
        self, conflict: str, options: Mapping[str, Sequence[Any] | None], answer: str
    ) -> dict[str, Any]:
        """
        Read how the user wants to resolve ``conflict``. ``options`` maps each
        changeable field to its allowed values (None = free value). Return
        {"field": str | None, "value": Any}.
        """

    def chat_json(
        self, system: str, user: str, max_tokens: int = 256, temperature: float = 0.0
    ) -> dict[str, Any]:
        """Free-form JSON completion (used by experiments and tooling)."""


@runtime_checkable
class UserSimulationModel(Protocol):
    """A model that can role-play the user (for automatic evaluation)."""

    def simulate_user(self, scenario: str, history: History, question: str) -> str:
        """Answer ``question`` in the role described by ``scenario``."""


@runtime_checkable
class UserChannel(Protocol):
    """Whoever answers the pipeline's clarification questions."""

    def ask(self, question: Question, history: History) -> str:
        """Return the user's free-text answer."""


@runtime_checkable
class ConflictDetector(Protocol):
    """Decides which slot-value combinations conflict."""

    def detect(self, class_id: str, satisfied: Mapping[str, SlotValue]) -> list[ConflictFlag]:
        """Return a flag for every conflict among the satisfied slots."""
