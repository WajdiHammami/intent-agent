"""
LangChain implementation of the LanguageModel and UserSimulationModel ports.

Each task is an LCEL chain ``prompt | chat model | parser``. The chat model
comes from a factory, so any LangChain chat model works — Ollama in
production (see ``ollama.py``), a fake model in tests.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from adapters.llm import prompts
from adapters.llm.json_parsing import extract_json
from core.ports import History
from models.ontology import SlotDefinition

#: (max_tokens, json_mode, temperature) -> chat model
ChatModelFactory = Callable[[int, bool, float], BaseChatModel]

TO_JSON: Runnable = StrOutputParser() | RunnableLambda(extract_json)
TO_TEXT: Runnable = StrOutputParser() | RunnableLambda(str.strip)


def to_messages(history: History) -> list[BaseMessage]:
    """Convert [{"role", "content"}] dicts into LangChain messages."""
    kinds = {"system": SystemMessage, "assistant": AIMessage}
    return [kinds.get(t["role"], HumanMessage)(content=t["content"]) for t in history]


def cast_value(dtype: str, value: Any) -> Any:
    """Coerce a parsed value to the slot's declared type."""
    if value is None:
        return None
    if dtype == "float":
        return float(value)
    if dtype == "int":
        return int(value)
    if dtype == "bool" and isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return value


class LangChainLanguageModel:
    def __init__(self, chat_model_factory: ChatModelFactory) -> None:
        self._factory = chat_model_factory
        self._models: dict[tuple, BaseChatModel] = {}
        self._classify = prompts.classify.PROMPT | self.chat_model(256, json_mode=True) | TO_JSON
        self._extract = prompts.extract.PROMPT | self.chat_model(512, json_mode=True) | TO_JSON
        self._parse = prompts.parse_answer.PROMPT | self.chat_model(128, json_mode=True) | TO_JSON
        self._resolve = prompts.parse_resolution.PROMPT | self.chat_model(128, json_mode=True) | TO_JSON
        self._simulate = prompts.simulate_user.PROMPT | self.chat_model(128) | TO_TEXT

    def chat_model(
        self, max_tokens: int = 256, json_mode: bool = False, temperature: float = 0.0
    ) -> BaseChatModel:
        """Return a cached chat model configured for one kind of call."""
        key = (max_tokens, json_mode, temperature)
        if key not in self._models:
            self._models[key] = self._factory(max_tokens, json_mode, temperature)
        return self._models[key]

    # ── generic calls ─────────────────────────────────────────────────────────

    def chat(
        self, system: str, user: str, max_tokens: int = 256,
        temperature: float = 0.0, json_mode: bool = False,
    ) -> str:
        model = self.chat_model(max_tokens, json_mode=json_mode, temperature=temperature)
        return (model | TO_TEXT).invoke([SystemMessage(content=system), HumanMessage(content=user)])

    def chat_json(
        self, system: str, user: str, max_tokens: int = 256, temperature: float = 0.0
    ) -> dict[str, Any]:
        return extract_json(self.chat(system, user, max_tokens, temperature, json_mode=True))

    # ── LanguageModel port ────────────────────────────────────────────────────

    def classify_intent(self, intent: str) -> dict[str, Any]:
        return self._classify.invoke(
            {"intent": intent, "class_list": prompts.classify.format_class_list()}
        )

    def extract_slots(self, intent: str, slots: Sequence[SlotDefinition]) -> dict[str, Any]:
        return self._extract.invoke(
            {"intent": intent, "slot_defs": prompts.extract.format_slot_definitions(slots)}
        )

    def parse_answer(self, slot: SlotDefinition, answer: str) -> dict[str, Any]:
        result = self._parse.invoke({
            "field": slot.field,
            "type_hint": prompts.parse_answer.type_hint(slot),
            "question": slot.question,
            "answer": answer,
        })
        result["value"] = cast_value(slot.dtype, result.get("value"))
        return result

    def parse_resolution(
        self, conflict: str, options: Mapping[str, Sequence[Any] | None], answer: str
    ) -> dict[str, Any]:
        result = self._resolve.invoke({
            "conflict": conflict,
            "options": prompts.parse_resolution.format_options(options),
            "answer": answer,
        })
        return {"field": result.get("field"), "value": result.get("value")}

    # ── UserSimulationModel port ──────────────────────────────────────────────

    def simulate_user(self, scenario: str, history: History, question: str) -> str:
        return self._simulate.invoke(
            {"scenario": scenario, "history": to_messages(history), "question": question}
        )
