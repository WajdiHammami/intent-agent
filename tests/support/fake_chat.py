"""
RoutingChatModel — a LangChain chat model that answers by prompt type.

It lets the *real* LangChain adapter (prompts, chains, JSON parsing) run end
to end without Ollama: classification, extraction, answer parsing and user
simulation each get a deterministic reply.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class RoutingChatModel(BaseChatModel):
    respond: Callable[[str, str], str]
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "routing-fake"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        system = next((m.content for m in messages if m.type == "system"), "")
        last = messages[-1].content
        self.calls.append(system[:40])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.respond(system, last)))])


def parse_reply(human: str) -> dict:
    """Answer an answer-parsing prompt with the first valid value of the field's type."""
    type_hint = re.search(r"\(type: (.*)\)\n", human).group(1)
    if type_hint.startswith("enum"):
        options = ast.literal_eval(type_hint[type_hint.index("["):])
        return {"value": options[0], "confidence": "high"}
    if type_hint.startswith(("float", "int")):
        return {"value": 10, "confidence": "high"}
    if type_hint.startswith("bool"):
        return {"value": False, "confidence": "high"}
    return {"value": "Balanced", "confidence": "high"}


def resolution_reply(human: str) -> dict:
    """Answer a conflict-resolution prompt by changing the first field to its first option."""
    options = json.loads(re.search(r"Options: (\{.*\})\n", human).group(1))
    field, values = next(iter(options.items()))
    return {"field": field, "value": values[0] if isinstance(values, list) else "Balanced"}


def scenario_responder(class_id: str, extracted: dict | None = None,
                       answer: str = "Yes, that works.") -> Callable[[str, str], str]:
    """A responder for one intent: fixed class, fixed extraction, cooperative user."""
    def respond(system: str, human: str) -> str:
        if "service class classifier" in system:
            return json.dumps({"class_id": class_id, "confidence": 0.9, "reason": "test"})
        if "parameter extractor" in system:
            return f"```json\n{json.dumps(extracted or {})}\n```"     # exercise fence parsing
        if "conflict resolution parser" in system:
            return json.dumps(resolution_reply(human))
        if "value parser" in system:
            return json.dumps(parse_reply(human))
        if "network engineer providing requirements" in system:
            return answer
        return "{}"
    return respond


def factory_for(respond: Callable[[str, str], str]):
    """A ChatModelFactory for LangChainLanguageModel."""
    return lambda max_tokens, json_mode, temperature: RoutingChatModel(respond=respond, calls=[])
