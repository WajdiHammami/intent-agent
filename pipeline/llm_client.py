"""
LLM Client

Four focused LLM calls used by the dialogue runner:
  1. classify_intent    — soft-classify intent into one of the 4 service classes
  2. extract_slots      — extract slot values directly from the intent text
  3. simulate_user      — play the user role, answering a KB question in context
  4. parse_answer       — parse a free-text answer into a typed SlotValue

Backend selection (in order of preference):
  1. LlamaClient  — local llama_cpp model (no API key needed, default)
  2. AnthropicClient — Claude API (requires ANTHROPIC_API_KEY)

All calls use temperature=0 for determinism. Structured output is requested
as JSON inside a fenced block; a fallback parser handles edge cases.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ontology.types import RequiresConditional, RequiresParameter

_DEFAULT_MODEL_PATH = "models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

_CLASS_DESCRIPTIONS = {
    "TimeCritical": (
        "Sub-millisecond industrial/vehicular control. "
        "Latency ≤1 ms, reliability 99.9999%. "
        "Robotics, grid protection relays, V2X. Always dedicated slice."
    ),
    "MissionCritical": (
        "Resilience-first connectivity where availability matters most. "
        "Latency ≤10 ms, availability 99.999%. "
        "Public safety, emergency response, critical infrastructure. "
        "Mandatory TN/NTN fallback support."
    ),
    "RemoteConnectivity": (
        "NTN-primary (satellite/UAV) coverage for areas without terrestrial networks. "
        "Latency ≤500 ms, availability 99.9%. "
        "Maritime, rural, aerial, polar, city fringe."
    ),
    "MassiveConnectivity": (
        "Massive numbers of low-power, low-rate IoT devices. "
        "Density ≥10⁶ dev/km², throughput 1 Kbps/device, latency tolerant. "
        "Smart metering, industrial sensors, precision agriculture."
    ),
}


def _extract_json(text: str) -> dict:
    """Extract JSON dict from a response that may contain markdown fences."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON found in LLM response:\n{text}")


def _chat(llm_model, system: str, user: str, max_tokens: int = 256) -> str:
    """Single chat call via llama_cpp. Returns the assistant text."""
    response = llm_model.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"].strip()


def _chat_with_history(
    llm_model, system: str, history: list[dict], question: str, max_tokens: int = 128
) -> str:
    """Chat call with conversation history via llama_cpp."""
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"Question: {question}"})
    response = llm_model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"].strip()


class LLMClient:
    """
    Wraps the local Llama-3-8B-Instruct (llama_cpp) for the four
    dialogue-runner call types. Loads the model once on first use.
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH) -> None:
        self._model_path = model_path
        self._model = None  # lazy load

    def _get_model(self):
        if self._model is None:
            from llama_cpp import Llama
            self._model = Llama(
                model_path=self._model_path,
                n_gpu_layers=-1,     # offload all layers to GPU
                n_ctx=4096,
                verbose=False,
            )
        return self._model

    # ── 1. Intent classification ───────────────────────────────────────────────

    def classify_intent(self, intent: str) -> dict[str, Any]:
        """Returns {"class_id": str, "confidence": float, "reason": str}."""
        class_list = "\n".join(
            f'  - "{cid}": {desc}' for cid, desc in _CLASS_DESCRIPTIONS.items()
        )
        system = (
            "You are a 5G/6G service class classifier. "
            "Given a natural-language service intent, choose the single most appropriate class. "
            "Respond with a JSON object only — no prose, no markdown fences."
        )
        user = (
            f"Service intent: \"{intent}\"\n\n"
            f"Available classes:\n{class_list}\n\n"
            'Respond with JSON: {"class_id": "<ID>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}'
        )
        text = _chat(self._get_model(), system, user, max_tokens=256)
        return _extract_json(text)

    # ── 2. Slot extraction ─────────────────────────────────────────────────────

    def extract_slots(
        self,
        intent: str,
        slots: list[RequiresParameter | RequiresConditional],
    ) -> dict[str, Any]:
        """Returns {field_name: value} for extractable/implied slots."""
        slot_defs = []
        for s in slots:
            entry = f'  "{s.field}" ({s.dtype}'
            if s.dtype == "enum":
                entry += f', options: {s.enum_values}'
            if hasattr(s, "unit") and s.unit:
                entry += f', unit: {s.unit}'
            entry += f'): {s.question}'
            slot_defs.append(entry)

        system = (
            "You are a parameter extractor for a 5G/6G network configuration system. "
            "Extract slot values explicitly stated or clearly implied by the intent. "
            "For enum fields use ONLY one of the listed options exactly as written. "
            "Respond with a JSON object only — no prose, no markdown fences."
        )
        user = (
            f"Service intent: \"{intent}\"\n\n"
            "Slots to extract:\n" + "\n".join(slot_defs) + "\n\n"
            'Return JSON with only extractable fields. Example: {"geographic_scope": "Campus"}'
        )
        text = _chat(self._get_model(), system, user, max_tokens=512)
        return _extract_json(text)

    # ── 3. User simulation ─────────────────────────────────────────────────────

    def simulate_user(
        self,
        scenario: str,
        history: list[dict[str, str]],
        question: str,
    ) -> str:
        """Plays the user role, answering a KB question given the scenario."""
        system = (
            "You are a network engineer providing requirements for a 5G/6G deployment. "
            f"Your deployment scenario: {scenario}\n\n"
            "Answer clarification questions concisely and directly. "
            "If given options, choose the most appropriate one. "
            "Do not ask questions back. Do not explain — just answer."
        )
        return _chat_with_history(
            self._get_model(), system, history, question, max_tokens=128
        )

    # ── 4. Answer parsing ──────────────────────────────────────────────────────

    def parse_answer(
        self,
        slot: RequiresParameter | RequiresConditional,
        answer: str,
    ) -> dict[str, Any]:
        """Parses a free-text answer into a typed slot value."""
        type_hint = slot.dtype
        if slot.dtype == "enum":
            type_hint = f'enum — must be exactly one of {slot.enum_values}'
        elif hasattr(slot, "unit") and slot.unit:
            type_hint = f'{slot.dtype} in {slot.unit}'

        system = (
            "You are a value parser for a network configuration system. "
            "Extract the typed slot value from the user's answer. "
            "Respond with JSON only — no prose, no markdown fences."
        )
        user = (
            f'Field: "{slot.field}" (type: {type_hint})\n'
            f'Question asked: "{slot.question}"\n'
            f'User answer: "{answer}"\n\n'
            'Return JSON: {"value": <extracted value with correct type>, "confidence": "high|medium|low"}'
        )
        text = _chat(self._get_model(), system, user, max_tokens=128)
        result = _extract_json(text)

        # Cast to correct Python type
        if slot.dtype == "float" and result.get("value") is not None:
            result["value"] = float(result["value"])
        elif slot.dtype == "int" and result.get("value") is not None:
            result["value"] = int(result["value"])
        elif slot.dtype == "bool" and isinstance(result.get("value"), str):
            result["value"] = result["value"].lower() in ("true", "yes", "1")

        return result
