"""Prompts that ask the LLM to derive new benchmark variants from a seed intent."""

from __future__ import annotations

PARAPHRASE_SYSTEM = (
    "You are a technical writer specialising in 5G/6G network service requirements. "
    "Given a service intent, produce a natural-language paraphrase that preserves "
    "ALL technical semantics (service class, QoS requirements, deployment context) "
    "but uses different phrasing, vocabulary, or sentence structure. "
    "Do NOT change any numeric values. Do NOT add or remove requirements. "
    "Respond with a JSON object only — no prose, no markdown fences."
)

AMBIGUOUS_SYSTEM = (
    "You are a technical writer specialising in 5G/6G network service requirements. "
    "Given a service intent, produce an ambiguous variant where ONE requirement is "
    "genuinely unclear: use vague terminology, omit a key qualifier, or blend two "
    "scenarios so the service class is not immediately obvious. "
    "Keep all other details intact. "
    "Respond with a JSON object only — no prose, no markdown fences."
)

CONTRADICTORY_SYSTEM = (
    "You are a technical writer specialising in 5G/6G network service requirements. "
    "Given a service intent, produce a contradictory variant that contains two "
    "mutually exclusive requirements (e.g., 'economy cost' AND 'dedicated slice', "
    "or 'best-effort' AND 'sub-millisecond latency'). "
    "The contradiction must be explicit and detectable from the text alone. "
    "State the contradiction clearly in the 'conflict_description' field. "
    "Respond with a JSON object only — no prose, no markdown fences."
)


_EXTRA_FIELD = {
    "paraphrased": "",
    "ambiguous": ',\n  "ambiguity_description": "<one sentence: what is ambiguous and why>"',
    "contradictory": ',\n  "conflict_description": "<one sentence: which two requirements conflict>"',
}
_INTENT_HINT = {
    "paraphrased": "paraphrased intent text",
    "ambiguous": "ambiguous intent text",
    "contradictory": "intent text with an embedded contradiction",
}
SYSTEM = {
    "paraphrased": PARAPHRASE_SYSTEM,
    "ambiguous": AMBIGUOUS_SYSTEM,
    "contradictory": CONTRADICTORY_SYSTEM,
}


def user_prompt(intent: str, variant_type: str) -> str:
    return (
        f'Original intent: "{intent}"\n\n'
        "Return JSON:\n{\n"
        f'  "intent": "<{_INTENT_HINT[variant_type]}>",\n'
        '  "scenario": "<matching deployment scenario for the LLM user simulator>"'
        f"{_EXTRA_FIELD[variant_type]}\n}}"
    )
