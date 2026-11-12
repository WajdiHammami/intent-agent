"""Stage 3 prompt — read which field the user changes to resolve a conflict, and to what."""

import json
from typing import Any, Mapping, Sequence

from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a conflict resolution parser for a 5G/6G network configuration system. "
     "The user was told two settings conflict and was asked which one to change. "
     "Pick the field the user wants to change and the new value. Use ONLY the listed "
     "fields; for fields with listed values, use one of them exactly as written. "
     "If the answer does not say how to resolve the conflict, return null for both. "
     "Respond with JSON only — no prose, no markdown fences."),
    ("human",
     'Conflict: "{conflict}"\n'
     "Options: {options}\n"
     'User answer: "{answer}"\n\n'
     'Return JSON: {{"field": "<field or null>", "value": <new value or null>}}'),
])


def format_options(options: Mapping[str, Sequence[Any] | None]) -> str:
    """JSON map of field → allowed values ("any" for free-form fields)."""
    return json.dumps({f: (list(v) if v is not None else "any") for f, v in options.items()})
