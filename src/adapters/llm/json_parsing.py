"""Tolerant JSON extraction from model output."""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict:
    """Return the first JSON object in ``text``, even inside prose or markdown fences."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON found in LLM response:\n{text[:300]}")
