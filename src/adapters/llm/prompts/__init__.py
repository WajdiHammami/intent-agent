"""Prompt templates, one module per LLM task."""

from adapters.llm.prompts import (
    classify,
    extract,
    parse_answer,
    parse_resolution,
    simulate_user,
)

__all__ = ["classify", "extract", "parse_answer", "parse_resolution", "simulate_user"]
