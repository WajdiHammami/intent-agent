"""
Composition root — the only place that wires concrete adapters into the core.
The CLI and the experiments build pipelines through these helpers.
"""

from __future__ import annotations

from typing import Any

from adapters.llm.ollama import OllamaLanguageModel
from adapters.users.simulated import SimulatedUser
from config import Settings
from core.pipeline.runner import IntentPipeline


def create_language_model(settings: Settings, check: bool = True) -> OllamaLanguageModel:
    """Build the Ollama model; with ``check`` fail fast if it is unusable."""
    llm = OllamaLanguageModel(model=settings.model, base_url=settings.ollama_url)
    if check:
        llm.check_connection()
    return llm


def simulated_pipeline(llm: Any, scenario: str, **options: Any) -> IntentPipeline:
    """A pipeline whose questions are answered by the LLM playing ``scenario``."""
    return IntentPipeline(llm, SimulatedUser(llm, scenario), **options)
