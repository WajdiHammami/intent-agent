"""Runtime settings, read from the environment with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from intent_agent.adapters.llm.host import normalize_host

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_HOST = "localhost:11434"
DEFAULT_RESULTS_DIR = "results"


@dataclass(frozen=True)
class Settings:
    """
    model        Ollama model tag                       (env OLLAMA_MODEL)
    ollama_url   Ollama server URL                      (env OLLAMA_HOST)
    results_dir  where experiments write their output   (env INTENT_AGENT_RESULTS)
    """
    model: str = DEFAULT_MODEL
    ollama_url: str = normalize_host(DEFAULT_OLLAMA_HOST)
    results_dir: Path = Path(DEFAULT_RESULTS_DIR)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
            ollama_url=normalize_host(os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)),
            results_dir=Path(os.environ.get("INTENT_AGENT_RESULTS", DEFAULT_RESULTS_DIR)),
        )

    def with_overrides(self, **changes) -> "Settings":
        changes = {k: v for k, v in changes.items() if v is not None}
        if "ollama_url" in changes:
            changes["ollama_url"] = normalize_host(changes["ollama_url"])
        if "results_dir" in changes:
            changes["results_dir"] = Path(changes["results_dir"])
        return replace(self, **changes)
