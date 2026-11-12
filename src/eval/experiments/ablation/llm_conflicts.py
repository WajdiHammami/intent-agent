"""LLMConflictDetector — the ConflictDetector port implemented by asking the LLM."""

from __future__ import annotations

from typing import Mapping

from core.ports import LanguageModel
from eval.experiments.ablation import prompts
from models.dialogue import ConflictFlag, SlotValue


class LLMConflictDetector:
    def __init__(self, llm: LanguageModel) -> None:
        self.llm = llm

    def detect(self, class_id: str, satisfied: Mapping[str, SlotValue]) -> list[ConflictFlag]:
        if not satisfied:
            return []
        slots = "\n".join(f"  {k}: {v.value}" for k, v in satisfied.items())
        try:
            data = self.llm.chat_json(prompts.SYSTEM, prompts.user_prompt(class_id, slots))
        except Exception:
            return []
        if not data.get("has_conflict"):
            return []
        return [
            ConflictFlag(
                conflict_type="kpi_conflict",
                description=c.get("description", "LLM-detected conflict"),
                subject_field=c.get("field_a", ""),
                object_field=c.get("field_b", ""),
            )
            for c in data.get("conflicts", [])
            if isinstance(c, dict)
        ]
