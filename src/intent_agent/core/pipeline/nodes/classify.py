"""Stage 1 — soft-classify the intent into a service class (LLM)."""

from __future__ import annotations

import logging

from intent_agent.core.pipeline.context import PipelineContext
from intent_agent.core.pipeline.state import PipelineState

log = logging.getLogger("intent_agent.pipeline")

FALLBACK_CLASS = "MissionCritical"


class ClassifyNode:
    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx

    def __call__(self, state: PipelineState) -> dict:
        result = self.ctx.llm.classify_intent(state["intent"])
        class_id = result.get("class_id")
        log.info("[S1] Classified as %r (confidence=%.2f)",
                 class_id, float(result.get("confidence", 0.0) or 0.0))
        if class_id not in self.ctx.kg.all_class_ids():
            log.info("[S1] Unknown class %r — defaulting to %s", class_id, FALLBACK_CLASS)
            class_id = FALLBACK_CLASS
        return {"class_id": class_id}
