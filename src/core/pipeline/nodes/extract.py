"""Stage 2 — fill slots from the intent text (LLM) and open the dialogue state."""

from __future__ import annotations

import logging

from core.pipeline.context import PipelineContext
from core.pipeline.state import PipelineState
from models.dialogue import DialogueState, SingleClass, SlotValue
from models.ontology import SlotDefinition

log = logging.getLogger("intent_agent.pipeline")


class ExtractNode:
    """
    ``use_llm=False`` skips extraction entirely, so every mandatory slot must
    be asked (ablation: no S2).
    """

    def __init__(self, ctx: PipelineContext, use_llm: bool = True) -> None:
        self.ctx = ctx
        self.use_llm = use_llm

    def __call__(self, state: PipelineState) -> dict:
        intent, class_id = state["intent"], state["class_id"]
        extracted = self.ctx.llm.extract_slots(intent, self._slots(class_id)) if self.use_llm else {}
        log.info("[S2] Extracted from intent: %s", list(extracted))
        return {"dialogue": self._open_dialogue(intent, class_id, extracted)}

    def _slots(self, class_id: str) -> list[SlotDefinition]:
        kg = self.ctx.kg
        slots: list[SlotDefinition] = [*kg.get_required_slots(class_id),
                                       *kg.get_class(class_id).requires_conditional]
        vertical = kg.get_slot_definition(class_id, "vertical_domain")
        if vertical:
            slots.insert(0, vertical)
        return slots

    def _open_dialogue(self, intent: str, class_id: str, extracted: dict) -> DialogueState:
        kg = self.ctx.kg
        satisfied = {
            f: SlotValue(field=f, value=v, source="implied", confidence="high")
            for f, v in kg.get_implied_fields(class_id).items()
        }
        for f, v in extracted.items():
            if v is not None:
                satisfied[f] = SlotValue(field=f, value=v, source="extracted", confidence="high")

        missing = [f for f in kg.init_missing(class_id) if f not in satisfied]
        if "vertical_domain" not in satisfied:
            missing.insert(0, "vertical_domain")
        log.info("[S2] Missing after extraction: %s", missing)

        return DialogueState(
            intent=intent,
            classification=SingleClass(
                class_id=class_id, group=kg.get_group(class_id), confidence_proxy=0.9,
            ),
            current_class=class_id,
            satisfied=satisfied,
            missing=missing,
            conditionally_activated=[],
            max_turns=self.ctx.max_turns,
        )
