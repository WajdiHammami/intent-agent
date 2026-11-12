"""Stages 5–6 — assemble the ServiceIntent; the KG assigns the deployment mode."""

from __future__ import annotations

import logging

from intent_agent.core.pipeline.context import PipelineContext
from intent_agent.core.pipeline.state import PipelineState
from intent_agent.models.dialogue import DialogueState
from intent_agent.models.service import ServiceIntent

log = logging.getLogger("intent_agent.pipeline")

UNIVERSAL_FIELDS = frozenset(
    {"geographic_scope", "isolation_level", "priority_level", "cost_profile", "vertical_domain"}
)


class AssembleNode:
    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx

    def __call__(self, s: PipelineState) -> dict:
        state = s["dialogue"]
        service_intent = assemble_profile(self.ctx, state)
        state.service_intent = service_intent
        state.is_complete = True
        return {"service_intent": service_intent, "dialogue": state}


def assemble_profile(ctx: PipelineContext, state: DialogueState) -> ServiceIntent:
    class_id = state.current_class
    node = ctx.kg.get_class(class_id)

    def value(field: str, default: object = "") -> object:
        sv = state.satisfied.get(field)
        return sv.value if sv else default

    class_slots = {k: v.value for k, v in state.satisfied.items() if k not in UNIVERSAL_FIELDS}

    if "deployment_mode" not in class_slots:
        scenario = class_slots.get("deployment_scenario") or "Urban"
        class_slots["deployment_mode"] = ctx.kg.assign_deployment_mode(class_id, str(scenario))
        log.info("[S6] Deployment mode assigned by KG: %s (scenario=%s)",
                 class_slots["deployment_mode"], scenario)
    else:
        log.info("[S6] Deployment mode from user preference: %s", class_slots["deployment_mode"])

    return ServiceIntent(
        service_class=class_id,
        vertical_domain=str(value("vertical_domain", "Unknown")),
        kpi_profile=node.kpi_profile,
        traffic_pattern=node.traffic_pattern,
        geographic_scope=str(value("geographic_scope")),
        isolation_level=str(value("isolation_level")),
        priority_level=str(value("priority_level")),
        cost_profile=str(value("cost_profile")),
        class_slots=class_slots,
    )
