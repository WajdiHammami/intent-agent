"""
Clarification loop, routing node — KG bookkeeping after every update.

Refreshes conflicts, activated conditionals and escalation, then decides the
next action: stop (turn limit or Algorithm-1 termination), resolve a
conflict, or ask for the next missing slot.
"""

from __future__ import annotations

import logging

from core.pipeline.context import PipelineContext
from core.pipeline.state import ElicitState

log = logging.getLogger("intent_agent.pipeline")


class KGCheckNode:
    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx

    def __call__(self, s: ElicitState) -> dict:
        state, kg = s["dialogue"], self.ctx.kg
        class_id = state.current_class

        if state.is_at_turn_limit():
            return {"action": "done"}

        detected = self.ctx.conflicts.detect(class_id, state.satisfied)
        for flag in detected:
            # still present after a "resolution" → it is open again, not done
            if not state.reopen_conflict(flag.description):
                state.add_conflict(flag)
        # an open conflict the detector no longer reports was fixed by another change
        current = {f.description for f in detected}
        for c in state.conflicts:
            if not c.resolved and c.description not in current:
                state.resolve_conflict(c.description, "no longer applies")

        for cond in kg.get_activated_conditionals(class_id, state.satisfied):
            if cond.field not in state.satisfied and cond.field not in state.missing:
                state.missing.append(cond.field)
                state.conditionally_activated.append(cond.field)

        if not state.escalation_proposed:
            esc = kg.get_escalation_candidates(class_id, state.satisfied)
            if esc:
                log.info("[S3] Escalation candidate: %s — %s", esc[0].target_class, esc[0].reason)
                state.escalation_proposed = True

        if kg.check_termination(class_id, state):
            log.info("[S3] Termination condition met.")
            return {"action": "done"}
        if any(not c.resolved for c in state.conflicts):
            return {"action": "resolve_conflict"}
        if not state.missing:
            return {"action": "done"}
        return {"action": "ask_slot"}
