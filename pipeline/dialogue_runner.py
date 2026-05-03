"""
TMP 3.0 — Dialogue Runner

Drives the full intent → service profile pipeline:
  Stage 1: Classify intent (LLM soft classifier)
  Stage 2: Extract slots directly from intent text (LLM)
  Stage 3: KB-driven elicitation loop (LLM plays user)
  Stage 5: Assemble ServiceIntent

The KB (KGOntology) makes all decisions:
  - what questions to ask (get_required_slots, get_activated_conditionals)
  - when to stop (check_termination / Algorithm 1)
  - when to flag conflicts (check_conflicts)
  - when to propose escalation (get_escalation_candidates)

The LLM is used only for NLP tasks:
  - soft classification
  - slot extraction from free text
  - answering questions (user simulation)
  - parsing typed values from answers
"""

from __future__ import annotations

import logging
from typing import Any

from ontology.kg import KGOntology
from ontology.ontology_v3 import ALL_CLASSES
from ontology.types import KPIProfile, RequiresConditional, RequiresParameter
from pipeline.llm_client import LLMClient
from pipeline.models import (
    AmbiguousClass,
    ConflictFlag,
    DialogueState,
    ServiceIntent,
    SingleClass,
    SlotValue,
)

logger = logging.getLogger(__name__)


class DialogueRunner:
    """
    Orchestrates the TMP 3.0 pipeline for a single intent.

    Parameters
    ----------
    scenario : str
        A short description of the deployment context given to the LLM
        when it plays the user role (e.g. "offshore oil platforms in the
        North Sea needing broadband for 150 vessels").
    llm : LLMClient
        The LLM wrapper. Injected so tests can substitute a mock.
    kg : KGOntology | None
        The knowledge graph. Defaults to the standard TMP 3.0 ontology.
    verbose : bool
        Print dialogue turns to stdout when True.
    """

    def __init__(
        self,
        scenario: str,
        llm: LLMClient,
        kg: KGOntology | None = None,
        verbose: bool = False,
    ) -> None:
        self.scenario = scenario
        self.llm = llm
        self.kg = kg or KGOntology(ALL_CLASSES)
        self.verbose = verbose

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, intent: str) -> tuple[ServiceIntent, DialogueState]:
        """
        Run the full pipeline for a single natural-language intent.
        Returns the final ServiceIntent and the DialogueState (for metrics).
        """
        # Stage 1: Classify
        class_id = self._stage1_classify(intent)

        # Stage 2: Extract slots from intent text
        state = self._stage2_extract(intent, class_id)

        # Stage 3: Elicitation loop
        state = self._stage3_elicit(intent, state)

        # Stage 5: Assemble final output
        service_intent = self._stage5_assemble(state)
        state.service_intent = service_intent
        state.is_complete = True

        return service_intent, state

    # ── Stage 1: Classify ─────────────────────────────────────────────────────

    def _stage1_classify(self, intent: str) -> str:
        result = self.llm.classify_intent(intent)
        class_id = result["class_id"]
        confidence = result.get("confidence", 0.0)
        self._log(f"[S1] Classified as {class_id!r} (confidence={confidence:.2f})")
        # Validate class_id
        valid = self.kg.all_class_ids()
        if class_id not in valid:
            # fallback: pick closest by name or default
            self._log(f"[S1] Unknown class {class_id!r} — defaulting to MissionCritical")
            class_id = "MissionCritical"
        return class_id

    # ── Stage 2: Extract ──────────────────────────────────────────────────────

    def _stage2_extract(self, intent: str, class_id: str) -> DialogueState:
        # Build the full slot list to attempt extraction on.
        # vertical_domain is synthetic (not in REQUIRES edges) — add it explicitly.
        required = self.kg.get_required_slots(class_id)
        all_slots: list[RequiresParameter | RequiresConditional] = list(required)
        node = self.kg.get_class(class_id)
        all_slots += node.requires_conditional
        vd_slot = self.kg.get_slot_definition(class_id, "vertical_domain")
        if vd_slot:
            all_slots.insert(0, vd_slot)

        extracted_raw = self.llm.extract_slots(intent, all_slots)
        self._log(f"[S2] Extracted from intent: {list(extracted_raw.keys())}")

        # Fill implied fields first
        satisfied: dict[str, SlotValue] = {}
        for field, value in self.kg.get_implied_fields(class_id).items():
            satisfied[field] = SlotValue(
                field=field, value=value, source="implied", confidence="high"
            )

        # Add LLM-extracted slots
        for field, value in extracted_raw.items():
            if value is None:
                continue
            # Determine confidence: if field is in required mandatory slots it matters more
            satisfied[field] = SlotValue(
                field=field, value=value, source="extracted", confidence="high"
            )

        # Compute initial missing list (mandatory slots not yet satisfied)
        missing = [
            f for f in self.kg.init_missing(class_id)
            if f not in satisfied
        ]
        # vertical_domain must always be present — add to missing if not extracted
        if "vertical_domain" not in satisfied:
            missing.insert(0, "vertical_domain")

        state = DialogueState(
            intent=intent,
            classification=SingleClass(
                class_id=class_id,
                group=self.kg.get_group(class_id),
                confidence_proxy=0.9,
            ),
            current_class=class_id,
            satisfied=satisfied,
            missing=missing,
            conditionally_activated=[],
        )
        self._log(f"[S2] Missing after extraction: {missing}")
        return state

    # ── Stage 3: Elicitation loop ─────────────────────────────────────────────

    def _stage3_elicit(self, intent: str, state: DialogueState) -> DialogueState:
        class_id = state.current_class
        history: list[dict[str, str]] = []  # for LLM user simulation context

        while not state.is_at_turn_limit():
            # Re-check conflicts after each update
            new_conflicts = self.kg.check_conflicts(class_id, state.satisfied)
            for flag in new_conflicts:
                state.add_conflict(flag)

            # Re-check activated conditionals — add to missing if newly triggered
            activated = self.kg.get_activated_conditionals(class_id, state.satisfied)
            for cond in activated:
                if cond.field not in state.satisfied and cond.field not in state.missing:
                    state.missing.append(cond.field)
                    state.conditionally_activated.append(cond.field)

            # Check escalation (propose but don't auto-apply in this simulation)
            if not state.escalation_proposed:
                esc = self.kg.get_escalation_candidates(class_id, state.satisfied)
                if esc:
                    self._log(
                        f"[S3] Escalation candidate: {esc[0].target_class} — {esc[0].reason}"
                    )
                    state.escalation_proposed = True

            # Termination check
            if self.kg.check_termination(class_id, state):
                self._log("[S3] Termination condition met.")
                break

            # Resolve any unresolved conflict by asking the user
            unresolved = [c for c in state.conflicts if not c.resolved]
            if unresolved:
                conflict = unresolved[0]
                q = (
                    f"There is a conflict: {conflict.description} "
                    f"Please choose: keep '{conflict.subject_field}' as is "
                    f"or change '{conflict.object_field}'?"
                )
                answer, history = self._ask(q, history, state)
                # Resolve: update the object_field slot based on answer
                parsed = self.llm.parse_answer(
                    _synthetic_slot(conflict.object_field), answer
                )
                if parsed.get("value"):
                    state.satisfy_slot(SlotValue(
                        field=conflict.object_field,
                        value=parsed["value"],
                        source="user_provided",
                        confidence=parsed.get("confidence", "medium"),
                    ))
                state.resolve_conflict(conflict.description, answer)
                continue

            # Next missing mandatory slot
            if not state.missing:
                break

            field = state.missing[0]
            slot_def = self.kg.get_slot_definition(class_id, field)
            if slot_def is None:
                # Unknown slot — skip
                state.missing.remove(field)
                continue

            answer, history = self._ask(slot_def.question, history, state)
            parsed = self.llm.parse_answer(slot_def, answer)

            value = parsed.get("value")
            confidence = parsed.get("confidence", "medium")
            if value is not None:
                state.satisfy_slot(SlotValue(
                    field=field,
                    value=value,
                    source="user_provided",
                    confidence=confidence,
                ))
            else:
                # Could not parse — remove from missing to avoid infinite loop
                self._log(f"[S3] Could not parse answer for {field!r} — skipping")
                state.missing.remove(field)

        return state

    def _ask(
        self,
        question: str,
        history: list[dict[str, str]],
        state: DialogueState,
    ) -> tuple[str, list[dict[str, str]]]:
        """Ask a question to the LLM user simulation and append to history."""
        answer = self.llm.simulate_user(self.scenario, history, question)
        state.add_turn("system", question)
        state.add_turn("user", answer)
        history = history + [
            {"role": "assistant", "content": f"Question: {question}"},
            {"role": "user", "content": answer},
        ]
        self._log(f"[Q] {question}")
        self._log(f"[A] {answer}")
        return answer, history

    # ── Stage 5+6: Assemble ServiceIntent + Assign Deployment Mode ──────────────

    def _stage5_assemble(self, state: DialogueState) -> ServiceIntent:
        class_id = state.current_class
        node = self.kg.get_class(class_id)

        def _get(field: str, default: Any = "") -> Any:
            sv = state.satisfied.get(field)
            return sv.value if sv else default

        # Universal fields
        geographic_scope = _get("geographic_scope")
        isolation_level  = _get("isolation_level")
        priority_level   = _get("priority_level")
        cost_profile     = _get("cost_profile")
        vertical_domain  = _get("vertical_domain", "Unknown")

        # Class-specific slots — everything not in the universal set
        universal = {"geographic_scope", "isolation_level", "priority_level",
                     "cost_profile", "vertical_domain"}
        class_slots = {
            k: v.value
            for k, v in state.satisfied.items()
            if k not in universal
        }

        # Stage 6: assign deployment_mode from KG if user did not express a preference
        if "deployment_mode" not in class_slots:
            scenario = class_slots.get("deployment_scenario", "")
            if not scenario:
                sv = state.satisfied.get("deployment_scenario")
                scenario = sv.value if sv else "Urban"
            assigned_mode = self.kg.assign_deployment_mode(class_id, str(scenario))
            class_slots["deployment_mode"] = assigned_mode
            self._log(f"[S6] Deployment mode assigned by KG: {assigned_mode} (scenario={scenario})")
        else:
            self._log(f"[S6] Deployment mode from user preference: {class_slots['deployment_mode']}")

        return ServiceIntent(
            service_class=class_id,
            vertical_domain=vertical_domain,
            kpi_profile=node.kpi_profile,
            traffic_pattern=node.traffic_pattern,
            geographic_scope=str(geographic_scope),
            isolation_level=str(isolation_level),
            priority_level=str(priority_level),
            cost_profile=str(cost_profile),
            class_slots=class_slots,
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        logger.debug(msg)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_slot(field_name: str) -> RequiresParameter:
    """Create a minimal RequiresParameter for an ad-hoc field (conflict resolution)."""
    return RequiresParameter(
        field=field_name,
        dtype="str",
        question=f"Please provide a new value for {field_name}.",
    )
