"""Each LangGraph node in isolation, with scripted dependencies."""

import pytest

from adapters.llm.scripted import ScriptedLanguageModel
from adapters.users.scripted import ScriptedUser
from core.knowledge import NullConflictDetector
from core.pipeline.nodes import (
    AskSlotNode,
    AssembleNode,
    ClassifyNode,
    ExtractNode,
    KGCheckNode,
    ResolveConflictNode,
)
from models.dialogue import ConflictFlag


class _Classifier(ScriptedLanguageModel):
    def __init__(self, answer):
        super().__init__()
        self.answer = answer

    def classify_intent(self, intent):
        return self.answer


# ── classify ──────────────────────────────────────────────────────────────────

def test_classify_returns_the_llm_class(make_ctx):
    node = ClassifyNode(make_ctx(llm=_Classifier({"class_id": "TimeCritical", "confidence": 0.9})))
    assert node({"intent": "robots"}) == {"class_id": "TimeCritical"}


@pytest.mark.parametrize("answer", [{"class_id": "Teleportation"}, {}])
def test_classify_falls_back_on_an_unknown_class(make_ctx, answer):
    assert ClassifyNode(make_ctx(llm=_Classifier(answer)))({"intent": "?"}) == {"class_id": "MissionCritical"}


# ── extract ───────────────────────────────────────────────────────────────────

def test_extract_merges_implied_and_extracted_values(make_ctx):
    llm = ScriptedLanguageModel(extracted={"cost_profile": "Premium", "ntn_tier": None})
    state = ExtractNode(make_ctx(llm=llm))({"intent": "x", "class_id": "TimeCritical"})["dialogue"]
    assert state.satisfied["isolation_level"].source == "implied"
    assert state.satisfied["cost_profile"].source == "extracted"
    assert "ntn_tier" not in state.satisfied              # None values are dropped
    assert state.missing[0] == "vertical_domain"
    assert "cost_profile" not in state.missing


def test_extract_disabled_skips_the_llm(make_ctx):
    llm = ScriptedLanguageModel(extracted={"cost_profile": "Premium"})
    state = ExtractNode(make_ctx(llm=llm), use_llm=False)({"intent": "x", "class_id": "TimeCritical"})["dialogue"]
    assert "extract" not in llm.calls and "cost_profile" in state.missing


# ── kg_check ──────────────────────────────────────────────────────────────────

def _elicit(state):
    return {"dialogue": state, "history": [], "action": "done"}


def test_kg_check_asks_for_missing_slots(make_ctx, make_state):
    assert KGCheckNode(make_ctx())(_elicit(make_state(missing=["cost_profile"])))["action"] == "ask_slot"


def test_kg_check_detects_and_routes_to_conflict_resolution(make_ctx, make_state):
    state = make_state("TimeCritical", priority_level="Critical", cost_profile="LowCost")
    assert KGCheckNode(make_ctx())(_elicit(state))["action"] == "resolve_conflict"
    assert state.conflicts


def test_kg_check_uses_the_injected_detector(make_ctx, make_state):
    state = make_state("TimeCritical", priority_level="Critical", cost_profile="LowCost")
    KGCheckNode(make_ctx(detector=NullConflictDetector()))(_elicit(state))
    assert state.conflicts == []


def test_kg_check_stops_at_the_turn_limit(make_ctx, make_state):
    state = make_state(missing=["cost_profile"])
    state.turn_count = state.max_turns
    assert KGCheckNode(make_ctx())(_elicit(state))["action"] == "done"


def test_kg_check_activates_conditionals(make_ctx, make_state):
    state = make_state("TimeCritical", cycle_time_ms=0.5)
    KGCheckNode(make_ctx())(_elicit(state))
    assert "sync_accuracy_us" in state.missing and "sync_accuracy_us" in state.conditionally_activated


def test_kg_check_flags_escalation_once(make_ctx, make_state):
    state = make_state("MissionCritical", cycle_time_ms=2.0)
    KGCheckNode(make_ctx())(_elicit(state))
    assert state.escalation_proposed


# ── ask_slot ──────────────────────────────────────────────────────────────────

def test_ask_slot_records_the_parsed_answer(make_ctx, make_state):
    user = ScriptedUser(["Premium please"])
    state = make_state("TimeCritical", missing=["cost_profile"])
    out = AskSlotNode(make_ctx(user=user))({"dialogue": state, "history": [], "action": "ask_slot"})
    assert state.satisfied["cost_profile"].value == "LowCost"   # scripted: first enum option
    assert state.satisfied["cost_profile"].source == "user_provided"
    assert user.questions[0].kind == "slot" and user.questions[0].field == "cost_profile"
    assert [t.role for t in state.turn_history] == ["system", "user"]
    assert out["history"][-1] == {"role": "user", "content": "Premium please"}


def test_ask_slot_drops_unparseable_answers(make_ctx, make_state):
    class Unparseable(ScriptedLanguageModel):
        def parse_answer(self, slot, answer):
            return {"value": None}
    state = make_state(missing=["cost_profile"])
    AskSlotNode(make_ctx(llm=Unparseable()))({"dialogue": state, "history": [], "action": "ask_slot"})
    assert state.missing == [] and "cost_profile" not in state.satisfied


def test_ask_slot_skips_unknown_slots_without_asking(make_ctx, make_state):
    user = ScriptedUser()
    state = make_state(missing=["no_such_slot"])
    AskSlotNode(make_ctx(user=user))({"dialogue": state, "history": [], "action": "ask_slot"})
    assert state.missing == [] and user.questions == []


# ── resolve_conflict ──────────────────────────────────────────────────────────

def test_resolve_conflict_updates_the_object_field(make_ctx, make_state):
    user = ScriptedUser(["upgrade to Balanced"])
    state = make_state("TimeCritical", cost_profile="LowCost")
    state.add_conflict(ConflictFlag("kpi_conflict", "too cheap", "priority_level", "cost_profile"))
    ResolveConflictNode(make_ctx(user=user))({"dialogue": state, "history": [], "action": "resolve_conflict"})
    assert state.conflicts[0].resolved
    assert state.satisfied["cost_profile"].value == "Balanced"
    assert user.questions[0].kind == "conflict"


# ── assemble ──────────────────────────────────────────────────────────────────

def test_assemble_assigns_the_kg_deployment_mode(make_ctx, make_state):
    state = make_state("TimeCritical", deployment_scenario="Urban", vertical_domain="Manufacturing",
                       geographic_scope="Campus", cost_profile="Premium")
    out = AssembleNode(make_ctx())({"dialogue": state})
    si = out["service_intent"]
    assert si.class_slots["deployment_mode"] == "TN_Only"
    assert si.vertical_domain == "Manufacturing" and "vertical_domain" not in si.class_slots
    assert state.is_complete and state.service_intent is si


def test_assemble_keeps_a_user_chosen_mode(make_ctx, make_state):
    state = make_state("TimeCritical", deployment_scenario="Urban", deployment_mode="Hybrid")
    assert AssembleNode(make_ctx())({"dialogue": state})["service_intent"].class_slots["deployment_mode"] == "Hybrid"
