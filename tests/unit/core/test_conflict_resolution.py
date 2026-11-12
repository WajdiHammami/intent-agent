"""Conflict resolution: offer only valid changes, apply only valid answers, never fake a fix."""

from intent_agent.adapters.llm.scripted import ScriptedLanguageModel
from intent_agent.adapters.users.scripted import ScriptedUser
from intent_agent.core.pipeline import IntentPipeline
from intent_agent.core.pipeline.nodes import KGCheckNode, ResolveConflictNode
from intent_agent.models.dialogue import ConflictFlag


class Resolver(ScriptedLanguageModel):
    """Scripted model whose conflict answers come from a list (one per question)."""

    def __init__(self, choices, **kw):
        super().__init__(**kw)
        self.choices = list(choices)
        self.seen_options = []

    def parse_resolution(self, conflict, options, answer):
        self.seen_options.append(options)
        return self.choices.pop(0) if self.choices else super().parse_resolution(conflict, options, answer)


class Stubborn(ScriptedLanguageModel):
    """The user never says how to resolve anything."""

    def parse_resolution(self, conflict, options, answer):
        return {"field": None, "value": None}


def _elicit(state):
    return {"dialogue": state, "history": [], "action": "resolve_conflict"}


def _conflict(state, kg):
    for flag in kg.check_conflicts(state.current_class, state.satisfied):
        state.add_conflict(flag)
    return state.conflicts[0]


def test_question_offers_only_changeable_fields_and_new_values(make_ctx, make_state, kg):
    state = make_state("TimeCritical", priority_level="Critical", isolation_level="Dedicated", cost_profile="LowCost")          # Critical (implied) + LowCost
    _conflict(state, kg)
    llm, user = Resolver([{"field": "cost_profile", "value": "Premium"}]), ScriptedUser()
    ResolveConflictNode(make_ctx(llm=llm, user=user))(_elicit(state))
    assert llm.seen_options == [{"cost_profile": ["Balanced", "Premium"]}]   # no priority, no LowCost
    assert "cost_profile (Balanced, Premium)" in user.questions[0].text
    assert user.questions[0].kind == "conflict"


def test_valid_answer_changes_the_chosen_field(make_ctx, make_state, kg):
    state = make_state("MissionCritical", isolation_level="Shared", deployment_mode="TN_Only")
    conflict = _conflict(state, kg)
    llm = Resolver([{"field": "isolation_level", "value": "dedicated"}])
    ResolveConflictNode(make_ctx(llm=llm))(_elicit(state))
    assert state.satisfied["isolation_level"].value == "Dedicated"
    assert state.satisfied["deployment_mode"].value == "TN_Only"           # untouched
    assert conflict.resolved


def test_invalid_answers_leave_the_conflict_open(make_ctx, make_state, kg):
    for choice in ({"field": None, "value": None},
                   {"field": "deployment_mode", "value": "Dedicated"},      # the old CF-43 bug
                   {"field": "isolation_level", "value": "Shared"},         # not a change
                   {"field": "priority_level", "value": "Low"}):            # not offered
        state = make_state("MissionCritical", isolation_level="Shared", deployment_mode="TN_Only")
        conflict = _conflict(state, kg)
        ResolveConflictNode(make_ctx(llm=Resolver([choice])))(_elicit(state))
        assert not conflict.resolved, choice
        assert state.satisfied["deployment_mode"].value == "TN_Only"
        assert state.turn_count == 1                                       # the question was asked


def test_conflict_without_changeable_fields_does_not_loop(make_ctx, make_state):
    state = make_state("TimeCritical")
    state.add_conflict(ConflictFlag("kpi_conflict", "made up", "priority_level", "mystery_field"))
    user = ScriptedUser()
    ResolveConflictNode(make_ctx(user=user))(_elicit(state))
    assert state.conflicts[0].resolved and not state.conflicts[0].resolvable
    assert user.questions == []


def test_kg_check_reopens_a_conflict_that_is_still_there(make_ctx, make_state, kg):
    state = make_state("TimeCritical", priority_level="Critical", isolation_level="Dedicated", cost_profile="LowCost")
    conflict = _conflict(state, kg)
    state.resolve_conflict(conflict.description, "claimed")
    out = KGCheckNode(make_ctx())({"dialogue": state, "history": [], "action": "done"})
    assert not conflict.resolved and out["action"] == "resolve_conflict"


def test_kg_check_closes_conflicts_fixed_by_another_change(make_ctx, make_state, kg):
    state = make_state("TimeCritical", priority_level="Critical", isolation_level="Dedicated", cost_profile="LowCost")
    _conflict(state, kg)
    assert len(state.conflicts) == 2                      # Critical+LowCost and Dedicated+LowCost
    state.satisfied["cost_profile"].value = "Premium"     # one change fixes both
    KGCheckNode(make_ctx())({"dialogue": state, "history": [], "action": "done"})
    assert all(c.resolved for c in state.conflicts)


def test_pipeline_reasks_until_a_valid_resolution(kg):
    llm = Resolver([{"field": "deployment_mode", "value": "Dedicated"},
                    {"field": "isolation_level", "value": "Dedicated"}],
                   class_id="MissionCritical",
                   extracted={"isolation_level": "Shared", "deployment_mode": "TN_Only"})
    result = IntentPipeline(llm, ScriptedUser(), kg=kg).run("police dispatch, shared, TN only")
    conflict_questions = [t for t in result.state.turn_history
                          if t.role == "system" and t.content.startswith("There is a conflict: TN-only shared")]
    assert len(conflict_questions) == 2
    assert result.service_intent.isolation_level == "Dedicated"
    assert result.service_intent.class_slots["deployment_mode"] == "TN_Only"
    assert all(c.resolved for c in result.state.conflicts)


def test_a_user_who_never_resolves_cannot_produce_a_valid_profile(kg):
    llm = Stubborn(class_id="TimeCritical", extracted={"cost_profile": "LowCost"})
    result = IntentPipeline(llm, ScriptedUser(), kg=kg, max_turns=4).run("cheap robots")
    assert result.state.turn_count == 4
    assert any(not c.resolved for c in result.state.conflicts)
    assert not kg.check_termination("TimeCritical", result.state)
