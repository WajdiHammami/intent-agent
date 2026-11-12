"""
Pure-LLM replacement nodes: the LLM, not the KG, decides what is missing,
what conflicts, and when to stop — the baseline the KG is compared against.

    elicit:   START → assess ─┬─ resolve → assess
                              ├─ ask ─────→ assess
                              └─ done → END
    assemble: the LLM picks the deployment mode, then the standard assembly runs.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from intent_agent.core.pipeline.context import PipelineContext
from intent_agent.core.pipeline.nodes import ElicitNode, assemble_profile
from intent_agent.core.pipeline.nodes._dialogue import ask_user
from intent_agent.eval.experiments.pure_llm import prompts
from intent_agent.models.dialogue import ConflictFlag, DialogueState, Question, SlotValue

MODES = ("TN_Only", "NTN_Only", "Hybrid")
EMPTY_ASSESSMENT = {"is_complete": False, "missing_fields": [], "questions": {}, "conflicts": []}


class PureLLMState(TypedDict, total=False):
    dialogue: DialogueState
    history: list[dict[str, str]]
    action: Literal["resolve", "ask", "done"]
    assessment: dict[str, Any]


def _json(ctx: PipelineContext, system: str, user: str, max_tokens: int, default: dict) -> dict:
    try:
        return ctx.llm.chat_json(system, user, max_tokens=max_tokens)
    except Exception:
        return dict(default)


def _user_value(field: str, value: Any) -> SlotValue:
    return SlotValue(field=field, value=value, source="user_provided", confidence="medium")


def assess_node(ctx: PipelineContext):
    def node(s: PureLLMState) -> dict:
        state = s["dialogue"]
        if state.is_at_turn_limit():
            return {"action": "done"}
        assessment = _json(ctx, prompts.ASSESS_SYSTEM, prompts.assess_user(state), 600,
                           EMPTY_ASSESSMENT)
        for c in assessment.get("conflicts", []):
            desc = (c.get("description") or "").strip() if isinstance(c, dict) else ""
            if desc:
                state.add_conflict(ConflictFlag(
                    conflict_type="kpi_conflict", description=desc,
                    subject_field=c.get("field_a", ""), object_field=c.get("field_b", ""),
                ))
        if any(not c.resolved for c in state.conflicts):
            action = "resolve"
        elif assessment.get("is_complete") or not assessment.get("missing_fields"):
            action = "done"
        else:
            action = "ask"
        return {"action": action, "assessment": assessment}
    return node


def resolve_node(ctx: PipelineContext):
    def node(s: PureLLMState) -> dict:
        state = s["dialogue"]
        conflict = next(c for c in state.conflicts if not c.resolved)
        question = Question(
            text=(f"There is a conflict: {conflict.description} Please choose a way to "
                  "resolve it — which value would you like to change?"),
            kind="conflict",
        )
        answer, history = ask_user(ctx, question, s["history"], state)
        chosen = _json(ctx, prompts.RESOLUTION_SYSTEM, prompts.resolution_user(conflict, answer),
                       64, {})
        if chosen.get("field") and chosen.get("value") is not None:
            state.satisfy_slot(_user_value(chosen["field"], chosen["value"]))
        state.resolve_conflict(conflict.description, answer)
        return {"dialogue": state, "history": history}
    return node


def ask_node(ctx: PipelineContext):
    def node(s: PureLLMState) -> dict:
        state, assessment = s["dialogue"], s["assessment"]
        field = assessment["missing_fields"][0]
        text = assessment.get("questions", {}).get(
            field, f"What is the {field.replace('_', ' ')} for this service deployment?")
        answer, history = ask_user(ctx, Question(text=text, field=field), s["history"], state)
        value = _json(ctx, prompts.PARSE_FIELD_SYSTEM,
                      prompts.parse_field_user(field, state.current_class, answer), 64, {}).get("value")
        if value is not None:
            state.satisfy_slot(_user_value(field, value))
        return {"dialogue": state, "history": history}
    return node


def build_elicit_graph(ctx: PipelineContext):
    g = StateGraph(PureLLMState)
    g.add_node("assess", assess_node(ctx))
    g.add_node("resolve", resolve_node(ctx))
    g.add_node("ask", ask_node(ctx))
    g.add_edge(START, "assess")
    g.add_conditional_edges("assess", lambda s: s["action"],
                            {"resolve": "resolve", "ask": "ask", "done": END})
    g.add_edge("resolve", "assess")
    g.add_edge("ask", "assess")
    return g.compile()


def elicit_factory(ctx: PipelineContext) -> ElicitNode:
    return ElicitNode(build_elicit_graph(ctx))


def assemble_factory(ctx: PipelineContext):
    def node(s: dict) -> dict:
        state = s["dialogue"]
        if "deployment_mode" not in state.satisfied:
            mode = _json(ctx, prompts.MODE_SYSTEM, prompts.mode_user(state), 64, {}).get(
                "deployment_mode", "Hybrid")
            state.satisfy_slot(_user_value("deployment_mode", mode if mode in MODES else "Hybrid"))
        service_intent = assemble_profile(ctx, state)
        state.service_intent, state.is_complete = service_intent, True
        return {"service_intent": service_intent, "dialogue": state}
    return node


NODE_OVERRIDES = {"elicit": elicit_factory, "assemble": assemble_factory}
