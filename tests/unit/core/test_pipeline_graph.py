"""Graph wiring and the IntentPipeline facade (scripted dependencies, no LLM)."""

import pytest

from adapters.llm.scripted import ScriptedLanguageModel
from adapters.users.scripted import ScriptedUser
from core.pipeline import IntentPipeline
from core.pipeline.graph import PIPELINE_ORDER, build_elicitation_graph


def _pipeline(**kw):
    return IntentPipeline(ScriptedLanguageModel("RemoteConnectivity"), ScriptedUser(), **kw)


def test_pipeline_nodes_run_in_order():
    graph = _pipeline().graph.get_graph()
    assert set(PIPELINE_ORDER) <= set(graph.nodes)
    edges = {(e.source, e.target) for e in graph.edges}
    assert {("classify", "extract"), ("extract", "elicit"), ("elicit", "assemble")} <= edges


def test_clarification_loop_routes_back_to_kg_check(make_ctx):
    edges = {(e.source, e.target) for e in build_elicitation_graph(make_ctx()).get_graph().edges}
    assert {("ask_slot", "kg_check"), ("resolve_conflict", "kg_check"),
            ("kg_check", "ask_slot"), ("kg_check", "resolve_conflict"), ("kg_check", "__end__")} <= edges


def test_stream_yields_every_stage():
    assert [node for node, _ in _pipeline().stream("ships")] == list(PIPELINE_ORDER)


def test_run_produces_a_complete_profile():
    result = _pipeline().run("internet for our ships")
    assert result.service_intent.service_class == "RemoteConnectivity"
    assert result.state.is_complete and result.state.turn_count > 0


def test_node_overrides_replace_a_stage():
    def fixed_assemble(ctx):
        from core.pipeline.nodes import AssembleNode
        inner = AssembleNode(ctx)
        def node(s):
            s["dialogue"].satisfied.pop("deployment_scenario", None)
            return inner(s)
        return node
    si = _pipeline(node_overrides={"assemble": fixed_assemble}).run("x").service_intent
    assert si.class_slots["deployment_mode"] == "TN_Only"   # RC + default Urban


def test_unknown_override_is_rejected():
    with pytest.raises(KeyError):
        _pipeline(node_overrides={"teleport": lambda ctx: None})


def test_max_turns_is_respected():
    result = _pipeline(max_turns=2).run("x")
    assert result.state.turn_count <= 2
