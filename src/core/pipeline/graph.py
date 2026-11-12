"""
LangGraph wiring.

    pipeline:            START → classify → extract → elicit → assemble → END

    clarification loop:  START → kg_check ─┬─ resolve_conflict ─→ kg_check
                                           ├─ ask_slot ─────────→ kg_check
                                           └─ done → END

Any top-level node can be replaced by passing a factory in ``overrides``
(e.g. the Pure-LLM baseline swaps ``elicit`` and ``assemble``).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from langgraph.graph import END, START, StateGraph

from core.pipeline.context import PipelineContext
from core.pipeline.nodes import (
    AskSlotNode,
    AssembleNode,
    ClassifyNode,
    ElicitNode,
    ExtractNode,
    KGCheckNode,
    ResolveConflictNode,
)
from core.pipeline.state import ElicitState, PipelineState

Node = Callable[[Any], dict]
NodeFactory = Callable[[PipelineContext], Node]

PIPELINE_ORDER = ("classify", "extract", "elicit", "assemble")


def build_elicitation_graph(ctx: PipelineContext):
    g = StateGraph(ElicitState)
    g.add_node("kg_check", KGCheckNode(ctx))
    g.add_node("resolve_conflict", ResolveConflictNode(ctx))
    g.add_node("ask_slot", AskSlotNode(ctx))
    g.add_edge(START, "kg_check")
    g.add_conditional_edges(
        "kg_check",
        lambda s: s["action"],
        {"resolve_conflict": "resolve_conflict", "ask_slot": "ask_slot", "done": END},
    )
    g.add_edge("resolve_conflict", "kg_check")
    g.add_edge("ask_slot", "kg_check")
    return g.compile()


def default_nodes(ctx: PipelineContext, extraction: bool = True) -> dict[str, Node]:
    return {
        "classify": ClassifyNode(ctx),
        "extract": ExtractNode(ctx, use_llm=extraction),
        "elicit": ElicitNode(build_elicitation_graph(ctx)),
        "assemble": AssembleNode(ctx),
    }


def build_pipeline_graph(
    ctx: PipelineContext,
    *,
    extraction: bool = True,
    overrides: Mapping[str, NodeFactory] | None = None,
):
    nodes = default_nodes(ctx, extraction)
    for name, factory in (overrides or {}).items():
        if name not in nodes:
            raise KeyError(f"Unknown pipeline node {name!r}; expected one of {PIPELINE_ORDER}")
        nodes[name] = factory(ctx)

    g = StateGraph(PipelineState)
    for name in PIPELINE_ORDER:
        g.add_node(name, nodes[name])
    g.add_edge(START, PIPELINE_ORDER[0])
    for a, b in zip(PIPELINE_ORDER, PIPELINE_ORDER[1:]):
        g.add_edge(a, b)
    g.add_edge(PIPELINE_ORDER[-1], END)
    return g.compile()
