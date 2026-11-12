"""Stage 3 — run the clarification-loop subgraph until the KG says stop."""

from __future__ import annotations

from intent_agent.core.pipeline.state import RECURSION_LIMIT, PipelineState


class ElicitNode:
    def __init__(self, subgraph) -> None:
        self.subgraph = subgraph

    def __call__(self, state: PipelineState) -> dict:
        final = self.subgraph.invoke(
            {"dialogue": state["dialogue"], "history": [], "action": "done"},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        return {"dialogue": final["dialogue"]}
