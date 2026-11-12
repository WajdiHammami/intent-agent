"""IntentPipeline — the public entry point to the core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

from core.knowledge.conflicts import KGConflictDetector
from core.knowledge.graph import KnowledgeGraph
from core.pipeline.context import PipelineContext
from core.pipeline.graph import NodeFactory, build_elicitation_graph, build_pipeline_graph
from core.pipeline.state import RECURSION_LIMIT
from core.ports import ConflictDetector, LanguageModel, UserChannel
from models.dialogue import DialogueState
from models.service import ServiceIntent


@dataclass(frozen=True)
class PipelineResult:
    service_intent: ServiceIntent
    state: DialogueState


class IntentPipeline:
    """
    Turn one natural-language intent into a validated ServiceIntent.

        pipeline = IntentPipeline(llm, user)
        result = pipeline.run("Dedicated slice for 40 welding robots, 1 ms loop")

    Parameters
    ----------
    llm               : NLP adapter (classification, extraction, answer parsing)
    user              : who answers clarification questions
    kg                : knowledge graph (default ontology if omitted)
    conflict_detector : conflict strategy (KG rules if omitted)
    extraction        : False skips Stage 2 (ablation)
    node_overrides    : replace top-level nodes, {name: factory(ctx) -> node}
    """

    def __init__(
        self,
        llm: LanguageModel,
        user: UserChannel,
        *,
        kg: KnowledgeGraph | None = None,
        conflict_detector: ConflictDetector | None = None,
        extraction: bool = True,
        node_overrides: Mapping[str, NodeFactory] | None = None,
        max_turns: int = 12,
    ) -> None:
        kg = kg or KnowledgeGraph()
        self.context = PipelineContext(
            kg=kg,
            llm=llm,
            user=user,
            conflicts=conflict_detector or KGConflictDetector(kg),
            max_turns=max_turns,
        )
        self.graph = build_pipeline_graph(
            self.context, extraction=extraction, overrides=node_overrides
        )

    def run(self, intent: str) -> PipelineResult:
        final = self.graph.invoke({"intent": intent}, config={"recursion_limit": RECURSION_LIMIT})
        return PipelineResult(final["service_intent"], final["dialogue"])

    def stream(self, intent: str) -> Iterator[tuple[str, dict]]:
        """Yield (node_name, state_update) as each top-level node finishes."""
        for update in self.graph.stream(
            {"intent": intent},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="updates",
        ):
            yield from update.items()

    def mermaid(self) -> dict[str, str]:
        """Mermaid diagrams of the pipeline and the clarification loop."""
        return {
            "pipeline": self.graph.get_graph().draw_mermaid(),
            "clarification loop": build_elicitation_graph(self.context).get_graph().draw_mermaid(),
        }
