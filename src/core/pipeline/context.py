"""Dependencies shared by every pipeline node."""

from __future__ import annotations

from dataclasses import dataclass

from core.knowledge.graph import KnowledgeGraph
from core.ports import ConflictDetector, LanguageModel, UserChannel


@dataclass
class PipelineContext:
    """Everything a node may use. Nodes never construct their own dependencies."""
    kg: KnowledgeGraph
    llm: LanguageModel
    user: UserChannel
    conflicts: ConflictDetector
    max_turns: int = 12
