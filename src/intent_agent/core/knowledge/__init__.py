"""The knowledge layer: ontology data, graph queries and conflict detection."""

from intent_agent.core.knowledge.conflicts import KGConflictDetector, NullConflictDetector
from intent_agent.core.knowledge.graph import KnowledgeGraph

__all__ = ["KGConflictDetector", "KnowledgeGraph", "NullConflictDetector"]
