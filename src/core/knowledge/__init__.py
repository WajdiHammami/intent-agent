"""The knowledge layer: ontology data, graph queries and conflict detection."""

from core.knowledge.conflicts import KGConflictDetector, NullConflictDetector
from core.knowledge.graph import KnowledgeGraph

__all__ = ["KGConflictDetector", "KnowledgeGraph", "NullConflictDetector"]
