"""
The ontology itself: the four service classes, the paper's tables, and the
universal slots and conflict rules. Pure data — queried via KnowledgeGraph.
"""

from intent_agent.core.knowledge.catalog.massive_connectivity import MASSIVE_CONNECTIVITY
from intent_agent.core.knowledge.catalog.mission_critical import MISSION_CRITICAL
from intent_agent.core.knowledge.catalog.remote_connectivity import REMOTE_CONNECTIVITY
from intent_agent.core.knowledge.catalog.tables import (
    DEPLOYMENT_MODES,
    DEPLOYMENT_RECOMMENDATIONS,
    SCENARIOS,
)
from intent_agent.core.knowledge.catalog.time_critical import TIME_CRITICAL
from intent_agent.core.knowledge.catalog.universal import (
    UNIVERSAL_CONFLICTS,
    UNIVERSAL_REQUIRES,
    VERTICAL_DOMAIN_PARAM,
)
from intent_agent.models.ontology import ServiceClassNode

ALL_CLASSES: list[ServiceClassNode] = [
    TIME_CRITICAL,
    MISSION_CRITICAL,
    REMOTE_CONNECTIVITY,
    MASSIVE_CONNECTIVITY,
]

__all__ = [
    "ALL_CLASSES", "DEPLOYMENT_MODES", "DEPLOYMENT_RECOMMENDATIONS", "MASSIVE_CONNECTIVITY",
    "MISSION_CRITICAL", "REMOTE_CONNECTIVITY", "SCENARIOS", "TIME_CRITICAL",
    "UNIVERSAL_CONFLICTS", "UNIVERSAL_REQUIRES", "VERTICAL_DOMAIN_PARAM",
]
