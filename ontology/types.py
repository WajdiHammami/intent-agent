"""
KG Edge Type Definitions

All dataclasses representing edge types in the knowledge graph.

Key elements:
  - KPIProfile fields are numeric (float/int) for arithmetic comparisons.
  - ScenarioNode: four TN/NTN deployment scenarios (Table 1 of the paper).
  - DeploymentModeNode: TN_Only / NTN_Only / Hybrid performance profiles (Table 5).
  - ServiceClassNode gains primary_objective, dominant_kpi, deployment_tendency
    metadata (Tables 2 & 3) and the DEPLOYMENT_RECOMMENDATIONS matrix (Table 4)
    is stored in ontology_v3.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Union


# ── Edge: REQUIRES_PARAMETER ─────────────────────────────────────────────────

@dataclass
class RequiresParameter:
    """
    A slot that must be filled for this service class.
    The dialogue elicits this from the user if it cannot be extracted
    from the initial intent text.

    mandatory=False means: ask only if there are remaining turns after
    mandatory slots are satisfied.
    """
    field: str
    dtype: Literal["str", "float", "int", "enum", "bool"]
    question: str
    unit: Optional[str] = None
    enum_values: list[str] = field(default_factory=list)
    mandatory: bool = True


# ── Edge: REQUIRES_CONDITIONAL ────────────────────────────────────────────────

@dataclass
class RequiresConditional:
    """
    A slot activated only when a prior answer satisfies a condition.
    Implements dynamic graph expansion: new slots appear mid-dialogue
    based on what the user has already revealed.

    condition_op semantics:
        "eq"  → satisfied[condition_field].value == condition_value
        "gt"  → satisfied[condition_field].value > condition_value
        "lt"  → satisfied[condition_field].value < condition_value
        "gte" → satisfied[condition_field].value >= condition_value
        "lte" → satisfied[condition_field].value <= condition_value
        "in"  → satisfied[condition_field].value in condition_value (list)
    """
    field: str
    dtype: Literal["str", "float", "int", "enum", "bool"]
    condition_field: str
    condition_op: Literal["eq", "gt", "lt", "gte", "lte", "in"]
    condition_value: Union[str, float, int, list]
    question: str
    unit: Optional[str] = None
    enum_values: list[str] = field(default_factory=list)


# ── Edge: SIMILAR_TO ──────────────────────────────────────────────────────────

@dataclass
class SimilarTo:
    """
    When a satisfied slot's value matches the trigger, the LLM proposes
    revising the current class hypothesis to target_class.

    Class revision is ALWAYS user-confirmed — never automatic.
    similarity_score is a hint for ranking multiple revision proposals.
    """
    target_class: str
    trigger_field: str
    trigger_value: Union[str, float, bool, list]
    similarity_score: float = 0.7


# ── Edge: ACCOMPANIES ─────────────────────────────────────────────────────────

@dataclass
class Accompanies:
    """
    This class frequently co-occurs with companion_class in a single
    business intent. When the classifier scores both classes in the top-2
    with small margin AND this edge exists between them, the system
    proposes multi-class decomposition instead of forcing one class.
    """
    companion_class: str
    scenario: str = ""


# ── Edge: ESCALATES_TO ────────────────────────────────────────────────────────

@dataclass
class EscalatesTo:
    """
    When a user's answer satisfies the condition, the intent exceeds
    this class's scope. The system proposes escalation to target_class.

    Unlike SimilarTo (which is within a KPI group), EscalatesTo
    crosses group boundaries — e.g. MissionCritical → TimeCritical when
    cycle_time_ms < 5.
    """
    target_class: str
    condition_field: str
    condition_op: Literal["eq", "gt", "lt", "gte", "lte", "in"]
    condition_value: Union[str, float, int, list]
    reason: str = ""


# ── Edge: CONFLICT_RULE ───────────────────────────────────────────────────────

@dataclass
class ConflictRule:
    """
    A conflict detected when two satisfied slot values are incompatible.
    Checked after every slot update in the dialogue loop.

    Both subject and object refer to field names + expected string values
    (exact match via str() cast). For numeric comparisons use EscalatesTo
    or handle in stage logic.

    Example: priority_level="Critical" conflicts with cost_profile="LowCost"
    """
    subject_field: str
    subject_value: str
    object_field: str
    object_value: str
    message: str = ""


# ── Scenario Node (Table 1) ──────────────────────────────────────────────────

@dataclass
class ScenarioNode:
    """
    One of the four TN/NTN deployment scenarios used in the paper evaluation.
    Encodes Table 1: Representative integrated TN/NTN evaluation scenarios.
    """
    id: str                  # Urban | RuralRemote | Disaster | Maritime
    tn_availability: str     # qualitative: High / Low-Medium / Degraded / Very Low / None
    ntn_availability: str    # qualitative: Available / High
    characteristics: str


# ── Deployment Mode Node (Table 5) ───────────────────────────────────────────

@dataclass
class DeploymentModeNode:
    """
    Performance tendencies of a TN/NTN deployment mode.
    Encodes Table 5: Qualitative performance tendencies of deployment modes.
    """
    id: str              # TN_Only | NTN_Only | Hybrid
    coverage: str
    availability: str
    complexity_cost: str


# ── KPI Profile ───────────────────────────────────────────────────────────────

@dataclass
class KPIProfile:
    """
    The numeric KPI signature of a service class.
    Auto-filled into ServiceIntent without asking the user.

    Note: all fields are numeric (float/int) instead of string
    enums, enabling arithmetic comparisons in conflict and escalation rules.

    Sources per class — see the standards cited below:
        TimeCritical     : TS 22.104 §4.2, TS 22.261 Table 7.2-1
        MissionCritical  : TS 22.261 §6.2, TS 22.280 §5
        RemoteConnectivity: TR 38.821 §5.2
        MassiveConnectivity: ITU-R M.2083 §4.4, TS 22.261 §7.4
    """
    latency_max_ms: float          # hard upper bound on end-to-end latency
    reliability: float             # packet/message success probability [0,1]
    availability: float            # service uptime fraction [0,1]
    throughput_min_mbps: float     # minimum user-plane rate per endpoint
    connection_density_per_km2: int  # peak device density supported

    def to_dict(self) -> dict:
        return {
            "latency_max_ms": self.latency_max_ms,
            "reliability": self.reliability,
            "availability": self.availability,
            "throughput_min_mbps": self.throughput_min_mbps,
            "connection_density_per_km2": self.connection_density_per_km2,
        }


# ── Service Class Node ────────────────────────────────────────────────────────

@dataclass
class ServiceClassNode:
    """
    A complete service class definition with all KG edges.
    This is the atomic unit of the knowledge graph.

    Fields auto-filled into ServiceIntent (no user input needed):
        kpi_profile, traffic_pattern, implied_priority, implied_isolation

    Fields elicited via dialogue (defined in requires / requires_conditional):
        geographic_scope, cost_profile, isolation_level, class-specific slots
    """
    id: str
    group: Literal["A", "B", "C", "D"]
    display_name: str
    description: str
    typical_verticals: list[str]
    kpi_profile: KPIProfile
    traffic_pattern: str

    implied_priority: Optional[str] = None
    implied_isolation: Optional[str] = None

    # Paper metadata — Tables 2 and 3
    primary_objective: str = ""
    dominant_kpi: list[str] = field(default_factory=list)
    deployment_tendency: str = ""

    requires: list[RequiresParameter] = field(default_factory=list)
    requires_conditional: list[RequiresConditional] = field(default_factory=list)
    similar_to: list[SimilarTo] = field(default_factory=list)
    accompanies: list[Accompanies] = field(default_factory=list)
    escalates_to: list[EscalatesTo] = field(default_factory=list)
    conflicts: list[ConflictRule] = field(default_factory=list)
