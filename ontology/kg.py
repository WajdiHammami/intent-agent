"""
TMP 3.0 — KGOntology

Typed query interface over the TMP 3.0 Knowledge Graph.

TMP 3.0 changes vs TMP 2.0:
  1. _UNIVERSAL_REQUIRES adds deployment_scenario (Urban/RuralRemote/Disaster/Maritime)
     encoding the four TN/NTN evaluation scenarios (paper Table 1).
  2. geographic_scope enum extended to cover RemoteConnectivity values.
  3. New universal conflict: Maritime + TN_Only, and GEO + Critical.
  4. New KG query methods: get_scenario_profile, get_deployment_mode_profile,
     recommend_deployment_mode — expose Tables 1, 4, 5 for system reasoning.

Algorithm 1 (termination condition) is implemented in check_termination().
"""

from __future__ import annotations

from typing import Literal

from ontology.ontology_v3 import (
    DEPLOYMENT_MODES,
    DEPLOYMENT_RECOMMENDATIONS,
    SCENARIOS,
)
from ontology.types import (
    Accompanies,
    ConflictRule,
    DeploymentModeNode,
    EscalatesTo,
    RequiresConditional,
    RequiresParameter,
    ScenarioNode,
    ServiceClassNode,
    SimilarTo,
)
from pipeline.models import (
    ConflictFlag,
    DialogueState,
    SlotValue,
)


# Synthetic slot for vertical_domain — almost always extractable from intent,
# but needed as a fallback question when Stage 2 extraction fails.
_VERTICAL_DOMAIN_PARAM = RequiresParameter(
    field="vertical_domain",
    dtype="enum",
    enum_values=[
        "Manufacturing", "Industry", "Energy", "Transportation",
        "Government", "Media", "Entertainment", "SmartCity", "Automotive",
        "Maritime", "Agriculture",
    ],
    question=(
        "What vertical industry or sector does this deployment serve? "
        "(Manufacturing, Industry, Energy, Transportation, Government, "
        "Media, Entertainment, SmartCity, Automotive, Maritime, or Agriculture)"
    ),
)

# Universal slots applied to every class unless overridden by the class node.
_UNIVERSAL_REQUIRES: list[RequiresParameter] = [
    RequiresParameter(
        field="deployment_scenario",
        dtype="enum",
        enum_values=["Urban", "RuralRemote", "Disaster", "Maritime"],
        question=(
            "What is the deployment scenario context? "
            "Urban (strong TN), RuralRemote (sparse TN, coverage extension needed), "
            "Disaster (degraded TN, resilience critical), or "
            "Maritime (very low/no TN, NTN-dominant)?"
        ),
    ),
    RequiresParameter(
        field="geographic_scope",
        dtype="enum",
        enum_values=["Campus", "Urban", "CityWide", "Regional", "National",
                     "Maritime", "Rural", "Aerial", "Polar", "CityFringe"],
        question=(
            "What is the geographic coverage needed? "
            "(Campus, Urban, CityWide, Regional, National, "
            "Maritime, Rural, Aerial, Polar, or CityFringe)"
        ),
    ),
    RequiresParameter(
        field="isolation_level",
        dtype="enum",
        enum_values=["Shared", "Dedicated"],
        question=(
            "Do you need a dedicated network slice (higher cost, guaranteed SLA) "
            "or is a shared slice acceptable?"
        ),
    ),
    RequiresParameter(
        field="cost_profile",
        dtype="enum",
        enum_values=["LowCost", "Balanced", "Premium"],
        question=(
            "What cost profile fits your deployment? "
            "(LowCost / Balanced / Premium)"
        ),
    ),
]

# Universal conflict rules checked for every class.
_UNIVERSAL_CONFLICTS: list[ConflictRule] = [
    ConflictRule(
        subject_field="priority_level",
        subject_value="Critical",
        object_field="cost_profile",
        object_value="LowCost",
        message=(
            "Critical priority is incompatible with a LowCost profile — "
            "the SLA cannot be guaranteed. Use Balanced or Premium."
        ),
    ),
    ConflictRule(
        subject_field="isolation_level",
        subject_value="Dedicated",
        object_field="cost_profile",
        object_value="LowCost",
        message=(
            "A dedicated slice with LowCost profile is not offered — "
            "dedicated slices require at least a Balanced cost profile."
        ),
    ),
    # Maritime scenario + TN-only mode: TN availability is very low / none
    ConflictRule(
        subject_field="deployment_scenario",
        subject_value="Maritime",
        object_field="deployment_mode",
        object_value="TN_Only",
        message=(
            "TN-only deployment is not viable in Maritime scenarios where "
            "terrestrial network availability is very low or absent. "
            "Use NTN-only or Hybrid."
        ),
    ),
    # GEO tier + Critical priority: ~540 ms RTT cannot meet ≤1 ms / ≤10 ms
    ConflictRule(
        subject_field="ntn_tier",
        subject_value="GEO",
        object_field="priority_level",
        object_value="Critical",
        message=(
            "GEO satellite RTT (~540 ms) cannot meet Critical-priority latency "
            "requirements. Use LEO (10-40 ms RTT) or UAV/HAPS."
        ),
    ),
    # Disaster scenario + TN-only mode: TN is degraded/unreliable
    ConflictRule(
        subject_field="deployment_scenario",
        subject_value="Disaster",
        object_field="deployment_mode",
        object_value="TN_Only",
        message=(
            "TN-only deployment is not viable in Disaster scenarios where "
            "terrestrial infrastructure may be degraded or unavailable. "
            "Use Hybrid or NTN-only to ensure resilience."
        ),
    ),
]


def _evaluate_condition(
    op: str,
    slot_value: object,
    condition_value: object,
) -> bool:
    """Evaluate a single condition_op against a slot value."""
    try:
        if op == "eq":
            return slot_value == condition_value
        if op == "gt":
            return float(slot_value) > float(condition_value)  # type: ignore[arg-type]
        if op == "lt":
            return float(slot_value) < float(condition_value)  # type: ignore[arg-type]
        if op == "gte":
            return float(slot_value) >= float(condition_value)  # type: ignore[arg-type]
        if op == "lte":
            return float(slot_value) <= float(condition_value)  # type: ignore[arg-type]
        if op == "in":
            return slot_value in condition_value  # type: ignore[operator]
    except (TypeError, ValueError):
        return False
    return False


class KGOntology:
    """
    Typed query interface over the TMP 3.0 Knowledge Graph.

    Usage:
        from ontology.ontology_v3 import ALL_CLASSES
        kg = KGOntology(ALL_CLASSES)
        slots = kg.get_required_slots("TimeCritical")
    """

    def __init__(self, classes: list[ServiceClassNode]) -> None:
        self._classes: dict[str, ServiceClassNode] = {c.id: c for c in classes}

    # ── Class access ──────────────────────────────────────────────────────────

    def get_class(self, class_id: str) -> ServiceClassNode:
        if class_id not in self._classes:
            raise KeyError(f"Unknown service class: {class_id!r}")
        return self._classes[class_id]

    def all_class_ids(self) -> list[str]:
        return list(self._classes.keys())

    def get_group(self, class_id: str) -> Literal["A", "B", "C", "D"]:
        return self.get_class(class_id).group

    def classes_in_group(self, group: Literal["A", "B", "C", "D"]) -> list[str]:
        return [c.id for c in self._classes.values() if c.group == group]

    def get_display_name(self, class_id: str) -> str:
        return self.get_class(class_id).display_name

    def get_description(self, class_id: str) -> str:
        return self.get_class(class_id).description

    # ── Scenario / Mode / Recommendation queries ──────────────────────────────

    def get_scenario_profile(self, scenario: str) -> ScenarioNode:
        """Return Table 1 data for a deployment scenario."""
        if scenario not in SCENARIOS:
            raise KeyError(f"Unknown deployment scenario: {scenario!r}. "
                           f"Valid: {list(SCENARIOS)}")
        return SCENARIOS[scenario]

    def get_deployment_mode_profile(self, mode: str) -> DeploymentModeNode:
        """Return Table 5 performance tendencies for a deployment mode."""
        if mode not in DEPLOYMENT_MODES:
            raise KeyError(f"Unknown deployment mode: {mode!r}. "
                           f"Valid: {list(DEPLOYMENT_MODES)}")
        return DEPLOYMENT_MODES[mode]

    def recommend_deployment_mode(
        self, class_id: str, scenario: str
    ) -> tuple[str, str]:
        """
        Return (recommended_mode_string, reason) from Table 4.
        Both values are qualitative strings (may contain ' / ' for dual options).
        Returns ('Unknown', '') if the pair is not in the table.
        """
        return DEPLOYMENT_RECOMMENDATIONS.get((scenario, class_id), ("Unknown", ""))

    def assign_deployment_mode(self, class_id: str, scenario: str) -> str:
        """
        Stage 6: assign the primary KG-recommended deployment mode for
        (class_id, scenario). Returns the first option from the recommendation
        string (e.g. "TN_Only / Hybrid" → "TN_Only").
        Falls back to "Hybrid" for unknown combinations.
        """
        rec_str, _ = self.recommend_deployment_mode(class_id, scenario)
        if not rec_str or rec_str == "Unknown":
            return "Hybrid"
        return rec_str.split("/")[0].strip()

    def all_scenarios(self) -> list[str]:
        return list(SCENARIOS.keys())

    def all_deployment_modes(self) -> list[str]:
        return list(DEPLOYMENT_MODES.keys())

    # ── Slot queries ──────────────────────────────────────────────────────────

    def get_required_slots(self, class_id: str) -> list[RequiresParameter]:
        """
        All REQUIRES_PARAMETER slots for this class, including universal slots.

        If the class defines its own entry for a universal field (e.g.
        RemoteConnectivity overrides geographic_scope), the class-specific
        definition wins — the universal entry is dropped.

        For Group A: isolation_level is excluded (always implied as Dedicated).
        """
        node = self.get_class(class_id)

        # Field names explicitly defined by the class (these override universals)
        class_fields = {r.field for r in node.requires}

        # Build universal list, excluding Group A isolation and any class overrides
        universals: list[RequiresParameter] = []
        for r in _UNIVERSAL_REQUIRES:
            if node.group == "A" and r.field == "isolation_level":
                continue
            if r.field in class_fields:
                continue  # class-specific definition takes precedence
            universals.append(r)

        return universals + node.requires

    def get_mandatory_slot_names(self, class_id: str) -> set[str]:
        """Set of field names that must be satisfied for termination."""
        return {r.field for r in self.get_required_slots(class_id) if r.mandatory}

    # ── Conditional activation ────────────────────────────────────────────────

    def get_activated_conditionals(
        self,
        class_id: str,
        satisfied: dict[str, SlotValue],
    ) -> list[RequiresConditional]:
        """
        Evaluate all REQUIRES_CONDITIONAL edges for class_id.
        Returns those whose condition is met by the current satisfied slots.
        Already-satisfied conditionals are excluded.
        """
        node = self.get_class(class_id)
        activated: list[RequiresConditional] = []

        for cond in node.requires_conditional:
            if cond.field in satisfied:
                continue
            if cond.condition_field not in satisfied:
                continue
            slot_val = satisfied[cond.condition_field].value
            if _evaluate_condition(cond.condition_op, slot_val, cond.condition_value):
                activated.append(cond)

        return activated

    # ── Conflict detection ────────────────────────────────────────────────────

    def check_conflicts(
        self,
        class_id: str,
        satisfied: dict[str, SlotValue],
    ) -> list[ConflictFlag]:
        """
        Evaluate all ConflictRule edges (class-specific + universal).
        Returns ConflictFlag instances for each triggered rule.
        Called after every slot update in the dialogue loop.
        """
        node = self.get_class(class_id)
        all_rules = _UNIVERSAL_CONFLICTS + node.conflicts
        flags: list[ConflictFlag] = []

        for rule in all_rules:
            subject_present = (
                rule.subject_field in satisfied
                and str(satisfied[rule.subject_field].value) == rule.subject_value
            )
            object_present = (
                rule.object_field in satisfied
                and str(satisfied[rule.object_field].value) == rule.object_value
            )
            if subject_present and object_present:
                flags.append(
                    ConflictFlag(
                        conflict_type="kpi_conflict",
                        description=rule.message or (
                            f"{rule.subject_field}={rule.subject_value} conflicts with "
                            f"{rule.object_field}={rule.object_value}"
                        ),
                        subject_field=rule.subject_field,
                        object_field=rule.object_field,
                    )
                )

        return flags

    # ── Class revision ────────────────────────────────────────────────────────

    def get_revision_candidates(
        self,
        class_id: str,
        satisfied: dict[str, SlotValue],
    ) -> list[SimilarTo]:
        """
        Check SIMILAR_TO edges. Return those whose trigger slot is satisfied
        with a matching value. Sorted by similarity_score descending.
        """
        node = self.get_class(class_id)
        triggered: list[SimilarTo] = []

        for edge in node.similar_to:
            if edge.trigger_field not in satisfied:
                continue
            slot_val = satisfied[edge.trigger_field].value
            trigger = edge.trigger_value

            if isinstance(trigger, list):
                match = slot_val in trigger
            else:
                match = slot_val == trigger

            if match:
                triggered.append(edge)

        return sorted(triggered, key=lambda e: e.similarity_score, reverse=True)

    # ── Escalation ────────────────────────────────────────────────────────────

    def get_escalation_candidates(
        self,
        class_id: str,
        satisfied: dict[str, SlotValue],
    ) -> list[EscalatesTo]:
        """
        Check ESCALATES_TO edges. Return those whose condition is satisfied.
        Escalation crosses group boundaries — more significant than revision.
        """
        node = self.get_class(class_id)
        triggered: list[EscalatesTo] = []

        for edge in node.escalates_to:
            if edge.condition_field not in satisfied:
                continue
            slot_val = satisfied[edge.condition_field].value
            if _evaluate_condition(edge.condition_op, slot_val, edge.condition_value):
                triggered.append(edge)

        return triggered

    # ── Companions (MultiClass detection) ─────────────────────────────────────

    def get_companions(self, class_id: str) -> list[Accompanies]:
        return self.get_class(class_id).accompanies

    def has_accompanies_edge(self, class_a: str, class_b: str) -> bool:
        companions = {a.companion_class for a in self.get_companions(class_a)}
        return class_b in companions

    # ── KPI / Traffic resolution ──────────────────────────────────────────────

    def get_kpi_profile_dict(self, class_id: str) -> dict:
        return self.get_class(class_id).kpi_profile.to_dict()

    def get_traffic_pattern(self, class_id: str) -> str:
        return self.get_class(class_id).traffic_pattern

    def get_implied_fields(self, class_id: str) -> dict[str, str]:
        """
        Returns slot values implied by the class and filled automatically
        in Stage 2 without asking the user.

        priority_level is always implied (class group determines default).
        isolation_level is implied for Group A (always Dedicated).
        """
        _GROUP_DEFAULT_PRIORITY = {
            "A": "Critical",
            "B": "High",
            "C": "Normal",
            "D": "Low",
        }
        node = self.get_class(class_id)
        implied: dict[str, str] = {}
        implied["priority_level"] = (
            node.implied_priority
            if node.implied_priority is not None
            else _GROUP_DEFAULT_PRIORITY[node.group]
        )
        if node.implied_isolation is not None:
            implied["isolation_level"] = node.implied_isolation
        return implied

    # ── Termination condition (Algorithm 1) ───────────────────────────────────

    def check_termination(
        self,
        class_id: str,
        state: DialogueState,
    ) -> bool:
        """
        Algorithm 1 — Dialogue termination condition.

        Returns True when ALL of the following hold:
            1. vertical_domain is filled (always required)
            2. All mandatory REQUIRES_PARAMETER slots are satisfied
            3. All activated REQUIRES_CONDITIONAL slots are satisfied
            4. No ConflictFlag in state.conflicts is unresolved
        """
        if "vertical_domain" not in state.satisfied:
            return False

        mandatory = self.get_mandatory_slot_names(class_id)
        if not mandatory.issubset(set(state.satisfied.keys())):
            return False

        activated = self.get_activated_conditionals(class_id, state.satisfied)
        activated_fields = {c.field for c in activated}
        if not activated_fields.issubset(set(state.satisfied.keys())):
            return False

        if any(c for c in state.conflicts if not c.resolved):
            return False

        return True

    # ── Dialogue state initialisation ─────────────────────────────────────────

    def init_missing(self, class_id: str) -> list[str]:
        """
        Return the initial list of missing slot names for a class.
        Excludes slots that will be auto-filled (implied fields).
        """
        required = self.get_required_slots(class_id)
        implied = set(self.get_implied_fields(class_id).keys())
        return [r.field for r in required if r.mandatory and r.field not in implied]

    def get_slot_definition(
        self, class_id: str, field_name: str
    ) -> RequiresParameter | RequiresConditional | None:
        """
        Look up the full slot definition by field name.
        vertical_domain has a synthetic definition for fallback questioning.
        """
        if field_name == "vertical_domain":
            return _VERTICAL_DOMAIN_PARAM
        for r in self.get_required_slots(class_id):
            if r.field == field_name:
                return r
        node = self.get_class(class_id)
        for r in node.requires_conditional:
            if r.field == field_name:
                return r
        return None
