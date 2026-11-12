"""
KnowledgeGraph — typed query interface over the ontology.

Answers every decision the dialogue needs: which slots a class requires,
which conditionals a set of answers activates, which value combinations
conflict, when to escalate, which deployment mode fits, and when the
dialogue may stop (Algorithm 1, ``check_termination``).
"""

from __future__ import annotations

from typing import Iterable, Literal, Mapping, Sequence

from core.knowledge.catalog import (
    ALL_CLASSES,
    DEPLOYMENT_MODES,
    DEPLOYMENT_RECOMMENDATIONS,
    SCENARIOS,
    UNIVERSAL_CONFLICTS,
    UNIVERSAL_REQUIRES,
    VERTICAL_DOMAIN_PARAM,
)
from core.knowledge.conditions import evaluate_condition
from models.dialogue import ConflictFlag, DialogueState, SlotValue
from models.ontology import (
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


def _matches(satisfied: Mapping[str, SlotValue], field: str, value: str) -> bool:
    return field in satisfied and str(satisfied[field].value) == value


def _flag(rule: ConflictRule) -> ConflictFlag:
    return ConflictFlag(
        conflict_type="kpi_conflict",
        description=rule.message or (
            f"{rule.subject_field}={rule.subject_value} conflicts with "
            f"{rule.object_field}={rule.object_value}"
        ),
        subject_field=rule.subject_field,
        object_field=rule.object_field,
    )


class KnowledgeGraph:
    """
    Typed query interface over the knowledge graph.

    Usage:
        kg = KnowledgeGraph()                     # the default ontology
        slots = kg.get_required_slots("TimeCritical")
    """

    def __init__(self, classes: Iterable[ServiceClassNode] | None = None) -> None:
        self._classes: dict[str, ServiceClassNode] = {
            c.id: c for c in (ALL_CLASSES if classes is None else classes)
        }

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
        for r in UNIVERSAL_REQUIRES:
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
        satisfied: Mapping[str, SlotValue],
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
            if evaluate_condition(cond.condition_op, slot_val, cond.condition_value):
                activated.append(cond)

        return activated

    # ── Conflict detection ────────────────────────────────────────────────────

    def conflict_rules(self, class_id: str) -> list[ConflictRule]:
        """All rules checked for a class, in priority order: universal first."""
        return UNIVERSAL_CONFLICTS + self.get_class(class_id).conflicts

    def check_conflicts(
        self,
        class_id: str,
        satisfied: Mapping[str, SlotValue],
        rules: Sequence[ConflictRule] | None = None,
    ) -> list[ConflictFlag]:
        """
        Evaluate ConflictRule edges against the satisfied slots and return a
        ConflictFlag per triggered rule. ``rules`` defaults to every rule for
        the class (universal + class-specific).
        """
        return [
            _flag(rule)
            for rule in (self.conflict_rules(class_id) if rules is None else rules)
            if _matches(satisfied, rule.subject_field, rule.subject_value)
            and _matches(satisfied, rule.object_field, rule.object_value)
        ]

    # ── Class revision ────────────────────────────────────────────────────────

    def get_revision_candidates(
        self,
        class_id: str,
        satisfied: Mapping[str, SlotValue],
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
        satisfied: Mapping[str, SlotValue],
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
            if evaluate_condition(edge.condition_op, slot_val, edge.condition_value):
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
            return VERTICAL_DOMAIN_PARAM
        for r in self.get_required_slots(class_id):
            if r.field == field_name:
                return r
        node = self.get_class(class_id)
        for r in node.requires_conditional:
            if r.field == field_name:
                return r
        return None
