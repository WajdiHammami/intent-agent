import pytest

from intent_agent.core.knowledge import KGConflictDetector, NullConflictDetector
from intent_agent.models.dialogue import SlotValue


def _sat(**values):
    return {k: SlotValue(k, v, "extracted", "high") for k, v in values.items()}


# One case per distinct conflict type in the ontology (5 universal + 5 class-specific).
CASES = [
    ("TimeCritical", dict(priority_level="Critical", cost_profile="LowCost"), "LowCost"),
    ("MissionCritical", dict(isolation_level="Dedicated", cost_profile="LowCost"), "dedicated"),
    ("RemoteConnectivity", dict(deployment_scenario="Maritime", deployment_mode="TN_Only"), "Maritime"),
    ("TimeCritical", dict(ntn_tier="GEO", priority_level="Critical"), "GEO"),
    ("MassiveConnectivity", dict(deployment_scenario="Disaster", deployment_mode="TN_Only"), "Disaster"),
    ("MissionCritical", dict(ntn_tier="GEO", latency_sensitive="True"), "latency"),
    ("MissionCritical", dict(isolation_level="Shared", deployment_mode="TN_Only"), "shared"),
    ("RemoteConnectivity", dict(ntn_tier="GEO", interactive_traffic="True"), "interactive"),
    ("MassiveConnectivity", dict(isolation_level="Dedicated", cost_profile="Premium"), "Premium"),
    ("MassiveConnectivity", dict(priority_level="Low", cost_profile="Premium"), "Low-priority"),
]


@pytest.mark.parametrize("class_id, values, keyword", CASES)
def test_every_conflict_type_is_detected(kg, class_id, values, keyword):
    flags = KGConflictDetector(kg).detect(class_id, _sat(**values))
    assert flags, f"no conflict for {values}"
    assert any(keyword.lower() in f.description.lower() for f in flags)


def test_no_conflict_for_a_consistent_profile(kg):
    sat = _sat(priority_level="Critical", cost_profile="Premium", isolation_level="Dedicated")
    assert KGConflictDetector(kg).detect("TimeCritical", sat) == []


def test_bool_values_match_their_string_rule(kg):
    flags = KGConflictDetector(kg).detect("RemoteConnectivity", _sat(ntn_tier="GEO", interactive_traffic=True))
    assert flags


@pytest.mark.parametrize("coverage, expected", [(0.0, 0), (0.25, 2), (0.5, 4), (1.0, 7)])
def test_coverage_limits_active_rules(kg, coverage, expected):
    assert KGConflictDetector(kg, coverage).active_rule_count("MissionCritical") == expected


def test_zero_coverage_detects_nothing(kg):
    sat = _sat(priority_level="Critical", cost_profile="LowCost")
    assert KGConflictDetector(kg, coverage=0.0).detect("TimeCritical", sat) == []
    assert KGConflictDetector(kg, coverage=0.25).detect("TimeCritical", sat)   # universal rule #1


def test_coverage_must_be_a_fraction(kg):
    with pytest.raises(ValueError):
        KGConflictDetector(kg, coverage=1.5)


def test_null_detector_never_flags(kg):
    assert NullConflictDetector().detect("TimeCritical", _sat(priority_level="Critical",
                                                              cost_profile="LowCost")) == []
