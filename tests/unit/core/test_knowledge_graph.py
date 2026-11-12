import pytest

from intent_agent.models.dialogue import ConflictFlag, SlotValue


def test_four_service_classes(kg):
    assert kg.all_class_ids() == ["TimeCritical", "MissionCritical",
                                  "RemoteConnectivity", "MassiveConnectivity"]


def test_unknown_class_raises(kg):
    with pytest.raises(KeyError):
        kg.get_class("Telepathy")


def test_group_a_never_asks_for_isolation(kg):
    fields = {s.field for s in kg.get_required_slots("TimeCritical")}
    assert "isolation_level" not in fields
    assert {"application_type", "cycle_time_ms", "deployment_scenario"} <= fields


def test_class_definition_overrides_the_universal_slot(kg):
    scope = [s for s in kg.get_required_slots("RemoteConnectivity") if s.field == "geographic_scope"]
    assert len(scope) == 1
    assert scope[0].enum_values == ["Maritime", "Rural", "Aerial", "Polar", "CityFringe"]


@pytest.mark.parametrize("class_id, implied", [
    ("TimeCritical", {"priority_level": "Critical", "isolation_level": "Dedicated"}),
    ("MissionCritical", {"priority_level": "High"}),
    ("RemoteConnectivity", {"priority_level": "Normal"}),
    ("MassiveConnectivity", {"priority_level": "Low"}),
])
def test_implied_fields(kg, class_id, implied):
    assert kg.get_implied_fields(class_id) == implied


def test_init_missing_excludes_implied_fields(kg):
    missing = kg.init_missing("TimeCritical")
    assert "priority_level" not in missing and "isolation_level" not in missing


def _sat(**values):
    return {k: SlotValue(k, v, "extracted", "high") for k, v in values.items()}


def test_conditional_slot_activates_on_its_condition(kg):
    activated = kg.get_activated_conditionals("TimeCritical", _sat(cycle_time_ms=0.5))
    assert "sync_accuracy_us" in {c.field for c in activated}
    assert "sync_accuracy_us" not in {
        c.field for c in kg.get_activated_conditionals("TimeCritical", _sat(cycle_time_ms=2.0))}


def test_satisfied_conditionals_are_not_reactivated(kg):
    activated = kg.get_activated_conditionals(
        "TimeCritical", _sat(cycle_time_ms=0.5, sync_accuracy_us=1.0))
    assert "sync_accuracy_us" not in {c.field for c in activated}


def test_escalation_from_mission_critical_to_time_critical(kg):
    esc = kg.get_escalation_candidates("MissionCritical", _sat(cycle_time_ms=2.0))
    assert [e.target_class for e in esc] == ["TimeCritical"]


@pytest.mark.parametrize("class_id, scenario, mode", [
    ("TimeCritical", "Urban", "TN_Only"),
    ("MissionCritical", "Urban", "Hybrid"),
    ("RemoteConnectivity", "Maritime", "NTN_Only"),
    ("MassiveConnectivity", "RuralRemote", "NTN_Only"),
    ("TimeCritical", "Atlantis", "Hybrid"),        # unknown → safe default
])
def test_deployment_mode_assignment(kg, class_id, scenario, mode):
    assert kg.assign_deployment_mode(class_id, scenario) == mode


def test_slot_definition_lookup(kg):
    assert kg.get_slot_definition("TimeCritical", "vertical_domain").dtype == "enum"
    assert kg.get_slot_definition("TimeCritical", "sync_accuracy_us").unit == "µs"
    assert kg.get_slot_definition("TimeCritical", "no_such_slot") is None


def _complete_mc(make_state, kg):
    values = {f: "x" for f in kg.get_mandatory_slot_names("MissionCritical")}
    values["vertical_domain"] = "Government"
    return make_state("MissionCritical", **values)


def test_termination_when_everything_is_filled(make_state, kg):
    assert kg.check_termination("MissionCritical", _complete_mc(make_state, kg))


def test_no_termination_without_vertical_domain(make_state, kg):
    state = _complete_mc(make_state, kg)
    del state.satisfied["vertical_domain"]
    assert not kg.check_termination("MissionCritical", state)


def test_no_termination_with_an_unresolved_conflict(make_state, kg):
    state = _complete_mc(make_state, kg)
    state.add_conflict(ConflictFlag("kpi_conflict", "open"))
    assert not kg.check_termination("MissionCritical", state)
    state.resolve_conflict("open", "fixed")
    assert kg.check_termination("MissionCritical", state)


def test_universal_rules_come_first(kg):
    rules = kg.conflict_rules("MassiveConnectivity")
    assert len(rules) == 7
    assert rules[0].subject_value == "Critical" and rules[-1].subject_value == "Low"
