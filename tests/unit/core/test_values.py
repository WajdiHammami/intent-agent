import pytest

from intent_agent.core.knowledge.values import InvalidValue, allowed_values, coerce_value
from intent_agent.models.ontology import RequiresParameter

ENUM = RequiresParameter("cost_profile", "enum", "?", enum_values=["LowCost", "Balanced", "Premium"])
BOOL = RequiresParameter("battery_powered", "bool", "?")
FLOAT = RequiresParameter("cycle_time_ms", "float", "?", unit="ms")


def test_allowed_values():
    assert allowed_values(ENUM) == ["LowCost", "Balanced", "Premium"]
    assert allowed_values(BOOL) == [True, False]
    assert allowed_values(FLOAT) is None


@pytest.mark.parametrize("slot, raw, value", [
    (ENUM, "balanced", "Balanced"),          # canonical casing
    (ENUM, " Premium ", "Premium"),
    (BOOL, "yes", True), (BOOL, False, False),
    (FLOAT, "2.5", 2.5),
])
def test_coerce_valid(slot, raw, value):
    assert coerce_value(slot, raw) == value


@pytest.mark.parametrize("slot, raw", [
    (ENUM, "Maritime"), (ENUM, None), (ENUM, ""), (BOOL, "maybe"), (FLOAT, "fast"),
])
def test_coerce_invalid(slot, raw):
    with pytest.raises(InvalidValue):
        coerce_value(slot, raw)
