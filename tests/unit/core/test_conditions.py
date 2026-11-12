import pytest

from core.knowledge.conditions import evaluate_condition


@pytest.mark.parametrize("op, value, target, expected", [
    ("eq", "V2X", "V2X", True),
    ("eq", "V2X", "GridProtection", False),
    ("gt", 150000, 100000, True),
    ("lt", 0.5, 1.0, True),
    ("lt", "0.5", 1.0, True),          # numeric strings are compared as numbers
    ("gte", 5, 5, True),
    ("lte", 6, 5, False),
    ("in", "Maritime", ["Maritime", "Disaster"], True),
    ("in", "Urban", ["Maritime", "Disaster"], False),
])
def test_operators(op, value, target, expected):
    assert evaluate_condition(op, value, target) is expected


def test_non_numeric_comparison_is_false_not_an_error():
    assert evaluate_condition("gt", "lots", 10) is False


def test_unknown_operator_is_false():
    assert evaluate_condition("approx", 1, 1) is False
