"""Evaluation of the condition operators used by conditional and escalation edges."""

from __future__ import annotations


def evaluate_condition(
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
