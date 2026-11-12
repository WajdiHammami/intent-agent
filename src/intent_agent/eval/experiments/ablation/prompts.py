"""Prompt for the LLM conflict detector (ablation: LLM instead of KG rules)."""

SYSTEM = (
    "You are a 5G/6G network configuration validator. "
    "Given a service class and a set of collected parameters, identify any "
    "contradictory or mutually incompatible combinations. "
    "Examples of conflicts: Critical priority with LowCost profile, "
    "Dedicated slice with LowCost profile, GEO satellite with Critical priority, "
    "Maritime scenario with TN-only deployment mode. "
    "Respond with a JSON object only — no prose, no markdown fences."
)


def user_prompt(class_id: str, slots: str) -> str:
    return (
        f"Service class: {class_id}\n"
        f"Collected parameters:\n{slots}\n\n"
        "Are there any contradictory or incompatible parameter combinations?\n"
        'Return JSON: {"has_conflict": true/false, "conflicts": '
        '[{"field_a": "...", "field_b": "...", "description": "..."}]}'
    )
