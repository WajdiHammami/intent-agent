"""Prompts for the Pure-LLM baseline: assess, parse a field, parse a resolution, pick a mode."""

from __future__ import annotations

import json

from intent_agent.eval.experiments.pure_llm.knowledge import CLASS_INFO, CONFLICT_RULES
from intent_agent.models.dialogue import ConflictFlag, DialogueState

ASSESS_SYSTEM = (
    "You are an expert 6G service intent validator with knowledge of "
    "TN/NTN network requirements. Analyze the service profile and return "
    "a JSON assessment. Be concise. Return ONLY the JSON object."
)

ASSESS_SCHEMA = """{
  "is_complete": <true if all required fields are present AND no unresolved conflicts>,
  "missing_fields": ["<field_name>", ...],
  "questions": {
    "<field_name>": "<concise natural-language question to ask the user>"
  },
  "conflicts": [
    {
      "field_a": "<field>",
      "value_a": "<value>",
      "field_b": "<field>",
      "value_b": "<value>",
      "description": "<one-sentence conflict explanation>"
    }
  ]
}"""


def assess_user(state: DialogueState) -> str:
    class_id = state.current_class
    info = CLASS_INFO.get(class_id, {})
    filled = "\n".join(f"  {k}: {v.value}" for k, v in state.satisfied.items()) or "  (none yet)"
    note = info.get("conditional_note", "")
    return f"""Service class: {class_id}
Class description: {info.get("description", "")}
Original intent: "{state.intent}"

Currently satisfied attributes:
{filled}

Required fields for {class_id}: {", ".join(info.get("required_fields", []))}
{f"Additional conditional fields: {note}" if note else ""}
Implied fields (already set, do NOT ask): {info.get("implied", {})}

{CONFLICT_RULES}

Assess the current profile. Return JSON:
{ASSESS_SCHEMA}"""


PARSE_FIELD_SYSTEM = (
    "You are a value extractor for a 6G network configuration system. "
    "Extract the typed value from the user's answer. "
    "Return ONLY a JSON object."
)


def parse_field_user(field: str, class_id: str, answer: str) -> str:
    return (
        f"Service class: {class_id}\n"
        f'Field: "{field}"\n'
        f'User answer: "{answer}"\n\n'
        'Return JSON: {"value": <extracted typed value or null if unclear>}'
    )


RESOLUTION_SYSTEM = (
    "You are a conflict resolution parser for a 6G configuration system. "
    "Return ONLY a JSON object."
)


def resolution_user(conflict: ConflictFlag, answer: str) -> str:
    return (
        f'Conflict: "{conflict.description}"\n'
        f'Fields involved: "{conflict.subject_field}", "{conflict.object_field}"\n'
        f'User response: "{answer}"\n\n'
        "Which field did the user agree to change, and to what new value? "
        'Return JSON: {"field": "<field_name>", "value": "<new_value>"}'
    )


MODE_SYSTEM = (
    "You are a 6G deployment mode recommender. "
    "Choose the most appropriate deployment mode. Return ONLY JSON."
)


def mode_user(state: DialogueState) -> str:
    scenario = state.satisfied.get("deployment_scenario")
    filled = {k: str(v.value) for k, v in state.satisfied.items()}
    return (
        f"Service class: {state.current_class}\n"
        f"Deployment scenario: {scenario.value if scenario else 'Unknown'}\n"
        f"Satisfied attributes: {json.dumps(filled)}\n\n"
        "Choose deployment mode — TN_Only, NTN_Only, or Hybrid — using these guidelines:\n"
        "  - Urban + TimeCritical/MissionCritical: TN_Only or Hybrid\n"
        "  - Maritime or Remote + RemoteConnectivity/MassiveConnectivity: NTN_Only or Hybrid\n"
        "  - Disaster scenario: Hybrid (TN may be degraded)\n"
        "  - RuralRemote + any class: Hybrid or NTN_Only\n"
        "  - MissionCritical always needs fallback: Hybrid unless explicitly NTN_Only\n"
        'Return JSON: {"deployment_mode": "TN_Only|NTN_Only|Hybrid"}'
    )
