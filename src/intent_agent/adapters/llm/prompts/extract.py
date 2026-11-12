"""Stage 2 prompt — extract slot values stated or implied by the intent."""

from typing import Sequence

from langchain_core.prompts import ChatPromptTemplate

from intent_agent.models.ontology import SlotDefinition

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a parameter extractor for a 5G/6G network configuration system. "
     "Extract slot values explicitly stated or clearly implied by the intent. "
     "For enum fields use ONLY one of the listed options exactly as written. "
     "Respond with a JSON object only — no prose, no markdown fences."),
    ("human",
     'Service intent: "{intent}"\n\n'
     "Slots to extract:\n{slot_defs}\n\n"
     'Return JSON with only extractable fields. Example: {{"geographic_scope": "Campus"}}'),
])


def format_slot_definitions(slots: Sequence[SlotDefinition]) -> str:
    lines = []
    for s in slots:
        entry = f'  "{s.field}" ({s.dtype}'
        if s.dtype == "enum":
            entry += f", options: {s.enum_values}"
        if getattr(s, "unit", None):
            entry += f", unit: {s.unit}"
        lines.append(entry + f"): {s.question}")
    return "\n".join(lines)
