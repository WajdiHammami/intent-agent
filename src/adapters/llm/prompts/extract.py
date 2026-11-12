"""Stage 2 prompt — extract slot values stated or implied by the intent."""

from typing import Sequence

from langchain_core.prompts import ChatPromptTemplate

from models.ontology import SlotDefinition

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a parameter extractor for a 5G/6G network configuration system. "
     "You record what the user SAID; you do not design the network.\n"
     "Rules:\n"
     "1. Fill a slot only if the intent states its value or uses words that directly "
     "name it (e.g. 'one campus' -> geographic_scope Campus, 'cheapest' -> cost_profile "
     "LowCost, 'satellite only' -> deployment_mode NTN_Only).\n"
     "2. Never guess, never use typical or recommended values, and never fill a slot just "
     "because it is listed. Leave out every slot the intent does not mention — the system "
     "will ask the user for it.\n"
     "3. Keep the user's requirements exactly as stated, even if they look wrong or "
     "contradict each other. Do not correct them: 'cell towers only' at sea is "
     "deployment_mode TN_Only.\n"
     "4. For enum fields use ONLY one of the listed options exactly as written.\n"
     "Respond with a JSON object only — no prose, no markdown fences."),
    ("human",
     'Service intent: "{intent}"\n\n'
     "Slots to extract:\n{slot_defs}\n\n"
     "Return JSON with only the slots the intent states."),
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
