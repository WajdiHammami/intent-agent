"""Stage 3 prompt — parse a free-text answer into a typed slot value."""

from langchain_core.prompts import ChatPromptTemplate

from intent_agent.models.ontology import SlotDefinition

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a value parser for a network configuration system. "
     "Extract the typed slot value from the user's answer. "
     "Respond with JSON only — no prose, no markdown fences."),
    ("human",
     'Field: "{field}" (type: {type_hint})\n'
     'Question asked: "{question}"\n'
     'User answer: "{answer}"\n\n'
     'Return JSON: {{"value": <extracted value with correct type>, "confidence": "high|medium|low"}}'),
])


def type_hint(slot: SlotDefinition) -> str:
    if slot.dtype == "enum":
        return f"enum — must be exactly one of {slot.enum_values}"
    if getattr(slot, "unit", None):
        return f"{slot.dtype} in {slot.unit}"
    return slot.dtype
