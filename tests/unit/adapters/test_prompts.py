"""Prompt templates render with their variables (and literal JSON braces survive)."""

from intent_agent.adapters.llm import prompts
from intent_agent.models.ontology import RequiresParameter

SLOT = RequiresParameter(field="cost_profile", dtype="enum", question="Which cost profile?",
                         enum_values=["LowCost", "Balanced", "Premium"])


def _text(prompt, **kw):
    return "\n".join(m.content for m in prompt.format_messages(**kw))


def test_classify_prompt_lists_all_classes():
    text = _text(prompts.classify.PROMPT, intent="robots", class_list=prompts.classify.format_class_list())
    for cid in prompts.classify.CLASS_DESCRIPTIONS:
        assert f'"{cid}"' in text
    assert '{"class_id": "<ID>"' in text


def test_extract_prompt_describes_enum_options_and_units():
    unit_slot = RequiresParameter(field="cycle_time_ms", dtype="float", question="Cycle?", unit="ms")
    defs = prompts.extract.format_slot_definitions([SLOT, unit_slot])
    assert "options: ['LowCost', 'Balanced', 'Premium']" in defs and "unit: ms" in defs
    assert '{"geographic_scope": "Campus"}' in _text(prompts.extract.PROMPT, intent="x", slot_defs=defs)


def test_parse_answer_type_hints():
    assert prompts.parse_answer.type_hint(SLOT).startswith("enum — must be exactly one of")
    assert prompts.parse_answer.type_hint(
        RequiresParameter(field="t", dtype="float", question="?", unit="ms")) == "float in ms"


def test_simulate_user_prompt_includes_history():
    from langchain_core.messages import AIMessage, HumanMessage
    msgs = prompts.simulate_user.PROMPT.format_messages(
        scenario="You are X.", history=[AIMessage(content="Q1"), HumanMessage(content="A1")],
        question="Q2")
    assert [m.type for m in msgs] == ["system", "ai", "human", "human"]
    assert "You are X." in msgs[0].content and msgs[-1].content == "Question: Q2"


def test_parse_resolution_prompt_lists_options_as_json():
    import json
    opts = prompts.parse_resolution.format_options({"cost_profile": ["Balanced"], "cycle_time_ms": None})
    assert json.loads(opts) == {"cost_profile": ["Balanced"], "cycle_time_ms": "any"}
    text = _text(prompts.parse_resolution.PROMPT, conflict="c", options=opts, answer="a")
    assert '{"field": "<field or null>"' in text
