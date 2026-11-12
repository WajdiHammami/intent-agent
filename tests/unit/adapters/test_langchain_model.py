"""The LangChain adapter with a fake chat model: chains, parsing, casting, caching."""

import json

import pytest

from intent_agent.adapters.llm.langchain_model import LangChainLanguageModel, cast_value, to_messages
from intent_agent.models.ontology import RequiresParameter
from tests.support.fake_chat import factory_for, scenario_responder


@pytest.fixture
def model():
    return LangChainLanguageModel(factory_for(scenario_responder("TimeCritical", {"cost_profile": "Premium"},
                                                                 answer="Premium, please")))


def test_classify(model):
    assert model.classify_intent("robots")["class_id"] == "TimeCritical"


def test_extract_parses_fenced_json(model):
    assert model.extract_slots("x", []) == {"cost_profile": "Premium"}


def test_parse_answer_casts_to_the_slot_type(model):
    slot = RequiresParameter(field="cycle_time_ms", dtype="float", question="?", unit="ms")
    assert model.parse_answer(slot, "ten") == {"value": 10.0, "confidence": "high"}


def test_simulate_user_returns_plain_text(model):
    assert model.simulate_user("persona", [{"role": "assistant", "content": "Q"}], "Q2") == "Premium, please"


def test_chat_json_and_model_cache():
    seen = []
    def respond(system, human):
        seen.append((system, human))
        return json.dumps({"ok": True})
    m = LangChainLanguageModel(factory_for(respond))
    assert m.chat_json("sys", "user") == {"ok": True}
    assert m.chat_model(64, True) is m.chat_model(64, True)
    assert m.chat_model(64, True) is not m.chat_model(64, False)


@pytest.mark.parametrize("dtype, raw, out", [
    ("float", "2.5", 2.5), ("int", "7", 7), ("bool", "yes", True),
    ("bool", "no", False), ("enum", "LEO", "LEO"), ("float", None, None),
])
def test_cast_value(dtype, raw, out):
    assert cast_value(dtype, raw) == out


def test_to_messages_maps_roles():
    msgs = to_messages([{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"},
                        {"role": "user", "content": "u"}])
    assert [m.type for m in msgs] == ["system", "ai", "human"]


def test_parse_resolution_returns_field_and_value(model):
    out = model.parse_resolution("too cheap", {"cost_profile": ["Balanced", "Premium"]}, "go Balanced")
    assert out == {"field": "cost_profile", "value": "Balanced"}
