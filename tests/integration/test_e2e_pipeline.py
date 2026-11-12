"""
End to end through the real stack — LangChain adapter (prompts, chains, JSON
parsing) → LangGraph pipeline → knowledge graph → simulated user — with only
the chat model faked, so it runs without Ollama.
"""

import pytest

from adapters.llm.langchain_model import LangChainLanguageModel
from bootstrap import simulated_pipeline
from eval.checks import label_checks
from eval.dataset import ALL_SAMPLES, get_sample
from eval.metrics import aggregate, evaluate_sample
from tests.support.fake_chat import factory_for, scenario_responder


def _llm(class_id, extracted=None):
    return LangChainLanguageModel(factory_for(scenario_responder(class_id, extracted)))


@pytest.mark.parametrize("sample_id", ["SEED-TC-01", "SEED-MC-02", "SEED-RC-03", "SEED-DC-04"])
def test_one_intent_per_class(sample_id, kg):
    sample = get_sample(sample_id)
    llm = _llm(sample.expected_class, {"vertical_domain": "Energy"})
    result = simulated_pipeline(llm, sample.scenario, kg=kg).run(sample.intent)

    si, state = result.service_intent, result.state
    assert si.service_class == sample.expected_class
    assert si.vertical_domain == "Energy"                       # came from extraction
    assert state.turn_count > 0                                  # the rest was asked
    assert all(t.content == "Yes, that works." for t in state.turn_history if t.role == "user")
    assert si.class_slots["deployment_mode"] in ("TN_Only", "NTN_Only", "Hybrid")
    assert evaluate_sample(sample, result, kg).policy_valid


def test_contradiction_is_flagged_asked_and_resolved(kg):
    sample = get_sample("CF-01")
    llm = _llm("TimeCritical", {"cost_profile": "LowCost", "vertical_domain": "Manufacturing"})
    result = simulated_pipeline(llm, sample.scenario, kg=kg).run(sample.intent)

    questions = [t.content for t in result.state.turn_history if t.role == "system"]
    assert questions[0].startswith("There is a conflict:")
    assert all(c.resolved for c in result.state.conflicts)
    checks = {c.name: c.passed for c in label_checks(sample, result, evaluate_sample(sample, result, kg))}
    assert checks["conflict detected"] and checks["class"]


def test_the_whole_benchmark_runs_through_the_stack(kg):
    """Every one of the 200 intents completes and yields a valid profile."""
    results = []
    for s in ALL_SAMPLES:
        result = simulated_pipeline(_llm(s.expected_class), s.scenario, kg=kg).run(s.intent)
        results.append(evaluate_sample(s, result, kg))
    agg = aggregate(results)
    assert agg.n_samples == 200
    assert agg.classification_accuracy == 1.0
    assert agg.proper_termination_rate == 1.0
    assert agg.avg_profile_completeness == 1.0
