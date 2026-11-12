"""
Against a real Ollama server — skipped unless the server is up and the
configured model (OLLAMA_MODEL, default llama3.1:8b) is pulled.

    pytest -m ollama
"""

import pytest

from adapters.llm.ollama import ModelUnavailableError
from bootstrap import create_language_model, simulated_pipeline
from config import Settings
from eval.dataset import get_sample

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def llm():
    try:
        return create_language_model(Settings.from_env())
    except ModelUnavailableError as exc:
        pytest.skip(f"Ollama not available: {exc}")


def test_classify_with_the_real_model(llm):
    assert llm.classify_intent("Smart meters reporting hourly across the city")["class_id"]


def test_one_contradictory_intent_end_to_end(llm, kg):
    sample = get_sample("CF-43")
    result = simulated_pipeline(llm, sample.scenario, kg=kg).run(sample.intent)
    assert result.service_intent.service_class
    assert result.state.is_complete
