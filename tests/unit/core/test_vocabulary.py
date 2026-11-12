import pytest

from core.knowledge.values import InvalidValue
from models.dialogue import SlotValue


@pytest.fixture
def vocab(kg):
    return kg.vocabulary()


def test_vocabulary_merges_options_across_classes(vocab):
    assert {"LEO", "GEO", "UAV_HAPS", "Any"} <= set(vocab.options("ntn_tier"))
    assert set(vocab.options("priority_level")) == {"Critical", "High", "Normal", "Low"}
    assert vocab.options("cycle_time_ms") is None          # numeric, free value


@pytest.mark.parametrize("field, raw, value", [
    ("cost_profile", "balanced", "Balanced"),
    ("latency_sensitive", "True", True),
    ("device_density_per_km2", "1500", 1500),
    ("deployment_mode", "TN_Only", "TN_Only"),
])
def test_coerce(vocab, field, raw, value):
    assert vocab.coerce(field, raw) == value


@pytest.mark.parametrize("field, raw", [
    ("cost_profile", "Maritime"), ("vertical_domain", "Healthcare"),
    ("cycle_time_ms", "fast"), ("no_such_field", 1),
])
def test_coerce_rejects(vocab, field, raw):
    with pytest.raises(InvalidValue):
        vocab.coerce(field, raw)


def test_invalid_fields_ignores_unknown_fields(vocab):
    satisfied = {k: SlotValue(k, v, "extracted", "high") for k, v in
                 {"cost_profile": "Maritime", "vertical_domain": "Energy", "invented": "x"}.items()}
    assert vocab.invalid_fields(satisfied) == ["cost_profile"]


def test_vocabulary_is_built_once(kg):
    assert kg.vocabulary() is kg.vocabulary()
