import json
from collections import Counter

import pytest

from eval.dataset import (
    ADVERSARIAL,
    ALL_SAMPLES,
    CONTRADICTORY_SUBSET,
    WELL_FORMED,
    DatasetError,
    EvalSample,
    get_sample,
    load_samples,
    select,
    validate,
)


def test_two_hundred_intents_half_adversarial():
    assert len(ALL_SAMPLES) == 200
    assert len(WELL_FORMED) == len(ADVERSARIAL) == 100
    assert len(CONTRADICTORY_SUBSET) == 60


def test_class_by_type_balance():
    counts = Counter((s.expected_class, s.request_type) for s in ALL_SAMPLES)
    per_class = {"seed": 15, "paraphrased": 10, "ambiguous": 10, "contradictory": 15}
    for cls in ("TimeCritical", "MissionCritical", "RemoteConnectivity", "MassiveConnectivity"):
        for rtype, n in per_class.items():
            assert counts[(cls, rtype)] == n, (cls, rtype)


def test_every_conflict_type_is_covered():
    kinds = {tuple(s.expected_conflicts) for s in CONTRADICTORY_SUBSET}
    assert len(kinds) == 10


def test_select_filters_compose():
    picked = select(ALL_SAMPLES, cls="RC", rtype="contradictory")
    assert len(picked) == 15 and all(s.expected_class == "RemoteConnectivity" for s in picked)
    assert {s.request_type for s in select(ALL_SAMPLES, adversarial=False)} == {"seed", "paraphrased"}
    assert [s.id for s in select(ALL_SAMPLES, ids=["cf-43"])] == ["CF-43"]


def test_get_sample_is_case_insensitive():
    assert get_sample("cf-01").id == "CF-01"
    with pytest.raises(KeyError):
        get_sample("NOPE-1")


def test_sample_round_trips_through_dict():
    s = get_sample("CF-43")
    assert EvalSample.from_dict(s.to_dict()) == s and s.is_adversarial


def _write(tmp_path, rows_by_type):
    for rtype in ("seed", "paraphrased", "ambiguous", "contradictory"):
        (tmp_path / f"{rtype}.json").write_text(json.dumps(rows_by_type.get(rtype, [])))


def test_loader_reads_a_custom_directory(tmp_path):
    _write(tmp_path, {"seed": [{"id": "S1", "intent": "i", "scenario": "s", "expected_class": "TimeCritical"}]})
    assert [s.id for s in load_samples(tmp_path)] == ["S1"]


@pytest.mark.parametrize("bad", [
    [EvalSample("A", "i", "s", "TimeCritical"), EvalSample("A", "i", "s", "TimeCritical")],
    [EvalSample("A", "i", "s", "Telepathy")],
    [EvalSample("A", "i", "s", "TimeCritical", request_type="contradictory")],
    [EvalSample("A", "i", "s", "TimeCritical", expected_conflicts=["x"])],
])
def test_validate_rejects_broken_data(bad):
    with pytest.raises(DatasetError):
        validate(bad)
