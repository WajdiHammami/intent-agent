import pytest

from adapters.llm.scripted import ScriptedLanguageModel
from adapters.users.scripted import ScriptedUser
from core.pipeline import IntentPipeline
from eval.checks import label_checks
from eval.dataset import get_sample
from eval.metrics import aggregate, evaluate_sample, print_metric_matrix, print_results_table


def _run(sample_id, class_id=None, extracted=None):
    s = get_sample(sample_id)
    llm = ScriptedLanguageModel(class_id or s.expected_class, extracted)
    return s, IntentPipeline(llm, ScriptedUser()).run(s.intent)


def test_correct_class_and_complete_profile(kg):
    s, run = _run("SEED-TC-01")
    m = evaluate_sample(s, run, kg)
    assert m.classification_correct and m.profile_completeness == 1.0
    assert m.proper_termination and m.policy_valid
    assert m.extraction_coverage < 1.0 and m.turns_to_termination > 0


def test_wrong_class_is_counted(kg):
    s, run = _run("SEED-TC-01", class_id="MassiveConnectivity")
    assert not evaluate_sample(s, run, kg).classification_correct


def test_conflict_keywords_are_matched(kg):
    s, run = _run("CF-01", extracted={"isolation_level": "Dedicated", "cost_profile": "LowCost"})
    m = evaluate_sample(s, run, kg)
    assert m.detected_conflicts == 2 and m.conflict_detection_rate == 1.0


def test_aggregate_and_breakdowns(kg):
    results = [evaluate_sample(*_run(i), kg) for i in ("SEED-TC-01", "SEED-MC-01", "CF-43")]
    agg = aggregate(results)
    assert agg.n_samples == 3 and agg.classification_accuracy == 1.0
    assert set(agg.per_request_type) == {"seed", "contradictory"}
    assert agg.contradiction_detection_accuracy == 0.0     # scripted model extracts nothing for CF-43
    assert agg.to_dict()["M1_classification_accuracy"] == 1.0


def test_aggregate_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        aggregate([])


def test_reports_print(capsys, kg):
    results = [evaluate_sample(*_run("SEED-RC-01"), kg)]
    agg = aggregate(results)
    print_results_table(results, agg)
    print_metric_matrix("T", {"a": agg.to_dict(), "b": agg.to_dict()}, baseline="a")
    out = capsys.readouterr().out
    assert "SEED-RC-01" in out and "(+0.0%)" in out


def test_label_checks(kg):
    s, run = _run("CF-01", extracted={"isolation_level": "Dedicated", "cost_profile": "LowCost"})
    checks = {c.name: c for c in label_checks(s, run, evaluate_sample(s, run, kg))}
    assert checks["class"].passed and checks["conflict detected"].passed
    assert "deployment mode" in checks and "policy valid" in checks
