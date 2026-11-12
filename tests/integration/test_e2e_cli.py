"""End to end through the command line, with the Ollama model swapped for a fake chat model."""

import json

import pytest

from intent_agent.adapters.llm.langchain_model import LangChainLanguageModel
from intent_agent.cli.main import main
from tests.support.fake_chat import factory_for, scenario_responder


class FakeOllama(LangChainLanguageModel):
    """Stands in for OllamaLanguageModel: same API, fake chat model."""

    def __init__(self, class_id="MissionCritical", extracted=None):
        super().__init__(factory_for(scenario_responder(class_id, extracted)))
        self.model, self.base_url = "fake", "http://fake"

    def list_models(self):
        return ["fake"]

    def check_connection(self):
        pass


@pytest.fixture
def fake_model(monkeypatch):
    def install(class_id="MissionCritical", extracted=None):
        llm = FakeOllama(class_id, extracted)
        factory = lambda settings, check=True: llm
        monkeypatch.setattr("intent_agent.cli.common.create_language_model", factory)
        monkeypatch.setattr("intent_agent.eval.experiments.common.create_language_model", factory)
        return llm
    return install


def test_run_command_grades_a_sample(fake_model, capsys):
    fake_model("MissionCritical")
    code = main(["run", "SEED-MC-01"])
    out = capsys.readouterr().out
    assert "Checks against labels" in out and "Service profile · MissionCritical" in out
    assert code in (0, 1)


def test_ask_with_a_simulated_persona(fake_model, tmp_path, capsys):
    fake_model("RemoteConnectivity")
    out_file = tmp_path / "profile.json"
    assert main(["ask", "Internet for our ships", "--scenario", "You run a fleet.",
                 "--output", str(out_file)]) == 0
    assert json.loads(out_file.read_text())["service_class"] == "RemoteConnectivity"


def test_bench_eval_writes_results(fake_model, tmp_path, capsys):
    fake_model("TimeCritical")
    assert main(["--results-dir", str(tmp_path), "bench", "eval", "--ids", "SEED-TC-01", "SEED-TC-02"]) == 0
    results = json.loads((tmp_path / "results_full.json").read_text())
    assert results["aggregate"]["n_samples"] == 2
    assert results["aggregate"]["M1_classification_accuracy"] == 1.0
    assert len(json.loads((tmp_path / "transcripts.json").read_text())) == 2

    assert main(["--results-dir", str(tmp_path), "bench", "eval", "--ids", "SEED-TC-01", "--compare"]) == 0
    assert "PREVIOUS vs NEW" in capsys.readouterr().out


def test_bench_ablation_runs_every_variant(fake_model, tmp_path):
    fake_model("TimeCritical", {"cost_profile": "LowCost"})
    assert main(["--results-dir", str(tmp_path), "bench", "ablation", "--ids", "CF-01"]) == 0
    saved = json.loads((tmp_path / "ablation_results.json").read_text())
    assert set(saved) == {"baseline", "no_s2", "no_kg", "no_s2_no_kg", "llm_conflicts"}
    m8 = {v: saved[v]["aggregate"]["M8_contradiction_detection_accuracy"] for v in saved}
    assert m8["baseline"] == 1.0 and m8["no_kg"] == 0.0     # the KG is what catches it


def test_bench_sweep_and_pure_llm(fake_model, tmp_path):
    fake_model("TimeCritical", {"cost_profile": "LowCost"})
    assert main(["--results-dir", str(tmp_path), "bench", "sweep", "--ids", "CF-01",
                 "--fractions", "0", "100"]) == 0
    sweep = json.loads((tmp_path / "kg_sweep_results.json").read_text())
    assert sweep["0"]["aggregate"]["M8_contradiction_detection_accuracy"] == 0.0
    assert sweep["100"]["aggregate"]["M8_contradiction_detection_accuracy"] == 1.0

    assert main(["--results-dir", str(tmp_path), "bench", "pure-llm", "--ids", "CF-01"]) == 0
    assert json.loads((tmp_path / "pure_llm_results.json").read_text())["n_samples"] == 1


def test_shell_replays_a_sample(fake_model, capsys, tmp_path, monkeypatch):
    pytest.importorskip("prompt_toolkit")
    from intent_agent.cli.shell import app as shell_app
    from intent_agent.config import Settings

    monkeypatch.setattr(shell_app, "HISTORY_FILE", tmp_path / "history")
    shell = shell_app.Shell(Settings())
    shell.llm, shell.llm_ready = FakeOllama("MissionCritical"), True
    shell.run_samples([__import__("intent_agent.eval.dataset", fromlist=["get_sample"]).get_sample("SEED-MC-01")])
    out = capsys.readouterr().out
    assert "Service class  MissionCritical" in out and "Checks against labels" in out
    assert "(simulated)" in out
