"""CLI wiring: parsing, settings, and the commands that need no model."""

import json

import pytest

from cli.main import build_parser, main
from config import Settings


def test_global_options_build_settings(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    s = Settings.from_env()
    assert s.ollama_url == "http://localhost:11434" and s.model == "mistral"
    assert s.with_overrides(model=None, results_dir="out").results_dir.name == "out"


def test_every_command_is_registered():
    parser = build_parser()
    for argv in (["check"], ["selftest"], ["samples"], ["run", "CF-43"], ["ask", "x"],
                 ["bench", "eval"], ["bench", "sweep"], ["graph"], ["chat"]):
        assert callable(parser.parse_args(argv).func)


def test_no_command_means_the_shell():
    assert not hasattr(build_parser().parse_args([]), "func")


def test_selftest_passes(capsys):
    assert main(["selftest", "--class", "TC", "--type", "seed"]) == 0
    assert "Self-test passed" in capsys.readouterr().out


def test_samples_json_respects_filters(capsys):
    assert main(["samples", "--json", "--class", "MaC", "--adversarial"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 25 and {r["request_type"] for r in rows} == {"ambiguous", "contradictory"}


def test_samples_show_unknown_id_fails(capsys):
    assert main(["samples", "--show", "NOPE"]) == 1


def test_graph_prints_mermaid(capsys):
    assert main(["graph"]) == 0
    out = capsys.readouterr().out
    assert "classify" in out and "kg_check" in out


def test_bench_generate_dry_run(capsys):
    assert main(["bench", "generate", "--dry-run", "--ids", "SEED-TC-01", "--n", "1"]) == 0
    assert "would generate" in capsys.readouterr().out


def test_run_rejects_unknown_ids():
    assert main(["run", "NOPE-1"]) == 2


def test_shell_slash_commands_without_model(monkeypatch, capsys):
    pytest.importorskip("prompt_toolkit")
    from cli.shell.commands import COMMANDS, EXIT, _filters
    assert _filters(["RC", "ambiguous", "adversarial"]) == {
        "cls": "RemoteConnectivity", "rtype": "ambiguous", "adversarial": True}
    assert COMMANDS["/exit"].handler(None, []) is EXIT
    COMMANDS["/samples"].handler(None, ["TC", "contradictory"])
    assert "CF-01" in capsys.readouterr().out
