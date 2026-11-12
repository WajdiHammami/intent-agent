"""
Layering rules, enforced by scanning imports:

    models   → nothing else in the package
    core     → models (and LangGraph); never adapters, eval, cli or LangChain models
    adapters → core, models
    eval     → never cli
"""

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "src" / "intent_agent"


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _layer_imports(layer: str) -> dict[str, set[str]]:
    return {str(p.relative_to(PKG)): _imports(p) for p in (PKG / layer).rglob("*.py")}


FORBIDDEN = {
    "models": ("intent_agent.core", "intent_agent.adapters", "intent_agent.eval", "intent_agent.cli",
               "langchain", "langgraph"),
    "core": ("intent_agent.adapters", "intent_agent.eval", "intent_agent.cli",
             "intent_agent.bootstrap", "langchain_ollama", "langchain_core", "ollama"),
    "adapters": ("intent_agent.eval", "intent_agent.cli", "intent_agent.bootstrap"),
    "eval": ("intent_agent.cli",),
}


@pytest.mark.parametrize("layer", list(FORBIDDEN))
def test_layer_dependencies(layer):
    violations = [
        f"{file} imports {name}"
        for file, names in _layer_imports(layer).items()
        for name in names
        if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN[layer])
    ]
    assert not violations, "\n".join(violations)


def test_one_node_per_file():
    nodes = PKG / "core" / "pipeline" / "nodes"
    for f in nodes.glob("*.py"):
        if f.name.startswith("_"):
            continue
        classes = [n for n in ast.parse(f.read_text()).body if isinstance(n, ast.ClassDef)]
        assert len(classes) <= 1, f"{f.name} defines {len(classes)} classes"


def test_prompts_live_in_their_own_modules():
    prompt_files = {p.stem for p in (PKG / "adapters" / "llm" / "prompts").glob("*.py")}
    assert {"classify", "extract", "parse_answer", "simulate_user"} <= prompt_files
    model_src = (PKG / "adapters" / "llm" / "langchain_model.py").read_text()
    assert "You are a" not in model_src, "prompt text belongs in adapters/llm/prompts/"


@pytest.mark.parametrize("path", sorted(str(p.relative_to(PKG)) for p in PKG.rglob("*.py")))
def test_modules_stay_small(path):
    limit = 400
    assert len((PKG / path).read_text().splitlines()) <= limit, f"{path} is over {limit} lines"
