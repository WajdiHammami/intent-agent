"""`intent-agent check` — verify packages, knowledge graph, dataset and the Ollama model."""

from __future__ import annotations

import argparse
import importlib

from adapters.llm.ollama import ModelUnavailableError, OllamaLanguageModel
from cli.theme import console
from config import Settings

PACKAGES = [("langchain_core", "langchain-core"), ("langchain_ollama", "langchain-ollama"),
            ("langgraph", "langgraph"), ("ollama", "ollama"),
            ("rich", "rich"), ("prompt_toolkit", "prompt_toolkit")]


def _ok(msg: str) -> None:
    console.print(f"  [green]✓[/] {msg}")


def _fail(msg: str) -> None:
    console.print(f"  [red]✗[/] {msg}")


def run_checks(settings: Settings) -> int:
    problems = 0
    console.print("\n[bold]Python packages[/]")
    for module, name in PACKAGES:
        try:
            m = importlib.import_module(module)
            _ok(f"{name} {getattr(m, '__version__', '')}".rstrip())
        except ImportError:
            _fail(f"{name} missing — run: pip install -e .")
            problems += 1

    console.print("\n[bold]Knowledge graph & dataset[/]")
    try:
        from core.knowledge import KnowledgeGraph
        from eval.dataset import ADVERSARIAL, ALL_SAMPLES, WELL_FORMED
        _ok(f"ontology: {', '.join(KnowledgeGraph().all_class_ids())}")
        _ok(f"dataset: {len(ALL_SAMPLES)} intents ({len(WELL_FORMED)} well-formed, "
            f"{len(ADVERSARIAL)} adversarial)")
    except Exception as exc:
        _fail(f"failed to load: {exc}")
        problems += 1

    console.print("\n[bold]Ollama[/]")
    llm = OllamaLanguageModel(model=settings.model, base_url=settings.ollama_url)
    models = llm.list_models()
    if not models:
        _fail(f"cannot reach Ollama at {llm.base_url} — start it with: ollama serve")
        return 1
    _ok(f"server reachable at {llm.base_url} ({len(models)} models)")
    try:
        with console.status(f"[dim]Loading {llm.model}…[/]"):
            llm.check_connection()
            reply = llm.chat_json("Respond with JSON only.", 'Return {"status": "ok"}', max_tokens=20)
        _ok(f"model {llm.model} answers in JSON mode: {reply}")
    except ModelUnavailableError:
        _fail(f"model {llm.model} not pulled — run: ollama pull {llm.model}")
        console.print(f"[dim]      available: {', '.join(models)}  (use --model <tag>)[/]")
        problems += 1
    except Exception as exc:
        _fail(f"model call failed: {exc}")
        problems += 1

    console.print("\n" + ("[green]All checks passed.[/]" if not problems
                          else f"[red]{problems} problem(s) found.[/]"))
    return 1 if problems else 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("check", help="verify packages, knowledge graph, dataset and Ollama model")
    p.set_defaults(func=lambda args, settings: run_checks(settings))
