"""`intent-agent graph` — print the LangGraph diagrams as Mermaid."""

from __future__ import annotations

import argparse

from intent_agent.adapters.llm.scripted import ScriptedLanguageModel
from intent_agent.adapters.users.scripted import ScriptedUser
from intent_agent.core.pipeline import IntentPipeline


def diagrams() -> dict[str, str]:
    return IntentPipeline(ScriptedLanguageModel(), ScriptedUser()).mermaid()


def run(args: argparse.Namespace, settings) -> int:
    for title, mermaid in diagrams().items():
        print(f"%% {title}\n{mermaid}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("graph", help="print the LangGraph diagrams (Mermaid)").set_defaults(func=run)
