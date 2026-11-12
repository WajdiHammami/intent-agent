"""ConsoleUser — the UserChannel for one-shot terminal use (`intent-agent ask`)."""

from __future__ import annotations

from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt

from intent_agent.cli.theme import console
from intent_agent.core.ports import History
from intent_agent.models.dialogue import Question


class ConsoleUser:
    def ask(self, question: Question, history: History) -> str:
        if question.kind == "conflict":
            console.print(Panel(escape(question.text), title="[bold yellow]⚠ Conflict[/]",
                                title_align="left", border_style="yellow"))
        else:
            console.print(f"[bold cyan]?[/] {escape(question.text)}")
        answer = ""
        while not answer:
            answer = Prompt.ask("  [green]›[/]", console=console).strip()
        return answer
