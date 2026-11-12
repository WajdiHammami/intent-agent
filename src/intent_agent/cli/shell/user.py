"""UserChannels for the shell: you answer (ShellUser) or the LLM does, shown live (ObservedUser)."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.markup import escape
from rich.panel import Panel

from intent_agent.cli.shell.spinner import Spinner
from intent_agent.cli.shell.style import PROMPT_STYLE
from intent_agent.cli.theme import console
from intent_agent.core.ports import History, UserChannel
from intent_agent.models.dialogue import Question


class Cancelled(Exception):
    """The user cancelled the running intent."""


def show_question(question: Question) -> None:
    if question.kind == "conflict":
        body = question.text.removeprefix("There is a conflict:").strip()
        console.print(Panel(escape(body), title="[bold yellow]⚠ Conflict[/]", title_align="left",
                            border_style="yellow", padding=(0, 1)))
    else:
        console.print(f"[bold cyan]?[/] {escape(question.text)}")


class ShellUser:
    """You answer each question at an inline prompt; /cancel or Ctrl+C aborts."""

    def __init__(self, spinner: Spinner) -> None:
        self.spinner = spinner
        self.session = PromptSession(style=PROMPT_STYLE)

    def ask(self, question: Question, history: History) -> str:
        self.spinner.hide()
        show_question(question)
        while True:
            try:
                answer = self.session.prompt(
                    [("class:answer", "  › ")],
                    placeholder=HTML("<placeholder>your answer  (/cancel to stop)</placeholder>"),
                ).strip()
            except (KeyboardInterrupt, EOFError):
                raise Cancelled
            if answer.lower() in ("/cancel", "/exit", "/quit"):
                raise Cancelled
            if answer:
                self.spinner.show("Reading the answer…")
                return answer


class ObservedUser:
    """Wraps another channel (e.g. SimulatedUser) and shows the exchange as it happens."""

    def __init__(self, inner: UserChannel, spinner: Spinner) -> None:
        self.inner = inner
        self.spinner = spinner

    def ask(self, question: Question, history: History) -> str:
        self.spinner.hide()
        show_question(question)
        self.spinner.show("Simulated user is answering…")
        answer = self.inner.ask(question, history)
        self.spinner.hide()
        console.print(f"  [green]›[/] {escape(answer)}  [dim](simulated)[/]")
        self.spinner.show("Reading the answer…")
        return answer
