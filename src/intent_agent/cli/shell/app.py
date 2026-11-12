"""The interactive shell: a prompt loop that streams the LangGraph pipeline."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Sequence

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from intent_agent.adapters.llm.ollama import ModelUnavailableError, OllamaLanguageModel
from intent_agent.adapters.users.simulated import SimulatedUser
from intent_agent.cli.render import checks_panel, profile_panel
from intent_agent.cli.shell.commands import COMMANDS, EXIT
from intent_agent.cli.shell.completer import ShellCompleter
from intent_agent.cli.shell.spinner import Spinner
from intent_agent.cli.shell.style import PROMPT_STYLE
from intent_agent.cli.shell.user import Cancelled, ObservedUser, ShellUser
from intent_agent.cli.theme import ACCENT, console
from intent_agent.config import Settings
from intent_agent.core.knowledge import KnowledgeGraph
from intent_agent.core.pipeline import IntentPipeline, PipelineResult
from intent_agent.core.ports import UserChannel
from intent_agent.eval.checks import label_checks
from intent_agent.eval.dataset import ADVERSARIAL, ALL_SAMPLES, WELL_FORMED, EvalSample
from intent_agent.eval.experiments import EXPERIMENTS
from intent_agent.eval.metrics import evaluate_sample
from intent_agent.models.dialogue import DialogueState

HISTORY_FILE = Path.home() / ".intent_agent_history"
TIPS = [
    ("describe a service", "build a profile; you answer the follow-up questions"),
    ("/run CF-43", "replay a benchmark intent with the simulated user"),
    ("/random 3 adversarial", "spot-check random hard cases"),
    ("/help", "all commands"),
]


class _ConsoleTrace(logging.Handler):
    """Prints pipeline log lines (verbose mode) above the spinner."""

    def emit(self, record: logging.LogRecord) -> None:
        console.print(f"[dim]{escape(record.getMessage())}[/]")


class Shell:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.kg = KnowledgeGraph()
        self.llm = OllamaLanguageModel(settings.model, settings.ollama_url)
        self.llm_ready = False
        self.verbose = False
        self.json_mode = False
        self.scenario: str | None = None
        self.last_profile: dict | None = None
        self.spinner = Spinner()
        self.session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            completer=ShellCompleter(COMMANDS, self.llm_models, list(EXPERIMENTS)),
            complete_while_typing=True,
            style=PROMPT_STYLE,
            bottom_toolbar=self._toolbar,
        )
        self._trace = _ConsoleTrace()

    # ── model ─────────────────────────────────────────────────────────────────

    def llm_models(self) -> list[str]:
        return self.llm.list_models()

    def connect(self) -> bool:
        self.spinner.show(f"Loading {self.llm.model}…")
        try:
            self.llm.check_connection()
            self.llm_ready = True
        except ModelUnavailableError as exc:
            self.llm_ready = False
            self.spinner.hide()
            console.print(Panel(escape(str(exc)), title="[bold red]Model unavailable[/]",
                                title_align="left", border_style="red"))
        finally:
            self.spinner.hide()
        return self.llm_ready

    def switch_model(self, tag: str) -> None:
        self.llm = OllamaLanguageModel(tag, self.llm.base_url)
        self.settings = self.settings.with_overrides(model=tag)
        self.llm_ready = False
        if self.connect():
            console.print(f"[green]●[/] Now using [bold]{escape(tag)}[/]")

    # ── pipeline ──────────────────────────────────────────────────────────────

    def run_intent(self, intent: str, user: UserChannel) -> PipelineResult | None:
        """Stream the pipeline, rendering each stage as it finishes."""
        if not (self.llm_ready or self.connect()):
            return None
        log = logging.getLogger("intent_agent.pipeline")
        if self.verbose:
            log.addHandler(self._trace)
            log.setLevel(logging.INFO)
        pipeline = IntentPipeline(self.llm, user, kg=self.kg)
        si = state = None
        self.spinner.show("Classifying the intent…")
        try:
            for node, update in pipeline.stream(intent):
                self.spinner.hide()
                if node == "classify":
                    console.print(f"[green]●[/] Service class  [bold]{update['class_id']}[/]")
                    self.spinner.show("Extracting parameters from the text…")
                elif node == "extract":
                    self._show_extracted(update["dialogue"])
                    self.spinner.show("Checking requirements against the knowledge graph…")
                elif node == "elicit":
                    self._show_clarified(update["dialogue"])
                    self.spinner.show("Assembling the service profile…")
                elif node == "assemble":
                    si, state = update["service_intent"], update["dialogue"]
        except (Cancelled, KeyboardInterrupt):
            self.spinner.hide()
            console.print("[yellow]■ Cancelled.[/]\n")
            return None
        except Exception as exc:
            self.spinner.hide()
            console.print(Panel(escape(f"{type(exc).__name__}: {exc}"), title="[bold red]Error[/]",
                                title_align="left", border_style="red"))
            return None
        finally:
            self.spinner.hide()
            log.removeHandler(self._trace)

        console.print(profile_panel(si, as_json=self.json_mode))
        console.print("[dim]  /export to save as JSON · /json to toggle raw output[/]\n")
        self.last_profile = si.to_dict()
        return PipelineResult(si, state)

    def run_samples(self, samples: Sequence[EvalSample]) -> None:
        passed = 0
        for s in samples:
            console.print(Rule(f"[bold]{s.id}[/] · {s.expected_class} · {s.request_type}",
                               style="dim", align="left"))
            console.print(f"[bold]›[/] {escape(s.intent)}\n")
            result = self.run_intent(s.intent, ObservedUser(SimulatedUser(self.llm, s.scenario),
                                                            self.spinner))
            if result is None:
                if not self.llm_ready:
                    return
                continue
            checks = label_checks(s, result, evaluate_sample(s, result, self.kg))
            console.print(checks_panel(checks))
            passed += all(c.passed for c in checks)
        if len(samples) > 1:
            console.print(f"[bold]{passed}/{len(samples)}[/] sample(s) passed every check.\n")

    def _show_extracted(self, state: DialogueState) -> None:
        extracted = [(k, v) for k, v in state.satisfied.items() if v.source == "extracted"]
        implied = len(state.satisfied) - len(extracted)
        console.print(f"[green]●[/] Understood {len(extracted)} parameter(s) from the text"
                      + (f", {implied} implied by the class" if implied else ""))
        shown = extracted if self.verbose else extracted[:6]
        for k, v in shown:
            console.print(f"  [dim]⎿[/] {k} [dim]=[/] {escape(str(v.value))}")
        if len(shown) < len(extracted):
            console.print(f"  [dim]⎿ … {len(extracted) - len(shown)} more (/verbose shows all)[/]")
        if state.missing:
            console.print(f"  [dim]⎿ still needed: {', '.join(state.missing)}[/]")

    def _show_clarified(self, state: DialogueState) -> None:
        asked = sum(1 for t in state.turn_history if t.role == "system")
        if not asked:
            console.print("[green]●[/] Nothing missing or conflicting — no questions needed")
        else:
            conflicts = f", {len(state.conflicts)} conflict(s) resolved" if state.conflicts else ""
            console.print(f"[green]●[/] Clarified in {asked} question(s){conflicts}")

    # ── prompt loop ───────────────────────────────────────────────────────────

    def _toolbar(self):
        return HTML(
            f" <b>{escape_html(self.llm.model)}</b> ({'ready' if self.llm_ready else 'offline'})"
            f" · answers: {'simulated' if self.scenario else 'you'}"
            f" · verbose {'on' if self.verbose else 'off'} · json {'on' if self.json_mode else 'off'}"
            "    <b>/</b> commands  <b>↑</b> history  <b>Ctrl+D</b> exit"
        )

    def banner(self) -> None:
        body = Text.assemble(
            ("✻ ", ACCENT), ("Intent Agent", "bold"),
            ("\n\nTurn a plain-English 6G service request into a validated profile.\n", ""),
            ("The LLM reads; the knowledge graph decides what to ask and what conflicts.\n\n", "dim"),
            ("model    ", "dim"), (self.llm.model, "bold"), (f"  @ {self.llm.base_url}\n", "dim"),
            ("dataset  ", "dim"), (f"{len(ALL_SAMPLES)} intents ", ""),
            (f"({len(WELL_FORMED)} well-formed, {len(ADVERSARIAL)} adversarial)", "dim"),
        )
        console.print(Panel(body, border_style=ACCENT, padding=(1, 2), expand=False))
        tips = Table.grid(padding=(0, 2))
        tips.add_column(style="dim")
        tips.add_column()
        for row in TIPS:
            tips.add_row(*row)
        console.print(tips, "")

    def dispatch(self, line: str) -> object:
        try:
            name, *args = shlex.split(line)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            return None
        matches = [c for c in COMMANDS if c.startswith(name.lower())]
        command = COMMANDS.get(name.lower()) or (COMMANDS[matches[0]] if len(matches) == 1 else None)
        if command is None:
            console.print(f"[red]Unknown command {escape(name)}[/] — try /help")
            return None
        try:
            return command.handler(self, args)
        except KeyboardInterrupt:
            self.spinner.hide()
            console.print("[yellow]■ Interrupted.[/]")
            return None

    def loop(self) -> None:
        self.banner()
        self.connect()
        console.print()
        while True:
            try:
                text = self.session.prompt(
                    [("class:prompt", "› ")],
                    placeholder=HTML("<placeholder>Describe a service, or type / for commands</placeholder>"),
                ).strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            if not text:
                continue
            if text.startswith("/"):
                if self.dispatch(text) is EXIT:
                    break
                continue
            console.print()
            user = (ObservedUser(SimulatedUser(self.llm, self.scenario), self.spinner)
                    if self.scenario else ShellUser(self.spinner))
            self.run_intent(text, user)
        console.print("[dim]Bye.[/]")


def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
