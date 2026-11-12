"""Tab completion: slash commands, then their arguments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from prompt_toolkit.completion import Completer, Completion

from intent_agent.eval.dataset import ALL_SAMPLES

if TYPE_CHECKING:
    from intent_agent.cli.shell.commands import SlashCommand

FILTER_TOKENS = ["TC", "MC", "RC", "MaC", "seed", "paraphrased", "ambiguous",
                 "contradictory", "adversarial", "well-formed"]


class ShellCompleter(Completer):
    def __init__(self, commands: dict[str, "SlashCommand"],
                 models: Callable[[], list[str]], experiments: list[str]) -> None:
        self.commands = commands
        self.arguments = {
            "/run": lambda: [s.id for s in ALL_SAMPLES],
            "/show": lambda: [s.id for s in ALL_SAMPLES],
            "/samples": lambda: FILTER_TOKENS,
            "/random": lambda: FILTER_TOKENS,
            "/selftest": lambda: FILTER_TOKENS,
            "/model": models,
            "/bench": lambda: experiments,
            "/scenario": lambda: ["off"],
        }

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        parts = text.split(" ")
        if len(parts) == 1:
            for name, cmd in self.commands.items():
                if name.startswith(text.lower()):
                    yield Completion(name, start_position=-len(text),
                                     display=f"{name} {cmd.usage}".rstrip(), display_meta=cmd.help)
            return
        word = parts[-1]
        for option in self.arguments.get(parts[0].lower(), lambda: [])():
            if option.lower().startswith(word.lower()):
                yield Completion(option, start_position=-len(word))
