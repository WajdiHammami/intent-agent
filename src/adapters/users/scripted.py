"""ScriptedUser — answers from a fixed list or a function (tests, demos)."""

from __future__ import annotations

from typing import Callable, Iterable

from core.ports import History
from models.dialogue import Question


class ScriptedUser:
    def __init__(
        self,
        answers: Iterable[str] | Callable[[Question], str] = (),
        default: str = "(scripted answer)",
    ) -> None:
        self._fn = answers if callable(answers) else None
        self._answers = iter(()) if callable(answers) else iter(list(answers))
        self.default = default
        self.questions: list[Question] = []

    def ask(self, question: Question, history: History) -> str:
        self.questions.append(question)
        if self._fn is not None:
            return self._fn(question)
        return next(self._answers, self.default)
