"""A pausable status spinner (paused while the user types)."""

from __future__ import annotations

from rich.markup import escape

from cli.theme import ACCENT, console


class Spinner:
    def __init__(self) -> None:
        self._status = console.status("", spinner="dots", spinner_style=ACCENT)
        self._on = False

    def show(self, message: str) -> None:
        self._status.update(f"[dim]{escape(message)}[/]")
        if not self._on:
            self._status.start()
            self._on = True

    def hide(self) -> None:
        if self._on:
            self._status.stop()
            self._on = False
