"""prompt_toolkit styling for the shell."""

from prompt_toolkit.styles import Style

from intent_agent.cli.theme import ACCENT

PROMPT_STYLE = Style.from_dict({
    "prompt": f"{ACCENT} bold",
    "answer": "ansicyan bold",
    "placeholder": "#777777 italic",
    "bottom-toolbar": "noreverse #888888",
    "completion-menu.completion": "bg:#262626 #d0d0d0",
    "completion-menu.completion.current": f"bg:{ACCENT} #000000",
    "completion-menu.meta.completion": "bg:#262626 #808080",
    "completion-menu.meta.completion.current": f"bg:{ACCENT} #000000",
})
