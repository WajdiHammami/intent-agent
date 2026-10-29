#!/usr/bin/env python3
"""
run_interactive.py — Interactive 6G Intent-to-Service-Profile Demo

Start with a natural-language service intent. The framework will:
  1. Classify it into a service class  (TimeCritical / MissionCritical /
     RemoteConnectivity / MassiveConnectivity)
  2. Extract every slot it can directly from the intent text
  3. Ask targeted clarification questions for missing or conflicting slots
  4. Output a validated, orchestration-ready JSON service profile

Usage
-----
    python run_interactive.py
    python run_interactive.py --model /path/to/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
    python run_interactive.py --output profile.json
    python run_interactive.py --verbose

Type  quit  or  exit  at any prompt to stop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the package importable when run from the export/ directory
sys.path.insert(0, str(Path(__file__).parent))

from ontology.kg import KGOntology
from ontology.ontology_v3 import ALL_CLASSES
from pipeline.dialogue_runner import DialogueRunner
from pipeline.llm_client import LLMClient


# ── Terminal colour helpers ────────────────────────────────────────────────────

_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_RED   = "\033[91m"
_GREEN = "\033[92m"
_BLUE  = "\033[94m"
_CYAN  = "\033[96m"
_RESET = "\033[0m"


def _header(text: str) -> None:
    bar = "─" * 62
    print(f"\n{_BOLD}{_BLUE}{bar}{_RESET}")
    print(f"{_BOLD}{_BLUE}  {text}{_RESET}")
    print(f"{_BOLD}{_BLUE}{bar}{_RESET}")


def _info(label: str, value: str) -> None:
    print(f"  {_DIM}{label}:{_RESET}  {value}")


def _ok(text: str) -> None:
    print(f"  {_GREEN}✓{_RESET}  {text}")


def _warn(text: str) -> None:
    print(f"\n  {_RED}{_BOLD}⚠  CONFLICT:{_RESET}  {text}")


def _ask_user(prompt: str) -> str:
    """Print a question and read the user's answer from stdin."""
    print(f"\n  {_BOLD}{_CYAN}?{_RESET}  {prompt}")
    try:
        answer = input(f"  {_BOLD}>{_RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"\n{_RED}Interrupted.{_RESET}")
        sys.exit(0)
    if answer.lower() in ("quit", "exit", "q"):
        print(f"\n{_DIM}Exiting.{_RESET}")
        sys.exit(0)
    return answer


# ── Interactive runner ─────────────────────────────────────────────────────────

class InteractiveRunner(DialogueRunner):
    """
    Subclass of DialogueRunner that replaces LLM user simulation with real
    terminal input.  Every clarification question is printed to stdout;
    the user types their answer and presses Enter.

    All other pipeline stages (classification, extraction, assembly) run
    exactly as in the evaluation version.
    """

    def _ask(
        self,
        question: str,
        history: list[dict],
        state,
    ) -> tuple[str, list[dict]]:
        """Ask the real user and record the turn in the dialogue state."""
        answer = _ask_user(question)
        state.add_turn("system", question)
        state.add_turn("user", answer)
        updated_history = history + [
            {"role": "assistant", "content": f"Question: {question}"},
            {"role": "user",      "content": answer},
        ]
        return answer, updated_history

    def _log(self, msg: str) -> None:
        """Show stage-level progress in verbose mode; suppress LLM trace noise."""
        if self.verbose and msg.startswith("["):
            tag, _, rest = msg.partition("] ")
            print(f"  {_DIM}{tag}]{_RESET} {rest}")


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_intent(
    intent: str,
    llm: LLMClient,
    kg: KGOntology,
    verbose: bool = False,
) -> dict:
    """
    Run the full pipeline for one intent and return the output as a dict.

    Stages
    ------
    S1  Soft classification  — which of the 4 service classes?
    S2  Slot extraction      — what can be parsed directly from the text?
    S3  Clarification loop   — ask only what is genuinely missing or conflicting
    S5  Profile assembly     — build the structured ServiceIntent
    S6  Deployment mode      — assign TN/NTN mode from the knowledge base
    """
    runner = InteractiveRunner(
        scenario="",   # scenario context not needed — real user answers
        llm=llm,
        kg=kg,
        verbose=verbose,
    )

    # Stage 1
    _header("Stage 1 · Intent Classification")
    class_id = runner._stage1_classify(intent)
    _info("Detected class", f"{_BOLD}{class_id}{_RESET}")

    # Stage 2
    _header("Stage 2 · Slot Extraction from Intent Text")
    state = runner._stage2_extract(intent, class_id)

    if state.satisfied:
        print()
        for field, sv in state.satisfied.items():
            _ok(f"{_BOLD}{field}{_RESET} = {sv.value}  {_DIM}[{sv.source}]{_RESET}")
    else:
        print(f"  {_DIM}No slots extracted directly — all will be asked.{_RESET}")

    if state.missing:
        print(f"\n  {_DIM}Still needed: {state.missing}{_RESET}")

    # Stage 3  — only if something is missing or a conflict was found
    needs_dialogue = bool(state.missing) or any(not c.resolved for c in state.conflicts)
    if needs_dialogue:
        _header("Stage 3 · Clarification Dialogue")
        print(f"  {_DIM}The framework will ask only what is missing or conflicting.{_RESET}")
        state = runner._stage3_elicit(intent, state)
    else:
        _header("Stage 3 · Clarification Dialogue")
        print(f"  {_GREEN}Fast-path:{_RESET} all required slots were extracted — no questions needed.")

    # Stage 5 + 6
    _header("Stage 5–6 · Profile Assembly & Deployment Mode")
    si = runner._stage5_assemble(state)
    state.service_intent = si
    state.is_complete = True

    turns = state.turn_count
    _info("Dialogue turns used", str(turns))
    _info("Deployment mode assigned", si.class_slots.get("deployment_mode", "—"))

    return si.to_dict()


# ── Main ───────────────────────────────────────────────────────────────────────

_BANNER = f"""
{_BOLD}╔══════════════════════════════════════════════════════════════════╗
║  6G Intent → Service Profile Framework       Intent Agent        ║
║  Knowledge-Guided LLM · Integrated TN/NTN Orchestration          ║
╚══════════════════════════════════════════════════════════════════╝{_RESET}

Describe your 6G service requirement in plain English.
The framework classifies it, extracts parameters, asks only
the clarification questions that are genuinely needed, and
outputs a validated JSON service profile.

Type  {_BOLD}quit{_RESET}  at any prompt to exit.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive 6G intent-to-service-profile demo"
    )
    parser.add_argument(
        "--model",
        default="models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        help="Path to the GGUF model file (default: models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Save the output JSON to FILE after each intent (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stage-level pipeline messages",
    )
    args = parser.parse_args()

    print(_BANNER)

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"{_RED}Model file not found: {model_path}{_RESET}")
        print(f"Pass the correct path with:  --model /path/to/model.gguf")
        sys.exit(1)

    # Load LLM (once — reused across all intents in the session)
    print(f"{_DIM}Loading LLM ({model_path.name}) ...{_RESET}", end="", flush=True)
    llm = LLMClient(model_path=str(model_path))
    llm._get_model()   # trigger load now so the first intent starts instantly
    print(f"  {_GREEN}ready{_RESET}")

    kg = KGOntology(ALL_CLASSES)

    # Main loop — run until the user quits
    while True:
        _header("New Service Intent")
        try:
            intent = input(f"\n  {_BOLD}Your intent:{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not intent:
            print(f"  {_DIM}Empty input — try again or type quit.{_RESET}")
            continue
        if intent.lower() in ("quit", "exit", "q"):
            break

        try:
            profile = run_intent(intent, llm, kg, verbose=args.verbose)
        except KeyboardInterrupt:
            print(f"\n{_RED}Interrupted mid-run.{_RESET}\n")
            continue
        except Exception as exc:
            print(f"\n{_RED}Pipeline error: {exc}{_RESET}\n")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue

        # Print output
        _header("Output · Orchestration-Ready Service Profile")
        print(json.dumps(profile, indent=2))

        # Optionally save
        if args.output:
            out_path = Path(args.output)
            out_path.write_text(json.dumps(profile, indent=2))
            print(f"\n  {_GREEN}Profile saved → {out_path}{_RESET}")

        # Loop or quit
        print()
        try:
            again = input(f"  Run another intent? [{_BOLD}Y{_RESET}/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if again in ("n", "no"):
            break

    print(f"\n{_DIM}Session ended.{_RESET}\n")


if __name__ == "__main__":
    main()
