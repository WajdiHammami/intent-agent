"""Plumbing shared by the experiments: sample filters, model connection, logging."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Protocol, Sequence

from adapters.llm.ollama import ModelUnavailableError, OllamaLanguageModel
from bootstrap import create_language_model
from config import Settings
from eval.dataset import REQUEST_TYPES, EvalSample, select


class Experiment(Protocol):
    NAME: str
    HELP: str

    def configure(self, parser: argparse.ArgumentParser) -> None: ...
    def run(self, args: argparse.Namespace, settings: Settings) -> int: ...


def add_sample_filters(parser: argparse.ArgumentParser, *, with_type: bool = True) -> None:
    parser.add_argument("--class", dest="cls", metavar="CLASS",
                        help="service class: TC, MC, RC, MaC (or full name)")
    if with_type:
        parser.add_argument("--type", dest="rtype", choices=REQUEST_TYPES)
    parser.add_argument("--ids", nargs="+", metavar="ID", help="only these sample ids")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the pipeline trace")


def pick(args: argparse.Namespace, pool: Sequence[EvalSample]) -> list[EvalSample]:
    samples = select(pool, cls=args.cls, rtype=getattr(args, "rtype", None), ids=args.ids)
    if not samples:
        sys.exit("No samples matched the filters — nothing to run.")
    return samples


def connect(settings: Settings) -> OllamaLanguageModel:
    print(f"Connecting to Ollama ({settings.model} @ {settings.ollama_url}) ...")
    try:
        return create_language_model(settings)
    except ModelUnavailableError as exc:
        sys.exit(str(exc))


def configure_logging(verbose: bool) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO, format="    %(message)s")
        logging.getLogger("intent_agent").setLevel(logging.INFO)
