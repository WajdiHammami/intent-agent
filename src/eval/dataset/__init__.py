"""
The 200-intent benchmark: 50 per service class, half well-formed (seed +
paraphrased), half adversarial (ambiguous + contradictory). Data lives in
``data/*.json``; collections are loaded once at import.
"""

from eval.dataset.loader import DATA_DIR, DatasetError, load_samples, validate
from eval.dataset.schema import (
    ADVERSARIAL_TYPES,
    CLASS_ABBREVIATIONS,
    REQUEST_TYPES,
    SERVICE_CLASSES,
    WELL_FORMED_TYPES,
    EvalSample,
)
from eval.dataset.selection import resolve_class, select

ALL_SAMPLES: list[EvalSample] = load_samples()
WELL_FORMED: list[EvalSample] = select(ALL_SAMPLES, adversarial=False)
ADVERSARIAL: list[EvalSample] = select(ALL_SAMPLES, adversarial=True)
CONTRADICTORY_SUBSET: list[EvalSample] = select(ALL_SAMPLES, rtype="contradictory")
NON_CONTRADICTORY: list[EvalSample] = [s for s in ALL_SAMPLES if s.request_type != "contradictory"]


def get_sample(sample_id: str) -> EvalSample:
    for s in ALL_SAMPLES:
        if s.id.lower() == sample_id.lower():
            return s
    raise KeyError(f"Unknown sample id {sample_id!r}")


__all__ = [
    "ADVERSARIAL", "ADVERSARIAL_TYPES", "ALL_SAMPLES", "CLASS_ABBREVIATIONS",
    "CONTRADICTORY_SUBSET", "DATA_DIR", "DatasetError", "EvalSample", "NON_CONTRADICTORY",
    "REQUEST_TYPES", "SERVICE_CLASSES", "WELL_FORMED", "WELL_FORMED_TYPES", "get_sample",
    "load_samples", "resolve_class", "select", "validate",
]
