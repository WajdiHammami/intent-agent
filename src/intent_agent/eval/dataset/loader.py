"""Load the benchmark from its JSON files (one per request type) and validate it."""

from __future__ import annotations

import json
from pathlib import Path

from intent_agent.eval.dataset.schema import REQUEST_TYPES, SERVICE_CLASSES, EvalSample

DATA_DIR = Path(__file__).parent / "data"


class DatasetError(ValueError):
    """The dataset files violate a structural invariant."""


def load_samples(data_dir: Path = DATA_DIR) -> list[EvalSample]:
    """All samples, ordered seed → paraphrased → ambiguous → contradictory."""
    samples: list[EvalSample] = []
    for rtype in REQUEST_TYPES:
        path = Path(data_dir) / f"{rtype}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        samples.extend(EvalSample.from_dict(row, request_type=rtype) for row in rows)
    validate(samples)
    return samples


def validate(samples: list[EvalSample]) -> None:
    ids = [s.id for s in samples]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise DatasetError(f"duplicate sample ids: {dupes}")
    for s in samples:
        if s.expected_class not in SERVICE_CLASSES:
            raise DatasetError(f"{s.id}: unknown class {s.expected_class!r}")
        if s.request_type == "contradictory" and not s.expected_conflicts:
            raise DatasetError(f"{s.id}: contradictory sample without expected_conflicts")
        if s.request_type != "contradictory" and s.expected_conflicts:
            raise DatasetError(f"{s.id}: only contradictory samples may expect conflicts")
