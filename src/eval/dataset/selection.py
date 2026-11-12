"""Filtering helpers shared by the CLI and the experiments."""

from __future__ import annotations

from typing import Iterable, Sequence

from eval.dataset.schema import ADVERSARIAL_TYPES, CLASS_CODES, WELL_FORMED_TYPES, EvalSample


def resolve_class(name: str | None) -> str | None:
    """Accept a class code (TC, MC, RC, MaC/DC) or a full class name."""
    if not name:
        return None
    return CLASS_CODES.get(name.upper(), name)


def select(
    samples: Iterable[EvalSample],
    *,
    cls: str | None = None,
    rtype: str | None = None,
    ids: Sequence[str] | None = None,
    id_prefix: str | None = None,
    adversarial: bool | None = None,
) -> list[EvalSample]:
    """
    Filter samples. ``adversarial=True`` keeps ambiguous + contradictory,
    ``False`` keeps seed + paraphrased, ``None`` keeps both.
    """
    cls = resolve_class(cls)
    wanted_ids = {i.lower() for i in ids} if ids else None
    allowed = None if adversarial is None else (ADVERSARIAL_TYPES if adversarial else WELL_FORMED_TYPES)
    return [
        s for s in samples
        if (cls is None or s.expected_class == cls)
        and (rtype is None or s.request_type == rtype)
        and (wanted_ids is None or s.id.lower() in wanted_ids)
        and (id_prefix is None or s.id.startswith(id_prefix))
        and (allowed is None or s.request_type in allowed)
    ]


def dev_test_split(samples: Iterable[EvalSample]) -> tuple[list[EvalSample], list[EvalSample]]:
    """
    Deterministic 50/50 split, stratified by class × request type: within each
    group (in file order) samples alternate dev, test, dev, … Use dev to
    develop prompts and test to report — so a fix isn't tuned on the numbers
    it is judged by.
    """
    groups: dict[tuple[str, str], list[EvalSample]] = {}
    for s in samples:
        groups.setdefault((s.expected_class, s.request_type), []).append(s)
    dev_ids: set[str] = set()
    odd_groups = 0
    for group in groups.values():
        start = 0
        if len(group) % 2:            # odd group: alternate which half gets the extra sample
            start, odd_groups = odd_groups % 2, odd_groups + 1
        dev_ids.update(s.id for s in group[start::2])
    samples = [s for group in groups.values() for s in group]
    order = {s.id: i for i, s in enumerate(samples)}
    dev = sorted((s for s in samples if s.id in dev_ids), key=lambda s: order[s.id])
    test = sorted((s for s in samples if s.id not in dev_ids), key=lambda s: order[s.id])
    return dev, test
