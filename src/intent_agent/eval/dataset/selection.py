"""Filtering helpers shared by the CLI and the experiments."""

from __future__ import annotations

from typing import Iterable, Sequence

from intent_agent.eval.dataset.schema import ADVERSARIAL_TYPES, CLASS_CODES, WELL_FORMED_TYPES, EvalSample


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
