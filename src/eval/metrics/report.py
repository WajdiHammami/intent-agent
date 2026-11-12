"""Plain-text result tables printed by the experiments."""

from __future__ import annotations

from typing import Mapping, Sequence

from eval.metrics.aggregate import AggregateMetrics
from eval.metrics.sample import SampleResult

#: (key in AggregateMetrics.to_dict(), label, is_rate)
METRIC_ROWS: list[tuple[str, str, bool]] = [
    ("M1_classification_accuracy", "M1  Classification Accuracy", True),
    ("M2_avg_extraction_coverage", "M2  Extraction Coverage", True),
    ("M3_avg_turns", "M3  Avg Turns", False),
    ("M4_avg_profile_completeness", "M4  Profile Completeness", True),
    ("M5_conflict_detection_rate", "M5  Conflict Detection Rate", True),
    ("M6_proper_termination_rate", "M6  Proper Termination Rate", True),
    ("M7_policy_validity_rate", "M7  Policy Validity Rate", True),
    ("M8_contradiction_detection_accuracy", "M8  Contradiction Det. Acc.", True),
]


def _fmt(value: object, is_rate: bool) -> str:
    if isinstance(value, float):
        return f"{value:.1%}" if is_rate else f"{value:.2f}"
    return "—" if value is None else str(value)


def print_metric_matrix(
    title: str,
    columns: Mapping[str, Mapping[str, object]],
    rows: Sequence[tuple[str, str, bool]] = METRIC_ROWS,
    baseline: str | Mapping[str, object] | None = None,
    col_width: int = 16,
) -> None:
    """
    One row per metric, one column per run (variant, KG coverage, …).
    ``baseline`` — a column name or an aggregate dict — adds a delta to every
    other column.
    """
    base = columns.get(baseline) if isinstance(baseline, str) else baseline
    width = 42 + col_width * len(columns)
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(f"{'Metric':<42}" + "".join(f"{name[:col_width - 1]:>{col_width}}" for name in columns))
    print("-" * width)
    for key, label, is_rate in rows:
        line = f"  {label:<40}"
        for name, agg in columns.items():
            val = agg.get(key)
            cell = _fmt(val, is_rate)
            ref = base.get(key) if base and base is not agg else None
            if isinstance(val, float) and isinstance(ref, float):
                cell += f"({val - ref:+.1%})" if is_rate else f"({val - ref:+.2f})"
            line += f"{cell:>{col_width}}"
        print(line)
    print("=" * width)


def print_results_table(results: list[SampleResult], agg: AggregateMetrics) -> None:
    """Print a formatted results table to stdout."""

    # ── Per-sample table ──────────────────────────────────────────────────────
    W = 120
    print("\n" + "=" * W)
    print(
        f"{'ID':<12} {'Expected':<22} {'Predicted':<22} {'Cls':>4} "
        f"{'Cov':>6} {'Turns':>6} {'Compl':>6} {'ConflDR':>8} "
        f"{'ProperT':>8} {'PolV':>5} {'Type':<14}"
    )
    print("-" * W)
    for r in results:
        cls_ok  = "Y" if r.classification_correct else "N"
        pt_ok   = "Y" if r.proper_termination else "N"
        pv_ok   = "Y" if r.policy_valid else "N"
        print(
            f"{r.sample_id:<12} {r.expected_class:<22} {r.predicted_class:<22} "
            f"{cls_ok:>4} {r.extraction_coverage:>6.2f} {r.turns_to_termination:>6} "
            f"{r.profile_completeness:>6.2f} {r.conflict_detection_rate:>8.2f} "
            f"{pt_ok:>8} {pv_ok:>5} {r.request_type:<14}"
        )

    # ── Step 1–2 aggregate (original metrics) ─────────────────────────────────
    print("=" * W)
    print(f"\n{'AGGREGATE — Steps 1 & 2 (Core Metrics)'}")
    print(f"  Samples                         : {agg.n_samples}")
    print(f"  M1  Classification Accuracy     : {agg.classification_accuracy:.1%}")
    print(f"  M2  Extraction Coverage         : {agg.avg_extraction_coverage:.1%}  (±{agg.std_extraction_coverage:.2f})")
    print(f"  M3  Avg Turns to Termination    : {agg.avg_turns:.2f}  (±{agg.std_turns:.2f})")
    print(f"  M4  Avg Profile Completeness    : {agg.avg_profile_completeness:.1%}")
    print(f"  M5  Conflict Detection Rate     : {agg.overall_conflict_detection_rate:.1%}")
    print(f"  M6  Proper Termination Rate     : {agg.proper_termination_rate:.1%}")

    # ── Step 3 aggregate (new metrics) ────────────────────────────────────────
    print(f"\n{'AGGREGATE — Step 3 (Policy & Contradiction Metrics)'}")
    print(f"  M7  Policy Validity Rate        : {agg.policy_validity_rate:.1%}")
    print(f"  M8  Contradiction Det. Accuracy : {agg.contradiction_detection_accuracy:.1%}  (contradictory samples only)")

    # ── Per-request-type breakdown ────────────────────────────────────────────
    print(f"\n{'Per-request-type breakdown':}")
    rt_header = (
        f"  {'Type':<16} {'n':>3} {'ClsAcc':>7} {'PolValid':>9} "
        f"{'AvgTurns':>9}"
    )
    print(rt_header)
    print("  " + "-" * (len(rt_header) - 2))
    for rtype, m in agg.per_request_type.items():
        print(
            f"  {rtype:<16} {m['n']:>3} "
            f"{m['classification_accuracy']:>7.1%} "
            f"{m['policy_validity_rate']:>9.1%} "
            f"{m['avg_turns']:>9.2f}"
        )

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print(f"\n{'Per-class breakdown':}")
    cls_header = (
        f"  {'Class':<24} {'n':>3} {'ClsAcc':>7} {'Coverage':>9} "
        f"{'Turns':>6} {'Compl':>6} {'ProperT':>8} {'PolValid':>9}"
    )
    print(cls_header)
    print("  " + "-" * (len(cls_header) - 2))
    for cls, m in agg.per_class.items():
        print(
            f"  {cls:<24} {m['n']:>3} "
            f"{m['classification_accuracy']:>7.1%} "
            f"{m['avg_extraction_coverage']:>9.2f} "
            f"{m['avg_turns']:>6.2f} "
            f"{m['avg_profile_completeness']:>6.2f} "
            f"{m['proper_termination_rate']:>8.1%} "
            f"{m['policy_validity_rate']:>9.1%}"
        )
    print()
