"""Paper metrics M1–M8: per-sample computation, aggregation and reporting."""

from eval.metrics.aggregate import AggregateMetrics, aggregate
from eval.metrics.report import METRIC_ROWS, print_metric_matrix, print_results_table
from eval.metrics.sample import SampleResult, compute_sample_metrics, evaluate_sample

__all__ = [
    "METRIC_ROWS", "AggregateMetrics", "SampleResult", "aggregate", "compute_sample_metrics",
    "evaluate_sample", "print_metric_matrix", "print_results_table",
]
