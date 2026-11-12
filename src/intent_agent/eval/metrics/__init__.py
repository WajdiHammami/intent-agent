"""Paper metrics M1–M8: per-sample computation, aggregation and reporting."""

from intent_agent.eval.metrics.aggregate import AggregateMetrics, aggregate
from intent_agent.eval.metrics.report import METRIC_ROWS, print_metric_matrix, print_results_table
from intent_agent.eval.metrics.sample import SampleResult, compute_sample_metrics, evaluate_sample

__all__ = [
    "METRIC_ROWS", "AggregateMetrics", "SampleResult", "aggregate", "compute_sample_metrics",
    "evaluate_sample", "print_metric_matrix", "print_results_table",
]
