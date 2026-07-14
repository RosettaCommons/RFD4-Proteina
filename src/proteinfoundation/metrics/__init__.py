"""Metrics for model evaluation.

Re-exports the base metric framework so ``from proteinfoundation.metrics import
MetricManager`` keeps working; the concrete metrics live in submodules.
"""

from proteinfoundation.metrics.metric import Metric, MetricInputError, MetricManager

__all__ = [
    "Metric",
    "MetricInputError",
    "MetricManager",
]
