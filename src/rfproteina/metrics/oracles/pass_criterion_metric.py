"""Pass/fail criterion metric for design evaluation.

Evaluates a single threshold criterion against an already-computed metrics dict
(the merged oracle + refolding output). Used as entries in the pass_criteria
MetricManager, which runs last in the validation flow.
"""

from collections.abc import Callable, Sequence
from statistics import fmean
from typing import Any

from rfproteina.metrics.metric import Metric
from rfproteina.utils.logging import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)

# Reductions a threshold spec may apply to a list of metric keys instead of naming one.
_REDUCERS: dict[str, Callable[[Sequence[float]], float]] = {
    "min_of": min,
    "max_of": max,
    "mean_of": fmean,
}


def _resolve_reduction(label: str, spec: dict[str, Any]) -> tuple[list[str], Callable[[Sequence[float]], float] | None]:
    """Resolve which metric keys a threshold spec covers, and how to reduce them.

    Args:
        label: The ``metric_thresholds`` key; the metric key itself if *spec* has no reducer.
        spec: Threshold spec, optionally carrying one :data:`_REDUCERS` key.

    Returns:
        Metric keys to look up, and the reduction over their values, or ``None`` if
        *spec* has no reducer -- there's a single key and nothing to reduce.
    """
    reducers = [name for name in _REDUCERS if name in spec]
    if len(reducers) > 1:
        raise ValueError(f"Threshold spec '{label}' carries more than one of {sorted(_REDUCERS)}: {reducers}.")
    if not reducers:
        return [label], None
    name = reducers[0]
    keys = list(spec[name])
    if not keys:
        raise ValueError(f"Threshold spec '{label}' has an empty '{name}' key list.")
    return keys, _REDUCERS[name]


class PassCriterionMetric(Metric):
    """Evaluate a pass criterion against an already-computed metrics dict.

    Intended as entries in a ``pass_criteria`` :py:class:`~rfproteina.metrics.metric.MetricManager`
    that runs after the oracle and refolding MetricManagers. Receives the merged
    oracle + refolding output via ``extra_info["metrics"]`` and returns a binary
    pass/fail flag.

    Args:
        metric_thresholds: Dict mapping metric keys to threshold specs (dicts with
            optional ``"min"`` / ``"max"``). All thresholds must pass. A spec may
            instead carry ``"min_of"`` / ``"max_of"`` / ``"mean_of"``, a list of
            metric keys reduced to a single value before thresholding; the outer
            key is then just a label. Every listed key must be present, or the
            result is indeterminate.

    Example:
        >>> PassCriterionMetric(metric_thresholds={"standard.plddt_coi": {"min": 70.0}}).compute(
        ...     {"standard.plddt_coi": 85.0}
        ... )
        {"pass": 1.0, "all_metrics_present": 1.0}

        >>> PassCriterionMetric(
        ...     metric_thresholds={
        ...         "ipae": {"min_of": ["oracle.ipae_C", "oracle.ipae_D"], "max": 2.0}
        ...     }
        ... ).compute({"oracle.ipae_C": 3.1, "oracle.ipae_D": 1.7})
        {"pass": 1.0, "all_metrics_present": 1.0}
    """

    def __init__(
        self,
        metric_thresholds: dict[str, dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric_thresholds = metric_thresholds
        # Resolved here so a malformed spec fails at config load rather than mid-validation.
        self._reductions = {label: _resolve_reduction(label, spec) for label, spec in metric_thresholds.items()}

    @property
    def kwargs_to_compute_args(self) -> dict:
        return {"metrics": ("extra_info", "metrics")}

    def compute(self, metrics: dict) -> dict[str, float]:
        """Evaluate the criteria and return a pass/fail flag.

        Args:
            metrics: Merged oracle + refolding metrics dict, keyed by
                ``{manager_name}.{metric_name}`` dot notation.

        Returns:
            Dict with ``"pass"`` (``1.0``/``0.0``) and ``"all_metrics_present"``
            (``0.0`` if any required key was missing, in which case ``"pass"``
            is always ``0.0`` but should be treated as indeterminate).
        """
        all_present = True
        passed = True
        for label, thresholds in self.metric_thresholds.items():
            keys, reduce = self._reductions[label]
            values = [metrics.get(key) for key in keys]
            if missing := [key for key, val in zip(keys, values, strict=True) if val is None]:
                logger.warning(
                    f"Metric key(s) {missing} absent from metrics. Treating as indeterminate (not a hard fail)."
                )
                all_present = False
                passed = False
            else:
                val = values[0] if reduce is None else reduce(values)
                min_val = thresholds.get("min")
                max_val = thresholds.get("max")
                if not ((min_val is None or val >= min_val) and (max_val is None or val <= max_val)):
                    passed = False

        return {"pass": float(passed), "all_metrics_present": float(all_present)}
