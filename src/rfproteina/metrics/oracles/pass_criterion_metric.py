"""Pass/fail criterion metric for design evaluation.

Evaluates a single threshold criterion against an already-computed metrics dict
(the merged oracle + refolding output). Used as entries in the pass_criteria
MetricManager, which runs last in the validation flow.
"""

from typing import Any

from rfproteina.metrics.metric import Metric
from rfproteina.utils.logging import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class PassCriterionMetric(Metric):
    """Evaluate a pass criterion against an already-computed metrics dict.

    Intended as entries in a ``pass_criteria`` :py:class:`~rfproteina.metrics.metric.MetricManager`
    that runs after the oracle and refolding MetricManagers. Receives the merged
    oracle + refolding output via ``extra_info["metrics"]`` and returns a binary
    pass/fail flag.

    Args:
        metric_thresholds: A dict mapping metric keys to threshold specs. Each value is
            a dict with optional ``"min"`` and ``"max"`` keys. All thresholds must pass
            for the criterion to be satisfied.

    Example:
        >>> crit = PassCriterionMetric(
        ...     metric_thresholds={
        ...         "standard.plddt_coi": {"min": 70.0},
        ...         "refolded_backbone_rmsd.rmsd": {"max": 2.0},
        ...     }
        ... )
        >>> crit.compute({"standard.plddt_coi": 85.0, "refolded_backbone_rmsd.rmsd": 1.5})
        {"pass": 1.0, "all_metrics_present": 1.0}
        >>> crit.compute({"standard.plddt_coi": 85.0, "refolded_backbone_rmsd.rmsd": 3.0})
        {"pass": 0.0, "all_metrics_present": 1.0}
        >>> crit.compute({"standard.plddt_coi": 85.0})  # missing refolded_backbone_rmsd.rmsd
        {"pass": 0.0, "all_metrics_present": 0.0}
    """

    def __init__(
        self,
        metric_thresholds: dict[str, dict[str, float]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric_thresholds = metric_thresholds

    @property
    def kwargs_to_compute_args(self) -> dict:
        return {"metrics": ("extra_info", "metrics")}

    def compute(self, metrics: dict) -> dict[str, float]:
        """Evaluate the criteria and return a pass/fail flag.

        Args:
            metrics: Merged oracle + refolding metrics dict, keyed by
                ``{manager_name}.{metric_name}`` dot notation.

        Returns:
            Dict with two keys:

            - ``"pass"``: ``1.0`` if all thresholds are met, ``0.0`` otherwise.
            - ``"all_metrics_present"``: ``1.0`` if every required metric key was
              found in *metrics*, ``0.0`` if at least one was absent. When
              ``"all_metrics_present"`` is ``0.0`` the ``"pass"`` value is always
              ``0.0`` but callers should treat the result as indeterminate rather
              than a hard failure.
        """
        all_present = True
        passed = True
        for key, thresholds in self.metric_thresholds.items():
            val = metrics.get(key)
            if val is None:
                logger.warning(
                    f"Metric key '{key}' is absent from metrics. Treating as indeterminate (not a hard fail)."
                )
                all_present = False
                passed = False
            else:
                min_val = thresholds.get("min")
                max_val = thresholds.get("max")
                if not ((min_val is None or val >= min_val) and (max_val is None or val <= max_val)):
                    passed = False

        return {"pass": float(passed), "all_metrics_present": float(all_present)}
