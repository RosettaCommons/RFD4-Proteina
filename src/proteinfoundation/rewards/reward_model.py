"""Reward model that aggregates metrics into scalar rewards for search/refinement."""

from typing import Any

import torch

from foundry.metrics.metric import Metric, MetricManager


class RewardModel:
    """Aggregates Metrics into scalar reward for search/refinement.

    This is a lightweight wrapper around MetricManager that computes weighted sums
    instead of returning dictionaries. All computation logic lives in Metric classes.

    Args:
        metrics: Dictionary mapping metric names to Metric instances, or a MetricManager
        weights: Dictionary mapping "metric_name.key" to weight values
        differentiable: Whether this reward can be used for gradient-based refinement

    Example:
        >>> from proteinfoundation.metrics.hbonds_hbplus_metrics import HbondMetrics
        >>> reward_model = RewardModel(
        ...     metrics={"hbonds": HbondMetrics(cutoff_HA_dist=2.2)},
        ...     weights={
        ...         "hbonds.num_hbonds": 1.0,
        ...         "hbonds.avg_hbond_energy": -0.5,
        ...     },
        ... )
        >>> reward = reward_model.aggregate_metrics_output(
        ...     reward_model.metric_manager(predicted_atom_array=structure)
        ... )
    """

    def __init__(
        self, metrics: dict[str, Metric] | MetricManager, weights: dict[str, float], differentiable: bool = False
    ):
        """Initialize reward model.

        Args:
            metrics: Metrics to compute, or existing MetricManager
            weights: Weight for each metric output key
            differentiable: If True, can be used for gradient-based refinement
        """
        if isinstance(metrics, MetricManager):
            self.metric_manager = metrics
        else:
            self.metric_manager = MetricManager(metrics, raise_errors=True)

        self.weights = weights
        self.differentiable = differentiable

    def aggregate_metrics_output(self, metrics_output: dict[str, Any]) -> float:
        """Aggregate a metrics output dict into a scalar reward using configured weights.

        Args:
            metrics_output: Dictionary of metric results, as returned by MetricManager
                or :py:func:`~proteinfoundation.metrics.metrics_utils.run_metrics_for_result`.

        Returns:
            Weighted scalar reward.
        """
        reward = 0.0
        for key, value in metrics_output.items():
            if key == "example_id":
                continue
            weight = self.weights.get(key, 0.0)
            if weight != 0.0:
                reward += weight * value
        return reward

    def compute_components(self, **kwargs: Any) -> dict[str, float]:
        """Return detailed breakdown of reward components.

        This delegates directly to MetricManager, returning the full dictionary
        of metric outputs before aggregation.

        Args:
            **kwargs: Arguments passed to MetricManager

        Returns:
            Dictionary of metric results
        """
        return self.metric_manager(**kwargs)

    def compute_with_gradients(self, **kwargs: Any) -> dict[str, Any]:
        """Compute reward with gradients for differentiable optimization.

        This method aggregates metrics using configured weights AND scales
        gradients by the same weights, ensuring consistency.

        Only works if metrics support compute_with_gradients() method.
        Currently only AF2Metric supports this.

        Args:
            **kwargs: Arguments passed to metric's compute_with_gradients()
                     (e.g., pdb_path, target_chain, binder_sequence_logits)

        Returns:
            Dictionary containing:
                - 'total_reward': Weighted sum of metrics
                - 'grad': Dict with weighted gradients
                - 'metrics': Individual metric values (before weighting)

        Raises:
            ValueError: If reward model is not differentiable
        """
        if not self.differentiable:
            raise ValueError(
                "compute_with_gradients requires differentiable=True. " "Set this in the reward model config."
            )

        # Compute metrics with gradients from each metric
        # For now, assume only one metric (af2) supports gradients
        all_metrics = {}
        all_grads = {}

        for metric_name, metric in self.metric_manager.metrics.items():
            if hasattr(metric, "compute_with_gradients"):
                # Call differentiable compute
                result = metric.compute_with_gradients(**kwargs)

                # Store metrics with metric_name prefix
                for key, value in result["metrics"].items():
                    prefixed_key = f"{metric_name}.{key}" if "." not in key else key
                    all_metrics[prefixed_key] = value

                # Store gradients
                if "grad" in result:
                    all_grads[metric_name] = result["grad"]

        # Aggregate reward using configured weights
        total_reward = 0.0
        for key, value in all_metrics.items():
            weight = self.weights.get(key, 0.0)
            if weight != 0.0:
                total_reward += weight * value

        # Aggregate gradients using the same weights
        # For sequence gradients, we sum weighted gradients from all metrics
        aggregated_grad = {}

        # Aggregate sequence gradients
        if all_grads:
            # Start with zeros
            first_metric_grad = next(iter(all_grads.values()))
            if "sequence" in first_metric_grad:
                aggregated_seq_grad = torch.zeros_like(first_metric_grad["sequence"])

                # Sum weighted gradients
                for metric_name, grad_dict in all_grads.items():
                    if "sequence" in grad_dict:
                        # Scale by weights corresponding to this metric
                        for key, weight in self.weights.items():
                            if key.startswith(f"{metric_name}.") and weight != 0.0:
                                # This metric's gradient contributes with this weight
                                aggregated_seq_grad = aggregated_seq_grad + weight * grad_dict["sequence"]
                                break  # Only apply once per metric

                aggregated_grad["sequence"] = aggregated_seq_grad

        return {
            "total_reward": total_reward,
            "grad": aggregated_grad,
            "metrics": all_metrics,
        }
