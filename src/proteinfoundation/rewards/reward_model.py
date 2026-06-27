"""Reward model that aggregates metrics into scalar rewards for search/refinement."""

from typing import Any

from foundry.utils.logging import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=False)


class RewardModel:
    """Aggregates metric outputs into a scalar reward for search/refinement.

    :py:attr:`reward_metrics_tags` controls which metrics fire during reward
    scoring, independently of the dataset-level ``metrics_tags``.

    Args:
        weights: Maps ``"metric_name.output_key"`` to a float weight.
        differentiable: If ``True``, can be used for gradient-based refinement.
        reward_metrics_tags: Tags passed to metric managers during reward scoring.
            Metrics with ``required_tags_any`` only fire if at least one of these
            tags is present. An empty list means only tag-unconditional metrics run
            (oracles are skipped, design metrics still run). Defaults to ``[]``.

    Example:
        >>> reward_model = RewardModel(
        ...     weights={"rf3.plddt": 1.0, "rf3.iptm": 2.0, "rf3.ipae": -1.0},
        ...     reward_metrics_tags=["unconditional_refolding"],
        ... )
    """

    def __init__(
        self,
        weights: dict[str, float],
        differentiable: bool = False,
        reward_metrics_tags: list[str] | None = None,
    ):
        """Initialize reward model.

        Args:
            weights: Weight for each metric output key.
            differentiable: If True, can be used for gradient-based refinement.
            reward_metrics_tags: Tags controlling which configured metrics fire
                during reward scoring. Defaults to ``[]`` (tag-unconditional only).
        """
        self.weights = weights
        self.differentiable = differentiable
        self.reward_metrics_tags = list(reward_metrics_tags) if reward_metrics_tags else []

    def aggregate_metrics_output(self, metrics_output: dict[str, Any]) -> float:
        """Aggregate a metrics output dict into a scalar reward using configured weights.

        Args:
            metrics_output: Dictionary of metric results keyed by
                ``"metric_name.output_key"`` (e.g. ``"rf3.plddt"``).

        Returns:
            Weighted scalar reward.
        """
        reward = 0.0
        for reward_key, weight in self.weights.items():
            if not weight:
                continue
            if reward_key in metrics_output:
                reward += weight * metrics_output[reward_key]
            else:
                ranked_logger.warning(
                    "Reward metric output key '%s' not found in metrics output "
                    "(oracle may have failed). Returning -inf reward.",
                    reward_key,
                )
                return float("-inf")

        return reward
