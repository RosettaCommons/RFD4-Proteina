"""Reward model that aggregates metrics into scalar rewards for search."""

from typing import Any

from rfproteina.utils.logging import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=False)

#: Key under which the aggregated scalar reward is published into a sample's metrics dict
SEARCH_REWARD_KEY = "search.reward"


class RewardModel:
    """Aggregates metric outputs into a scalar reward for search.

    :py:attr:`reward_metrics_tags` controls which metrics fire during reward
    scoring, independently of the dataset-level ``metrics_tags``.

    Args:
        weights: Maps ``"metric_name.output_key"`` to a float weight.
        reward_metrics_tags: Tags passed to metric managers during reward scoring.
            Metrics with ``required_tags_any`` only fire if at least one of these
            tags is present. An empty list means only tag-unconditional metrics run
            (oracles are skipped, design metrics still run). Defaults to ``[]``.
        clip_values: Maps a weighted key to ``{"min": lo, "max": hi}`` (either bound
            optional). The value is clipped to those bounds *before* its weight is
            applied, so the reward stops responding once the metric passes the bound.
            The metric itself is still reported raw — clipping affects the reward only.

    Example:
        >>> reward_model = RewardModel(
        ...     weights={"rf3.plddt": 1.0, "rf3.iptm": 2.0, "rf3.ipae": -1.0},
        ...     reward_metrics_tags=["unconditional_refolding"],
        ...     clip_values={"rf3.ipae": {"max": 10.0}},  # ipAE past 10 carries no signal
        ... )
    """

    def __init__(
        self,
        weights: dict[str, float],
        reward_metrics_tags: list[str] | None = None,
        clip_values: dict[str, dict[str, float]] | None = None,
    ):
        """Initialize reward model.

        Args:
            weights: Weight for each metric output key.
            reward_metrics_tags: Tags controlling which configured metrics fire
                during reward scoring. Defaults to ``[]`` (tag-unconditional only).
            clip_values: Per-key ``min``/``max`` bounds applied before weighting.

        Raises:
            ValueError: If a clipped key carries no weight, or its bounds are inverted.
        """
        self.weights = weights
        self.reward_metrics_tags = list(reward_metrics_tags) if reward_metrics_tags else []
        self.clip_values = {key: dict(bounds) for key, bounds in (clip_values or {}).items()}

        for key, bounds in self.clip_values.items():
            if key not in self.weights:
                raise ValueError(
                    f"clip_values key '{key}' has no entry in weights, so clipping it would have "
                    f"no effect. Weighted keys: {sorted(self.weights)}."
                )
            lo, hi = bounds.get("min"), bounds.get("max")
            if lo is None and hi is None:
                raise ValueError(f"clip_values['{key}'] must set at least one of 'min' or 'max'.")
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(f"clip_values['{key}'] has min={lo} > max={hi}.")

    def _clip(self, key: str, value: float) -> float:
        """Clip a metric value to its configured reward bounds (identity if unbounded)."""
        bounds = self.clip_values.get(key)
        if bounds is None:
            return float(value)
        clipped = float(value)
        if (lo := bounds.get("min")) is not None:
            clipped = max(clipped, lo)
        if (hi := bounds.get("max")) is not None:
            clipped = min(clipped, hi)
        return clipped

    def aggregate_metrics_output(self, metrics_output: dict[str, Any]) -> float:
        """Aggregate a metrics output dict into a scalar reward using configured weights.

        Values are clipped per :py:attr:`clip_values` before weighting. *metrics_output*
        is never mutated, so downstream reporting still sees the raw metric values.

        Args:
            metrics_output: Dictionary of metric results keyed by
                ``"metric_name.output_key"`` (e.g. ``"rf3.plddt"``).

        Returns:
            Weighted scalar reward, or ``-inf`` if a weighted key is missing.
        """
        reward = 0.0
        for reward_key, weight in self.weights.items():
            if not weight:
                continue
            if reward_key in metrics_output:
                reward += weight * self._clip(reward_key, metrics_output[reward_key])
            else:
                ranked_logger.warning(
                    "Reward metric output key '%s' not found in metrics output "
                    "(oracle may have failed). Returning -inf reward.",
                    reward_key,
                )
                return float("-inf")

        return reward
