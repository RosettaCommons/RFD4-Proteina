"""Aggregate metric outputs into scalar rewards for search checkpoints."""

import math
from collections.abc import Mapping
from typing import Any

from rfproteina.utils.logging import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=False)

#: Key under which the aggregated scalar reward is published into a sample's metrics dict
SEARCH_REWARD_KEY = "search.reward"


class RewardModel:
    """Aggregate metric outputs for one search checkpoint.

    Args:
        rewards: Map metric.output keys to weight, clip, and normalize_by settings.
            Values are clipped, divided by normalize_by, and multiplied by weight.
            Defaults are weight=1 and normalize_by=1, with no clipping.

    The search strategy owns a separate instance for each checkpoint and runs
    the metrics explicitly listed there before aggregating their outputs.
    """

    def __init__(self, rewards: Mapping[str, Any]) -> None:
        weights, clips = {}, {}
        if not isinstance(rewards, Mapping):
            raise ValueError("Checkpoint rewards must map metric.output to reward terms.")
        for key, term in rewards.items():
            if not isinstance(key, str) or "." not in key:
                raise ValueError(f"Reward key must be metric.output; got {key!r}.")
            if not isinstance(term, Mapping) or set(term) - {"weight", "clip", "normalize_by"}:
                raise ValueError(f"Reward {key!r} accepts weight, clip, and normalize_by.")
            weight, scale = term.get("weight", 1.0), term.get("normalize_by", 1.0)
            if (
                any(
                    isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
                    for v in (weight, scale)
                )
                or scale <= 0
            ):
                raise ValueError(f"Reward {key!r} requires a finite weight and positive finite normalize_by.")
            weights[key] = weight / scale
            if "clip" in term:
                bounds = term["clip"]
                if not isinstance(bounds, Mapping) or set(bounds) - {"min", "max"}:
                    raise ValueError(f"Reward {key!r} clip accepts min and max.")
                if any(
                    isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
                    for v in bounds.values()
                ):
                    raise ValueError(f"Reward {key!r} clip bounds must be finite numbers.")
                if not bounds:
                    raise ValueError(f"Reward {key!r} clip must set min and/or max.")
                if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
                    raise ValueError(f"Reward {key!r} clip min must not exceed max.")
                clips[key] = dict(bounds)
        self.weights = weights
        self.clip_values = clips

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
