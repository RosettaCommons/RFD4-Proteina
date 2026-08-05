"""Reward model that aggregates metrics into scalar rewards for search."""

from collections.abc import Mapping
from typing import Any

from rfproteina.utils.logging import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=False)

#: Key under which the aggregated scalar reward is published into a sample's metrics dict
SEARCH_REWARD_KEY = "search.reward"


#: Fields a per-checkpoint override may redefine (each falls back to the base reward)
OVERRIDABLE_FIELDS = ("weights", "clip_values", "reward_metrics_tags")


def _normalize_metrics_tags(tags: "list[str] | str | None") -> list[str]:
    """Coerce a ``reward_metrics_tags`` config value into a list of tag strings.

    A bare string is split on commas (matching how the top-level ``metrics_tags`` is
    handled in :py:mod:`rfproteina.generate`) rather than iterated into characters.
    """
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    if not isinstance(tags, list | tuple | set) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError(f"reward_metrics_tags must be a list of tags or one comma-separated string; got {tags!r}.")
    return list(tags)


class RewardModel:
    """Aggregates metric outputs into a scalar reward for search.

    :py:attr:`reward_metrics_tags` controls which metrics fire during reward
    scoring, independently of the dataset-level ``metrics_tags``.

    The reward can also *change mid-trajectory*: :py:attr:`checkpoint_overrides`
    maps a search checkpoint step to a partial reward spec, and
    :py:meth:`at_step` returns the reward model to score with at that step. Each
    field an override names **replaces** the base one wholesale; fields it omits
    are inherited unchanged.

    Args:
        weights: Maps ``"metric_name.output_key"`` to a float weight. A key whose
            metric does not run scores ``-inf``, so every weighted key must be
            produced under the active :py:attr:`reward_metrics_tags`.
        reward_metrics_tags: Tags passed to metric managers during reward scoring.
            Metrics with ``required_tags_any`` only fire if at least one of these
            tags is present. An empty list means only tag-unconditional metrics run
            (oracles are skipped, design metrics still run). Defaults to ``[]``.
        clip_values: Maps a weighted key to ``{"min": lo, "max": hi}`` (either bound
            optional). The value is clipped to those bounds *before* its weight is
            applied, so the reward stops responding once the metric passes the bound.
            The metric itself is still reported raw — clipping affects the reward only.
        checkpoint_overrides: Maps a search checkpoint step to a partial spec with any
            of ``weights``, ``clip_values``, ``reward_metrics_tags``. Each named field
            replaces the base one outright — in particular an override that switches
            ``reward_metrics_tags`` must restate the ``weights`` that go with the new
            metrics, since keeping a weight whose metric no longer fires would make
            every candidate at that checkpoint score ``-inf`` — which the search raises
            on rather than pruning on nothing.

    Example:
        >>> reward_model = RewardModel(
        ...     weights={"rf3.plddt": 1.0, "rf3.iptm": 2.0, "rf3.ipae": -1.0},
        ...     reward_metrics_tags=["unconditional_refolding"],
        ...     clip_values={"rf3.ipae": {"max": 10.0}},  # ipAE past 10 carries no signal
        ...     # late: ipTM alone (the base weights are replaced, not merged into)
        ...     checkpoint_overrides={100: {"weights": {"rf3.iptm": 4.0}, "clip_values": {}}},
        ... )
        >>> sorted(reward_model.at_step(100).weights)
        ['rf3.iptm']
    """

    def __init__(
        self,
        weights: dict[str, float],
        reward_metrics_tags: list[str] | str | None = None,
        clip_values: dict[str, dict[str, float]] | None = None,
        checkpoint_overrides: dict[int, dict[str, Any]] | None = None,
    ):
        """Initialize reward model.

        Args:
            weights: Weight for each metric output key.
            reward_metrics_tags: Tags controlling which configured metrics fire
                during reward scoring. Accepts a list of tags or one comma-separated
                string. Defaults to ``[]`` (tag-unconditional only).
            clip_values: Per-key ``min``/``max`` bounds applied before weighting.
            checkpoint_overrides: Per-checkpoint reward specs, keyed by step.

        Raises:
            ValueError: If a clipped key carries no weight, its bounds are inverted,
                or a config value (tags, weights, override key or spec) is malformed.
        """
        if not isinstance(weights, Mapping):
            raise ValueError(f"weights must map 'metric.output_key' to a number; got {weights!r}.")
        self.weights = dict(weights)
        null_weights = sorted(key for key, weight in self.weights.items() if weight is None)
        if null_weights:
            raise ValueError(f"weights {null_weights} are null; give a number (0.0 disables a term).")

        self.reward_metrics_tags = _normalize_metrics_tags(reward_metrics_tags)

        if clip_values is not None and not isinstance(clip_values, Mapping):
            raise ValueError(f"clip_values must map a weighted key to its bounds; got {clip_values!r}.")
        self.clip_values = {}
        for key, bounds in (clip_values or {}).items():
            if not isinstance(bounds, Mapping):
                raise ValueError(f"clip_values['{key}'] must set 'min' and/or 'max'; got {bounds!r}.")
            self.clip_values[key] = dict(bounds)

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

        self.checkpoint_overrides: dict[int, dict[str, Any]] = {}
        for step, spec in (checkpoint_overrides or {}).items():
            # YAML gives int keys, Hydra CLI overrides give digit strings; anything else
            # (a bool, a float, a name) would silently coerce, so reject it here.
            if isinstance(step, bool) or not (isinstance(step, int) or (isinstance(step, str) and step.isdigit())):
                raise ValueError(
                    f"checkpoint_overrides keys must be integer steps; got {step!r} ({type(step).__name__})."
                )
            if not isinstance(spec, Mapping):
                raise ValueError(
                    f"checkpoint_overrides[{step}] must be a mapping of {list(OVERRIDABLE_FIELDS)}; got {spec!r}."
                )
            self.checkpoint_overrides[int(step)] = dict(spec)

        # Build every per-checkpoint variant up front so a malformed override fails at
        # config time rather than hours into a run.
        self._models_by_step = {
            step: self._build_variant(step, spec) for step, spec in self.checkpoint_overrides.items()
        }

    def _build_variant(self, step: int, spec: dict[str, Any]) -> "RewardModel":
        """Build the reward model for one checkpoint; fields *spec* names replace the base's."""
        unknown = sorted(set(spec) - set(OVERRIDABLE_FIELDS))
        if unknown:
            raise ValueError(
                f"checkpoint_overrides[{step}] has unsupported field(s) {unknown}; "
                f"supported fields: {list(OVERRIDABLE_FIELDS)}."
            )
        try:
            return RewardModel(
                weights=spec.get("weights", self.weights),
                reward_metrics_tags=spec.get("reward_metrics_tags", self.reward_metrics_tags),
                clip_values=spec.get("clip_values", self.clip_values),
            )
        except ValueError as e:
            raise ValueError(f"checkpoint_overrides[{step}] is invalid: {e}") from e

    def at_step(self, step: int) -> "RewardModel":
        """Return the reward model to score with at search checkpoint *step*."""
        return self._models_by_step.get(int(step), self)

    def validate_checkpoint_steps(self, scoring_steps: list[int]) -> None:
        """Check that every override targets a step the search actually scores at."""
        unknown = sorted(set(self.checkpoint_overrides) - set(scoring_steps))
        if unknown:
            raise ValueError(
                f"search.rewards.checkpoint_overrides targets step(s) {unknown}, which the search "
                f"never scores at. Overrides must key on one of {sorted(scoring_steps)}."
            )

    @property
    def reward_metrics_tag_sets(self) -> list[set[str]]:
        """One tag set per reward variant (base plus each override), deduped."""
        tag_sets: list[set[str]] = []
        for model in (self, *self._models_by_step.values()):
            tags = set(model.reward_metrics_tags)
            if tags and tags not in tag_sets:
                tag_sets.append(tags)
        return tag_sets

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
