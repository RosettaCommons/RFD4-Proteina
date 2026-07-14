"""Linear soft threshold scores for search rewards and metrics.

Each bound yields a score in [0, 1]:
  - 1.0 when the value satisfies the bound (distance <= upper, or value >= lower)
  - 0.0 when the violation equals ``margin`` (e.g. 1 Å above an upper bound, or ``margin``
    below a lower bound for dimensionless scores such as pLDDT)
  - linear between

Aggregate with :func:`apply_soft_bounds_to_outputs` (default ``aggregate="min"`` so the
bundle gets full credit only when every bound passes).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np


class SoftBoundSpec(TypedDict, total=False):
    """YAML/Hydra-friendly spec for one soft bound on a metric output key."""

    key: str
    upper_bound: float
    lower_bound: float
    margin: float


def soft_score_upper_bound(value: float, upper_bound: float, margin: float = 1.0) -> float:
    """Score for ``value <= upper_bound``, decaying to 0 over ``margin`` above the bound."""
    if margin <= 0:
        return float(value <= upper_bound)
    excess = max(0.0, float(value) - float(upper_bound))
    return float(max(0.0, 1.0 - excess / margin))


def soft_score_lower_bound(value: float, lower_bound: float, margin: float = 1.0) -> float:
    """Score for ``value >= lower_bound``, decaying to 0 over ``margin`` below the bound."""
    if margin <= 0:
        return float(value >= lower_bound)
    deficit = max(0.0, float(lower_bound) - float(value))
    return float(max(0.0, 1.0 - deficit / margin))


def soft_score_for_bound(
    value: float,
    *,
    upper_bound: float | None = None,
    lower_bound: float | None = None,
    margin: float = 1.0,
) -> float:
    """Dispatch to upper- or lower-bound soft scoring."""
    if upper_bound is not None and lower_bound is not None:
        raise ValueError("Specify only one of upper_bound or lower_bound")
    if upper_bound is not None:
        return soft_score_upper_bound(value, upper_bound, margin)
    if lower_bound is not None:
        return soft_score_lower_bound(value, lower_bound, margin)
    raise ValueError("Either upper_bound or lower_bound is required")


def aggregate_soft_min(scores: list[float]) -> float:
    """Minimum soft score across constraints (all must pass for 1.0)."""
    if not scores:
        return 1.0
    return float(np.min(scores))


def aggregate_soft_mean(scores: list[float]) -> float:
    """Mean soft score across constraints."""
    if not scores:
        return 1.0
    return float(np.mean(scores))


def aggregate_soft_product(scores: list[float]) -> float:
    """Product soft score across constraints."""
    if not scores:
        return 1.0
    return float(np.prod(scores))


_SOFT_AGGREGATORS: dict[str, Callable[[list[float]], float]] = {
    "min": aggregate_soft_min,
    "mean": aggregate_soft_mean,
    "product": aggregate_soft_product,
}


def apply_soft_bounds_to_outputs(
    metrics: dict[str, float],
    bounds: list[dict[str, Any] | SoftBoundSpec],
    *,
    output_prefix: str = "",
    aggregate: str = "min",
) -> dict[str, float]:
    """Add ``{prefix}{key}_soft_score`` and ``{prefix}soft_min_score`` from a metrics dict.

    Each entry in ``bounds`` supports:
      - ``key``: prefix for output ``{key}_soft_score`` (and aggregate key)
      - ``value_key``: metric dict key to read (defaults to ``key``)
      - ``upper_bound`` or ``lower_bound``
      - ``margin`` (default 1.0; same units as the bound / raw metric)

    Args:
        aggregate: How to combine per-bound scores: ``min`` (default), ``mean``, or ``product``.
    """
    agg_fn = _SOFT_AGGREGATORS.get(aggregate)
    if agg_fn is None:
        raise ValueError(f"Unknown soft-bound aggregate {aggregate!r}; expected one of {sorted(_SOFT_AGGREGATORS)}")

    out: dict[str, float] = {}
    per_bound: list[float] = []
    for raw_spec in bounds:
        if "key" not in raw_spec:
            raise ValueError(f"Soft bound spec must include 'key': {raw_spec}")
        has_upper = raw_spec.get("upper_bound") is not None
        has_lower = raw_spec.get("lower_bound") is not None
        if has_upper == has_lower:
            raise ValueError(f"Soft bound spec must set exactly one of upper_bound or lower_bound: {raw_spec}")

        key = str(raw_spec["key"])
        value_key = str(raw_spec.get("value_key", key))
        margin = float(raw_spec.get("margin", 1.0))
        val = metrics.get(value_key)
        if val is None or not np.isfinite(val):
            per_bound.append(0.0)
            out[f"{output_prefix}{key}_soft_score"] = 0.0
            continue
        score = soft_score_for_bound(
            float(val),
            upper_bound=raw_spec.get("upper_bound"),
            lower_bound=raw_spec.get("lower_bound"),
            margin=margin,
        )
        per_bound.append(score)
        out[f"{output_prefix}{key}_soft_score"] = score
    out[f"{output_prefix}soft_min_score"] = agg_fn(per_bound)
    return out
