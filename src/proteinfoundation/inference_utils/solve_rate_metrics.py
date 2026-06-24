"""Per-target solve-rate metrics (e.g., percentage of targets with at least n passing design)."""

import numpy as np
import pandas as pd

from proteinfoundation.inference_utils.foldseek import _base_pass_rate_cols

# Percentage of targets with at least this pass rate is reported as solve_rate_{name} (e.g. solve_rate_5pct).
SOLVE_RATE_THRESHOLDS: dict[str, float] = {
    "1pct": 0.01,
    "5pct": 0.05,
    "10pct": 0.10,
    "20pct": 0.20,
}


def _metric_prefix(pass_col: str) -> str:
    """Namespace prefix for a pass-rate column: ``af3.pass`` -> ``af3.``, ``af3_pass_rate`` -> ``af3_``."""
    return pass_col[: -len("pass")] if pass_col.endswith(".pass") else pass_col[: -len("pass_rate")]


def compute_solve_rate_metrics(group_df: pd.DataFrame) -> dict[str, float]:
    """Compute the per-target solve-rate curve for one dataset.

    NOTE: Assumes ``group_df`` holds a *single* dataset

    Args:
        group_df: Per-design rows for a single dataset, with a ``task`` column (the target
            identifier) and one or more pass-rate columns (``*.pass`` or legacy ``*_pass_rate``)

    Returns:
        Mapping of solve-rate column name to value, one block per pass-rate column
    """
    if "task" not in group_df.columns:
        # No tasks (targets)
        return {}

    out: dict[str, float] = {}
    for pass_col in _base_pass_rate_cols(group_df):
        p_t = pd.to_numeric(group_df[pass_col], errors="coerce").groupby(group_df["task"]).mean().dropna()
        if len(p_t) == 0:
            continue

        prefix = _metric_prefix(pass_col)
        # Denominator for this oracle's rates: targets with valid pass data (post-dropna)
        out[f"{prefix}solve_rate_n_targets"] = float(len(p_t))
        out[f"{prefix}solve_rate_any"] = float((p_t > 0).mean())

        # Solve rate at each threshold, i.e. percentage of targets with at least this pass rate
        for name, thr in SOLVE_RATE_THRESHOLDS.items():
            out[f"{prefix}solve_rate_{name}"] = float((p_t >= thr).mean())

        # Mean across pass-rate thresholds
        out[f"{prefix}solve_rate_threshold_mean"] = float(
            np.mean([out[f"{prefix}solve_rate_{name}"] for name in SOLVE_RATE_THRESHOLDS])
        )

    return out
