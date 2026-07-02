"""Per-target solve-rate metrics (e.g., percentage of targets with at least n passing design)."""

import pandas as pd

from proteinfoundation.inference_utils.foldseek import _base_pass_cols

# Percentage of targets with at least this pass rate is reported as solve_rate_{name} (e.g. solve_rate_5pct).
SOLVE_RATE_THRESHOLDS: dict[str, float] = {
    "1pct": 0.01,
    "5pct": 0.05,
    "10pct": 0.10,
    "20pct": 0.20,
}


def _metric_prefix(pass_col: str) -> str:
    """Namespace prefix for a pass column: ``af3.pass`` -> ``af3.``."""
    return pass_col[: -len("pass")]


def compute_solve_rate_metrics(group_df: pd.DataFrame, pass_cols: list[str] | None = None) -> dict[str, float]:
    """Compute the per-target solve-rate curve for one dataset.

    NOTE: Assumes ``group_df`` holds a *single* dataset

    Args:
        group_df: Per-design rows for a single dataset, with a ``task`` column (the target
            identifier) and one or more ``{crit}.pass`` columns.
        pass_cols: Explicit solve-rate pass columns (see :py:func:`_base_pass_cols`).

    Returns:
        Mapping of solve-rate column name to value, one block per pass column
    """
    if "task" not in group_df.columns:
        # No tasks (targets)
        return {}

    out: dict[str, float] = {}
    for pass_col in _base_pass_cols(group_df, pass_cols):
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

        # Smooth, threshold-insensitive summary: normalized area under the solve-rate curve over
        # [0, max threshold], which equals the mean per-target pass fraction capped at that max.
        tmax = max(SOLVE_RATE_THRESHOLDS.values())
        out[f"{prefix}solve_rate_auc"] = float(p_t.clip(upper=tmax).mean() / tmax)

    return out
