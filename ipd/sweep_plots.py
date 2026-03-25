"""Sweep analysis plotting utilities.

Tools for visualizing how individual sweep parameters affect quality metrics
in concatenated sweep output DataFrames produced by ``concatenate_sweep_results.py``.

The primary input is ``sweep_per_example_metrics.csv`` from the sweep root directory.
This is the only file that retains all sweep-identity columns (e.g., ``ckpt_name``,
``nsteps``, ``bb_ca_noise_scale``) written by ``proteina.py``. Task-level aggregation
is done inside these plotting functions, not upfront.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ---------------------------------------------------------------------------
# Metric column resolution
# ---------------------------------------------------------------------------


def _resolve_metric(
    df: pd.DataFrame,
    metric_col: str | list[str],
) -> tuple[pd.DataFrame, str]:
    """Coalesce multiple possible metric columns into a single resolved column.

    If ``metric_col`` is a string, returns ``(df, metric_col)`` unchanged.
    If it is a list, validates that no row has more than one non-NaN value
    across the candidate columns, then coalesces them into a new column whose
    name is the candidates joined by ``" / "``.

    Args:
      df: Per-example metrics DataFrame.
      metric_col: Single column name or list of candidate column names.

    Returns:
      ``(df_out, resolved_name)`` where ``df_out`` may be a copy with an
      extra coalesced column appended.

    Raises:
      ValueError: If any row contains non-NaN values in more than one of
        the candidate columns.

    Examples:
      >>> df, col = _resolve_metric(df, ["plddt", "tm_score"])
    """
    if isinstance(metric_col, str):
        return df, metric_col

    present = [c for c in metric_col if c in df.columns]
    if not present:
        raise ValueError(f"None of the metric columns {metric_col} found in DataFrame.")

    non_nan_counts = df[present].notna().sum(axis=1)
    bad_rows = non_nan_counts > 1
    if bad_rows.any():
        bad_sample = df.loc[bad_rows, present].head(5)
        raise ValueError(
            f"Found {bad_rows.sum()} row(s) with non-NaN values in more than one "
            f"metric column {present}. Each row must populate at most one.\n"
            f"Example conflicts:\n{bad_sample}"
        )

    first_metric_colname_suffix = metric_col[0].split(".")[-1]
    if len(metric_col) == 1:
        resolved_name = metric_col[0]
    elif all(col.split(".")[-1] == first_metric_colname_suffix for col in metric_col):
        resolved_name = first_metric_colname_suffix
    else:
        resolved_name = " / ".join(metric_col)
    df = df.copy()
    coalesced = pd.Series(index=df.index, dtype=float)
    for c in present:
        coalesced = coalesced.combine_first(df[c])
    df[resolved_name] = coalesced
    return df, resolved_name


# ---------------------------------------------------------------------------
# Column suggestion
# ---------------------------------------------------------------------------


# Maps known sweep config key paths → df column names as written by proteina.py.
# Needed because proteina.py saves some config values under shorter aliases rather
# than the raw last path segment (e.g. sc_scale_noise → bb_ca_noise_scale).
_SWEEP_KEY_ALIASES: dict[str, str] = {
    "generation.model.bb_ca.simulation_step_params.sc_scale_noise": "bb_ca_noise_scale",
    "generation.model.local_latents.simulation_step_params.sc_scale_noise": "latents_noise_scale",
    "generation.args.nsteps": "nsteps",
}


def _resolve_sweep_key(key: str, df_cols: set[str]) -> tuple[str | None, str]:
    """Resolve a sweep config key to a DataFrame column name.

    Resolution order:
      1. Explicit alias in :py:data:`_SWEEP_KEY_ALIASES`
      2. Exact match in ``df_cols``
      3. Last dot-segment match

    Args:
      key: Dot-separated sweep config key (e.g. ``"generation.args.nsteps"``).
      df_cols: Set of column names present in the DataFrame.

    Returns:
      ``(col_name, status)`` where ``col_name`` is ``None`` if unresolved.
    """
    if key in _SWEEP_KEY_ALIASES:
        alias = _SWEEP_KEY_ALIASES[key]
        if alias in df_cols:
            return alias, "✓ alias"
        return None, f"✗ alias→'{alias}' not in df"
    if key in df_cols:
        return key, "✓ exact"
    tail = key.split(".")[-1]
    if tail in df_cols:
        return tail, "✓ last-segment"
    return None, "✗ unresolved"


def suggest_sweep_cols(
    df: pd.DataFrame,
    sweep_config_path: str | Path | None = None,
) -> list[str]:
    """Suggest which DataFrame columns correspond to sweep axes.

    Loads a preserved sweep config YAML (``sweep_config.yaml`` at the sweep
    root) and enumerates the keys inside the ``sweep`` block, which hold the
    Cartesian-product parameters. Each key is a dot-separated Hydra config
    path and is matched to a DataFrame column via:

    1. An explicit alias table (:py:data:`_SWEEP_KEY_ALIASES`) for paths that
       ``proteina.py`` saves under a shorter name (e.g.
       ``generation.model.bb_ca.simulation_step_params.sc_scale_noise``
       → ``bb_ca_noise_scale``).
    2. Exact match against ``df.columns``.
    3. Last dot-segment match.

    Keys from ``base_config_overrides`` and ``sweep_overrides`` are
    intentionally excluded: ``base_config_overrides`` are fixed (not swept),
    while ``sweep_overrides`` are Hydra override strings that generate separate
    sub-sweeps — their effect is captured by the ``dataset_name`` / ``task``
    columns already present in the DataFrame.

    If no config is given, falls back to a heuristic: any column with low
    cardinality that is not standard per-example metadata.

    Args:
      df: Per-example metrics DataFrame (``sweep_per_example_metrics.csv``).
      sweep_config_path: Optional path to the preserved sweep config YAML.

    Returns:
      List of matched column names. Inspect the printed table and edit this
      list before passing it to the plotting functions.

    Examples:
      >>> sweep_cols = suggest_sweep_cols(df, "sweep_dir/sweep_config.yaml")
      >>> # edit the list if the auto-match missed anything, then:
      >>> plot_marginal(df, sweep_col=sweep_cols[0], metric_col="plddt")
    """
    matched: list[str] = []

    if sweep_config_path is not None:
        with open(sweep_config_path) as f:
            cfg = yaml.safe_load(f)

        sweep_keys = list(cfg.get("sweep", {}).keys())
        sweep_overrides = cfg.get("sweep_overrides", [])
        df_cols = set(df.columns)

        print(f"Sweep config: {Path(sweep_config_path).name}")
        print(f"  sweep keys: {sweep_keys}")
        if sweep_overrides:
            print(f"  sweep_overrides ({len(sweep_overrides)} variants")
            for ov in sweep_overrides:
                print(f"    {ov}")
        print()

        print(f"  {'Sweep config key':<60} {'→':<3} {'DataFrame column':<30} Status")
        print("  " + "-" * 100)

        seen_cols: set[str] = set()
        for key in sweep_keys:
            col, status = _resolve_sweep_key(key, df_cols)
            print(f"  {key:<60} {'→':<3} {(col or '—'):<30} {status}")
            if col is not None and col not in seen_cols:
                matched.append(col)
                seen_cols.add(col)

        unresolved = [k for k in sweep_keys if _resolve_sweep_key(k, df_cols)[0] is None]
        if unresolved:
            numeric_cols = list(df.select_dtypes("number").columns)
            print(f"\n  Unresolved keys: {unresolved}")
            print(f"  Available numeric columns: {numeric_cols}")
            print("  Add missing entries to _SWEEP_KEY_ALIASES if the column exists under a different name.")
    else:
        _meta = {
            "example_id",
            "designed_structure_path",
            "dataset_name",
            "task",
            "chain_of_interest",
            "seed",
            "input_cif_path",
            "sweep_config",
            "sequence_redesigned_structure_path",
            "foldseek_cluster_id",
        }
        candidates = [c for c in df.columns if c not in _meta and df[c].nunique() < 20]
        print("No sweep config provided. Low-cardinality columns (possible sweep axes):")
        for c in candidates:
            vals = sorted(df[c].dropna().unique())
            print(f"  {c}: {vals}")
        matched = candidates

    print(f"\nSuggested sweep_cols = {matched}")
    return matched


# ---------------------------------------------------------------------------
# Core aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_to_task_level(
    df: pd.DataFrame,
    sweep_cols: list[str],
    metric_col: str | list[str],
    task_col: str = "task",
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate per-example metrics to the (sweep combo, task) grain.

    Groups ``df`` by ``sweep_cols + [task_col]`` and computes the mean (or
    median), standard deviation, and count of ``metric_col``.

    Args:
      df: Per-example metrics DataFrame.
      sweep_cols: Columns identifying sweep axes.
      metric_col: Metric column to aggregate. May be a list of candidate column
        names, at most one of which is non-NaN per row (see :py:func:`_resolve_metric`).
      task_col: Column identifying the task. Defaults to ``"task"``.
      agg: ``"mean"`` or ``"median"``. Defaults to ``"mean"``.

    Returns:
      DataFrame with columns: all sweep_cols, task_col, ``<metric_col>``,
      ``<metric_col>_std``, ``<metric_col>_n``.

    Examples:
      >>> agg_df = aggregate_to_task_level(df, ["nsteps", "bb_ca_noise_scale"], "plddt")
      >>> agg_df = aggregate_to_task_level(df, ["nsteps"], ["plddt", "tm_score"])
    """
    df, metric_col = _resolve_metric(df, metric_col)
    group_cols = [*sweep_cols, task_col]
    grouped = df.groupby(group_cols)[metric_col]
    agg_fn = grouped.median if agg == "median" else grouped.mean
    result = agg_fn().rename(metric_col).reset_index()
    result[f"{metric_col}_std"] = grouped.std().values
    result[f"{metric_col}_n"] = grouped.count().values
    return result


def _build_1d_agg(
    df: pd.DataFrame,
    target_col: str,
    metric_col: str,
    task_col: str,
    error: Literal["std", "sem", "ci95"] | None,
) -> pd.DataFrame:
    """Group by (target_col, task_col) and compute mean ± error.

    Grouping only on the target axis naturally marginalizes all other sweep
    dimensions.  Returns a DataFrame with columns:
    target_col, task_col, ``"mean"``, ``"err"``.
    """
    grouped = df.groupby([target_col, task_col])[metric_col]
    result = grouped.mean().rename("mean").reset_index()
    std = grouped.std().values
    n = grouped.count().values
    if error == "std":
        result["err"] = std
    elif error == "sem":
        result["err"] = std / np.sqrt(np.maximum(n, 1))
    elif error == "ci95":
        result["err"] = 1.96 * std / np.sqrt(np.maximum(n, 1))
    else:
        result["err"] = 0.0
    return result


def _build_2d_pivot(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    metric_col: str,
    task_col: str,
) -> dict[str, pd.DataFrame]:
    """Build one metric pivot table per task over a (col1 x col2) grid.

    Returns:
      Dict mapping task name → pivot DataFrame (col1 as index, col2 as columns),
      with cell values as the mean of metric_col.
    """
    return {
        task: task_df.pivot_table(index=col1, columns=col2, values=metric_col, aggfunc="mean")
        for task, task_df in df.groupby(task_col)
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _save_or_show(fig: Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def _display_label(v: object, col: str, value_labels: dict[str, dict] | None) -> str:
    """Return a short display label for a sweep value.

    Uses ``value_labels[col][v]`` if defined, otherwise truncates long strings.
    """
    if value_labels and col in value_labels:
        return str(value_labels[col].get(v, v))
    s = str(v)
    return s[-30:] if len(s) > 30 else s


def _apply_value_labels_to_df(
    df: pd.DataFrame,
    cols: list[str],
    value_labels: dict[str, dict] | None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with sweep values replaced by display labels."""
    if not value_labels:
        return df
    df = df.copy()
    for col in cols:
        if col in df.columns and col in value_labels:
            mapping = value_labels[col]
            df[col] = df[col].map(lambda v, m=mapping: m.get(v, v))
    return df


def _make_compact_fig(
    n_valid_rows: int,
    valid_j_per_row: dict[int, list[int]],
    n_facets: int,
    cell_h: float = 3.0,
    cell_w: float = 9.0,
    top: float = 0.92,
) -> tuple[Figure, dict[tuple[int, int], Axes]]:
    """Create a figure with nested GridSpec for compact, gap-free layout.

    Each row uses only the columns (facets) that have data. Rows with fewer
    valid facets still take the same vertical space, but their subplots stretch
    to fill the row width.

    Args:
      n_valid_rows: Number of rows with at least one valid cell.
      valid_j_per_row: Maps row_i (0-indexed in valid rows) → list of valid facet indices.
      n_facets: Total number of facets (determines figure width).
      cell_h: Height per row in inches. Defaults to ``3.0``.
      cell_w: Width per facet in inches. Defaults to ``9.0``.

    Returns:
      ``(fig, axes_dict)`` where ``axes_dict`` maps ``(row_i, j)`` → Axes.
    """
    fig_w = max(cell_w * min(n_facets, max(len(v) for v in valid_j_per_row.values())), cell_w)
    fig_h = cell_h * n_valid_rows + 0.6  # 0.6 for suptitle
    fig = plt.figure(figsize=(fig_w, fig_h))

    outer_gs = GridSpec(
        n_valid_rows,
        1,
        figure=fig,
        top=top,
        bottom=0.08,
        left=0.07,
        right=0.97,
        hspace=0.55,
    )

    axes_dict: dict[tuple[int, int], Axes] = {}
    for row_i in range(n_valid_rows):
        valid_j = valid_j_per_row[row_i]
        n_valid = len(valid_j)
        inner_gs = GridSpecFromSubplotSpec(1, n_valid, subplot_spec=outer_gs[row_i], wspace=0.25)
        first_ax = None
        for k, j in enumerate(valid_j):
            kw: dict = {"sharey": first_ax} if first_ax is not None else {}
            ax = fig.add_subplot(inner_gs[0, k], **kw)
            if first_ax is None:
                first_ax = ax
            axes_dict[(row_i, j)] = ax

    return fig, axes_dict


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------


def _iter_subsets(
    df: pd.DataFrame,
    cols_to_iterate: list[str],
) -> list[tuple[dict[str, object], pd.DataFrame]]:
    """Return (label_dict, sub_df) pairs for each unique value combination of cols_to_iterate."""
    result = []
    for keys, sub_df in df.groupby(cols_to_iterate, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        result.append((dict(zip(cols_to_iterate, keys, strict=False)), sub_df))
    return result


def _labelled_save_path(
    save_path: str | Path | None,
    label: dict[str, object],
) -> str | Path | None:
    """Append label key=value pairs to a save path stem.

    For example ``("out/fig.png", {"ckpt": "v1"})`` → ``"out/fig__ckpt=v1.png"``.
    Values are truncated to 20 characters to keep filenames reasonable.
    """
    if save_path is None or not label:
        return save_path
    path = Path(save_path)
    label_str = "_".join(f"{k}={str(v)[-20:]}" for k, v in sorted(label.items()))
    return path.with_stem(f"{path.stem}__{label_str}")


# ---------------------------------------------------------------------------
# 1-D marginal plot
# ---------------------------------------------------------------------------


def plot_marginal(
    df: pd.DataFrame,
    sweep_col: str,
    metric_col: str | list[str],
    *,
    task_col: str = "task",
    facet_col: str | None = None,
    agg: str = "mean",
    error: Literal["std", "sem", "ci95"] | None = "sem",
    value_labels: dict[str, dict] | None = None,
    metric_label: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    cols_to_iterate: list[str] | None = None,
    save_path: str | Path | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray] | list[tuple[Figure, np.ndarray]]:
    """Plot the marginal effect of one sweep axis on a metric, faceted by task.

    Marginalizes over all other sweep dimensions by grouping only on
    ``(sweep_col, task_col)``. Produces one subplot per task (rows). When
    ``facet_col`` is given, adds one column per unique facet value so task
    and dataset separations are both visible. Numeric ``sweep_col`` → line
    plot; categorical → bar chart.

    Subplots are created only for (task, facet) pairs that actually contain
    data, so the layout is always compact with no empty spaces.

    Args:
      df: Per-example metrics DataFrame.
      sweep_col: Sweep axis to show on the x-axis.
      metric_col: Metric to show on the y-axis. May be a list of candidate
        column names, at most one of which is non-NaN per row.
      task_col: Column identifying tasks. Defaults to ``"task"``.
      facet_col: If given, produce one column of subplots per unique value
        (e.g. ``"dataset_name"``). Defaults to ``None``.
      agg: ``"mean"`` or ``"median"``. Defaults to ``"mean"``.
      error: Error bar style: ``"std"``, ``"sem"``, ``"ci95"``, or ``None``.
        Defaults to ``"sem"``.
      value_labels: Optional mapping ``{column: {raw_value: display_label}}``
        used to replace long or cryptic sweep values with short readable labels.
      metric_label: Display name for the metric used in auto-generated titles
        and y-axis labels. Defaults to ``metric_col``.
      title: Figure suptitle override.
      ylabel: Y-axis label override.
      cols_to_iterate: If given, produce a separate figure for each unique value
        combination of these columns (rather than marginalizing over them).
        Returns a list of ``(fig, axes)`` pairs; filenames gain a ``__{k}={v}``
        suffix when ``save_path`` is set.
      save_path: If given, save the figure here instead of displaying.
      **kwargs: Forwarded to the underlying matplotlib plot/bar call.

    Returns:
      ``(fig, axes)`` where ``axes`` is a 2-D object array of shape
      ``(n_tasks, n_facets)`` with ``None`` for empty cells.  A list of such
      pairs is returned when ``cols_to_iterate`` is set.

    Examples:
      >>> fig, axes = plot_marginal(df, "nsteps", "plddt")
      >>> fig, axes = plot_marginal(
      ...     df,
      ...     "nsteps",
      ...     "plddt",
      ...     facet_col="dataset_name",
      ...     value_labels={"nsteps": {50: "fast", 200: "slow"}},
      ... )
      >>> figs = plot_marginal(df, "nsteps", "plddt", cols_to_iterate=["ckpt_path"])
    """
    if cols_to_iterate:
        results = []
        for label, sub_df in _iter_subsets(df, cols_to_iterate):
            label_str = "  |  ".join(f"{k}={_display_label(v, k, value_labels)}" for k, v in label.items())
            print(f"--- {label_str} ---")
            results.append(
                plot_marginal(
                    sub_df,
                    sweep_col,
                    metric_col,
                    task_col=task_col,
                    facet_col=facet_col,
                    agg=agg,
                    error=error,
                    value_labels=value_labels,
                    metric_label=metric_label,
                    title=title,
                    ylabel=ylabel,
                    cols_to_iterate=None,
                    save_path=_labelled_save_path(save_path, label),
                    **kwargs,
                )
            )
        return results
    df, metric_col = _resolve_metric(df, metric_col)
    is_numeric = pd.api.types.is_numeric_dtype(df[sweep_col])
    facets = sorted(df[facet_col].unique()) if facet_col else [None]
    n_facets = len(facets)
    mname = metric_label or metric_col
    suptitle = title or f"Effect of  {sweep_col}  on  {mname}"
    ylbl = ylabel or mname

    # Build global x-axis from full data for consistent categorical bar positions.
    all_x_raw = sorted(df[sweep_col].dropna().unique())
    all_x_labels = [_display_label(v, sweep_col, value_labels) for v in all_x_raw]

    def _draw_cell(ax: Axes, cell_df: pd.DataFrame, is_first_col: bool) -> bool:
        """Plot aggregated data into ax. Returns True if any data was drawn."""
        cell_df = cell_df.sort_values(sweep_col)
        if cell_df.empty:
            return False
        x_raw = cell_df[sweep_col].values
        y = cell_df["mean"].values
        yerr = cell_df["err"].values if error else None
        if is_numeric:
            ax.plot(x_raw, y, marker="o", **kwargs)
            if yerr is not None:
                ax.fill_between(x_raw, y - yerr, y + yerr, alpha=0.25)
        else:
            x_pos = np.array([all_x_raw.index(v) for v in x_raw], dtype=float)
            ax.bar(x_pos, y, **kwargs)
            if yerr is not None:
                ax.errorbar(x_pos, y, yerr=yerr, fmt="none", color="black", capsize=4)
            ax.set_xticks(range(len(all_x_raw)))
            ax.set_xticklabels(all_x_labels, rotation=0, ha="center", fontsize=7)
        ax.set_xlabel(sweep_col)
        if is_first_col:
            ax.set_ylabel(ylbl)
        ax.grid(True, alpha=0.3)
        return True

    def _sub_df(facet: Any) -> pd.DataFrame:
        return df[df[facet_col] == facet] if facet is not None else df

    # Per-task rows.
    tasks = sorted(df[task_col].unique())
    agg_by_facet = {facet: _build_1d_agg(_sub_df(facet), sweep_col, metric_col, task_col, error) for facet in facets}

    # Pre-compute which (task, facet) cells have data.
    def _cell_df(task: Any, facet: Any) -> pd.DataFrame:
        adf = agg_by_facet[facet]
        return adf[adf[task_col] == task]

    # Determine valid facets per task-row (all tasks in one figure).
    valid_j_per_row = {
        i: [j for j, facet in enumerate(facets) if not _cell_df(task, facet).empty] for i, task in enumerate(tasks)
    }
    valid_rows = [i for i in range(len(tasks)) if valid_j_per_row[i]]
    if not valid_rows:
        return plt.figure(), np.array([])

    fig, axes_dict = _make_compact_fig(
        len(valid_rows),
        {ri: valid_j_per_row[task_i] for ri, task_i in enumerate(valid_rows)},
        n_facets,
    )
    fig.suptitle(suptitle, fontsize=13)

    for ri, task_i in enumerate(valid_rows):
        task = tasks[task_i]
        valid_j = valid_j_per_row[task_i]
        for k, j in enumerate(valid_j):
            facet = facets[j]
            ax = axes_dict[(ri, j)]
            _draw_cell(ax, _cell_df(task, facet), is_first_col=(k == 0))
            task_label = _display_label(task, task_col, value_labels)
            if ri == 0 and facet is not None:
                ax.set_title(f"{facet}\n{task_label}", fontsize=10)
            else:
                ax.set_title(task_label, fontsize=10)

    _save_or_show(fig, save_path)
    axes_arr = np.full((len(tasks), n_facets), None, dtype=object)
    for ri, task_i in enumerate(valid_rows):
        for j in valid_j_per_row[task_i]:
            axes_arr[task_i, j] = axes_dict[(ri, j)]
    return fig, axes_arr


# ---------------------------------------------------------------------------
# 2-D heatmap
# ---------------------------------------------------------------------------


def plot_heatmap(
    df: pd.DataFrame,
    sweep_col1: str,
    sweep_col2: str,
    metric_col: str | list[str],
    *,
    task_col: str = "task",
    facet_col: str | None = None,
    agg: str = "mean",
    value_labels: dict[str, dict] | None = None,
    metric_label: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    cols_to_iterate: list[str] | None = None,
    save_path: str | Path | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray] | list[tuple[Figure, np.ndarray]]:
    """Plot a heatmap of a metric over two sweep axes, faceted by task.

    Marginalizes over any remaining sweep dimensions by averaging within
    cells. Produces one heatmap per (task x facet) combination, tiled 2 per
    row. Cell values are annotated numerically.

    Args:
      df: Per-example metrics DataFrame.
      sweep_col1: First sweep axis (rows of heatmap).
      sweep_col2: Second sweep axis (columns of heatmap).
      metric_col: Metric to display in cells. May be a list of candidate
        column names, at most one of which is non-NaN per row.
      task_col: Column identifying tasks. Defaults to ``"task"``.
      facet_col: If given, produce one heatmap per (task x facet_col value)
        combination (e.g. ``"dataset_name"``). Defaults to ``None``.
      agg: Aggregation within cells. Defaults to ``"mean"``.
      value_labels: Optional mapping ``{column: {raw_value: display_label}}``
        for axis tick labels on the heatmap.
      metric_label: Display name for the metric used in titles. Defaults to
        ``metric_col``.
      title: Figure suptitle override.
      ylabel: Y-axis label (heatmap row axis). Defaults to ``sweep_col1``.
      cols_to_iterate: If given, produce a separate heatmap figure for each
        unique value combination of these columns. Returns a list of
        ``(fig, axes)`` pairs.
      save_path: If given, save the figure here instead of displaying.
      **kwargs: Forwarded to :py:func:`seaborn.heatmap`.

    Returns:
      ``(fig, axes)`` where ``axes`` is a 2-D array of shape ``(n_rows, 2)``.
      A list of such pairs is returned when ``cols_to_iterate`` is set.

    Examples:
      >>> plot_heatmap(df, "nsteps", "bb_ca_noise_scale", "plddt")
      >>> plot_heatmap(
      ...     df, "nsteps", "bb_ca_noise_scale", ["plddt", "tm_score"], facet_col="dataset_name"
      ... )
      >>> figs = plot_heatmap(
      ...     df, "nsteps", "bb_ca_noise_scale", "plddt", cols_to_iterate=["ckpt_path"]
      ... )
    """
    if cols_to_iterate:
        results = []
        for label, sub_df in _iter_subsets(df, cols_to_iterate):
            label_str = "  |  ".join(f"{k}={_display_label(v, k, value_labels)}" for k, v in label.items())
            print(f"--- {label_str} ---")
            results.append(
                plot_heatmap(
                    sub_df,
                    sweep_col1,
                    sweep_col2,
                    metric_col,
                    task_col=task_col,
                    facet_col=facet_col,
                    agg=agg,
                    value_labels=value_labels,
                    metric_label=metric_label,
                    title=title,
                    ylabel=ylabel,
                    cols_to_iterate=None,
                    save_path=_labelled_save_path(save_path, label),
                    **kwargs,
                )
            )
        return results
    df, metric_col = _resolve_metric(df, metric_col)
    facets = sorted(df[facet_col].unique()) if facet_col else [None]
    mname = metric_label or metric_col
    suptitle = title or f"{mname}:  {sweep_col1}  x  {sweep_col2}"
    ylbl = ylabel or sweep_col1

    def _apply_vl(pivot: pd.DataFrame) -> pd.DataFrame:
        """Apply value_labels to a pivot table's index and columns."""
        if not value_labels:
            return pivot
        if sweep_col1 in value_labels:
            pivot = pivot.rename(index=lambda v: value_labels[sweep_col1].get(v, v))
        if sweep_col2 in value_labels:
            pivot = pivot.rename(columns=lambda v: value_labels[sweep_col2].get(v, v))
        return pivot

    # Build list of (label, pivot) cells.
    tasks = sorted(df[task_col].unique())
    cells: list[tuple[str, pd.DataFrame]] = []
    for facet in facets:
        sub_df = df[df[facet_col] == facet] if facet is not None else df
        pivots = _build_2d_pivot(sub_df, sweep_col1, sweep_col2, metric_col, task_col)
        for task in tasks:
            if task in pivots:
                task_label = _display_label(task, task_col, value_labels)
                label = f"{task_label}  —  {facet}" if facet is not None else task_label
                cells.append((label, _apply_vl(pivots[task])))

    all_vals = np.concatenate([c[1].values.ravel() for c in cells])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    n_cols = 2
    n_cells = len(cells)
    n_rows = math.ceil(n_cells / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols + 1, 5 * n_rows), squeeze=False)
    fig.suptitle(suptitle, fontsize=13)

    for idx, (cell_title, pivot) in enumerate(cells):
        ax = axes[idx // n_cols, idx % n_cols]
        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".3f",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            cbar=True,
            **kwargs,
        )
        ax.set_title(cell_title, fontsize=10)
        ax.set_xlabel(sweep_col2)
        ax.set_ylabel(ylbl)

    for idx in range(n_cells, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_or_show(fig, save_path)
    return fig, axes


# ---------------------------------------------------------------------------
# 3-D surface plot
# ---------------------------------------------------------------------------


def plot_surface(
    df: pd.DataFrame,
    sweep_col1: str,
    sweep_col2: str,
    metric_col: str | list[str],
    *,
    task_col: str = "task",
    agg: str = "mean",
    title: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """Plot a 3-D surface of a metric over two numeric sweep axes, faceted by task.

    Both ``sweep_col1`` and ``sweep_col2`` must be numeric. Complementary to
    :py:func:`plot_heatmap` — useful for seeing the curvature of a response
    surface. One surface per task, tiled horizontally.

    Args:
      df: Per-example metrics DataFrame.
      sweep_col1: First numeric sweep axis (y-axis of surface).
      sweep_col2: Second numeric sweep axis (x-axis of surface).
      metric_col: Metric on the z-axis. May be a list of candidate column
        names, at most one of which is non-NaN per row.
      task_col: Column identifying tasks. Defaults to ``"task"``.
      agg: Aggregation. Defaults to ``"mean"``.
      title: Figure suptitle. Defaults to ``"{metric_col}: {sweep_col1} x {sweep_col2}"``.
      ylabel: Y-axis label on each surface. Defaults to ``sweep_col1``.
      save_path: If given, save the figure here instead of displaying.
      **kwargs: Forwarded to :py:meth:`~mpl_toolkits.mplot3d.Axes3D.plot_surface`.

    Returns:
      ``(fig, axes)`` where ``axes`` is a 1-D array of 3-D Axes.

    Examples:
      >>> plot_surface(df, "nsteps", "bb_ca_noise_scale", "plddt")
      >>> plot_surface(df, "nsteps", "bb_ca_noise_scale", ["plddt", "tm_score"])
    """
    df, metric_col = _resolve_metric(df, metric_col)
    pivots = _build_2d_pivot(df, sweep_col1, sweep_col2, metric_col, task_col)
    tasks = sorted(pivots)
    n_tasks = len(tasks)

    fig = plt.figure(figsize=(6 * n_tasks, 5))
    fig.suptitle(title or f"{metric_col}:  {sweep_col1}  x  {sweep_col2}", fontsize=13)

    axes = []
    for i, task in enumerate(tasks, 1):
        pivot = pivots[task]
        ax = fig.add_subplot(1, n_tasks, i, projection="3d")
        x_vals = np.array(pivot.columns.astype(float))
        y_vals = np.array(pivot.index.astype(float))
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = pivot.values.astype(float)
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, **kwargs)
        ax.set_xlabel(sweep_col2, fontsize=8)
        ax.set_ylabel(ylabel or sweep_col1, fontsize=8)
        ax.set_zlabel(metric_col, fontsize=8)
        ax.set_title(task, fontsize=10)
        axes.append(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_or_show(fig, save_path)
    return fig, np.array(axes)


# ---------------------------------------------------------------------------
# Axis impact / importance ranking
# ---------------------------------------------------------------------------


def _axis_impact_on_ax(
    ax: Axes,
    df: pd.DataFrame,
    sweep_cols: list[str],
    sorted_cols: list[str],
    metric_col: str,
    task_col: str,
    tasks: list,
    colors: np.ndarray,
    value_labels: dict[str, dict] | None = None,
) -> None:
    """Draw grouped impact bars on a single Axes."""
    n_tasks = len(tasks)
    bar_w = 0.8 / n_tasks
    x_pos = np.arange(len(sorted_cols))

    impact: dict[str, dict] = {}
    for col in sweep_cols:
        agg_df = _build_1d_agg(df, col, metric_col, task_col, error=None)
        impact[col] = {}
        for task in tasks:
            task_means = agg_df[agg_df[task_col] == task]["mean"]
            impact[col][task] = float(task_means.max() - task_means.min()) if len(task_means) > 1 else 0.0

    for i, (task, color) in enumerate(zip(tasks, colors, strict=False)):
        values = [impact[col][task] for col in sorted_cols]
        offset = (i - n_tasks / 2 + 0.5) * bar_w
        task_label = _display_label(task, task_col, value_labels)
        ax.bar(x_pos + offset, values, width=bar_w, label=task_label, color=color)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_cols, rotation=0, ha="center", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def plot_axis_impact(
    df: pd.DataFrame,
    sweep_cols: list[str],
    metric_col: str | list[str],
    *,
    task_col: str = "task",
    facet_col: str | None = None,
    agg: str = "mean",
    value_labels: dict[str, dict] | None = None,
    metric_label: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    cols_to_iterate: list[str] | None = None,
    save_path: str | Path | None = None,
) -> tuple[Figure, np.ndarray] | list[tuple[Figure, np.ndarray]]:
    """Rank sweep axes by their impact on a metric (max - min of marginal mean).

    For each axis in ``sweep_cols``, computes the range of the marginal mean
    of ``metric_col`` (marginalizing over all other axes). Axes are sorted
    by overall average impact and shown as grouped vertical bars, colored by
    task. When ``facet_col`` is given, produces one subplot per facet value.

    Args:
      df: Per-example metrics DataFrame.
      sweep_cols: All sweep axis columns to evaluate.
      metric_col: Metric to evaluate. May be a list of candidate column names,
        at most one of which is non-NaN per row.
      task_col: Column identifying tasks. Defaults to ``"task"``.
      facet_col: If given, produce one subplot per unique value in this column
        (e.g. ``"dataset_name"``). Defaults to ``None``.
      agg: Aggregation. Defaults to ``"mean"``.
      metric_label: Display name for the metric used in auto-generated titles.
        Defaults to ``metric_col``.
      title: Figure suptitle override.
      ylabel: Y-axis label override. Defaults to
        ``"Range of mean {metric_col}  (max - min)"``.
      cols_to_iterate: If given, produce a separate figure for each unique value
        combination of these columns. Returns a list of ``(fig, axes)`` pairs.
      save_path: If given, save the figure here instead of displaying.

    Returns:
      ``(fig, axes)`` where ``axes`` is a 1-D array of Axes.  A list of such
      pairs is returned when ``cols_to_iterate`` is set.

    Examples:
      >>> plot_axis_impact(df, ["nsteps", "bb_ca_noise_scale", "ckpt_name"], "plddt")
      >>> plot_axis_impact(df, ["nsteps"], ["plddt", "tm_score"], facet_col="dataset_name")
      >>> figs = plot_axis_impact(df, sweep_cols, "plddt", cols_to_iterate=["ckpt_path"])
    """
    if cols_to_iterate:
        results = []
        for label, sub_df in _iter_subsets(df, cols_to_iterate):
            label_str = "  |  ".join(f"{k}={v}" for k, v in label.items())
            print(f"--- {label_str} ---")
            results.append(
                plot_axis_impact(
                    sub_df,
                    sweep_cols,
                    metric_col,
                    task_col=task_col,
                    facet_col=facet_col,
                    agg=agg,
                    value_labels=value_labels,
                    metric_label=metric_label,
                    title=title,
                    ylabel=ylabel,
                    cols_to_iterate=None,
                    save_path=_labelled_save_path(save_path, label),
                )
            )
        return results
    df, metric_col = _resolve_metric(df, metric_col)
    mname = metric_label or metric_col
    facets = sorted(df[facet_col].unique()) if facet_col is not None else [None]
    suptitle = title or f"Sweep axis impact on  {mname}"
    ylbl = ylabel or f"Range of mean {mname}  (max - min)"

    def _sub_df(facet: Any) -> pd.DataFrame:
        return df[df[facet_col] == facet] if facet is not None else df

    tasks = sorted(df[task_col].unique())
    n_tasks = len(tasks)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_tasks))

    avg_impact_global = {}
    for col in sweep_cols:
        agg_df = _build_1d_agg(df, col, metric_col, task_col, error=None)
        col_impacts = []
        for task in tasks:
            task_means = agg_df[agg_df[task_col] == task]["mean"]
            col_impacts.append(float(task_means.max() - task_means.min()) if len(task_means) > 1 else 0.0)
        avg_impact_global[col] = float(np.mean(col_impacts))
    sorted_cols = sorted(sweep_cols, key=lambda c: avg_impact_global[c], reverse=True)

    subplot_w = max(9.0, 2.0 * len(sorted_cols))
    n_facets = len(facets)
    fig, axes = plt.subplots(
        n_facets,
        1,
        figsize=(subplot_w, 4 * n_facets),
        squeeze=False,
        sharey=True,
    )
    fig.suptitle(suptitle, fontsize=13)

    for ax, facet in zip(axes[:, 0], facets, strict=False):
        _axis_impact_on_ax(
            ax, _sub_df(facet), sweep_cols, sorted_cols, metric_col, task_col, tasks, colors, value_labels
        )
        ax.set_ylabel(ylbl)
        if facet is not None:
            ax.set_title(str(facet), fontsize=10)

    axes[0, 0].legend(title=task_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_or_show(fig, save_path)
    return fig, axes[:, 0]


# ---------------------------------------------------------------------------
# Overview: all marginals in one compact figure
# ---------------------------------------------------------------------------


def plot_all_marginals(
    df: pd.DataFrame,
    sweep_cols: list[str],
    metric_col: str | list[str],
    *,
    task_col: str = "task",
    facet_col: str | None = None,
    agg: str = "mean",
    error: Literal["std", "sem", "ci95"] | None = "sem",
    value_labels: dict[str, dict] | None = None,
    metric_label: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    cols_to_iterate: list[str] | None = None,
    save_path: str | Path | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray] | list[tuple[Figure, np.ndarray]]:
    """Plot the marginal of every sweep axis in a single compact figure.

    Creates one row per sweep axis. Within each row all tasks are overlaid
    as colored lines or grouped bars. When ``facet_col`` is given, adds one
    column per unique facet value so dataset-level differences are visible
    alongside task-level differences.

    Only (sweep_col, facet) pairs that contain data produce subplots; empty
    combinations are omitted entirely so the layout has no gaps. Rows with
    fewer valid facets still span the figure width, with their cells expanded
    proportionally.

    Args:
      df: Per-example metrics DataFrame.
      sweep_cols: All sweep axis columns to plot.
      metric_col: Metric column. May be a list of candidate column names,
        at most one of which is non-NaN per row.
      task_col: Column identifying tasks. Defaults to ``"task"``.
      facet_col: If given, produce one column per unique value in this column
        (e.g. ``"dataset_name"``). Defaults to ``None``.
      agg: Aggregation. Defaults to ``"mean"``.
      error: Error bar style. Defaults to ``"sem"``.
      value_labels: Optional mapping ``{column: {raw_value: display_label}}``
        used to replace long or cryptic sweep values with short readable labels.
      metric_label: Display name for the metric used in auto-generated titles
        and y-axis labels. Defaults to ``metric_col``.
      title: Figure suptitle override.
      ylabel: Y-axis label override.
      cols_to_iterate: If given, produce a separate figure for each unique value
        combination of these columns. Returns a list of ``(fig, axes)`` pairs.
      save_path: If given, save the figure here instead of displaying.
      **kwargs: Forwarded to the underlying plot/bar call.

    Returns:
      ``(fig, axes)`` where ``axes`` is a 2-D object array of shape
      ``(n_sweep_cols, n_facets)`` with ``None`` for empty cells.  A list of
      such pairs is returned when ``cols_to_iterate`` is set.

    Examples:
      >>> plot_all_marginals(df, ["nsteps", "bb_ca_noise_scale"], "plddt")
      >>> plot_all_marginals(
      ...     df,
      ...     ["nsteps"],
      ...     "plddt",
      ...     facet_col="dataset_name",
      ...     value_labels={"nsteps": {50: "fast", 200: "slow"}},
      ... )
      >>> figs = plot_all_marginals(df, sweep_cols, "plddt", cols_to_iterate=["ckpt_path"])
    """
    if cols_to_iterate:
        results = []
        for label, sub_df in _iter_subsets(df, cols_to_iterate):
            label_str = "  |  ".join(f"{k}={_display_label(v, k, value_labels)}" for k, v in label.items())
            print(f"--- {label_str} ---")
            results.append(
                plot_all_marginals(
                    sub_df,
                    sweep_cols,
                    metric_col,
                    task_col=task_col,
                    facet_col=facet_col,
                    agg=agg,
                    error=error,
                    value_labels=value_labels,
                    metric_label=metric_label,
                    title=title,
                    ylabel=ylabel,
                    cols_to_iterate=None,
                    save_path=_labelled_save_path(save_path, label),
                    **kwargs,
                )
            )
        return results
    df, metric_col = _resolve_metric(df, metric_col)
    tasks = sorted(df[task_col].unique())
    n_tasks = len(tasks)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_tasks))
    facets = sorted(df[facet_col].unique()) if facet_col else [None]
    n_facets = len(facets)
    mname = metric_label or metric_col
    suptitle = title or f"All sweep axis marginals  —  {mname}"
    ylbl = ylabel or mname

    def _sub_df(facet: Any) -> pd.DataFrame:
        return df[df[facet_col] == facet] if facet is not None else df

    def _fill_cell(
        ax: Axes,
        sweep_col: str,
        sub_df: pd.DataFrame,
        is_first_col: bool,
        is_top_row: bool,
        facet: object,
        all_x_raw: list,
        all_x_labels: list,
    ) -> bool:
        """Draw one (sweep_col, facet) cell. Returns True if any data was drawn."""
        is_numeric = pd.api.types.is_numeric_dtype(df[sweep_col])

        agg_df = _build_1d_agg(sub_df, sweep_col, metric_col, task_col, error)
        has_data = False
        for task_i, (task, color) in enumerate(zip(tasks, colors, strict=False)):
            task_df = agg_df[agg_df[task_col] == task].sort_values(sweep_col)
            if task_df.empty:
                continue
            has_data = True
            task_label = _display_label(task, task_col, value_labels)
            x_raw = task_df[sweep_col].values
            y = task_df["mean"].values
            err = task_df["err"].values if error else None
            if is_numeric:
                ax.plot(x_raw, y, marker="o", label=task_label, color=color, **kwargs)
                if err is not None:
                    ax.fill_between(x_raw, y - err, y + err, alpha=0.15, color=color)
            else:
                bar_w = 0.8 / n_tasks
                offset = (task_i - n_tasks / 2 + 0.5) * bar_w
                x_pos = np.array([all_x_raw.index(v) for v in x_raw], dtype=float)
                ax.bar(x_pos + offset, y, width=bar_w, label=task_label, color=color, **kwargs)
                if err is not None:
                    ax.errorbar(x_pos + offset, y, yerr=err, fmt="none", color="black", capsize=3)
                ax.set_xticks(range(len(all_x_raw)))
                ax.set_xticklabels(all_x_labels, rotation=0, ha="center", fontsize=7)

        if not has_data:
            return False

        if is_first_col:
            ax.set_ylabel(ylbl)
        if is_top_row and facet is not None:
            ax.set_title(f"{facet}\n{sweep_col}", fontsize=10)
        else:
            ax.set_title(sweep_col, fontsize=10)
        ax.grid(True, alpha=0.3)
        return True

    # Pre-compute which (sweep_col, facet) pairs have data.
    def _has_data(sweep_col: str, facet: Any) -> bool:
        return not _sub_df(facet)[sweep_col].dropna().empty

    # All sweep_cols in one compact figure (no pagination).
    valid_j_per_row = {
        ai: [j for j, facet in enumerate(facets) if _has_data(sweep_col, facet)]
        for ai, sweep_col in enumerate(sweep_cols)
    }
    valid_rows = [ai for ai in range(len(sweep_cols)) if valid_j_per_row[ai]]
    if not valid_rows:
        return plt.figure(), np.array([])

    n_valid_rows = len(valid_rows)
    valid_j_per_valid_row = {ri: valid_j_per_row[ai] for ri, ai in enumerate(valid_rows)}

    fig, axes_dict = _make_compact_fig(n_valid_rows, valid_j_per_valid_row, n_facets, top=0.85)
    fig.suptitle(suptitle, fontsize=13)
    legend_ax: Axes | None = None

    for ri, ai in enumerate(valid_rows):
        sweep_col = sweep_cols[ai]
        all_x_raw = sorted(df[sweep_col].dropna().unique())
        all_x_labels = [_display_label(v, sweep_col, value_labels) for v in all_x_raw]
        valid_j = valid_j_per_valid_row[ri]

        for k, j in enumerate(valid_j):
            facet = facets[j]
            ax = axes_dict[(ri, j)]
            has_data = _fill_cell(
                ax,
                sweep_col,
                _sub_df(facet),
                is_first_col=(k == 0),
                is_top_row=(ri == 0),
                facet=facet,
                all_x_raw=all_x_raw,
                all_x_labels=all_x_labels,
            )
            if not has_data:
                ax.set_visible(False)
            elif ri == 0 and k == len(valid_j) - 1:
                legend_ax = ax

    if legend_ax is not None:
        legend_ax.legend(title=task_col, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title_fontsize=8)

    _save_or_show(fig, save_path)
    axes_arr = np.full((len(sweep_cols), n_facets), None, dtype=object)
    for ri, ai in enumerate(valid_rows):
        for j in valid_j_per_valid_row[ri]:
            axes_arr[ai, j] = axes_dict[(ri, j)]
    return fig, axes_arr


# ---------------------------------------------------------------------------
# Best configuration summary
# ---------------------------------------------------------------------------


def print_best_configs(
    df: pd.DataFrame,
    sweep_cols: list[str],
    metric_col: str | list[str],
    *,
    task_col: str = "task",
    facet_col: str | None = None,
    agg: str = "mean",
    higher_is_better: bool = True,
    value_labels: dict[str, dict] | None = None,
    metric_label: str | None = None,
    cols_to_iterate: list[str] | None = None,
) -> pd.DataFrame:
    """Print and return the best sweep configuration for each (task, facet) combo.

    For each unique combination of ``task_col`` (and ``facet_col`` if given),
    aggregates the metric across examples, then finds the sweep configuration
    that achieves the best mean metric value.

    Args:
      df: Per-example metrics DataFrame.
      sweep_cols: Sweep axis columns to include in the output table.
      metric_col: Metric to optimise. May be a list of candidate column names,
        at most one of which is non-NaN per row.
      task_col: Column identifying tasks. Defaults to ``"task"``.
      facet_col: Optional column for dataset separation (e.g. ``"dataset_name"``).
        Defaults to ``None``.
      agg: ``"mean"`` or ``"median"``. Defaults to ``"mean"``.
      higher_is_better: If ``True``, selects the configuration with the highest
        metric; if ``False``, the lowest. Defaults to ``True``.
      value_labels: Optional mapping ``{column: {raw_value: display_label}}``
        applied to the printed table for readability.
      metric_label: Display name for the metric column header. Defaults to
        ``metric_col``.
      cols_to_iterate: If given, print a separate best-config table for each
        unique value combination of these columns. Returns concatenated results.

    Returns:
      DataFrame with one row per (task, facet) combo showing the best
      sweep configuration and its metric value.

    Examples:
      >>> best = print_best_configs(df, ["nsteps", "bb_ca_noise_scale"], "plddt")
      >>> best = print_best_configs(
      ...     df,
      ...     sweep_cols,
      ...     "plddt",
      ...     facet_col="dataset_name",
      ...     value_labels={"ckpt_path": {"/very/long/path.pt": "v1"}},
      ... )
      >>> best = print_best_configs(df, sweep_cols, "plddt", cols_to_iterate=["ckpt_path"])
    """
    if cols_to_iterate:
        all_best = []
        for label, sub_df in _iter_subsets(df, cols_to_iterate):
            label_str = "  |  ".join(f"{k}={v}" for k, v in label.items())
            print(f"\n{'─' * 40}  {label_str}")
            all_best.append(
                print_best_configs(
                    sub_df,
                    sweep_cols,
                    metric_col=metric_col,
                    task_col=task_col,
                    facet_col=facet_col,
                    agg=agg,
                    higher_is_better=higher_is_better,
                    value_labels=value_labels,
                    metric_label=metric_label,
                    cols_to_iterate=None,
                )
            )
        return pd.concat(all_best, ignore_index=True)
    df, metric_col = _resolve_metric(df, metric_col)
    mname = metric_label or metric_col

    group_cols = sweep_cols + ([facet_col] if facet_col else []) + [task_col]
    key_cols = ([facet_col] if facet_col else []) + [task_col]

    grouped = df.groupby(group_cols)[metric_col]
    agg_fn = grouped.median if agg == "median" else grouped.mean
    agg_df = agg_fn().reset_index()

    if key_cols:
        best_idx = (
            agg_df.groupby(key_cols)[metric_col].idxmax()
            if higher_is_better
            else agg_df.groupby(key_cols)[metric_col].idxmin()
        )
        best_df = agg_df.iloc[list(best_idx)].sort_values(key_cols).reset_index(drop=True)
    else:
        best_row = agg_df[metric_col].idxmax() if higher_is_better else agg_df[metric_col].idxmin()
        best_df = agg_df.iloc[[best_row]].reset_index(drop=True)

    # Apply value_labels for display (sweep cols + task/facet cols).
    label_cols = sweep_cols + ([facet_col] if facet_col else []) + [task_col]
    display_df = _apply_value_labels_to_df(best_df, label_cols, value_labels)
    if mname != metric_col:
        display_df = display_df.rename(columns={metric_col: mname})

    direction = "higher" if higher_is_better else "lower"
    print(f"\nBest sweep configurations  (metric: {mname},  {direction} is better)")
    print("=" * 80)
    try:
        from tabulate import tabulate

        print(tabulate(display_df, headers="keys", tablefmt="rounded_outline", floatfmt=".4f", showindex=False))
    except ImportError:
        with pd.option_context(
            "display.max_columns", None, "display.width", None, "display.float_format", "{:.4f}".format
        ):
            print(display_df.to_string(index=False))

    return best_df


# ---------------------------------------------------------------------------
# All-in-one wrapper
# ---------------------------------------------------------------------------


def plot_sweep_summary(
    df: pd.DataFrame,
    sweep_cols: list[str],
    metric_col: str | list[str],
    *,
    marginal_sweep_cols: str | list[str],
    heatmap_col1: str | None = None,
    heatmap_col2: str | None = None,
    task_col: str = "task",
    facet_col: str | None = None,
    agg: str = "mean",
    error: Literal["std", "sem", "ci95"] | None = "sem",
    value_labels: dict[str, dict] | None = None,
    metric_label: str | None = None,
    ylabel: str | None = None,
    higher_is_better: bool = True,
    cols_to_iterate: list[str] | None = None,
    save_dir: str | Path | None = None,
) -> None:
    """Run the full suite of sweep analysis plots and print best configs.

    Convenience wrapper that calls :py:func:`print_best_configs`,
    :py:func:`plot_axis_impact`, :py:func:`plot_marginal`,
    :py:func:`plot_heatmap`, and :py:func:`plot_all_marginals` with shared
    common kwargs. When ``save_dir`` is given, each figure is saved there
    with an auto-generated filename; otherwise figures are displayed inline.

    Args:
      df: Per-example metrics DataFrame.
      sweep_cols: Sweep axis columns.
      metric_col: Metric column or list of candidate column names.
      heatmap_col1: First sweep axis for the 2-D heatmap (rows).
      heatmap_col2: Second sweep axis for the 2-D heatmap (columns).
      marginal_sweep_cols: Sweep axis (or list of axes) for 1-D marginal plots.
        One :py:func:`plot_marginal` call is made per column.
      task_col: Column identifying tasks. Defaults to ``"task"``.
      facet_col: If given, facet all plots by this column. Defaults to ``None``.
      agg: Aggregation function: ``"mean"`` or ``"median"``. Defaults to ``"mean"``.
      error: Error bar style for marginal plots: ``"std"``, ``"sem"``, ``"ci95"``,
        or ``None``. Defaults to ``"sem"``.
      value_labels: Optional mapping ``{column: {raw_value: display_label}}``
        used to replace long or cryptic sweep values with short readable labels.
        Forwarded to all plots and :py:func:`print_best_configs`.
      metric_label: Display name for the metric used in titles and axis labels.
        Defaults to ``metric_col``.
      ylabel: Y-axis label override passed to marginal and all-marginals plots.
        Axis-impact ylabel becomes ``"{ylabel} range width"``.
      higher_is_better: Passed to :py:func:`print_best_configs`. Defaults to ``True``.
      cols_to_iterate: If given, all plots are produced separately for each
        unique value combination of these columns. Forwarded to every
        underlying plot function.
      save_dir: Directory to save figures. If ``None``, figures are shown inline.

    Examples:
      >>> plot_sweep_summary(
      ...     df,
      ...     sweep_cols,
      ...     metric_col=pass_rate_cols,
      ...     heatmap_col1="bb_ca_noise_scale",
      ...     heatmap_col2="latents_noise_scale",
      ...     marginal_sweep_cols=["ckpt_path", "nsteps"],
      ...     task_col="dataset_name",
      ...     value_labels={"ckpt_path": {"/long/path/v1.pt": "v1"}},
      ...     cols_to_iterate=["seed"],
      ...     metric_label="pass_rate",
      ...     ylabel="pass_rate",
      ...     save_dir=Path("sweep_figures"),
      ... )
    """
    save_dir = Path(save_dir) if save_dir is not None else None
    marginal_cols = [marginal_sweep_cols] if isinstance(marginal_sweep_cols, str) else list(marginal_sweep_cols)

    def _save(name: str) -> Path | None:
        return (save_dir / name) if save_dir is not None else None

    print_best_configs(
        df,
        sweep_cols,
        metric_col=metric_col,
        task_col=task_col,
        facet_col=facet_col,
        agg=agg,
        value_labels=value_labels,
        metric_label=metric_label,
        higher_is_better=higher_is_better,
        cols_to_iterate=cols_to_iterate,
    )
    plot_axis_impact(
        df,
        sweep_cols,
        metric_col=metric_col,
        task_col=task_col,
        facet_col=facet_col,
        agg=agg,
        value_labels=value_labels,
        metric_label=metric_label,
        ylabel=f"{ylabel} range width" if ylabel else None,
        cols_to_iterate=cols_to_iterate,
        save_path=_save("axis_impact.png"),
    )
    for mcol in marginal_cols:
        plot_marginal(
            df,
            sweep_col=mcol,
            metric_col=metric_col,
            task_col=task_col,
            facet_col=facet_col,
            agg=agg,
            error=error,
            value_labels=value_labels,
            metric_label=metric_label,
            ylabel=ylabel,
            cols_to_iterate=cols_to_iterate,
            save_path=_save(f"{mcol}_marginal.png"),
        )
    if heatmap_col1 and heatmap_col2:
        plot_heatmap(
            df,
            heatmap_col1,
            heatmap_col2,
            metric_col=metric_col,
            task_col=task_col,
            facet_col=facet_col,
            agg=agg,
            value_labels=value_labels,
            metric_label=metric_label,
            cols_to_iterate=cols_to_iterate,
            save_path=_save(f"{heatmap_col1}_{heatmap_col2}_heatmap.png"),
        )
    plot_all_marginals(
        df,
        sweep_cols,
        metric_col=metric_col,
        task_col=task_col,
        facet_col=facet_col,
        agg=agg,
        error=error,
        value_labels=value_labels,
        metric_label=metric_label,
        ylabel=ylabel,
        cols_to_iterate=cols_to_iterate,
        save_path=_save("all_marginals.png"),
    )
