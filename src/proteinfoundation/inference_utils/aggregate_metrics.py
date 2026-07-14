#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/latent_exec.sh" "$0" "$@"'
"""Concatenate results from multi-job external evaluation runs.

Also provides shared aggregation logic used by external_eval.py.
"""

import os
import shutil
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import rootutils

from foundry.utils.env import load_dotenv
from foundry.utils.logging import RankedLogger, print_df_as_table

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

dotenv_path = os.path.join(os.environ["PROJECT_ROOT"], ".env")
load_dotenv(dotenv_path)

from proteinfoundation.inference_utils.foldseek import (  # noqa: E402  (load_dotenv must run first)
    add_diversity_summary_metrics,
    add_foldseek_diversity_to_per_example_df,
)
from proteinfoundation.inference_utils.solve_rate_metrics import (  # noqa: E402  (load_dotenv must run first)
    compute_solve_rate_metrics,
)

logger = RankedLogger(__name__, rank_zero_only=False)


def aggregate_metrics_from_per_example_df(
    per_example_df: pd.DataFrame,
    title: str = "Aggregated Metrics",
    group_by: str = "dataset_name",
) -> pd.DataFrame:
    """Aggregate per-example metrics by a grouping column.

    Args:
        per_example_df: DataFrame with columns including 'example_id', 'dataset_name',
            and metric columns.
        title: Title for the printed table.
        group_by: Column to group by.

    Returns:
        DataFrame with one row per group value, columns: <group_by>, n, and one column per metric.
    """
    # Ensure dataset_name column exists when grouping by it
    if group_by == "dataset_name" and "dataset_name" not in per_example_df.columns:
        per_example_df = per_example_df.copy()
        per_example_df["dataset_name"] = "external_eval"

    non_metric_cols = {
        "example_id",
        "dataset_name",
        "task",
        "designed_structure_path",
        "sequence_redesigned_structure_path",
        "chain_of_interest",
        "foldseek_cluster_id",
    }

    agg_rows = []
    for group_value, group_df in per_example_df.groupby(group_by):
        metric_cols = [c for c in group_df.select_dtypes(include="number").columns if c not in non_metric_cols]

        row: dict = {group_by: group_value, "n": len(group_df)}

        # Carry over identifier columns (non-metric, non-group_by) when they
        # have a unique value within the group (e.g. dataset_name per task).
        identifier_cols = non_metric_cols - {
            group_by,
            "example_id",
            "designed_structure_path",
            "sequence_redesigned_structure_path",
            "foldseek_cluster_id",
        }
        for col in identifier_cols:
            if col in group_df.columns:
                unique_vals = group_df[col].dropna().unique()
                if len(unique_vals) == 1:
                    row[col] = unique_vals[0]

        for metric_name in metric_cols:
            values = group_df[metric_name].dropna()
            if len(values) > 0:
                row[metric_name] = float(np.nanmean(values))

        # Dataset-level solve-rate statistics (e.g., percentage of targets with at least n% passing designs)
        if isinstance(group_by, str) and group_by == "dataset_name":
            row.update(compute_solve_rate_metrics(group_df))

        agg_rows.append(row)

    if not agg_rows:
        raise ValueError("No valid metrics were found after aggregation.")

    agg_df = pd.DataFrame(agg_rows)
    print_df_as_table(agg_df, title=title)

    return agg_df


def concatenate_results(
    input: str,
    cluster_with_foldseek: bool = False,
    cleanup: bool = True,
    cleanup_chains: bool = True,
    output_dir: str | None = None,
    foldseek_threads: int | None = None,
) -> None:
    """Concatenate per-example and aggregated metrics from multiple jobs.

    Can also be used to run foldseek post-hoc by calling on the output dataframe.

    Args:
        input: Path to either the directory containing the individual job result files,
            or the already-concatenated per_example_metrics.csv file.
        cluster_with_foldseek: Whether to cluster results using Foldseek.
        cleanup: If True, move individual job files to a subdirectory after concatenation.
        cleanup_chains: If True (default), delete the ``foldseek_input_chains/`` directory
            after clustering completes.  The chain CIF files are derived from the designed
            structures and can be regenerated, so removal is safe.
        output_dir: Directory where results should be saved. Defaults to the input directory
            (or parent directory when input is a file).
        foldseek_threads: Number of CPU threads to pass to foldseek. Defaults to all
            available CPUs. Set explicitly when running multiple clustering jobs in parallel
            on the same node to avoid resource contention.
    """
    input = Path(input)

    if input.is_file():
        concatenated_per_example = pd.read_csv(input)
    else:
        # Find all per-example metric files
        per_example_files = sorted(input.glob("*per_example_metrics*.csv"))
        per_example_files = [f for f in per_example_files if f.name != "per_example_metrics.csv"]

        if not per_example_files:
            logger.info(f"No per-example metric files found in {input}")
            logger.info("Nothing to concatenate - this may be a single-job run.")
            return

        logger.info(f"Found {len(per_example_files)} per-example metric files to concatenate")

        # 1. Concatenate per-example metrics
        all_per_example_dfs = []
        for file in per_example_files:
            df = pd.read_csv(file)
            all_per_example_dfs.append(df)
            logger.debug(f"  Loaded {file.name}: {len(df)} examples")

        concatenated_per_example = pd.concat(all_per_example_dfs, ignore_index=True)

    # 2. (Optional) Cluster results using Foldseek
    if cluster_with_foldseek:
        # Drop any pre-existing foldseek columns so re-runs don't produce _x/_y duplicates.
        _stale_tokens = ("foldseek_cluster_id", "clustered_pass_rate", "cluster_diversity")
        stale_cols = [c for c in concatenated_per_example.columns if any(tok in c for tok in _stale_tokens)]
        if stale_cols:
            concatenated_per_example = concatenated_per_example.drop(columns=stale_cols)
        # Chains dirs are discovered from output_base_dir in the DataFrame (no hardcoded path).
        concatenated_per_example = add_foldseek_diversity_to_per_example_df(
            concatenated_per_example, threads=foldseek_threads
        )
        concatenated_per_example = add_diversity_summary_metrics(concatenated_per_example)
        if cleanup_chains and "output_base_dir" in concatenated_per_example.columns:
            for base_dir in concatenated_per_example["output_base_dir"].dropna().unique():
                chains_dir = Path(base_dir) / "foldseek_input_chains"
                if chains_dir.is_dir():
                    shutil.rmtree(chains_dir)
                    logger.info(f"Removed foldseek input chains directory: {chains_dir}")

    # Save per-example metrics
    concatenated_per_example = concatenated_per_example.round(4)
    output_path = Path(output_dir) if output_dir is not None else (input.parent if input.is_file() else input)
    output_path.mkdir(parents=True, exist_ok=True)
    per_example_output = output_path / "per_example_metrics.csv"
    concatenated_per_example.to_csv(per_example_output, index=False)
    logger.info(f"Saved concatenated per-example metrics to {per_example_output}")
    logger.info(f"  Total examples: {len(concatenated_per_example)}")

    # 2. Re-aggregate metrics using shared function
    agg_df = aggregate_metrics_from_per_example_df(concatenated_per_example, title="Final Aggregated Metrics")

    # Save aggregated metrics
    agg_df = agg_df.round(4)
    agg_output = output_path / "aggregated_metrics.csv"
    agg_df.to_csv(agg_output, index=False)
    logger.info(f"Saved aggregated metrics to {agg_output}")

    # 3. Aggregate per-task metrics
    per_task_df = aggregate_metrics_from_per_example_df(
        concatenated_per_example, title="Per-Task Metrics", group_by="task"
    )

    per_task_df = per_task_df.round(4)
    per_task_output = output_path / "per_task_metrics.csv"
    per_task_df.to_csv(per_task_output, index=False)
    logger.info(f"Saved per-task metrics to {per_task_output}")

    # 4. Optionally clean up individual job files
    if not input.is_file() and cleanup:
        job_files_dir = output_path / "individual_job_results"
        job_files_dir.mkdir(exist_ok=True)

        logger.info(f"Moving individual job files to {job_files_dir}")

        # Move per-example files
        for file in per_example_files:
            file.rename(job_files_dir / file.name)
            logger.debug(f"  Moved {file.name}")

        # Move aggregated files
        agg_files = sorted(output_path.glob("*aggregated_metrics*.csv"))
        for file in agg_files:
            if file == agg_output:
                continue
            file.rename(job_files_dir / file.name)
            logger.debug(f"  Moved {file.name}")


def main() -> None:
    fire.Fire(concatenate_results)


if __name__ == "__main__":
    main()
