#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../.ipd/shebang/latent_exec.sh" "$0" "$@"'
"""Concatenate and aggregate per-example metrics from multi-job evaluation runs."""

import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import rootutils

from rfproteina.utils.env import load_dotenv
from rfproteina.utils.logging import RankedLogger, print_df_as_table

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

dotenv_path = os.path.join(os.environ["PROJECT_ROOT"], ".env")
load_dotenv(dotenv_path)

from atomworks.io.parser import parse  # noqa: E402  (load_dotenv must run first)
from atomworks.io.utils.selection import get_residue_starts  # noqa: E402

from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS  # noqa: E402  (load_dotenv must run first)
from rfproteina.inference_utils.foldseek import (  # noqa: E402  (load_dotenv must run first)
    add_diversity_summary_metrics,
    add_foldseek_diversity_to_per_example_df,
)
from rfproteina.inference_utils.solve_rate_metrics import (  # noqa: E402  (load_dotenv must run first)
    compute_solve_rate_metrics,
)
from rfproteina.metrics.sequence_adherence_metric import (  # noqa: E402  (load_dotenv must run first)
    sequence_probability_values_masks_and_outcomes,
)
from rfproteina.utils.parse_presets import (  # noqa: E402  (load_dotenv must run first)
    ANNOTATED_CIF,
)

logger = RankedLogger(__name__, rank_zero_only=False)


def _parse_sequence_probability_observations(
    task: tuple[str, str, str],
) -> tuple[list[tuple[str, str, str, int, str, str, float, int]], str | None]:
    """Read soft conditioning and outcomes keyed by task, output position, and identity."""
    dataset_name, task_name, designed_path_raw = task
    designed_path = Path(designed_path_raw)
    if not designed_path.is_file():
        return [], f"{designed_path}: file not found"

    try:
        predicted = parse(designed_path, config=ANNOTATED_CIF)["assemblies"]["1"][0]
        values, masks, outcomes = sequence_probability_values_masks_and_outcomes(predicted)
        residues = predicted[get_residue_starts(predicted)]
        positions = list(zip(residues.chain_id, residues.res_id, residues.ins_code, strict=True))
        if len(set(positions)) != len(positions):
            raise ValueError("Output chain ID, residue ID, and insertion code must uniquely identify each residue.")
    except Exception as error:
        return [], f"{designed_path}: {error}"

    soft_residue_indices, soft_token_indices = np.where(masks & (values > 0.0) & (values < 1.0))
    token_names = tuple(condition.token_name for condition in SEQUENCE_PROBABILITY_CONDITIONS)
    return [
        (
            dataset_name,
            task_name,
            str(residues.chain_id[residue_index]),
            int(residues.res_id[residue_index]),
            str(residues.ins_code[residue_index]),
            token_names[token_index],
            float(values[residue_index, token_index]),
            int(outcomes[residue_index, token_index]),
        )
        for residue_index, token_index in zip(soft_residue_indices, soft_token_indices, strict=True)
    ], None


def aggregate_sequence_probability_errors(
    per_example_df: pd.DataFrame,
    workers: int | None = None,
) -> pd.DataFrame:
    """Pool soft probability observations across designs before computing per-task errors.

    Returns:
        One row per dataset/task with n (number of pairs), absolute_mass_error
        (abs(sum(q - p))), mean_absolute_error, and median_absolute_error.
        Empty when no soft probability conditions were evaluated. Missing or
        unreadable structures are skipped with a warning.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be positive, got {workers}.")
    if "designed_structure_path" not in per_example_df:
        return pd.DataFrame()

    tasks = []
    for _, row in per_example_df.iterrows():
        path = row["designed_structure_path"]
        if not isinstance(path, str) or not path:
            continue
        task_name = row.get("task")
        if not isinstance(task_name, str) or not task_name:
            raise ValueError(f"Sequence-probability aggregation requires a nonempty task label for {path}.")
        dataset_name = row.get("dataset_name", "unknown")
        tasks.append((str(dataset_name) if pd.notna(dataset_name) else "unknown", task_name, path))
    if not tasks:
        return pd.DataFrame()

    if workers is None:
        allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        if not allocated_cpus:
            try:
                allocated_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                allocated_cpus = os.cpu_count() or 1
        workers = min(8, allocated_cpus)
    workers = min(workers, len(tasks))
    if workers == 1:
        parsed = map(_parse_sequence_probability_observations, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        chunksize = max(1, len(tasks) // (workers * 4))
        parsed = executor.map(_parse_sequence_probability_observations, tasks, chunksize=chunksize)

    observations: list[tuple[str, str, str, int, str, str, float, int]] = []
    errors: list[str] = []
    try:
        for records, error in parsed:
            observations.extend(records)
            if error is not None:
                errors.append(error)
    finally:
        if workers > 1:
            executor.shutdown()
    if errors:
        logger.warning(
            f"Failed to reconstruct sequence-probability observations for {len(errors)} output structure(s). "
            f"First failure: {errors[0]}"
        )

    if not observations:
        return pd.DataFrame()

    observed = pd.DataFrame(
        observations,
        columns=["dataset_name", "task", "chain_id", "res_id", "ins_code", "token", "probability", "outcome"],
    )
    pairs = observed.groupby(["dataset_name", "task", "chain_id", "res_id", "ins_code", "token"], sort=True)[
        ["probability", "outcome"]
    ].mean()

    summary_rows: list[dict[str, object]] = []
    for (dataset_name, task_name), group in pairs.groupby(level=["dataset_name", "task"], sort=True):
        errors = group["outcome"] - group["probability"]
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "task": task_name,
                "n": len(group),
                "absolute_mass_error": float(abs(errors.sum())),
                "mean_absolute_error": float(errors.abs().mean()),
                "median_absolute_error": float(errors.abs().median()),
            }
        )

    return pd.DataFrame(summary_rows)


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
        per_example_df["dataset_name"] = "unknown"

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
    sequence_probability_workers: int | None = None,
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
        sequence_probability_workers: Number of processes used to reconstruct probability
            observations from output CIFs. Defaults to up to eight CPUs from the SLURM allocation
            or process affinity; set explicitly to use more.
    """
    input = Path(input)

    if input.is_file():
        concatenated_per_example = pd.read_csv(input)
    else:
        # Find all per-example metric files including from previous concatenations.
        job_results_dir = input / "individual_job_results"
        per_example_files = list(input.glob("*per_example_metrics*.csv"))
        if job_results_dir.is_dir():
            per_example_files += list(job_results_dir.glob("*per_example_metrics*.csv"))
        per_example_files = [f for f in per_example_files if f.name != "per_example_metrics.csv"]
        # Order oldest→newest so that, when deduplicating by example_id below, the most recent
        # metrics for each design win (a rerun's fresh rows override stale ones).
        per_example_files = sorted(per_example_files, key=lambda f: f.stat().st_mtime)

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

        # The same design can appear in several files (e.g. an orphaned job's later eval-only
        # rerun, or an overlapping requeue). Keeping the most recent non-null values for each example_id.
        if "example_id" in concatenated_per_example.columns:
            n_before = len(concatenated_per_example)
            orig_cols = concatenated_per_example.columns
            concatenated_per_example = (
                concatenated_per_example.groupby("example_id", sort=False).last().reset_index()[orig_cols]
            )
            n_dropped = n_before - len(concatenated_per_example)
            if n_dropped:
                logger.info(f"Deduplicated {n_dropped} duplicate rows by example_id (merged, newest non-null wins).")

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

    probability_errors = aggregate_sequence_probability_errors(
        concatenated_per_example,
        workers=sequence_probability_workers,
    )
    if not probability_errors.empty:
        probability_errors_output = output_path / "sequence_probability_errors.csv"
        probability_errors.round(6).to_csv(probability_errors_output, index=False)
        logger.info(f"Saved sequence-probability errors to {probability_errors_output}")

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

        # Move per-example files (skip any already living in job_files_dir from a prior concat)
        for file in per_example_files:
            if file.parent == job_files_dir:
                continue
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
