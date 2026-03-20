#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/latent_exec.sh" "$0" "$@"'
"""Concatenate results from multi-job external evaluation runs.

Also provides shared aggregation logic used by external_eval.py.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import rootutils
from foundry.utils.logging import print_df_as_table, RankedLogger

import os
import shutil
import subprocess
from typing import List, Optional

from foundry.utils.env import load_dotenv

from biotite.structure import AtomArrayStack
from atomworks.io.utils.io_utils import load_any, to_pdb_string

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

dotenv_path = os.path.join(os.environ["PROJECT_ROOT"], "projects/proteinfoundation/.env")
load_dotenv(dotenv_path)

logger = RankedLogger(__name__, rank_zero_only=False)


def diversity_foldseek(
    list_of_pdb_paths: List[str],
    tm_threshold: float = 0.5,
    tmp_path: str = "./tmp/metrics/",
    clean_tmp: bool = True,
    chains: Optional[List[str]] = None,
    multimer_tm_threshold: float = 0.65,
    interface_lddt_threshold: float = 0.65,
    alignment_type: int = 1,
    min_seq_id: float = 0.0,
) -> Optional[List[int]]:
    """
    Evaluates diversity by clustering designable samples using foldseek. Diversity is measured as
    the number of clusters over number of samples. Supports both monomers and multimers.

    Args:
        list_of_pdb_paths: List of paths (strings) each one to a pdb file.
        tm_threshold: Threshold used to cluster with tm score (for monomers and chain-level in multimers)
        tmp_path: Path to store temp files during clustering
        clean_tmp: Whether to delete temp files created by the clustering after it is done
        chains: List of chain ids to use for clustering. If None, all chains are used.
        multimer_tm_threshold: TM threshold for multimer clustering (Foldseek multimer mode)
        interface_lddt_threshold: Interface lDDT threshold for multimer clustering
        alignment_type: 1 (structure) or 2 (structure+sequence), passed to Foldseek
        min_seq_id: Minimum sequence identity for clustering (monomer mode, alignment_type=2)

    Returns:
        Optional[List[int]], containing the cluster assignment for each input pdb file.
    """
    assert alignment_type in [
        1,
        2,
    ], f"alignment_type {alignment_type} not valid, should be either 1 or 2"
    if alignment_type == 1:
        assert min_seq_id == 0.0, f"alignment_type == 1 only admits min_seq_id = 0, but {min_seq_id} was given"
    if alignment_type == 2:
        assert min_seq_id > 0.0, f"alignment_type == 2 requires min_seq_id > 0, but {min_seq_id} was given"
    if len(list_of_pdb_paths) == 0:
        return None

    path_tmp = os.path.join(tmp_path, "diversity")
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    path_designable = os.path.join(path_tmp, "samples")
    if os.path.exists(path_designable):
        shutil.rmtree(path_designable)
    os.makedirs(path_designable, exist_ok=False)

    contain_multimers = False
    for i, fpdb in enumerate(list_of_pdb_paths):
        dest_f = os.path.join(path_designable, f"{i+1}.pdb")

        if chains is not None:
            atom_array = load_any(fpdb)
            if isinstance(atom_array, AtomArrayStack):
                assert atom_array.stack_depth() == 1, f"Expected single model in {fpdb}, found multiple"
                atom_array = atom_array[0]
            chain_identifier = "chain_iid" if "chain_iid" in atom_array else "chain_id"
            chain_annot = atom_array.get_annotation(chain_identifier)
            chains_mask = np.isin(chain_annot, chains)
            if not chains_mask.any():
                # Allow tolerance for the chain_of_interest being a chain_iid but the structure only containing chain_id
                modified_chains = [chain.split("_")[0] for chain in chains]
                chains_mask = np.isin(chain_annot, modified_chains)
                if not chains_mask.any():
                    raise ValueError(f"None of the specified chains {chains} found in {fpdb}")
            struct = atom_array[chains_mask]
            with open(dest_f, "w") as f:
                f.write(to_pdb_string(struct))
        else:
            shutil.copy(fpdb, dest_f)

        if not contain_multimers:
            atom_array = load_any(dest_f)
            chain_identifier = "chain_iid" if "chain_iid" in atom_array else "chain_id"
            if len(np.unique(atom_array.get_annotation(chain_identifier))) > 1:
                contain_multimers = True

    foldseek_exec = os.getenv("FOLDSEEK_EXEC")
    if foldseek_exec is None:
        raise ValueError("FOLDSEEK_EXEC environment variable not set")

    if contain_multimers:
        command = [
            foldseek_exec,
            "easy-multimercluster",
            path_designable,
            os.path.join(path_tmp, "res"),
            path_tmp,
            "--alignment-type",
            str(alignment_type),
            "--cov-mode",
            "0",
            "--multimer-tm-threshold",
            f"{multimer_tm_threshold}",
            "--chain-tm-threshold",
            f"{tm_threshold}",
            "--interface-lddt-threshold",
            f"{interface_lddt_threshold}",
        ]
    else:
        command = [
            foldseek_exec,
            "easy-cluster",
            path_designable,
            os.path.join(path_tmp, "res"),
            path_tmp,
            "--alignment-type",
            str(alignment_type),
            "--cov-mode",
            "0",
            "--min-seq-id",
            f"{min_seq_id}",
            "--tmscore-threshold",
            f"{tm_threshold}",
        ]

    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except Exception as e:
        logger.info(f"Foldseek diversity failed: {e} \n Returning None")
        if clean_tmp:
            shutil.rmtree(path_tmp, ignore_errors=True)
        return None

    res_file = os.path.join(path_tmp, "res_cluster.tsv")
    with open(res_file, "r") as f:
        reps = []
        els = []
        for line in f:
            line = line.strip()
            if line:
                words = line.split()
                if len(words) == 2:
                    word1, word2 = words
                    reps.append(word1)
                    els.append(word2)
                else:
                    logger.error(f"Strange line parsing Foldseek clustering results: {line}")
    nels = len(np.unique(els))
    cluster_ids = {rep: idx for idx, rep in enumerate(np.unique(reps))}
    cluster_assignments = [cluster_ids[rep] for rep in reps]

    assert nels == len(list_of_pdb_paths), f"Foldseek clustering clustered {nels} from {len(list_of_pdb_paths)} given"

    if clean_tmp:
        shutil.rmtree(path_tmp, ignore_errors=True)

    return cluster_assignments


def add_foldseek_diversity_to_per_example_df(per_example_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add Foldseek clustering information to per-example DataFrame output by validation or external eval.

    Clustering is computed only on unique designed_structure_path values within each task,
    then broadcast back to all rows with matching paths.

    Args:
        per_example_df: DataFrame with columns including 'designed_structure_path'.
        **kwargs: Additional arguments to pass to `diversity_foldseek`.

    Returns:
        DataFrame with additional column: foldseek_cluster_id.
    """
    # Validate input DataFrame
    if "designed_structure_path" not in per_example_df.columns:
        raise ValueError("designed_structure_path column not found in per_example_df")
    if "task" not in per_example_df.columns:
        raise ValueError("task column not found in per_example_df")
    if "chain_of_interest" not in per_example_df.columns:
        raise ValueError("chain_of_interest column not found in per_example_df")

    output_task_dfs = []
    for task_name, task_df in per_example_df.groupby("task"):
        # Get chain_of_interest (should be unique per task)
        chains_of_interest = task_df["chain_of_interest"].unique()
        if len(chains_of_interest) > 1:
            logger.warning(
                f"Expected only one chain_of_interest for task {task_name}, found {chains_of_interest}. "
                "This task will not get cluster ids."
            )
            task_df = task_df.copy()
            task_df["foldseek_cluster_id"] = None
            output_task_dfs.append(task_df)
            continue
        chain_of_interest = chains_of_interest[0]
        if "chains" in kwargs:
            if kwargs["chains"] != [chain_of_interest]:
                logger.warning(
                    f"chains argument {kwargs['chains']} does not match chain_of_interest {chain_of_interest} for task {task_name}. "
                    "chains argument will take precedence."
                )
                chain_of_interest = kwargs["chains"][0]
            kwargs.pop("chains")
        if pd.isna(chain_of_interest):
            logger.warning(f"chain_of_interest is missing for task {task_name}. This task will not get cluster ids")
            task_df = task_df.copy()
            task_df["foldseek_cluster_id"] = None
            output_task_dfs.append(task_df)
            continue

        # Get unique paths and cluster only those
        unique_paths = task_df["designed_structure_path"].unique().tolist()
        cluster_assignments = diversity_foldseek(
            list_of_pdb_paths=unique_paths,
            chains=[chain_of_interest],
            **kwargs,
        )
        if cluster_assignments is None:
            logger.warning(f"Foldseek clustering failed for task {task_name}. This task will not get cluster IDs")
            task_df = task_df.copy()
            task_df["foldseek_cluster_id"] = None
            output_task_dfs.append(task_df)
            continue

        # Create mapping from path to cluster ID and broadcast to full dataframe
        path_to_cluster = dict(zip(unique_paths, cluster_assignments))
        task_df = task_df.copy()
        task_df["foldseek_cluster_id"] = task_df["designed_structure_path"].map(path_to_cluster)
        output_task_dfs.append(task_df)

    return pd.concat(output_task_dfs, ignore_index=True)

def _base_pass_rate_cols(df: pd.DataFrame) -> list[str]:
    """Return pass_rate columns, excluding derived clustered/unclustered variants."""
    return [
        c for c in df.columns
        if c.endswith("pass_rate")
        and not c.endswith("clustered_pass_rate")
        and not c.endswith("unclustered_pass_rate")
    ]


def add_diversity_summary_metrics(per_example_df: pd.DataFrame) -> pd.DataFrame:
    """Broadcast cluster metrics (cluster_diversity, clustered_pass_rate, unclustered_pass_rate) onto per-example rows."""

    if "foldseek_cluster_id" not in per_example_df.columns:
        raise ValueError("foldseek_cluster_id column not found in per_example_df. Run add_foldseek_diversity_to_per_example_df first.")

    pass_rate_cols = _base_pass_rate_cols(per_example_df)

    # Add unclustered_pass_rate as a per-row coalesced value: the single non-NaN base pass_rate
    # value for that row, or NaN if zero or multiple base pass_rate columns are non-NaN.
    per_example_df = per_example_df.copy()
    if pass_rate_cols:
        pr_df = per_example_df[pass_rate_cols]
        non_nan_counts = pr_df.notna().sum(axis=1)
        coalesced = pr_df.bfill(axis=1).iloc[:, 0]
        per_example_df["unclustered_pass_rate"] = coalesced.where(non_nan_counts == 1)
    else:
        per_example_df["unclustered_pass_rate"] = float("nan")

    cluster_rows = []
    for task_name, task_df in per_example_df.groupby("task"):
        n = len(task_df)
        row: dict = {
            "task": task_name,
            "cluster_diversity": float(task_df["foldseek_cluster_id"].nunique()) / n,
        }
        cpr_values: dict = {}
        for pr_col in pass_rate_cols:
            prefix = pr_col[: -len("pass_rate")]
            cpr_col = f"{prefix}clustered_pass_rate"
            if task_df[pr_col].isna().all():
                cpr_val: float = float("nan")
            else:
                passing_clusters = (
                    task_df.loc[task_df[pr_col] == 1.0, "foldseek_cluster_id"].dropna().nunique()
                )
                cpr_val = float(passing_clusters) / n
            row[cpr_col] = cpr_val
            cpr_values[cpr_col] = cpr_val
        if pass_rate_cols:
            non_nan_cprs = [v for v in cpr_values.values() if pd.notna(v)]
            row["clustered_pass_rate"] = non_nan_cprs[0] if len(non_nan_cprs) == 1 else float("nan")
        cluster_rows.append(row)

    cluster_metrics_df = pd.DataFrame(cluster_rows)
    per_example_df = per_example_df.merge(cluster_metrics_df, on="task", how="left")

    return per_example_df

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
        metric_cols = [
            c
            for c in group_df.select_dtypes(include="number").columns
            if c not in non_metric_cols
        ]

        row: dict = {group_by: group_value, "n": len(group_df)}

        # Carry over identifier columns (non-metric, non-group_by) when they
        # have a unique value within the group (e.g. dataset_name per task).
        identifier_cols = non_metric_cols - {group_by, "example_id", "designed_structure_path",
                                             "sequence_redesigned_structure_path", "foldseek_cluster_id"}
        for col in identifier_cols:
            if col in group_df.columns:
                unique_vals = group_df[col].dropna().unique()
                if len(unique_vals) == 1:
                    row[col] = unique_vals[0]

        for metric_name in metric_cols:
            values = group_df[metric_name].dropna()
            if len(values) > 0:
                row[metric_name] = float(np.nanmean(values))

        agg_rows.append(row)

    if not agg_rows:
        raise ValueError("No valid metrics were found after aggregation.")

    agg_df = pd.DataFrame(agg_rows)
    print_df_as_table(agg_df, title=title)

    return agg_df


def concatenate_results(output_dir: str, cluster_with_foldseek: bool = False, cleanup: bool = True):
    """Concatenate per-example and aggregated metrics from multiple jobs.

    Args:
        output_dir: Directory containing the individual job result files.
        cluster_with_foldseek: Whether to cluster results using Foldseek.
        cleanup: If True, move individual job files to a subdirectory after concatenation.
    """
    output_path = Path(output_dir)

    # Find all per-example metric files
    per_example_files = sorted(output_path.glob("*per_example_metrics*.csv"))
    per_example_files = [f for f in per_example_files if f.name != "per_example_metrics.csv"]

    if not per_example_files:
        print(f"No per-example metric files found in {output_dir}")
        print("Nothing to concatenate - this may be a single-job run.")
        return

    print(f"Found {len(per_example_files)} per-example metric files to concatenate")

    # 1. Concatenate per-example metrics
    all_per_example_dfs = []
    for file in per_example_files:
        df = pd.read_csv(file)
        all_per_example_dfs.append(df)
        print(f"  Loaded {file.name}: {len(df)} examples")

    concatenated_per_example = pd.concat(all_per_example_dfs, ignore_index=True)

    # 2. (Optional) Cluster results using Foldseek
    if cluster_with_foldseek:
        concatenated_per_example = add_foldseek_diversity_to_per_example_df(concatenated_per_example)
        concatenated_per_example = add_diversity_summary_metrics(concatenated_per_example)


    # Save concatenated per-example metrics
    concatenated_per_example = concatenated_per_example.round(4)
    per_example_output = output_path / "per_example_metrics.csv"
    concatenated_per_example.to_csv(per_example_output, index=False)
    print(f"\nSaved concatenated per-example metrics to {per_example_output}")
    print(f"  Total examples: {len(concatenated_per_example)}")

    # 2. Re-aggregate metrics using shared function
    agg_df = aggregate_metrics_from_per_example_df(concatenated_per_example, title="Final Aggregated Metrics")

    # Save aggregated metrics
    agg_df = agg_df.round(4)
    agg_output = output_path / "aggregated_metrics.csv"
    agg_df.to_csv(agg_output, index=False)
    print(f"\nSaved aggregated metrics to {agg_output}")

    # 3. Aggregate per-task metrics
    per_task_df = aggregate_metrics_from_per_example_df(
        concatenated_per_example, title="Per-Task Metrics", group_by="task"
    )

    per_task_df = per_task_df.round(4)
    per_task_output = output_path / "per_task_metrics.csv"
    per_task_df.to_csv(per_task_output, index=False)
    print(f"\nSaved per-task metrics to {per_task_output}")

    # 4. Clean up individual job files
    if cleanup:
        job_files_dir = output_path / "individual_job_results"
        job_files_dir.mkdir(exist_ok=True)

        print(f"\nMoving individual job files to {job_files_dir}")

        # Move per-example files
        for file in per_example_files:
            file.rename(job_files_dir / file.name)
            print(f"  Moved {file.name}")

        # Move aggregated files
        agg_files = sorted(output_path.glob("*aggregated_metrics*.csv"))
        for file in agg_files:
            if file == agg_output:
                continue
            file.rename(job_files_dir / file.name)
            print(f"  Moved {file.name}")


def main():
    parser = argparse.ArgumentParser(description="Concatenate results from multi-job external evaluation runs.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing the individual job result files",
    )
    parser.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move individual job files into a subdirectory after concatenation",
    )
    parser.add_argument(
        "--cluster-with-foldseek",
        action="store_true",
        help="Whether to cluster results using Foldseek",
    )

    args = parser.parse_args()

    concatenate_results(
        output_dir=args.output_dir,
        cleanup=args.cleanup,
        cluster_with_foldseek=args.cluster_with_foldseek,
    )


if __name__ == "__main__":
    main()
