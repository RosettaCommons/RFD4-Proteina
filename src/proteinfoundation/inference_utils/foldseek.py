"""Foldseek diversity clustering: subprocess primitive, DataFrame wrappers, DDP entry."""

import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import torch.distributed as dist

from foundry.utils.logging import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


def diversity_foldseek(
    list_of_cif_paths: list[str],
    tm_threshold: float = 0.5,
    alignment_type: int = 1,
    min_seq_id: float = 0.0,
    timeout: float | None = None,
    threads: int | None = None,
) -> list[int] | None:
    """Cluster single-chain structures with Foldseek; return per-structure cluster IDs.

    Caller must guarantee each input file is already a single-chain structure.
    Files are ``shutil.copy``'d into a temporary staging dir and ``easy-cluster``
    is invoked directly. The staging dir is created on tmpfs (``$TMPDIR`` /
    ``/tmp``) and removed on exit.

    Args:
        list_of_cif_paths: paths to single-chain structures to cluster.
        tm_threshold: TM-score threshold for clustering.
        alignment_type: 1 (structure) or 2 (structure+sequence).
        min_seq_id: required iff ``alignment_type == 2``.
        timeout: total wall budget in seconds.
        threads: ``--threads`` for foldseek; defaults to foldseek's own
            default (which itself defaults to all cores).
    """
    assert alignment_type in (1, 2), f"alignment_type {alignment_type} not valid"
    if alignment_type == 1:
        assert min_seq_id == 0.0, f"alignment_type == 1 only admits min_seq_id = 0, got {min_seq_id}"
    if alignment_type == 2:
        assert min_seq_id > 0.0, f"alignment_type == 2 requires min_seq_id > 0, got {min_seq_id}"
    if not list_of_cif_paths:
        return None

    foldseek_exec = os.getenv("FOLDSEEK_EXEC")
    if foldseek_exec is None:
        raise ValueError("FOLDSEEK_EXEC environment variable not set")

    with tempfile.TemporaryDirectory(prefix="foldseek_diversity_") as path_tmp:
        path_designable = os.path.join(path_tmp, "samples")
        os.makedirs(path_designable)

        # staged_stems[i] is the basename Foldseek will emit for input i; defining
        # the convention once here means a future rename only touches one site.
        staged_stems: list[str] = []
        for i, fcif in enumerate(list_of_cif_paths):
            ext = os.path.splitext(fcif)[1] or ".cif"
            stem = f"{i + 1}"
            shutil.copy(fcif, os.path.join(path_designable, f"{stem}{ext}"))
            staged_stems.append(stem)

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
        if threads is not None:
            command += ["--threads", str(threads)]

        try:
            subprocess.run(command, check=True, text=True, capture_output=True, timeout=timeout)
        except Exception as e:
            logger.info(f"Foldseek diversity failed: {e} \n Returning None")
            return None

        res_file = os.path.join(path_tmp, "res_cluster.tsv")
        with open(res_file) as f:
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

    cluster_ids = {rep: idx for idx, rep in enumerate(np.unique(reps))}
    el_to_cluster = {el: cluster_ids[rep] for rep, el in zip(reps, els, strict=True)}

    # Loud failure if Foldseek's emitted stems don't exactly match what we staged
    # (drops a member, adds an unexpected one, or staging convention drifted).
    expected = set(staged_stems)
    got = set(el_to_cluster)
    if expected != got:
        raise RuntimeError(f"Foldseek output mismatch: missing={sorted(expected - got)} extra={sorted(got - expected)}")

    # Reorder by input position so callers can zip this list with list_of_cif_paths.
    return [el_to_cluster[stem] for stem in staged_stems]


def _base_pass_rate_cols(df: pd.DataFrame) -> list[str]:
    """Return pass-rate columns, supporting both legacy ``*_pass_rate`` and modern ``*.pass`` suffixes.

    Excludes derived ``clustered_pass_rate`` / ``unclustered_pass_rate`` variants.
    """
    return [
        c
        for c in df.columns
        if (
            c.endswith(".pass")
            or (
                c.endswith("pass_rate")
                and not c.endswith("clustered_pass_rate")
                and not c.endswith("unclustered_pass_rate")
            )
        )
    ]


def add_foldseek_diversity_to_per_example_df(
    per_example_df: pd.DataFrame,
    group_by: list[str] | None = None,
    timeout: float | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Add Foldseek clustering information to per-example DataFrame output by validation or external eval.

    Clustering is computed only on unique ``designed_chain_path`` values within each group,
    then broadcast back to all rows with matching paths. Groups are processed in parallel
    using threads — safe on GPU/DDP nodes because no Python processes are forked; each thread
    simply spawns an external Foldseek subprocess. Available CPUs are divided evenly across
    parallel processes via ``--threads``.

    Args:
        per_example_df: DataFrame with a ``designed_chain_path`` column pointing to
            single-chain structures (one chain per file).
        group_by: Columns to group by when clustering. Defaults to ``["task"]``.
        timeout: Wall-clock budget in seconds for the entire call. The remaining budget at
            submission time is passed to each per-group Foldseek subprocess. If ``None``,
            no timeout is applied.
        **kwargs: Additional arguments forwarded to :py:func:`diversity_foldseek`.

    Returns:
        DataFrame with additional column: foldseek_cluster_id.
    """
    if group_by is None:
        group_by = ["task"]

    if "designed_chain_path" not in per_example_df.columns:
        raise ValueError("designed_chain_path column not found in per_example_df")
    for col in group_by:
        if col not in per_example_df.columns:
            raise ValueError(f"{col} column not found in per_example_df")

    groups = list(per_example_df.groupby(group_by))
    n_groups = len(groups)

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if "threads" in kwargs and kwargs["threads"] is not None:
        available_cpus = kwargs.pop("threads")
    elif slurm_cpus is not None:
        available_cpus = int(slurm_cpus)
    else:
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 1

    # Run all groups in parallel; divide CPUs evenly so each foldseek subprocess
    # gets threads_per_worker threads rather than competing for all CPUs.
    n_parallel = min(n_groups, available_cpus)
    threads_per_worker = max(1, available_cpus // n_parallel)
    logger.info(
        f"Foldseek: {n_groups} groups, {n_parallel} parallel workers, "
        f"{threads_per_worker} threads/worker ({available_cpus} CPUs total)"
    )

    start_time = time.monotonic()

    def _run_group(key: Any, group_df: pd.DataFrame) -> pd.DataFrame:
        group_vals = key if isinstance(key, tuple) else (key,)
        group_label = "/".join(str(v) for v in group_vals)

        # NaN designed_chain_path indicates an upstream failure; exclude those rows
        # from foldseek but don't let them block the rest of the group.
        nan_mask = group_df["designed_chain_path"].isna()
        if nan_mask.any():
            logger.warning(
                f"Group {group_label} has {nan_mask.sum()} row(s) with missing designed_chain_path "
                "(upstream failure); these will be treated as failed designs."
            )
        valid_df = group_df[~nan_mask]

        if valid_df.empty:
            logger.warning(
                f"designed_chain_path is missing for all rows in group {group_label}."
                " This group will not get cluster ids."
            )
            out = group_df.copy()
            out["foldseek_cluster_id"] = None
            return out

        # Compute remaining budget at submission time; all groups start near-simultaneously
        # so this is a fair approximation of the per-group wall-clock limit.
        remaining = (timeout - (time.monotonic() - start_time)) if timeout is not None else None

        unique_paths = valid_df["designed_chain_path"].unique().tolist()
        cluster_assignments = diversity_foldseek(
            list_of_cif_paths=unique_paths,
            timeout=remaining,
            threads=threads_per_worker,
            **kwargs,
        )
        out = group_df.copy()
        out["foldseek_cluster_id"] = None  # NaN-path rows stay None (failed designs)
        if cluster_assignments is None:
            logger.warning(f"Foldseek clustering failed for group {group_label}. This group will not get cluster IDs")
        else:
            path_to_cluster = dict(zip(unique_paths, cluster_assignments, strict=False))
            out.loc[~nan_mask, "foldseek_cluster_id"] = out.loc[~nan_mask, "designed_chain_path"].map(path_to_cluster)
        return out

    # ThreadPoolExecutor is safe inside DDP rank-0: threads do not fork the Python
    # process, so no CUDA context is duplicated. Each thread spawns an independent
    # Foldseek subprocess that has no knowledge of CUDA.
    processed: list[pd.DataFrame] = [None] * n_groups  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        future_to_idx = {executor.submit(_run_group, key, gdf): i for i, (key, gdf) in enumerate(groups)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                processed[idx] = future.result()
            except Exception as e:
                key, gdf = groups[idx]
                group_vals = key if isinstance(key, tuple) else (key,)
                group_label = "/".join(str(v) for v in group_vals)
                logger.warning(f"Foldseek group {group_label} raised an exception: {e}; skipping cluster ids")
                out = gdf.copy()
                out["foldseek_cluster_id"] = None
                processed[idx] = out

    return pd.concat(processed, ignore_index=True)


def add_diversity_summary_metrics(
    per_example_df: pd.DataFrame,
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """Broadcast cluster metrics (cluster_diversity, clustered_pass_rate, unclustered_pass_rate)
    onto per-example rows.

    Args:
        per_example_df: DataFrame with a ``foldseek_cluster_id`` column (added by
            :py:func:`add_foldseek_diversity_to_per_example_df`).
        group_by: Columns that define the clustering groups. Must match the ``group_by``
            used in :py:func:`add_foldseek_diversity_to_per_example_df`. Defaults to
            ``["task"]``.

    Returns:
        DataFrame with additional columns: unclustered_pass_rate, cluster_diversity,
        clustered_pass_rate (and per-prefix variants).
    """
    if group_by is None:
        group_by = ["task"]

    if "foldseek_cluster_id" not in per_example_df.columns:
        raise ValueError(
            "foldseek_cluster_id column not found in per_example_df. "
            "Run add_foldseek_diversity_to_per_example_df first."
        )

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
    for group_key, group_df in per_example_df.groupby(group_by):
        group_vals = group_key if isinstance(group_key, tuple) else (group_key,)
        group_dict = dict(zip(group_by, group_vals, strict=False))

        n = len(group_df)
        all_clusters_missing = group_df["foldseek_cluster_id"].isna().all()
        row: dict = {
            **group_dict,
            "cluster_diversity": float("nan")
            if all_clusters_missing
            else float(group_df["foldseek_cluster_id"].nunique()) / n,
        }
        cpr_values: dict = {}
        for pr_col in pass_rate_cols:
            suffix = "pass" if pr_col.endswith(".pass") else "pass_rate"
            prefix = pr_col[: -len(suffix)]
            cpr_col = f"{prefix}clustered_pass_rate"
            if all_clusters_missing or group_df[pr_col].isna().all():
                cpr_val: float = float("nan")
            else:
                passing_clusters = group_df.loc[group_df[pr_col] == 1.0, "foldseek_cluster_id"].dropna().nunique()
                cpr_val = float(passing_clusters) / n
            row[cpr_col] = cpr_val
            cpr_values[cpr_col] = cpr_val
        if pass_rate_cols:
            non_nan_cprs = [v for v in cpr_values.values() if pd.notna(v)]
            row["clustered_pass_rate"] = non_nan_cprs[0] if len(non_nan_cprs) == 1 else float("nan")
        cluster_rows.append(row)

    cluster_metrics_df = pd.DataFrame(cluster_rows)
    per_example_df = per_example_df.merge(cluster_metrics_df, on=group_by, how="left")

    return per_example_df


def _partition_groups(df: pd.DataFrame) -> list[dict]:
    """Build the ``[{"key": "ds/task", "paths": [...]}]`` group list from a metrics DataFrame.

    Skips groups with no valid (non-NaN) ``designed_chain_path``. Returns ``[]``
    if required columns are missing or the DataFrame is empty.
    """
    required = {"designed_chain_path", "dataset_name", "task"}
    if df.empty or not required.issubset(df.columns):
        return []
    groups: list[dict] = []
    for key, gdf in df.groupby(["dataset_name", "task"]):
        valid = gdf[gdf["designed_chain_path"].notna()]
        if valid.empty:
            continue
        paths = valid["designed_chain_path"].unique().tolist()
        key_str = "/".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        groups.append({"key": key_str, "paths": paths})
    return groups


def _cluster_groups_locally(groups: list[dict], rank_label: str = "0") -> dict[tuple[str, str], int]:
    """Run ``diversity_foldseek`` on each group serially with all available CPUs."""
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback on non-Linux: SLURM env var → cpu_count → small safe default.
        available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or os.cpu_count() or 4

    results: dict[tuple[str, str], int] = {}
    for g in groups:
        t0 = time.perf_counter()
        try:
            cluster_assignments = diversity_foldseek(
                list_of_cif_paths=g["paths"],
                threads=available_cpus,
            )
            if cluster_assignments is not None:
                for path, cid in zip(g["paths"], cluster_assignments, strict=False):
                    results[(g["key"], path)] = int(cid)
            logger.info(
                f"[foldseek] rank{rank_label} group={g['key']} "
                f"n={len(g['paths'])} threads={available_cpus} "
                f"wall={time.perf_counter() - t0:.2f}s"
            )
        except Exception as e:
            logger.warning(f"[foldseek] rank{rank_label} group {g['key']} failed: {e}")
    return results


def run_foldseek_distributed(
    all_design_metrics: list[dict] | None,
    *,
    world_size: int,
    global_rank: int,
    is_inference: bool,
) -> dict[tuple[str, str], int] | None:
    """Cluster ``(dataset_name, task)`` foldseek groups, optionally across DDP ranks.

    Single-process (``world_size == 1``): builds groups locally and runs them
    serially in this process. Multi-process (``world_size > 1``): rank 0
    partitions groups round-robin, scatters slices to all ranks, each rank
    runs its slice, rank 0 gathers cluster IDs.

    Returns ``{(group_key, path): cluster_id}`` on rank 0 (or the sole rank),
    ``None`` on other ranks. Returns ``None`` (no-op) for inference or when
    ``FOLDSEEK_EXEC`` is unset.
    """
    if os.getenv("FOLDSEEK_EXEC") is None or is_inference:
        return None  # all ranks must take the same exit; collective would deadlock

    if world_size == 1:
        all_groups = _partition_groups(pd.DataFrame(all_design_metrics or []))
        return _cluster_groups_locally(all_groups)

    # Rank 0 builds per-rank work lists from the gathered DataFrame.
    per_rank_groups: list[list[dict]] | None = None
    if global_rank == 0:
        per_rank_groups = [[] for _ in range(world_size)]
        for i, group in enumerate(_partition_groups(pd.DataFrame(all_design_metrics or []))):
            per_rank_groups[i % world_size].append(group)

    output: list = [None]
    dist.scatter_object_list(output, per_rank_groups, src=0)
    my_groups: list[dict] = output[0] or []

    my_results = _cluster_groups_locally(my_groups, rank_label=str(global_rank))

    gathered: list | None = [None] * world_size if global_rank == 0 else None
    dist.gather_object(my_results, gathered, dst=0)
    if global_rank != 0:
        return None
    merged: dict[tuple[str, str], int] = {}
    for d in gathered or []:
        if d:
            merged.update(d)
    return merged


def apply_cluster_map_to_metrics(
    metrics: list[dict],
    cluster_map: dict[tuple[str, str], int] | None,
) -> None:
    """Write ``foldseek_cluster_id`` onto rows whose ``(key, path)`` is in ``cluster_map``.

    Mutates ``metrics`` in place. No-op when ``cluster_map`` is None or empty.
    Key format matches what :py:func:`run_foldseek_distributed` produces:
    ``f"{dataset_name}/{task}"``.
    """
    if not cluster_map:
        return
    for m in metrics:
        ds = str(m.get("dataset_name", ""))
        tk = str(m.get("task", ""))
        key = f"{ds}/{tk}"
        path = m.get("designed_chain_path") or m.get("designed_structure_path")
        cid = cluster_map.get((key, path))
        if cid is not None:
            m["foldseek_cluster_id"] = cid
