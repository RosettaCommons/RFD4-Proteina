#!/usr/bin/env python
"""
Pre-populate the PDB cache by parsing all structures in parallel.

This script reads the PDB interfaces parquet file and parses all structures
using multiple CPU workers, saving parsed structures to the cache directory.
This avoids the slow first epoch during training.

Usage:
    # With defaults
    python scripts/populate_pdb_cache.py

    # With custom settings
    python scripts/populate_pdb_cache.py \
        --parquet /path/to/interfaces_df_train.parquet \
        --cache-dir /path/to/.cache \
        --num-workers 64 \
        --batch-size 100

    # With SLURM (recommended for large datasets)
    sbatch scripts/populate_pdb_cache.sbatch
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from atomworks.io import parse  # noqa: E402


def parse_single_structure(args: tuple) -> dict[str, Any]:
    """Parse a single structure and save to cache.

    Args:
        args: Tuple of (example_id, path, cache_dir, ccd_mirror_path, skip_if_cached)

    Returns:
        Dictionary with status information.
    """
    example_id, path, cache_dir, ccd_mirror_path, skip_if_cached = args

    try:
        # Check if already cached (optional optimization)
        if skip_if_cached:
            from pathlib import Path

            from atomworks.io.parser import string_to_md5_hash

            # Replicate cache path construction from parse()
            parse_arguments = {
                "ccd_mirror_path": ccd_mirror_path,
                "add_missing_atoms": True,
                "add_id_and_entity_annotations": True,
                "add_bond_types_from_struct_conn": ["covale"],
                "remove_ccds": None,
                "remove_waters": True,
                "fix_ligands_at_symmetry_centers": True,
                "fix_arginines": True,
                "sanitize_bonds_and_charges": True,
                "fix_bond_types": True,
                "convert_mse_to_met": False,
                "hydrogen_policy": "keep",
            }
            args_string = ",".join(str(parse_arguments[k]) for k in parse_arguments)
            args_hash = string_to_md5_hash(args_string, truncate=8)
            assembly_info = "first"
            cache_file_path = Path(cache_dir) / args_hash / f"{Path(path).stem}_assembly_{assembly_info}.pkl.gz"

            if cache_file_path.exists():
                return {
                    "example_id": example_id,
                    "status": "cached",
                    "path": path,
                    "error": None,
                }

        # Parse with caching enabled
        _result = parse(
            path,
            build_assembly="first",
            load_from_cache=True,
            save_to_cache=True,
            cache_dir=cache_dir,
            ccd_mirror_path=ccd_mirror_path,
        )

        return {
            "example_id": example_id,
            "status": "success",
            "path": path,
            "error": None,
        }
    except Exception as e:
        return {
            "example_id": example_id,
            "status": "failed",
            "path": path,
            "error": str(e),
        }


def populate_cache(
    parquet_path: str,
    cache_dir: str,
    ccd_mirror_path: str,
    num_workers: int | None = None,
    batch_size: int = 100,
    max_structures: int | None = None,
    filters: list | None = None,
    array_task_id: int | None = None,
    array_total_tasks: int | None = None,
    skip_if_cached: bool = True,
) -> None:
    """Populate cache by parsing all structures in parallel.

    Args:
        parquet_path: Path to parquet file with structure metadata.
        cache_dir: Directory for caching parsed structures.
        ccd_mirror_path: Path to CCD mirror for ligand parsing.
        num_workers: Number of parallel workers (default: all CPUs).
        batch_size: Batch size for progress updates.
        max_structures: Maximum number of structures to parse (for testing).
        filters: List of pandas query strings to filter dataframe.
        array_task_id: SLURM array task ID (for distributed processing).
        array_total_tasks: Total number of SLURM array tasks.
    """
    print("=" * 80)
    if array_task_id is not None:
        print(f"PDB Cache Population (Array Task {array_task_id}/{array_total_tasks - 1})")
    else:
        print("PDB Cache Population")
    print("=" * 80)
    print(f"Parquet file: {parquet_path}")
    print(f"Cache directory: {cache_dir}")
    print(f"CCD mirror: {ccd_mirror_path}")

    # Array job configuration
    if array_task_id is not None:
        print("\nArray Job Mode:")
        print(f"  Task ID: {array_task_id}")
        print(f"  Total tasks: {array_total_tasks}")
        print(f"  Processing every {array_total_tasks}th structure starting at index {array_task_id}")
    else:
        print(f"Workers: {num_workers or cpu_count()}")
    print()

    # Create cache directory if needed
    os.makedirs(cache_dir, exist_ok=True)

    # Load parquet file
    print("Loading parquet file...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows")

    # Apply filters
    if filters:
        print("\nApplying filters:")
        for f in filters:
            print(f"  - {f}")
            df = df.query(f)
        print(f"After filtering: {len(df):,} rows")

    # Deduplicate by PDB path (multiple interfaces from same structure)
    print("\nDeduplicating by PDB path...")
    print(f"Rows before deduplication: {len(df):,}")
    unique_paths_df = df.drop_duplicates(subset=["path"], keep="first")
    print(f"Unique PDB structures: {len(unique_paths_df):,}")
    print(f"Reduction: {100 * (1 - len(unique_paths_df) / len(df)):.1f}%")
    df = unique_paths_df

    # Select subset for this array task
    if array_task_id is not None:
        print(f"\nSelecting subset for array task {array_task_id}...")
        # Select every Nth row starting at task_id
        # Task 0: rows 0, N, 2N, ...
        # Task 1: rows 1, N+1, 2N+1, ...
        # etc.
        df = df.iloc[array_task_id::array_total_tasks]
        print(f"This worker will process {len(df):,} unique structures")

    # Limit for testing
    if max_structures:
        df = df.head(max_structures)
        print(f"\nLimited to {max_structures:,} structures for testing")

    print(f"\nTotal unique structures to parse: {len(df):,}")
    if skip_if_cached:
        print("(Will skip already-cached structures)")
    print()

    # Prepare arguments for parsing
    parse_args = [
        (row["example_id"], row["path"], cache_dir, ccd_mirror_path, skip_if_cached) for _, row in df.iterrows()
    ]

    # Parse structures
    start_time = time.time()

    if array_task_id is not None:
        # Array job mode: parse sequentially (no multiprocessing within each worker)
        print(f"Starting sequential parsing (array task {array_task_id})...")
        print()

        results = []
        for args in tqdm(parse_args, desc="Parsing structures", unit="struct"):
            result = parse_single_structure(args)
            results.append(result)
    else:
        # Multiprocessing mode: use all CPUs
        num_workers = num_workers or cpu_count()

        print(f"Starting parallel parsing with {num_workers} workers...")
        print()

        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(parse_single_structure, parse_args, chunksize=batch_size),
                    total=len(parse_args),
                    desc="Parsing structures",
                    unit="struct",
                )
            )

    # Analyze results
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r["status"] == "success")
    cached = sum(1 for r in results if r["status"] == "cached")
    failed = sum(1 for r in results if r["status"] == "failed")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total structures: {len(results):,}")
    print(f"Newly parsed:     {successful:,} ({100 * successful / len(results):.1f}%)")
    print(f"Already cached:   {cached:,} ({100 * cached / len(results):.1f}%)")
    print(f"Failed:           {failed:,} ({100 * failed / len(results) if results else 0:.1f}%)")
    print(f"Elapsed time:     {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Rate:             {len(results) / elapsed:.1f} structures/sec")
    print(f"Cache directory:  {cache_dir}")
    print()

    # Show cache size
    try:
        import subprocess

        result = subprocess.run(
            ["du", "-sh", cache_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            cache_size = result.stdout.split()[0]
            print(f"Cache size:       {cache_size}")
    except Exception:
        pass

    # Save failed examples if any
    if failed > 0:
        failed_examples = [r for r in results if r["status"] == "failed"]
        failed_df = pd.DataFrame(failed_examples)

        output_path = Path(cache_dir) / "failed_parses.csv"
        failed_df.to_csv(output_path, index=False)

        print()
        print(f"⚠️  WARNING: {failed} structures failed to parse")
        print(f"Failed examples saved to: {output_path}")
        print()
        print("Sample errors:")
        for i, r in enumerate(failed_examples[:5]):
            print(f"  {i + 1}. {r['example_id']}: {r['error'][:100]}")
        if failed > 5:
            print(f"  ... and {failed - 5} more")

    print()
    print("✅ Cache population complete!")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-populate PDB cache by parsing structures in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Default paths
    default_parquet = (
        "/path/to/shared/storage"
    )
    default_cache = "/path/to/shared/storage"
    default_ccd = "/path/to/shared/storage"

    parser.add_argument(
        "--parquet",
        default=default_parquet,
        help=f"Path to parquet file (default: {default_parquet})",
    )
    parser.add_argument(
        "--cache-dir",
        default=default_cache,
        help=f"Cache directory (default: {default_cache})",
    )
    parser.add_argument(
        "--ccd-mirror",
        default=default_ccd,
        help=f"CCD mirror path (default: {default_ccd})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: all CPUs = {cpu_count()})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for multiprocessing chunks (default: 100)",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Maximum number of structures to parse (for testing)",
    )
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Disable default filters (parse all structures in parquet)",
    )
    parser.add_argument(
        "--array-task-id",
        type=int,
        default=None,
        help="SLURM array task ID (for distributed processing)",
    )
    parser.add_argument(
        "--array-total-tasks",
        type=int,
        default=None,
        help="Total number of SLURM array tasks",
    )
    parser.add_argument(
        "--no-skip-cached",
        action="store_true",
        help="Don't skip already-cached structures (slower but ensures re-parsing)",
    )

    args = parser.parse_args()

    # Default filters (same as training)
    filters = (
        None
        if args.no_filters
        else [
            "deposition_date < '2021-09-30'",
            "resolution < 9.0",
            "num_polymer_pn_units <= 300",
            "cluster.notnull()",
            "is_inter_molecule",
        ]
    )

    populate_cache(
        parquet_path=args.parquet,
        cache_dir=args.cache_dir,
        ccd_mirror_path=args.ccd_mirror,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        max_structures=args.max_structures,
        filters=filters,
        array_task_id=args.array_task_id,
        array_total_tasks=args.array_total_tasks,
        skip_if_cached=not args.no_skip_cached,
    )


if __name__ == "__main__":
    main()
