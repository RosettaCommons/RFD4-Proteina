#!/usr/bin/env python
"""Score structure quality for a dataframe of CIF file paths.

Reads a parquet or CSV with a ``path`` column pointing to CIF files, parses
each structure with ``add_missing_atoms=False``, runs the full set of
atomworks quality checks, and writes the enriched dataframe to a new file.

Added columns (prefixed ``quality_``):
  quality_passed                         Overall pass/fail
  quality_ramachandran_passed
  quality_ramachandran_kl_divergence
  quality_ramachandran_n_residues
  quality_chain_breaks_passed
  quality_chain_breaks_n_breaks
  quality_intra_residue_clashes_passed
  quality_intra_residue_clashes_n_clashing_residues
  quality_inter_residue_clashes_passed
  quality_inter_residue_clashes_n_clashing_pairs
  quality_global_bond_check_passed
  quality_global_bond_check_message
  quality_global_clash_check_passed
  quality_global_clash_check_n_clashing_pairs
  quality_error                          Exception string if parsing/scoring failed, else None

Usage:
    python score_structures_quality.py \\
        --input  path/to/df.parquet \\
        --output path/to/df_quality.parquet

    # Paths inside a squashfs archive
    python score_structures_quality.py \\
        --input   df.parquet \\
        --output  df_quality.parquet \\
        --squashfs /path/to/archive.sqsh

    # Subset of checks and explicit worker count
    python score_structures_quality.py \\
        --input  df.parquet \\
        --output df_quality.parquet \\
        --checks ramachandran chain_breaks \\
        --workers 16
"""

import argparse
import subprocess
import tempfile
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from atomworks.io import parse
from atomworks.io.tools.quality import CheckName, evaluate_quality


# ---------------------------------------------------------------------------
# Squashfs mount helper
# ---------------------------------------------------------------------------


@contextmanager
def _mount_squashfs(sqsh_path: str):
    """Mount a squashfs file via squashfuse and yield the mount point.

    The mount is torn down automatically on exit. If the script is killed
    before unmounting, run ``fusermount -u <mount_point>`` manually.

    Args:
        sqsh_path: Path to the ``.sqsh`` file.

    Yields:
        str: Path to the temporary mount point directory.
    """
    with tempfile.TemporaryDirectory() as mount_point:
        subprocess.run(["squashfuse", sqsh_path, mount_point], check=True)
        print(f"Mounted {sqsh_path} -> {mount_point}")
        try:
            yield mount_point
        finally:
            subprocess.run(["fusermount", "-u", mount_point], check=False)
            print(f"Unmounted {mount_point}")


def _remap_paths(paths: pd.Series, mount_point: str) -> pd.Series:
    """Replace the common path prefix with ``mount_point``.

    Finds the longest common filesystem prefix shared by all paths in the
    series and substitutes it with the squashfs mount point.

    Args:
        paths: Series of path strings from the ``path`` column.
        mount_point: Absolute path to the squashfs mount point.

    Returns:
        Series with paths remapped through the mount point.
    """
    # os.path.commonpath works on the actual path components (not just chars)
    import os

    common = os.path.commonpath(paths.tolist())
    print(f"  Detected common prefix: {common}")
    print(f"  Remapping to:           {mount_point}")
    return paths.apply(lambda p: os.path.join(mount_point, os.path.relpath(p, common)))


# ---------------------------------------------------------------------------
# Worker (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------


def _score_one(args: tuple) -> dict[str, Any]:
    """Parse one CIF and return a flat dict of quality columns.

    Args:
        args: Tuple of (original_row_index, path_str, checks_or_None).

    Returns:
        Dict with ``_idx`` for re-joining plus all ``quality_*`` columns.
    """
    idx, path_str, checks = args
    base: dict[str, Any] = {"_idx": idx, "quality_error": None}

    try:
        parsed = parse(path_str, config="lightweight")

        assemblies = parsed.get("assemblies", {})
        if "1" in assemblies:
            atom_array = assemblies["1"][0]
        else:
            atom_array = parsed["asym_unit"][0]

        report = evaluate_quality(atom_array, checks or None)

        base["quality_passed"] = report["passed"]
        for check_name, result in report["checks"].items():
            base[f"quality_{check_name}_passed"] = result["passed"]
            for detail_key, detail_val in result["detail"].items():
                base[f"quality_{check_name}_{detail_key}"] = detail_val

    except Exception as exc:
        base["quality_passed"] = None
        base["quality_error"] = str(exc)

    return base


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def score_dataframe(
    input_path: str,
    output_path: str,
    checks: list[CheckName] | None,
    num_workers: int,
    chunksize: int,
    squashfs_path: str | None = None,
) -> None:
    """Run quality scoring on every row of ``input_path`` and save results.

    Args:
        input_path: Path to input parquet or CSV file.
        output_path: Path to write enriched parquet or CSV file.
        checks: Subset of checks to run, or ``None`` for all six.
        num_workers: Number of parallel worker processes.
        chunksize: imap chunksize for multiprocessing.
        squashfs_path: Optional path to a squashfs archive. When provided the
            archive is mounted and the common prefix of all ``path`` values is
            remapped to the mount point before scoring.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    # Load
    print(f"Loading {in_path} ...")
    if in_path.suffix == ".csv":
        df = pd.read_csv(in_path)
    else:
        df = pd.read_parquet(in_path)
    print(f"  {len(df):,} rows")

    if "path" not in df.columns:
        raise ValueError("Input dataframe must have a 'path' column.")

    def _run(df: pd.DataFrame) -> pd.DataFrame:
        worker_args = [(i, str(row["path"]), checks) for i, row in df.iterrows()]

        print(f"Scoring with {num_workers} worker(s) ...")
        with Pool(processes=num_workers) as pool:
            rows: list[dict[str, Any]] = list(
                tqdm(
                    pool.imap_unordered(_score_one, worker_args, chunksize=chunksize),
                    total=len(worker_args),
                    desc="Scoring",
                    unit="struct",
                )
            )

        scores_df = pd.DataFrame(rows).set_index("_idx").sort_index()
        return df.join(scores_df)

    if squashfs_path:
        with _mount_squashfs(squashfs_path) as mount_point:
            df["path"] = _remap_paths(df["path"], mount_point)
            result = _run(df)
    else:
        result = _run(df)

    # Summary
    n_ok = int(result["quality_passed"].sum())
    n_err = int(result["quality_error"].notna().sum())
    n_fail = int((~result["quality_passed"].fillna(False)).sum()) - n_err
    print(f"\n  Passed:  {n_ok:,}")
    print(f"  Failed:  {n_fail:,}")
    print(f"  Errors:  {n_err:,}")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        result.to_csv(out_path, index=False)
    else:
        result.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score CIF structure quality and append results to a dataframe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Input parquet or CSV file with a 'path' column.")
    parser.add_argument("--output", required=True, help="Output parquet or CSV file path.")
    parser.add_argument(
        "--squashfs",
        default=None,
        metavar="SQSH",
        help="Path to a squashfs archive. The common prefix of all paths in the dataframe "
        "will be remapped to the temporary mount point automatically.",
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=[
            "ramachandran",
            "chain_breaks",
            "intra_residue_clashes",
            "inter_residue_clashes",
            "global_bond_check",
            "global_clash_check",
        ],
        default=None,
        help="Subset of checks to run (default: all six).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help=f"Number of parallel worker processes (default: {cpu_count()}).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4,
        help="imap chunksize for multiprocessing (default: 4).",
    )

    args = parser.parse_args()

    score_dataframe(
        input_path=args.input,
        output_path=args.output,
        checks=args.checks,
        num_workers=args.workers,
        chunksize=args.chunksize,
        squashfs_path=args.squashfs,
    )


if __name__ == "__main__":
    main()
