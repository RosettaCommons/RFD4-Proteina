"""
Simplified analysis script for protein generation results.

Main functions:
1. Concatenate all CSVs from a results directory
2. Filter successes based on the `pass_rate` column (from RF3Metric)
3. Copy successful structures to a success directory
4. Cluster successes via Foldseek
5. Generate summary statistics

Usage:
    python analysis_simple.py --results_dir ./results --success_dir ./successes --output ./summary.csv
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from loguru import logger

from foundry.utils.env import load_dotenv

# Try to import foldseek diversity function
try:
    from proteinfoundation.metrics.diversity import diversity_foldseek

    FOLDSEEK_AVAILABLE = True
except ImportError:
    FOLDSEEK_AVAILABLE = False
    logger.warning("Foldseek diversity function not available")


def concatenate_csvs(results_dir: Path, pattern: str = "*.csv") -> pd.DataFrame:
    """Concatenate all CSV files in results directory.

    Args:
        results_dir: Directory containing CSV result files
        pattern: Glob pattern for CSV files

    Returns:
        Concatenated DataFrame
    """
    csv_files = list(results_dir.glob(f"**/{pattern}"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {results_dir}")

    logger.info(f"Found {len(csv_files)} CSV files")

    dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["_source_file"] = str(csv_file)
            dataframes.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {csv_file}: {e}")

    if not dataframes:
        raise ValueError("No valid CSV files could be read")

    combined = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined DataFrame has {len(combined)} rows")

    return combined


def filter_successes(df: pd.DataFrame, pass_column: str = "pass_rate") -> pd.DataFrame:
    """Filter DataFrame to only successful designs.

    The pass_rate column is a boolean from RF3Metric.is_passing_design().

    Args:
        df: Input DataFrame
        pass_column: Column name containing pass/fail boolean

    Returns:
        DataFrame with only successful designs
    """
    if pass_column not in df.columns:
        # Try alternative column names
        alternatives = ["rf3_success", "is_success", "success", "pass"]
        for alt in alternatives:
            if alt in df.columns:
                pass_column = alt
                logger.info(f"Using alternative success column: {pass_column}")
                break
        else:
            available_cols = [c for c in df.columns if "pass" in c.lower() or "success" in c.lower()]
            raise ValueError(f"Column '{pass_column}' not found. Available pass/success columns: {available_cols}")

    # Handle different types of pass values
    if df[pass_column].dtype == bool:
        successes = df[df[pass_column] is True]
    elif df[pass_column].dtype in [int, float]:
        successes = df[df[pass_column] == 1]
    else:
        # String comparison
        successes = df[df[pass_column].astype(str).str.lower().isin(["true", "1", "yes", "pass"])]

    logger.info(f"Found {len(successes)} successes out of {len(df)} total ({100 * len(successes) / len(df):.1f}%)")

    return successes


def copy_successes(df: pd.DataFrame, success_dir: Path, path_column: str = "pdb_path") -> list[str]:
    """Copy successful structure files to success directory.

    Args:
        df: DataFrame containing successful designs
        success_dir: Directory to copy successful structures to
        path_column: Column containing file paths

    Returns:
        List of paths to copied files
    """
    success_dir.mkdir(parents=True, exist_ok=True)

    if path_column not in df.columns:
        alternatives = ["file_path", "structure_path", "cif_path"]
        for alt in alternatives:
            if alt in df.columns:
                path_column = alt
                break
        else:
            logger.warning(f"No path column found. Tried: {path_column}, {alternatives}")
            return []

    copied_files = []
    for _, row in df.iterrows():
        src = Path(row[path_column])
        if src.exists():
            dst = success_dir / src.name
            # Add suffix if file exists
            if dst.exists():
                dst = success_dir / f"{src.stem}_{hash(str(src)) % 10000}{src.suffix}"
            shutil.copy(src, dst)
            copied_files.append(str(dst))
        else:
            logger.warning(f"Source file not found: {src}")

    logger.info(f"Copied {len(copied_files)} successful structures to {success_dir}")
    return copied_files


def run_foldseek_clustering(
    pdb_paths: list[str],
    tmp_dir: Path | None = None,
    min_seq_id: float = 0.0,
    alignment_type: int = 1,
) -> dict:
    """Run Foldseek clustering on structure files.

    Args:
        pdb_paths: List of paths to PDB/CIF files
        tmp_dir: Temporary directory for Foldseek
        min_seq_id: Minimum sequence identity threshold
        alignment_type: 1=structure only, 2=structure+sequence

    Returns:
        Dictionary with clustering results
    """
    if not FOLDSEEK_AVAILABLE:
        logger.warning("Foldseek not available, skipping clustering")
        return {"n_clusters": None, "cluster_sizes": []}

    if not pdb_paths:
        return {"n_clusters": 0, "cluster_sizes": []}

    if tmp_dir is None:
        tmp_dir = Path("./tmp_foldseek")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        n_clusters = diversity_foldseek(
            list_of_pdb_paths=pdb_paths,
            tmp_path=str(tmp_dir),
            min_seq_id=min_seq_id,
            alignment_type=alignment_type,
        )
        logger.info(f"Foldseek found {n_clusters} unique clusters")
        return {"n_clusters": n_clusters, "cluster_sizes": []}
    except Exception as e:
        logger.error(f"Foldseek clustering failed: {e}")
        return {"n_clusters": None, "cluster_sizes": []}


def compute_summary(
    df: pd.DataFrame,
    successes: pd.DataFrame,
    groupby_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute summary statistics per parameter combination.

    Args:
        df: Full DataFrame with all results
        successes: DataFrame with only successful designs
        groupby_cols: Columns to group by for summary

    Returns:
        DataFrame with summary statistics
    """
    if groupby_cols is None:
        # Default: group by config name or similar
        potential_cols = ["config_name", "ckpt_name", "experiment"]
        groupby_cols = [c for c in potential_cols if c in df.columns]
        if not groupby_cols:
            # No grouping - compute overall stats
            return pd.DataFrame(
                [
                    {
                        "n_total": len(df),
                        "n_success": len(successes),
                        "success_rate": len(successes) / len(df) if len(df) > 0 else 0,
                    }
                ]
            )

    # Group by specified columns
    summary_rows = []

    for name, group in df.groupby(groupby_cols, dropna=False):
        if isinstance(name, tuple):
            row = dict(zip(groupby_cols, name, strict=False))
        else:
            row = {groupby_cols[0]: name}

        # Filter successes for this group
        if len(groupby_cols) == 1:
            group_successes = successes[successes[groupby_cols[0]] == name]
        else:
            mask = pd.Series(True, index=successes.index)
            for col, val in zip(groupby_cols, name, strict=False):
                mask = mask & (successes[col] == val)
            group_successes = successes[mask]

        row["n_total"] = len(group)
        row["n_success"] = len(group_successes)
        row["success_rate"] = len(group_successes) / len(group) if len(group) > 0 else 0

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified analysis for protein generation results")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing result CSV files")
    parser.add_argument(
        "--success_dir", type=str, default="./successes", help="Directory to copy successful structures"
    )
    parser.add_argument("--output", type=str, default="./summary.csv", help="Output summary CSV file")
    parser.add_argument(
        "--pass_column",
        type=str,
        default="pass_rate",
        help="Column name for success criteria (default: pass_rate from RF3Metric)",
    )
    parser.add_argument("--groupby", nargs="+", default=None, help="Columns to group by for summary statistics")
    parser.add_argument("--no_foldseek", action="store_true", help="Skip Foldseek clustering")
    parser.add_argument("--no_copy", action="store_true", help="Skip copying successful structures")
    args = parser.parse_args()

    load_dotenv()

    results_dir = Path(args.results_dir)
    success_dir = Path(args.success_dir)
    output_path = Path(args.output)

    # 1. Concatenate all CSVs
    logger.info(f"Step 1: Loading CSVs from {results_dir}")
    df = concatenate_csvs(results_dir)

    # 2. Filter to successes
    logger.info(f"Step 2: Filtering successes by '{args.pass_column}' column")
    successes = filter_successes(df, pass_column=args.pass_column)

    # 3. Copy successful structures
    if not args.no_copy and "pdb_path" in successes.columns:
        logger.info(f"Step 3: Copying successful structures to {success_dir}")
        copied_paths = copy_successes(successes, success_dir)
    else:
        copied_paths = []
        logger.info("Step 3: Skipping structure copy")

    # 4. Cluster with Foldseek
    if not args.no_foldseek and copied_paths:
        logger.info("Step 4: Running Foldseek clustering")
        cluster_results = run_foldseek_clustering(copied_paths)
        n_clusters = cluster_results.get("n_clusters", None)
    else:
        n_clusters = None
        logger.info("Step 4: Skipping Foldseek clustering")

    # 5. Compute summary
    logger.info("Step 5: Computing summary statistics")
    summary = compute_summary(df, successes, groupby_cols=args.groupby)

    # Add clustering info
    if n_clusters is not None:
        summary["n_unique_clusters"] = n_clusters

    # 6. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    logger.info(f"Summary saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total samples:     {len(df)}")
    print(f"Successful:        {len(successes)}")
    print(f"Success rate:      {100 * len(successes) / len(df):.1f}%")
    if n_clusters is not None:
        print(f"Unique clusters:   {n_clusters}")
    print("=" * 60 + "\n")

    return summary


if __name__ == "__main__":
    main()
