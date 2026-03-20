"""Concatenate inference results across all subdirectories of a sweep.

Walks the sweep base directory, finds per-config inference_metrics/ subdirectories,
and concatenates per_example_metrics.csv, per_task_metrics.csv, and
aggregated_metrics.csv into sweep-level files saved at the sweep root.

A `sweep_config` column is added to each row indicating which sweep config
produced that row (derived from the subdirectory name).

Usage:
    python concatenate_sweep_results.py --sweep-dir /path/to/sweep_root
"""

import argparse
import pandas as pd
from pathlib import Path


_METRICS_FILES = [
    ("per_example_metrics.csv", "sweep_per_example_metrics.csv"),
    ("per_task_metrics.csv", "sweep_per_task_metrics.csv"),
    ("aggregated_metrics.csv", "sweep_aggregated_metrics.csv"),
]


def concatenate_sweep_results(sweep_dir: str) -> None:
    """Concatenate per-config metrics CSVs into sweep-level files.

    Args:
        sweep_dir: Root directory of the sweep. Each immediate subdirectory
            that contains an inference_metrics/ folder is treated as one
            sweep config's output.
    """
    sweep_path = Path(sweep_dir)
    assert sweep_path.is_dir(), f"Sweep directory not found: {sweep_dir}"

    # Collect subdirectories that have inference_metrics/
    config_dirs = sorted(
        d for d in sweep_path.iterdir()
        if d.is_dir() and (d / "inference_metrics").is_dir()
    )

    if not config_dirs:
        print(f"No inference_metrics/ subdirectories found under {sweep_dir}")
        return

    print(f"Found {len(config_dirs)} sweep config(s) to concatenate:")
    for d in config_dirs:
        print(f"  {d.name}")

    for src_filename, dst_filename in _METRICS_FILES:
        dfs = []
        missing = []
        for config_dir in config_dirs:
            src = config_dir / "inference_metrics" / src_filename
            if not src.exists():
                missing.append(config_dir.name)
                continue
            df = pd.read_csv(src)
            df.insert(0, "sweep_config", config_dir.name)
            dfs.append(df)

        if missing:
            print(f"\nWarning: {src_filename} missing in {len(missing)} config(s): {missing}")

        if not dfs:
            print(f"No {src_filename} files found - skipping {dst_filename}")
            continue

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.round(4)

        dst = sweep_path / dst_filename
        combined.to_csv(dst, index=False)
        print(f"\nSaved {dst_filename}: {len(combined)} rows from {len(dfs)} config(s)")
        print(f"  -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Root directory of the sweep (contains per-config subdirectories)",
    )
    args = parser.parse_args()
    concatenate_sweep_results(args.sweep_dir)


if __name__ == "__main__":
    main()
