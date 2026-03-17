#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../.ipd/shebang/latent_exec.sh" "$0" "$@"'
"""Evaluate generated protein structures using configurable metrics.

This script loads structures from a directory and evaluates them using
metrics defined in Hydra configs. Results are saved to a CSV file.

Usage:
    # Use default design metrics
    ./proteinfoundation/evaluate.py input_dir=./inference/my_run

    # Use all metrics (design + refolding)
    ./proteinfoundation/evaluate.py metrics=all_metrics input_dir=./inference/my_run

    # Use refolding metrics only
    ./proteinfoundation/evaluate.py metrics=refolding_metrics input_dir=./inference/my_run

    # Custom output path
    ./proteinfoundation/evaluate.py input_dir=./inference/my_run output_csv=./results.csv

    # Parallel evaluation (job 0 of 4)
    ./proteinfoundation/evaluate.py input_dir=./inference/my_run job_id=0 njobs=4
"""

import os
from pathlib import Path

import hydra
import pandas as pd
import rootutils
from atomworks.io.parser import parse
from foundry.utils.env import load_dotenv
from foundry.metrics.metric import MetricManager
from foundry.utils.ddp import RankedLogger
from foundry.utils.logging import print_config_tree
from omegaconf import DictConfig
from tqdm import tqdm

from proteinfoundation.conditions import *

# Setup root dir and environment variables
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logger = RankedLogger(__name__, rank_zero_only=True)

# Config path
_config_path = os.path.join(os.environ["PROJECT_ROOT"], "projects/proteinfoundation/configs")


def discover_structure_files(
    input_dir: str | Path,
    file_pattern: str = "*.cif",
    job_id: int = 0,
    njobs: int = 1,
) -> list[Path]:
    """Discover structure files in directory and subdirectories.

    Args:
        input_dir: Directory containing structure files.
        file_pattern: Glob pattern for structure files. Defaults to ``*.cif``.
        job_id: Job ID for parallel processing (0-indexed). Defaults to ``0``.
        njobs: Total number of parallel jobs. Defaults to ``1``.

    Returns:
        List of paths to structure files for this job.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Recursively find all structure files
    # Sort for deterministic ordering (rglob order depends on filesystem)
    all_files = sorted(input_dir.rglob(file_pattern))

    if len(all_files) == 0:
        raise ValueError(f"No structure files found matching pattern '{file_pattern}' in {input_dir}")

    logger.info(f"Found {len(all_files)} structure files in {input_dir}")

    # Split files across jobs if parallel processing
    if njobs > 1:
        files_per_job = (len(all_files) - 1) // njobs + 1
        start_idx = job_id * files_per_job
        end_idx = min(start_idx + files_per_job, len(all_files))
        job_files = all_files[start_idx:end_idx]
        logger.info(f"Job {job_id}/{njobs}: Processing {len(job_files)} files " f"(indices {start_idx} to {end_idx})")
        return job_files
    else:
        return all_files


def evaluate_structure(
    file_path: Path,
    metric_manager: MetricManager,
    cif_parser_args: dict,
) -> dict:
    """Evaluate a single structure file.

    Args:
        file_path: Path to structure file.
        metric_manager: Instantiated MetricManager with metrics to compute.
        cif_parser_args: Arguments to pass to CIF parser.

    Returns:
        Dictionary of metric results with file path and example ID.
    """
    # Load structure using atomworks
    result_dict = parse(
        file_path,
        cif_parser_args=cif_parser_args,
    )
    atom_array = result_dict["atom_array"]
    example_id = file_path.stem

    # Prepare batch dict for metrics
    # Most metrics expect 'predicted_atom_array' as a list/stack
    batch = {
        "predicted_atom_array": [atom_array],
        "extra_info": {
            "example_id": example_id,
            "metrics_tags": [],  # No tags filtering for evaluation
        },
    }

    # Compute metrics
    results = metric_manager(**batch)

    # Add file path to results
    results["file_path"] = str(file_path)
    results["example_id"] = example_id

    return results


def aggregate_and_save_results(
    results: list[dict],
    output_csv: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Aggregate results into DataFrame and save to CSV.

    Args:
        results: List of metric result dictionaries.
        output_csv: Path to output CSV file.
        verbose: Log summary statistics if ``True``. Defaults to ``True``.

    Returns:
        DataFrame with all results.
    """
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"✓ Saved evaluation results to {output_csv}")

    # Log summary statistics
    if verbose and len(df) > 0:
        logger.info(f"Evaluated {len(df)} structures")

        # Identify numeric columns (metrics)
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            logger.info("\nSummary Statistics:")
            summary = df[numeric_cols].describe().loc[["mean", "std", "min", "max"]]
            logger.info(f"\n{summary}")

    return df


@hydra.main(version_base="1.3", config_path=_config_path, config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """Main evaluation function with Hydra config.

    Args:
        cfg: Hydra configuration.
    """
    load_dotenv()

    # Get config name from Hydra runtime
    print_config_tree(cfg)

    # Discover structure files
    structure_files = discover_structure_files(
        input_dir=cfg.input_dir,
        file_pattern=cfg.file_pattern,
        job_id=cfg.job_id,
        njobs=cfg.njobs,
    )

    # Instantiate metrics from config
    logger.info("Instantiating metrics...")
    metric_manager = MetricManager.instantiate_from_hydra(cfg.metrics)

    # Remove tag requirements since evaluation structures won't have metrics_tags
    for metric in metric_manager.metrics.values():
        metric.required_tags_all = set()
        metric.required_tags_any = set()
        metric.prohibited_tags = set()
        metric.setup()

    logger.info(f"Loaded {len(metric_manager.metrics)} metrics: {list(metric_manager.metrics.keys())}")

    # Evaluate each structure
    logger.info(f"Evaluating {len(structure_files)} structures...")
    results = []
    for file_path in tqdm(structure_files, desc="Evaluating structures"):
        result = evaluate_structure(file_path, metric_manager, cfg.cif_parser_args)
        results.append(result)

    for metric in metric_manager.metrics.values():
        metric.teardown()

    # Output path from config (already includes job_id)
    output_csv = Path(cfg.output_csv)

    # Aggregate and save results
    aggregate_and_save_results(results, output_csv, verbose=cfg.verbose)

    logger.info(f"✓ Evaluation complete! Results saved to {output_csv}")


if __name__ == "__main__":
    main()
