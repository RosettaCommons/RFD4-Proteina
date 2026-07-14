#!/usr/bin/env python
"""Extract unique CIF file paths and parser arguments from a training experiment config.

This script:
1. Loads a Hydra experiment config (like those used in train.py)
2. Instantiates all training datasets
3. Extracts unique CIF file paths from the parquet files
4. Generates a commands file to re-parse all CIF files with the correct arguments

Usage:
    python scripts/extract_cif_paths_from_experiment.py --experiment simon/rfd3-full-2d-stage0 --output commands.txt
"""

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
import rootutils
from atomworks.common import string_to_md5_hash
from atomworks.constants import CCD_MIRROR_PATH
from atomworks.io.parser import STANDARD_PARSER_ARGS
from omegaconf import DictConfig, OmegaConf

# Setup root dir
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

# setup dotenv
from rfproteina.utils.env import load_dotenv  # noqa: E402

load_dotenv()

# Set up project path for configs
_config_path = os.path.join(os.environ.get("PROJECT_PATH", os.environ["PROJECT_ROOT"]), "configs")

print(f"Config path: {_config_path}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cache_file_path(cif_path: Path, assembly_id: str, cif_parser_args: dict) -> Path:
    """Compute the cache file path for a given CIF file and parser arguments.

    This replicates the logic from atomworks.io.parser.parse().

    Args:
        cif_path: Path to CIF file
        assembly_id: Assembly ID to build
        cif_parser_args: Parser arguments dict

    Returns:
        Path to the cache file
    """
    cache_dir = Path(cif_parser_args["cache_dir"])

    # Merge with default parser args to get all parameters
    # CRITICAL: Must include ccd_mirror_path with same default as atomworks.io.parse()
    merged_args = {
        "ccd_mirror_path": CCD_MIRROR_PATH,  # Default from atomworks
        **STANDARD_PARSER_ARGS,
        **cif_parser_args,
    }

    # Build the args string from the relevant parameters (same as in atomworks)
    parse_arguments = {
        "ccd_mirror_path": merged_args.get("ccd_mirror_path"),
        "add_missing_atoms": merged_args.get("add_missing_atoms"),
        "add_id_and_entity_annotations": merged_args.get("add_id_and_entity_annotations"),
        "add_bond_types_from_struct_conn": merged_args.get("add_bond_types_from_struct_conn"),
        "remove_ccds": merged_args.get("remove_ccds"),
        "remove_waters": merged_args.get("remove_waters"),
        "fix_ligands_at_symmetry_centers": merged_args.get("fix_ligands_at_symmetry_centers"),
        "fix_arginines": merged_args.get("fix_arginines"),
        "fix_bond_types": merged_args.get("fix_bond_types"),
        "long_bond_policy": merged_args.get("long_bond_policy"),
        "convert_mse_to_met": merged_args.get("convert_mse_to_met"),
        "hydrogen_policy": merged_args.get("hydrogen_policy"),
    }

    # Compose args_string from parse_arguments values (in order)
    args_string = ",".join(str(parse_arguments[k]) for k in sorted(parse_arguments.keys()))
    args_hash = string_to_md5_hash(args_string, truncate=8)

    # Construct the cache file path
    assembly_info = assembly_id
    cache_file_path = cache_dir / args_hash / f"{cif_path.stem}_assembly_{assembly_info}.pkl.gz"

    return cache_file_path


def get_parquet_path_and_parser_args(
    dataset_cfg: DictConfig,
) -> tuple[Path | None, dict]:
    """Extract parquet path and cif_parser_args from a dataset config.

    Args:
        dataset_cfg: Dataset configuration from Hydra

    Returns:
        Tuple of (parquet_path, cif_parser_args)
    """
    # Check if this is a PandasDataset with a parquet file
    if "dataset" not in dataset_cfg:
        return None, {}

    dataset = dataset_cfg["dataset"]

    # Get the data path (parquet file)
    data_path = dataset.get("data", None)
    if data_path is None or not str(data_path).endswith(".parquet"):
        return None, {}

    # Extract parser_args from dataset.loader.parser_args (current structure)
    cif_parser_args = {}
    if "loader" in dataset and "parser_args" in dataset["loader"]:
        cif_parser_args = dict(dataset["loader"]["parser_args"])
    # Fallback: check for direct parser_args (old structure)
    elif "parser_args" in dataset:
        cif_parser_args = dict(dataset["parser_args"])

    return Path(data_path), cif_parser_args


def extract_unique_cif_paths(cfg: DictConfig, override: bool = False) -> dict[str, set[Path]]:
    """Extract unique CIF paths from all training datasets.

    Args:
        cfg: Full Hydra configuration
        override: If False, skip files that are already cached

    Returns:
        Dictionary mapping cif_parser_args_hash to set of CIF paths
    """
    paths_by_parser_args = defaultdict(set)

    # Only process training datasets
    if "train" not in cfg.datasets:
        logger.error("No training datasets found in config")
        return {}

    train_datasets = cfg.datasets.train

    # Iterate through all training datasets
    for dataset_name, dataset_cfg in train_datasets.items():
        if dataset_cfg is None:
            continue

        logger.info(f"Processing dataset: {dataset_name}")

        # Handle nested sub_datasets
        if "sub_datasets" in dataset_cfg:
            for sub_name, sub_cfg in dataset_cfg.sub_datasets.items():
                logger.info(f"  Processing sub-dataset: {sub_name}")
                parquet_path, cif_args = get_parquet_path_and_parser_args(sub_cfg)

                # Skip if cache_dir is null (typically distillation datasets)
                if cif_args.get("cache_dir") is None:
                    logger.info("    Skipping sub-dataset: cache_dir is null")
                    continue

                if parquet_path and parquet_path.exists():
                    _extract_from_parquet(parquet_path, sub_cfg, cif_args, paths_by_parser_args, override)
        else:
            parquet_path, cif_args = get_parquet_path_and_parser_args(dataset_cfg)

            # Skip if cache_dir is null (typically distillation datasets)
            if cif_args.get("cache_dir") is None:
                logger.info("  Skipping dataset: cache_dir is null")
                continue

            if parquet_path and parquet_path.exists():
                _extract_from_parquet(parquet_path, dataset_cfg, cif_args, paths_by_parser_args, override)

    return paths_by_parser_args


def _extract_from_parquet(
    parquet_path: Path,
    dataset_cfg: DictConfig,
    cif_args: dict,
    paths_by_parser_args: dict,
    override: bool = False,
) -> None:
    """Extract CIF paths from a single parquet file."""
    logger.info(f"    Reading parquet: {parquet_path}")

    # Check if dataset has assembly_id column
    dataset = dataset_cfg["dataset"]
    loader = dataset.get("loader", {})
    assembly_id_colname = loader.get("assembly_id_colname", None)
    # Handle both Python None and string "None" from configs
    has_assembly_id = assembly_id_colname is not None and str(assembly_id_colname).lower() != "none"

    # Read required columns
    columns = ["path"]
    if has_assembly_id:
        columns.append(assembly_id_colname)

    df = pd.read_parquet(parquet_path, columns=columns)
    logger.info(f"    Found {len(df)} examples")

    # Get unique (path, assembly_id) combinations
    if has_assembly_id:
        unique_combos = df.groupby(["path", assembly_id_colname]).size().reset_index(name="count")
        logger.info(f"    Found {len(unique_combos)} unique (path, {assembly_id_colname}) combinations in parquet")
    else:
        # No assembly_id, so just get unique paths and use default assembly_id "1"
        unique_combos = df.groupby(["path"]).size().reset_index(name="count")
        unique_combos["assembly_id"] = "1"
        logger.info(f"    Found {len(unique_combos)} unique paths in parquet (using default assembly_id='1')")

    # Create a hashable key from cif_parser_args (without build_assembly since that varies per file)
    base_args = {k: v for k, v in cif_args.items() if k != "build_assembly"}
    args_key = json.dumps(base_args, sort_keys=True)

    # Store tuples of (path, assembly_id)
    path_assembly_tuples = set()
    skipped_count = 0
    for _, row in unique_combos.iterrows():
        cif_path = Path(row["path"])
        # Use the correct column name for assembly_id
        if has_assembly_id:
            assembly_id = str(row[assembly_id_colname])
        else:
            assembly_id = str(row["assembly_id"])  # Default "1" we added above

        # If not overriding, check if cache file already exists
        if not override:
            cache_file = get_cache_file_path(cif_path, assembly_id, base_args)
            if cache_file.exists():
                skipped_count += 1
                continue

        path_assembly_tuples.add((cif_path, assembly_id))

    if skipped_count > 0:
        logger.info(f"    Skipped {skipped_count} already-cached files")
    logger.info(f"    Found {len(path_assembly_tuples)} unique (CIF path, assembly) combinations to process")

    if args_key not in paths_by_parser_args:
        paths_by_parser_args[args_key] = set()
    paths_by_parser_args[args_key].update(path_assembly_tuples)


def write_commands_file(paths_by_parser_args: dict, output_path: Path, batch_format: bool = False) -> None:
    """Write parsing commands to a file.

    Args:
        paths_by_parser_args: Dictionary mapping parser args to (path, assembly_id) tuples
        output_path: Path to output commands file
        batch_format: If True, write in simplified batch format for --command-file
    """
    with open(output_path, "w") as f:
        for args_json, path_assembly_tuples in paths_by_parser_args.items():
            base_args = json.loads(args_json)

            # CRITICAL: Merge with STANDARD_PARSER_ARGS just like atomworks loaders do
            merged_args = {"ccd_mirror_path": CCD_MIRROR_PATH, **STANDARD_PARSER_ARGS, **base_args}

            if batch_format:
                # Write header comment with MERGED parser args
                merged_args_json = json.dumps(merged_args, sort_keys=True)
                f.write(f"# CIF Parser Args (merged): {merged_args_json}\n")

                # Write simplified format for batch processing
                for path, assembly_id in sorted(path_assembly_tuples):
                    # Create parser args with build_assembly for this specific file
                    file_args = merged_args.copy()
                    file_args["build_assembly"] = [assembly_id]
                    file_args_json = json.dumps(file_args, sort_keys=True)

                    # Simplified format: <path> <assembly_id> <parser_args_json>
                    f.write(f"{path} {assembly_id} {file_args_json}\n")

                f.write("\n")
            else:
                # Write full python commands (original format)
                merged_args_json = json.dumps(merged_args, sort_keys=True)
                f.write(f"# CIF Parser Args (merged): {merged_args_json}\n")

                for path, assembly_id in sorted(path_assembly_tuples):
                    # Create parser args with build_assembly for this specific file
                    file_args = merged_args.copy()
                    file_args["build_assembly"] = [assembly_id]
                    file_args_json = json.dumps(file_args, sort_keys=True)

                    # Build command
                    cmd_parts = [
                        "python",
                        "scripts/parse_single_cif.py",
                        f"--cif-path '{path}'",
                        f"--assembly-id '{assembly_id}'",
                    ]

                    # Add parser args
                    cmd_parts.append(f"--cif-parser-args '{file_args_json}'")

                    f.write(" ".join(cmd_parts) + "\n")

                f.write("\n")

    logger.info(f"Wrote commands to {output_path}")


@hydra.main(config_path=_config_path, config_name="training_proteina", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Default output file
    output_file = Path("cif_parsing_commands.txt")

    # Check for override flag in Hydra-style override
    override_mode = any(arg == "+override=true" or arg == "override=true" for arg in sys.argv[1:])

    # Print config for debugging
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Extract unique CIF paths
    mode_desc = (
        "override mode (will re-parse all files)"
        if override_mode
        else "incremental mode (will skip already-cached files)"
    )

    # Print cache_dir from environment for debugging
    cache_dir = os.environ.get("CACHE_DIR", "NOT SET")
    logger.info(f"CACHE_DIR environment variable: {cache_dir}")

    logger.info(f"Extracting unique CIF paths from training datasets in {mode_desc}...")
    paths_by_parser_args = extract_unique_cif_paths(cfg, override=override_mode)

    # Count total unique files
    total_files = sum(len(paths) for paths in paths_by_parser_args.values())
    logger.info(f"Found {total_files} unique CIF files across {len(paths_by_parser_args)} parser configurations")

    # Write commands file in batch format (for use with parse_single_cif.py --command-file)
    logger.info("Writing commands in batch format...")
    write_commands_file(paths_by_parser_args, output_file, batch_format=True)
    logger.info(f"Done! Commands written to {output_file}")
    logger.info(f"To process these files, run: python scripts/parse_single_cif.py --command-file {output_file}")


if __name__ == "__main__":
    main()
