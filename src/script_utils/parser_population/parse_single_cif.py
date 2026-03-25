#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/latent_exec.sh" "$0" "$@"'
"""Parse CIF files with specified parser arguments and save to cache.

This script is designed to be called from the command line to parse CIF files
with the correct parser arguments and save them to cache. It can process either
a single CIF file or multiple files from a command file (to run many files in one
apptainer session).

Usage:
    # Single file
    python scripts/parse_single_cif.py --cif-path /path/to/file.cif.gz --assembly-id 1 \
        --cif-parser-args '{"cache_dir": "/path/to/cache", ...}'

    # Multiple files (batch mode)
    python scripts/parse_single_cif.py --command-file commands.txt

    Where commands.txt contains one command per line in the format:
    /path/to/file.cif.gz 1 {"cache_dir": "...", ...}
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import rootutils
from atomworks.common import string_to_md5_hash
from atomworks.constants import CCD_MIRROR_PATH
from atomworks.io import parse
from atomworks.io.parser import STANDARD_PARSER_ARGS

# setup dotenv
from foundry.utils.env import load_dotenv

load_dotenv()

# Setup root dir
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_cif_with_args(cif_path: Path, cif_parser_args: dict) -> None:
    """Parse a CIF file with the given arguments.

    Args:
        cif_path: Path to the CIF file
        cif_parser_args: Dictionary of parser arguments
    """
    # Force save_to_cache to True
    cif_parser_args = cif_parser_args.copy()
    cif_parser_args["save_to_cache"] = True

    # Make sure we're not trying to load from cache
    cif_parser_args["load_from_cache"] = False

    # CRITICAL: JSON converts tuples to lists, but we need tuples to match atomworks hash
    # Convert these specific fields back to tuples
    if "add_bond_types_from_struct_conn" in cif_parser_args:
        cif_parser_args["add_bond_types_from_struct_conn"] = tuple(cif_parser_args["add_bond_types_from_struct_conn"])
    if "remove_ccds" in cif_parser_args:
        cif_parser_args["remove_ccds"] = tuple(cif_parser_args["remove_ccds"])
    if "build_assembly" in cif_parser_args:
        cif_parser_args["build_assembly"] = tuple(cif_parser_args["build_assembly"])

    # Merge STANDARD_PARSER_ARGS with cif_parser_args
    # CRITICAL: Must include ccd_mirror_path with same default as atomworks.io.parse()
    cif_parser_args = {"ccd_mirror_path": CCD_MIRROR_PATH, **STANDARD_PARSER_ARGS, **cif_parser_args}

    logger.info(f"Parsing {cif_path}")
    logger.info(f"Parser args: {cif_parser_args}")

    # Compute hash the same way atomworks does (only the subset of args that affect parsing)
    parse_arguments = {
        "ccd_mirror_path": cif_parser_args.get("ccd_mirror_path"),
        "add_missing_atoms": cif_parser_args.get("add_missing_atoms"),
        "add_id_and_entity_annotations": cif_parser_args.get("add_id_and_entity_annotations"),
        "add_bond_types_from_struct_conn": cif_parser_args.get("add_bond_types_from_struct_conn"),
        "remove_ccds": cif_parser_args.get("remove_ccds"),
        "remove_waters": cif_parser_args.get("remove_waters"),
        "fix_ligands_at_symmetry_centers": cif_parser_args.get("fix_ligands_at_symmetry_centers"),
        "fix_arginines": cif_parser_args.get("fix_arginines"),
        "fix_formal_charges": cif_parser_args.get("fix_formal_charges"),
        "fix_bond_types": cif_parser_args.get("fix_bond_types"),
        "long_bond_policy": cif_parser_args.get("long_bond_policy"),
        "convert_mse_to_met": cif_parser_args.get("convert_mse_to_met"),
        "hydrogen_policy": cif_parser_args.get("hydrogen_policy"),
    }
    args_string = ",".join(str(parse_arguments[k]) for k in sorted(parse_arguments.keys()))
    logger.info(f"THIS IS THE HASH: {string_to_md5_hash(args_string, truncate=8)}")

    try:
        result = parse(cif_path, **cif_parser_args)
        logger.info(f"Successfully parsed {cif_path}")
        logger.info(f"  - Assemblies: {list(result['assemblies'].keys())}")
        logger.info(f"  - Chains: {list(result['chain_info'].keys())}")
    except Exception as e:
        logger.error(f"Failed to parse {cif_path}: {e}")
        raise


def parse_from_command_line(cif_path_str: str, assembly_id_str: str, parser_args_str: str) -> bool:
    """Parse a CIF file from command line arguments.

    Args:
        cif_path_str: Path to CIF file as string
        assembly_id_str: Assembly ID as string
        parser_args_str: JSON string of parser arguments

    Returns:
        True if successful, False otherwise
    """
    try:
        cif_path = Path(cif_path_str)

        # Validate CIF path exists
        if not cif_path.exists():
            logger.error(f"CIF file does not exist: {cif_path}")
            return False

        # Parse CIF parser args from JSON
        try:
            cif_parser_args = json.loads(parser_args_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cif-parser-args JSON: {e}")
            return False

        # Parse the CIF file
        parse_cif_with_args(cif_path, cif_parser_args)
        return True

    except Exception as e:
        logger.error(f"Failed to process {cif_path_str}: {e}")
        return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Parse CIF files with specified arguments")
    parser.add_argument("--cif-path", type=Path, help="Path to CIF file (for single file mode)")
    parser.add_argument("--assembly-id", type=str, help="Assembly ID to parse (for single file mode)")
    parser.add_argument(
        "--cif-parser-args",
        type=str,
        default="{}",
        help="JSON string of CIF parser arguments (for single file mode)",
    )
    parser.add_argument(
        "--command-file",
        type=Path,
        help=(
            "Path to file containing commands (for batch mode). "
            "Each line should be: <cif_path> <assembly_id> <parser_args_json>"
        ),
    )

    args = parser.parse_args()

    # Batch mode: process multiple files from command file
    if args.command_file:
        if not args.command_file.exists():
            logger.error(f"Command file does not exist: {args.command_file}")
            sys.exit(1)

        logger.info(f"Processing commands from file: {args.command_file}")

        total = 0
        success = 0
        failed = 0

        with open(args.command_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse the line: <cif_path> <assembly_id> <parser_args_json>
                parts = line.split(None, 2)  # Split on whitespace, max 3 parts
                if len(parts) != 3:
                    logger.warning(f"Line {line_num}: Invalid format (expected 3 parts, got {len(parts)}), skipping")
                    failed += 1
                    continue

                cif_path_str, assembly_id_str, parser_args_str = parts
                total += 1

                if parse_from_command_line(cif_path_str, assembly_id_str, parser_args_str):
                    success += 1
                else:
                    failed += 1

        logger.info(f"Batch processing complete: {total} total, {success} succeeded, {failed} failed")

        if failed > 0:
            sys.exit(1)

    # Single file mode
    else:
        if not args.cif_path:
            logger.error("Either --cif-path or --command-file must be provided")
            parser.print_help()
            sys.exit(1)

        # Parse CIF parser args from JSON
        try:
            cif_parser_args = json.loads(args.cif_parser_args)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cif-parser-args JSON: {e}")
            sys.exit(1)

        # Validate CIF path exists
        if not args.cif_path.exists():
            logger.error(f"CIF file does not exist: {args.cif_path}")
            sys.exit(1)

        # Parse the CIF file
        parse_cif_with_args(args.cif_path, cif_parser_args)


if __name__ == "__main__":
    main()
