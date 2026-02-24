#!/usr/bin/env python3
"""
Inference Sweep Config Generator
=================================

Generates inference and evaluation config files by sweeping over parameter combinations.
Replaces the old per-use-case scripts (generate_inference_configs.py, generate_binder_inference_configs.py, etc.)
with a single unified module.

Sweep parameters can be defined either:
  1. In a YAML sweep config file (recommended) — see configs/sweeps/*.yaml
  2. Programmatically via the Python API

Usage (CLI with YAML sweep config):
    python script_utils/sweep.py --sweep-config configs/sweeps/ppi_benchmark.yaml
    python script_utils/sweep.py --sweep-config configs/sweeps/ppi_benchmark.yaml --dry-run
    python script_utils/sweep.py --sweep-config configs/sweeps/ppi_benchmark.yaml --output-dir ./my_sweep_configs

Usage (CLI with inline overrides):
    python script_utils/sweep.py --base-config inference \
        --sweep 'ckpt_name=["ckpt_a.ckpt","ckpt_b.ckpt"]' \
        --sweep 'generation.args.nsteps=[100,400]'

Python API:
    from script_utils.sweep import generate_sweep_configs
    configs = generate_sweep_configs(
        base_config_name="inference",
        sweeper={"generation.args.nsteps": [100, 400], "nsamples": [50, 200]},
    )
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

import hydra
from hydra import initialize_config_dir
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from atomworks.ml.utils.nested_dict import unflatten


def load_sweep_config(sweep_config_path: str | Path) -> dict:
    """Load a YAML sweep config file.

    Expected format:
        base_config: inference          # Hydra config name to compose on
        base_config_overrides: []       # Optional: Hydra overrides for base config
        output_dir: ./configs/sweep_configs  # Where to save generated configs
        sweep:
            ckpt_path: ["/path/1", "/path/2"]
            generation.args.nsteps: [100, 400]

    Returns:
        Dictionary with sweep configuration.
    """
    with open(sweep_config_path) as f:
        cfg = yaml.safe_load(f)

    if "sweep" not in cfg:
        raise ValueError(f"Sweep config must have a 'sweep' key: {sweep_config_path}")

    return cfg


def generate_sweep_configs(
    base_config_name: str = "inference",
    base_config_overrides: list[str] | None = None,
    sweeper: dict[str, list] | None = None,
    output_dir: str = "./configs/sweep_configs",
    run_name_prefix: str | None = None,
    config_path: str = "../configs",
    clean_output: bool = True,
) -> list[Path]:
    """Generate config files for each combination in the sweeper.

    Args:
        base_config_name: Hydra config name to compose on (e.g., "inference").
        base_config_overrides: Additional Hydra overrides for the base config.
        sweeper: Dict mapping dot-notation keys to lists of values to sweep.
                 Empty dict or None generates a single config with defaults.
        output_dir: Directory to save generated config files.
        run_name_prefix: Optional prefix for generated run names.
        config_path: Relative path to Hydra config directory.
        clean_output: If True, remove existing output directory before generating.

    Returns:
        List of paths to generated config files.
    """
    if sweeper is None:
        sweeper = {}

    # Create output directory
    output_path = Path(output_dir)
    if clean_output and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base config via Hydra
    overrides = base_config_overrides or []
    config_path_resolved = str(config_path)
    if os.path.isabs(config_path_resolved):
        ctx = initialize_config_dir(config_dir=config_path_resolved, version_base=None)
    else:
        ctx = hydra.initialize(config_path=config_path_resolved, version_base=None)
    with ctx:
        cfg_base = hydra.compose(config_name=base_config_name, overrides=overrides)
        logger.info(f"Loaded base config: {base_config_name}")

    # Generate Cartesian product of sweep values
    keys = list(sweeper.keys())
    values = list(sweeper.values())
    combinations = list(itertools.product(*values)) if values else [()]
    combinations = [dict(zip(keys, combo)) for combo in combinations]

    if not combinations:
        combinations = [{}]

    generated_files = []
    for idx, combo in enumerate(combinations):
        # Convert dot-notation to nested dict and merge
        nested_override = unflatten(combo, split_keys=".")
        override_cfg = OmegaConf.create(nested_override)

        OmegaConf.set_struct(cfg_base, False)
        cfg_merged = OmegaConf.merge(cfg_base, override_cfg)

        # Set run_name if prefix is given
        if run_name_prefix:
            OmegaConf.set_struct(cfg_merged, False)
            cfg_merged["run_name_"] = f"{run_name_prefix}_{idx}"

        # Build descriptive filename from swept values
        name_parts = []
        for key, val in combo.items():
            short_key = key.split(".")[-1]  # Use last part of dot key
            short_val = str(val).replace("/", "_").replace(" ", "")
            # Truncate long values
            if len(short_val) > 30:
                short_val = short_val[:27] + "..."
            name_parts.append(f"{short_key}={short_val}")

        if name_parts:
            suffix = "__".join(name_parts)
            filename = f"inf_{idx:03d}__{suffix}.yaml"
        else:
            filename = f"inf_{idx:03d}.yaml"

        # Sanitize filename
        filename = filename.replace("'", "").replace('"', "")

        filepath = output_path / filename
        OmegaConf.save(config=cfg_merged, f=str(filepath))
        generated_files.append(filepath)
        logger.info(f"  Generated: {filename}")

    logger.info(f"Generated {len(generated_files)} config(s) in {output_dir}")
    return generated_files


# =============================================================================
# CLI
# =============================================================================


def parse_sweep_arg(sweep_str: str) -> tuple[str, list]:
    """Parse a --sweep argument like 'key=[val1,val2]' into (key, [val1, val2]).

    Supports:
        'nsteps=[100,400]'
        'ckpt_path=["/path/a","/path/b"]'
    """
    if "=" not in sweep_str:
        raise ValueError(f"Invalid --sweep format: '{sweep_str}'. Expected 'key=[val1,val2]'")

    key, value_str = sweep_str.split("=", 1)
    key = key.strip()

    # Try JSON parsing first (handles arrays, strings, numbers)
    try:
        values = json.loads(value_str)
        if isinstance(values, list):
            return key, values
        return key, [values]
    except json.JSONDecodeError:
        pass

    # Fallback: treat as comma-separated values
    if value_str.startswith("[") and value_str.endswith("]"):
        value_str = value_str[1:-1]

    values = [v.strip().strip("'\"") for v in value_str.split(",")]

    # Try to convert to numbers
    converted = []
    for v in values:
        try:
            converted.append(int(v))
        except ValueError:
            try:
                converted.append(float(v))
            except ValueError:
                if v.lower() == "true":
                    converted.append(True)
                elif v.lower() == "false":
                    converted.append(False)
                else:
                    converted.append(v)
    return key, converted


def main():
    parser = argparse.ArgumentParser(
        description="Generate inference configs by sweeping over parameter combinations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From a YAML sweep config (recommended):
  python sweep.py --sweep-config configs/sweeps/ppi_benchmark.yaml

  # Inline sweep:
  python sweep.py --base-config inference \\
      --sweep 'generation.args.nsteps=[100,400]' \\
      --sweep 'nsamples=[50,200]'

  # Dry run (shows what would be generated):
  python sweep.py --sweep-config configs/sweeps/ppi_benchmark.yaml --dry-run
        """,
    )

    parser.add_argument(
        "--sweep-config",
        type=str,
        default=None,
        help="Path to YAML sweep config file (see configs/sweeps/*.yaml for examples)",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="inference",
        help="Hydra config name to compose on (default: inference)",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        action="append",
        default=[],
        help="Sweep parameter: 'key=[val1,val2]'. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated configs (default: ./configs/sweep_configs)",
    )
    parser.add_argument(
        "--run-name-prefix",
        type=str,
        default=None,
        help="Prefix for generated run names",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without creating files",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="../configs",
        help="Relative path to Hydra config directory",
    )

    args = parser.parse_args()

    # Load sweep from YAML config or CLI args
    if args.sweep_config:
        sweep_cfg = load_sweep_config(args.sweep_config)
        base_config = sweep_cfg.get("base_config", "inference")
        base_overrides = sweep_cfg.get("base_config_overrides", [])
        sweeper = sweep_cfg.get("sweep", {})
        output_dir = args.output_dir or sweep_cfg.get("output_dir", "./configs/sweep_configs")
        run_name_prefix = args.run_name_prefix or sweep_cfg.get("run_name_prefix", None)
    else:
        base_config = args.base_config
        base_overrides = []
        sweeper = {}
        for sweep_str in args.sweep:
            key, values = parse_sweep_arg(sweep_str)
            sweeper[key] = values
        output_dir = args.output_dir or "./configs/sweep_configs"
        run_name_prefix = args.run_name_prefix

    # Show sweep info
    total_combos = 1
    for values in sweeper.values():
        total_combos *= len(values) if isinstance(values, list) else 1

    logger.info(f"Base config: {base_config}")
    logger.info(f"Sweep parameters: {len(sweeper)} keys")
    for key, values in sweeper.items():
        n = len(values) if isinstance(values, list) else 1
        logger.info(f"  {key}: {n} values")
    logger.info(f"Total combinations: {total_combos}")
    logger.info(f"Output directory: {output_dir}")

    if args.dry_run:
        logger.info("DRY RUN — no files will be created")
        # Show combinations
        keys = list(sweeper.keys())
        values = list(sweeper.values())
        combos = list(itertools.product(*values)) if values else [()]
        for i, combo in enumerate(combos):
            combo_dict = dict(zip(keys, combo))
            logger.info(f"  Config {i}: {combo_dict}")
        return

    # Generate configs
    generated = generate_sweep_configs(
        base_config_name=base_config,
        base_config_overrides=base_overrides,
        sweeper=sweeper,
        output_dir=output_dir,
        run_name_prefix=run_name_prefix,
        config_path=args.config_path,
    )

    logger.info(f"Done! Generated {len(generated)} config file(s)")
    for f in generated:
        logger.info(f"  {f}")


if __name__ == "__main__":
    main()
