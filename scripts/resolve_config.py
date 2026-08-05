#!/usr/bin/env python3
"""Resolve Hydra config and output values for shell scripts.

This script uses Hydra to properly resolve configs including:
- Config composition (defaults lists)
- Interpolations (${oc.env:VAR}, ${other.key})
- Command-line overrides

Usage:
    python resolve_config.py --config-name training_proteina [overrides...]
    python resolve_config.py --config-name training_proteina +experiment=ablation
    python resolve_config.py --config-name training_ae

Output format (shell-friendly variable assignments):
    RUN_NAME=my_run_name
    NGPUS_PER_NODE=8
    NNODES=4
    CPUS_PER_TASK=24

Shell usage:
    eval $(python resolve_config.py --config-name training_proteina)
"""

import os
import sys
from pathlib import Path
from typing import Any

import hydra
import rootutils
from hydra import initialize_config_dir
from omegaconf import DictConfig, OmegaConf

_AF3_ORACLE_TARGET = "rfproteina.metrics.oracles.af3_oracle.AF3Oracle"


def _has_af3_oracle_target(cfg: DictConfig) -> bool:
    """Recursively search the resolved config for a node whose ``_target_``
    equals ``rfproteina.metrics.oracles.af3_oracle.AF3Oracle``.
    """
    container = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)

    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            if node.get("_target_") == _AF3_ORACLE_TARGET:
                return True
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    return walk(container)


def main() -> int:
    args = sys.argv[1:]

    # Extract config name, optional config path, and overrides
    config_name = "training_proteina"
    config_path = None
    overrides = []

    i = 0
    while i < len(args):
        if args[i] == "--config-name" and i + 1 < len(args):
            config_name = args[i + 1]
            i += 2
        elif args[i] == "--config-path" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        else:
            overrides.append(args[i])
            i += 1

    # Config path is relative to this script's location (scripts/)
    script_dir = Path(__file__).parent
    configs_root = (script_dir / "../configs").resolve()

    # Ensure PROJECT_ROOT is set for configs that interpolate it (e.g. ``out_dir``,
    # ``inputs``, or ``hydra.searchpath: file://${oc.env:PROJECT_ROOT}/...``).
    repo_root = rootutils.find_root(indicator=".project-root", search_from=__file__)
    os.environ.setdefault("PROJECT_ROOT", str(repo_root))

    # Mirror Hydra exactly: --config-path is the search-path root (defaults to the
    # repo configs/ tree), --config-name is the config name within it
    config_dir = str(Path(config_path).resolve()) if config_path is not None else str(configs_root)
    config_base = config_name

    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(config_name=config_base, overrides=overrides)

            # Selectively resolve only the keys we need
            # This avoids failing on unresolved interpolations in other parts of config
            run_name = OmegaConf.select(cfg, "run_name_", throw_on_resolution_failure=False, default="")
            ngpus_per_node = OmegaConf.select(
                cfg, "hardware.ngpus_per_node_", throw_on_resolution_failure=False, default=1
            )
            nnodes = OmegaConf.select(cfg, "hardware.nnodes_", throw_on_resolution_failure=False, default=1)
            cpus_per_task = OmegaConf.select(
                cfg, "hardware.ncpus_per_task_train_", throw_on_resolution_failure=False, default=None
            )

            # Fallback for cpus_per_task: check dataloaders config
            if cpus_per_task is None:
                cpus_per_task = OmegaConf.select(
                    cfg, "dataloaders.num_workers", throw_on_resolution_failure=False, default=8
                )

            # Parallel generation jobs (used by launch_infer.sh)
            gen_njobs = OmegaConf.select(cfg, "gen_njobs", throw_on_resolution_failure=False, default=1)

            # Output directory (used by launch_infer.sh to set RUN_DIR)
            out_dir = OmegaConf.select(cfg, "out_dir", throw_on_resolution_failure=False, default=None)

            # Optional hardware overrides (only output if set)
            gpu_type = OmegaConf.select(cfg, "hardware.gpu_type_", throw_on_resolution_failure=False, default=None)
            slurm_constraint = OmegaConf.select(
                cfg, "hardware.slurm_constraint_", throw_on_resolution_failure=False, default=None
            )
            exclude_nodes = OmegaConf.select(
                cfg, "hardware.exclude_nodes_", throw_on_resolution_failure=False, default=None
            )

            # Detect whether any AF3Oracle is configured anywhere in the resolved config.
            # Used by launchers to gate the AF3-daemon pre-spawn block.
            try:
                af3_needed = _has_af3_oracle_target(cfg)
            except Exception:
                af3_needed = False
    except Exception as e:
        print(f"Error resolving config: {e}", file=sys.stderr)
        return 1

    # Output shell variable assignments
    # Quote values to handle special characters
    print(f"RUN_NAME='{run_name}'")
    if out_dir is not None:
        print(f"OUT_DIR='{out_dir}'")
    print(f"NGPUS_PER_NODE={ngpus_per_node}")
    print(f"NNODES={nnodes}")
    print(f"CPUS_PER_TASK={cpus_per_task}")
    print(f"GEN_NJOBS={gen_njobs}")
    print(f"AF3_NEEDED={'true' if af3_needed else 'false'}")

    # Optional overrides (only output if explicitly set in config)
    if gpu_type is not None:
        print(f"GPU_TYPE='{gpu_type}'")
    if slurm_constraint is not None:
        print(f"SLURM_EXTRA_OPTIONS='--constraint={slurm_constraint}'")
    if exclude_nodes is not None:
        print(f"EXCLUDE_NODES='{exclude_nodes}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
