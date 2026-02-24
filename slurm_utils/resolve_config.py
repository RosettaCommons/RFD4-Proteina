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

import sys

import hydra
from omegaconf import OmegaConf


def main() -> int:
    args = sys.argv[1:]

    # Extract config name and overrides
    config_name = "training_proteina"
    overrides = []

    i = 0
    while i < len(args):
        if args[i] == "--config-name" and i + 1 < len(args):
            config_name = args[i + 1]
            i += 2
        else:
            overrides.append(args[i])
            i += 1

    # Config path is relative to this script's location (slurm_utils/)
    # Hydra requires a relative path string
    config_path = "../configs"

    try:
        with hydra.initialize(config_path=config_path, version_base=None):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)

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

            # Optional hardware overrides (only output if set)
            gpu_type = OmegaConf.select(cfg, "hardware.gpu_type_", throw_on_resolution_failure=False, default=None)
            slurm_constraint = OmegaConf.select(
                cfg, "hardware.slurm_constraint_", throw_on_resolution_failure=False, default=None
            )
            exclude_nodes = OmegaConf.select(
                cfg, "hardware.exclude_nodes_", throw_on_resolution_failure=False, default=None
            )
    except Exception as e:
        print(f"Error resolving config: {e}", file=sys.stderr)
        return 1

    # Output shell variable assignments
    # Quote values to handle special characters
    print(f"RUN_NAME='{run_name}'")
    print(f"NGPUS_PER_NODE={ngpus_per_node}")
    print(f"NNODES={nnodes}")
    print(f"CPUS_PER_TASK={cpus_per_task}")

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
