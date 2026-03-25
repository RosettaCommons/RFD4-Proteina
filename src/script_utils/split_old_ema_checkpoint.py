"""
Split old-style EMA checkpoint into new two-file format.

Old format (single file):
- state_dict: training weights
- optimizer_states[0]: {opt, ema, current_step, decay, every_n_steps}

New format (two files):
1. Regular checkpoint (e.g., checkpoint.ckpt):
   - state_dict: training weights
   - optimizer_states[0]: just the inner optimizer state {state, param_groups}

2. EMA checkpoint (e.g., checkpoint-EMA.ckpt):
   - state_dict: EMA weights
   - optimizer_states[0]: full EMA structure {opt, ema, current_step, decay, every_n_steps}

3. Save as three separate files: <checkpoint_name>.ckpt, <checkpoint_name>-EMA.ckpt, and <checkpoint_name>-bad.ckpt

Usage:
    python split_old_ema_checkpoint.py <old_checkpoint.ckpt>
"""

import argparse
import copy
import os

import torch


def build_ema_state_dict_from_params(ema_params: list, state_dict: dict) -> dict:
    """
    Build a model state_dict from EMA parameters.

    Args:
        ema_params: List/tuple of EMA parameter tensors
        state_dict: State dictionary from the original model

    Returns:
        dict: A state_dict with EMA weights
    """
    non_autoencoder_state_dict_keys = [key for key in state_dict if not key.startswith("autoencoder")]
    if len(ema_params) != len(non_autoencoder_state_dict_keys):
        raise ValueError(
            f"Mismatch: {len(ema_params)} EMA params but "
            f"{len(non_autoencoder_state_dict_keys)} non-autoencoder state_dict keys"
        )

    ema_state_dict = copy.deepcopy(state_dict)
    for key, param in zip(non_autoencoder_state_dict_keys, ema_params, strict=False):
        assert (
            param.shape == state_dict[key].shape
        ), f"Mismatch: {param.shape} != {state_dict[key].shape} on parameter {key}"
        ema_state_dict[key] = param.clone()

    return ema_state_dict


def split_checkpoint(old_ckpt_path: str) -> None:
    """
    Split an old-style EMA checkpoint into new two-file format.

    Args:
        old_ckpt_path: Path to old checkpoint file
    """
    checkpoint = torch.load(old_ckpt_path, map_location="cpu", weights_only=False)

    opt_state = checkpoint["optimizer_states"][0]

    # Check if it's already in the new format
    if "opt" not in opt_state:
        print("WARNING: This checkpoint appears to already be in the new format (no 'opt' key)")
        print(f"Optimizer state keys: {list(opt_state.keys())}")
        print("Aborting.")
        return

    ema_params = opt_state["ema"]
    print(f"Found {len(ema_params)} EMA parameters")

    # Get model state dict keys
    state_dict = checkpoint["state_dict"]
    non_autoencoder_state_dict_keys = [key for key in state_dict if not key.startswith("autoencoder")]
    print(
        f"Model has {len(state_dict)} parameters in state_dict, "
        f"of which {len(non_autoencoder_state_dict_keys)} are non-autoencoder parameters"
    )

    # Create regular checkpoint
    regular_checkpoint = copy.deepcopy(checkpoint)
    regular_checkpoint["optimizer_states"][0] = opt_state["opt"]

    # Create EMA checkpoint
    ema_checkpoint = copy.deepcopy(checkpoint)

    # Replace state_dict with EMA weights
    try:
        ema_state_dict = build_ema_state_dict_from_params(ema_params, state_dict)
    except Exception as e:
        print(f"WARNING: Could not save EMA weights: {e}")
        print("Instead, will save regular training weights in the EMA checkpoint, resetting the EMA weights!!!")
        ema_state_dict = copy.deepcopy(state_dict)

    ema_checkpoint["state_dict"] = ema_state_dict

    # save all checkpoints
    os.rename(old_ckpt_path, old_ckpt_path.replace(".ckpt", "_OLD_AND_BAD.ckpt"))

    regular_ckpt_path = old_ckpt_path
    ema_ckpt_path = old_ckpt_path.replace(".ckpt", "-EMA.ckpt")

    torch.save(regular_checkpoint, regular_ckpt_path)
    torch.save(ema_checkpoint, ema_ckpt_path)

    print("Success!")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split old-style EMA checkpoint into new two-file format",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to old-style checkpoint file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file not found: {args.checkpoint}")
        return 1

    split_checkpoint(args.checkpoint)
    return 0


if __name__ == "__main__":
    main()
