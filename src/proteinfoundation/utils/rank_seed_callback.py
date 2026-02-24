"""Callback to re-seed RNGs with rank offset after DDP initialization."""

import random
import numpy as np
import lightning as L
import torch
import torch.distributed as dist


class RankSeedCallback(L.Callback):
    """Re-seed all RNGs with rank offset at training/validation boundaries.

    This ensures:
    1. Each GPU has different random state (rank offset)
    2. Deterministic behavior on resume (global_step offset)
    3. Consistent rank diversity at epoch and validation boundaries

    Args:
        base_seed: The base seed. Each rank will use base_seed + rank + epoch.
    """

    def __init__(self, base_seed: int):
        self.base_seed = base_seed

    def _reseed(self, trainer: L.Trainer, context: str) -> None:
        """Re-seed all RNGs with rank-specific seed."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        # Use epoch + rank for seed to ensure:
        # - Different ranks have different seeds (rank offset)
        # - Reproducible on resume (epoch-based, not step-based for cleaner resume)
        seed = self.base_seed + rank + epoch * 1000 + global_step
        L.seed_everything(seed, workers=False)

        # Sample some random values to verify diversity across ranks
        sample_python = random.random()
        sample_numpy = np.random.rand()
        sample_torch = torch.rand(1).item()

        print(
            f"[Rank {rank}] Re-seeded RNGs ({context}): seed={seed} "
            f"(base={self.base_seed}, rank={rank}, epoch={epoch}, step={global_step}) | "
            f"samples: py={sample_python:.6f}, np={sample_numpy:.6f}, torch={sample_torch:.6f}",
            flush=True
        )

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Re-seed all RNGs after DDP is initialized, before training starts."""
        self._reseed(trainer, "fit_start")

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Re-seed at the start of each training epoch for consistent rank diversity."""
        self._reseed(trainer, "train_epoch_start")

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Re-seed before validation to ensure different ranks generate different samples."""
        self._reseed(trainer, "validation_epoch_start")
