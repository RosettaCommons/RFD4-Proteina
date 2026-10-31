"""Independent, reproducible generation random streams for distributed ranks."""

import lightning as L
import numpy as np

from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class GenerationRankSeedCallback(L.Callback):
    """Seed generation after distributed setup and before dataloader iteration.

    ``job_seed`` is the existing ``base_seed + job_id``. Mix it with the global
    rank rather than adding the rank, which would reuse seeds across adjacent
    jobs. Leave single-process generation untouched, including its RNG position
    after model loading. This callback is only registered by the generation CLI;
    training keeps its own seeding policy.
    """

    def __init__(self, job_seed: int) -> None:
        self.job_seed = job_seed

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage != "validate" or trainer.world_size == 1:
            return

        # Lightning resets RNGs during DDP setup. Its setup hook runs afterwards,
        # before worker seeds and prefetched input transforms draw from these RNGs.
        seed = int(np.random.SeedSequence([self.job_seed, trainer.global_rank]).generate_state(1)[0])
        L.seed_everything(seed, workers=False, verbose=False)
        pl_module.inference_seed = seed
        logger.info(f"Generation seed: {seed} (job seed={self.job_seed}, global rank={trainer.global_rank})")
