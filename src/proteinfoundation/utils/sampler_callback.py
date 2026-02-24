"""Callback to handle sampler epoch setting in distributed training."""

import lightning as L
from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class SetSamplerEpochCallback(L.Callback):
    """Set epoch on distributed samplers when use_distributed_sampler=False.

    Lightning only automatically calls set_epoch() on samplers when it manages
    them (use_distributed_sampler=True). When using custom samplers, we must
    manually set the epoch to ensure proper shuffling between epochs.

    Without this callback, DistributedSampler will use the same random seed
    every epoch, resulting in identical batch ordering (equivalent to shuffle=False
    after the first epoch).
    """

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Set the epoch for train dataloader sampler.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: LightningModule being trained.
        """
        if trainer.train_dataloader is not None:
            sampler = trainer.train_dataloader.sampler
            if sampler is not None and hasattr(sampler, "set_epoch"):
                # Our distributed sampler MUST have a "set_epoch" function
                sampler.set_epoch(trainer.current_epoch)
                logger.info(f"[Rank {trainer.global_rank}] Set train sampler epoch to {trainer.current_epoch}")

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Set the epoch for validation dataloader samplers.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: LightningModule being validated.
        """
        if trainer.val_dataloaders is not None:
            val_dataloaders = trainer.val_dataloaders
            # Handle both single dataloader and list of dataloaders
            if not isinstance(val_dataloaders, list):
                val_dataloaders = [val_dataloaders]

            for idx, val_dataloader in enumerate(val_dataloaders):
                sampler = val_dataloader.sampler
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(trainer.current_epoch)
                    logger.info(
                        f"[Rank {trainer.global_rank}] Set validation sampler {idx} epoch to {trainer.current_epoch}"
                    )
