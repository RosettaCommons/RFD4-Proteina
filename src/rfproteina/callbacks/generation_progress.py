"""Progress display for generation through Lightning's validation loop."""

from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import RichProgressBar


class GenerationProgressBar(RichProgressBar):
    """Alternate the generation label each batch (one example at batch size one)."""

    @property
    def validation_description(self) -> str:
        return "sizzling... 🔥"

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Show the active batch's label before its generation begins."""
        super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
        if self.is_disabled or self.progress is None or self.val_progress_bar_id is None:
            return
        description = "sizzling... 🔥" if batch_idx % 2 == 0 else "crunching... 🔢"
        self.progress.update(self.val_progress_bar_id, description=description, visible=True)
        self.refresh()
