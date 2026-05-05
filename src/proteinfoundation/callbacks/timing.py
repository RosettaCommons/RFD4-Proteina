import lightning as L
import pandas as pd
import torch
from beartype.typing import Any

from foundry.utils.logging import print_df_as_table
from foundry.utils.torch import Timers


class LightningTimingCallback(L.Callback):
    """
    The hooks are a little different for Lightning vs. our typical custom fabric trainer / callback hooks.

    Lightning Training Loop - Callback Hook Timing Table

    +--------+-----------------------------------------------------------+---------------------------------------------+
    | Step   | Event                                                     | Callback Hook                               |
    +--------+-----------------------------------------------------------+---------------------------------------------+
    |   1    | Get batch from DataLoader                                 | on_train_batch_start                        |
    |   2    | Run training_step() (forward pass and loss calculation)   | training_step() (in LightningModule)        |
    |   3    | Before loss.backward()                                    | on_before_backward                          |
    |   4    | Run loss.backward() (gradient calculation)                | backward() (LightningModule/loss.backward())|
    |   5    | After loss.backward() (before optimizer steps)            | on_after_backward                           |
    |   6    | Before optimizer.step() (if not accumulating gradients)   | on_before_optimizer_step                    |
    |   7    | Run optimizer.step() (weight update)                      | (internal to Lightning)                     |
    |   8    | After optimizer.step(), before optimizer.zero_grad()      | on_before_zero_grad                         |
    |   9    | Run optimizer.zero_grad() (zero gradients)                | (internal to Lightning)                     |
    |  10    | End of the batch processing                               | on_train_batch_end                          |
    +--------+-----------------------------------------------------------+---------------------------------------------+

    To measure the time it takes to load a batch from the dataloader, we need to measure the time BEFORE
    on_train_batch_start .Either this is done by measuring the time from on_train_batch_end to on_train_batch_start,
    or by starting at on_train_epoch_start (for the first batch)
    """

    def __init__(self, log_every_n: int = 100):
        super().__init__()
        self.log_every_n = log_every_n
        self.timers = Timers()
        self.n_steps_since_last_log = 0

        # Pre-create all the timers to set use_barrier appropriately
        self.timers("train_batch_load", use_barrier=False)
        self.timers("forward_pass", use_barrier=False)
        self.timers("forward_backward_pass", use_barrier=False)
        self.timers("backward_pass", use_barrier=False)
        self.timers("optimizer_step", use_barrier=False)

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """
        At the start of a batch, we have to stop train loader iter timer since we're about to start the forward pass
        """
        # Ensure train_batch_load timer exists and is running before stopping
        if "train_batch_load" in self.timers.timers and self.timers.timers["train_batch_load"].started_:
            self.timers.stop("train_batch_load")
        self.timers.start("forward_pass")
        self.timers.start("forward_backward_pass")

    def on_before_backward(self, trainer: L.Trainer, pl_module: L.LightningModule, loss: torch.Tensor) -> None:
        self.timers.stop("forward_pass")
        self.timers.start("backward_pass")

    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.timers.stop("backward_pass")
        self.timers.stop("forward_backward_pass")

    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: torch.optim.Optimizer
    ) -> None:
        if not self.timers("optimizer_step").started_:
            self.timers.start("optimizer_step")

    def on_after_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: torch.optim.Optimizer
    ) -> None:
        """
        Override to do nothing since we're already stopping the optimizer step timer in on_train_batch_end
        """
        pass

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        # Only stop optimizer_step timer if it was started (i.e., optimizer step actually occurred)
        # With gradient accumulation, on_before_optimizer_step is only called every N batches
        if "optimizer_step" in self.timers.timers and self.timers.timers["optimizer_step"].started_:
            self.timers.stop("optimizer_step")

        step = trainer.global_step
        self.n_steps_since_last_log += 1

        if step % self.log_every_n == 0:
            timings = self.timers.elapsed(*self.timers.timers.keys(), reset=True)
            timings = {f"timings/{k}": v / self.n_steps_since_last_log for k, v in timings.items()}
            pl_module.log_dict(timings, on_step=True, on_epoch=False, rank_zero_only=True, sync_dist=False)

            if trainer.is_global_zero:
                self._print_timings(timings)

            self.n_steps_since_last_log = 0

        # Reset the timer for the next batch
        self.timers.start("train_batch_load")

    def _print_timings(self, timings: dict[str, float]) -> None:
        """Print timing statistics as a formatted table."""

        df = pd.DataFrame(timings.items(), columns=["Step", "Time (s)"])
        print_df_as_table(df, title=f"Timing stats (over {self.n_steps_since_last_log} steps)")
