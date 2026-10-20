import json
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class GradientCoverageCallback(Callback):
    """Record, per parameter, how many dim-0 rows ever receive a non-zero gradient.

    Tests ``grad != 0``, not ``grad is None``: a feature behind an all-false mask stays in the
    autograd graph with an exactly-zero grad, so DDP's unused-parameter check never fires.
    Reject nonfinite gradients rather than counting them as live rows.

    Args:
      output_path: JSON destination for the ``{param_name: {alive_rows, n_rows}}`` map.
    """

    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self._alive: dict[str, torch.Tensor] = {}

    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: torch.optim.Optimizer
    ) -> None:
        """Accumulate live rows. Fires after any DDP allreduce, so grads agree on every rank."""
        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue  # frozen by design (e.g. the autoencoder)
            n_rows = param.shape[0] if param.ndim else 1
            if param.grad is None:
                alive = torch.zeros(n_rows, dtype=torch.bool, device=param.device)
            else:
                if not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"Nonfinite gradient for {name}")
                alive = (param.grad.reshape(n_rows, -1) != 0).any(dim=1)  # [n_rows]
            key = name.replace("_orig_mod.", "")  # torch.compile wrapper prefix
            if key in self._alive:
                self._alive[key] |= alive
            else:
                self._alive[key] = alive

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Write the accumulated per-parameter row coverage to ``output_path`` on rank 0."""
        if trainer.global_rank != 0:
            return
        coverage = {
            name: {
                "alive_rows": int(alive.sum()),
                "n_rows": int(alive.numel()),
                # Small tables: name the live rows, so a mismatch says *which* row went dark.
                **({"alive_idx": alive.nonzero().flatten().tolist()} if alive.numel() <= 8 else {}),
            }
            for name, alive in sorted(self._alive.items())
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(coverage, indent=2))
        n_partial = sum(1 for v in coverage.values() if v["alive_rows"] < v["n_rows"])
        logger.info(
            f"Wrote gradient coverage for {len(coverage)} parameters "
            f"({n_partial} with unreached rows) to {self.output_path}"
        )


class LogEpochTimeCallback(Callback):
    """Simple callback that logs how long each epoch takes, in seconds, to a pytorch lightning log"""

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        curr_time = time.time()
        duration = curr_time - self.epoch_start
        pl_module.log(
            "train_info/epoch_duration_secs",
            duration,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        if pl_module.current_epoch % 10 == 0:
            logger.info(f"Done training epoch {pl_module.current_epoch}, epoch took {duration} seconds")


class LogStepTimeCallback(Callback):
    """Simple callback that logs how long each training step takes, in seconds, to a pytorch lightning log"""

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: dict, batch_idx: int
    ) -> None:
        self.step_start = time.time()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        curr_time = time.time()
        duration = curr_time - self.step_start
        pl_module.log(
            "train_info/step_duration_secs",
            torch.tensor(duration, device=pl_module.device),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )


class SkipNanGradCallback(Callback):
    """Callback to skip gradient updates with NaN in them."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.iter = 0

    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.iter += 1
        checks_by_device: dict[torch.device, list[torch.Tensor]] = {}
        for p in pl_module.parameters():
            if p.grad is not None:
                checks_by_device.setdefault(p.grad.device, []).append(p.grad.isnan().any())
        nan_flag = any(torch.stack(checks).any().item() for checks in checks_by_device.values())
        if nan_flag:
            self.count += 1
            logger.warning(
                f"NaN gradient detected at iteration {self.iter}, "
                f"skipping optimizer step. Total NaN skips: {self.count}"
            )
            pl_module.zero_grad()


class SkipLargeGradients(Callback):
    """
    This callback tracks the average max gradient for the last `moving_avg_size` steps
    and checks if the current max gradient exceeds the threshold.
    """

    def __init__(
        self,
        moving_avg_size: int = 100,
        factor_threshold: int = 5,
        min_opt_steps: int = 2000,
    ) -> None:
        self.max_g_history = []
        self.moving_avg_size = moving_avg_size
        self.factor_threshold = factor_threshold
        self.min_opt_steps = min_opt_steps

    def on_before_optimizer_step(self, trainer: Any, pl_module: Any, optimizer: Any) -> None:
        # calculate the maximum gradient
        max_g = max(param.grad.data.abs().max() for param in pl_module.parameters() if param.grad is not None)

        # Update the moving average history
        if len(self.max_g_history) >= self.moving_avg_size:
            self.max_g_history.pop(0)
        if not (max_g.isnan().any() or max_g.isinf().any()):
            self.max_g_history.append(max_g.item())

        if trainer.global_step > self.min_opt_steps and len(self.max_g_history) > 1:
            moving_average = sum(self.max_g_history) / len(self.max_g_history)

            # Check if the current max gradient exceeds factor_threshold * moving average
            if max_g > self.factor_threshold * moving_average:
                # Zero out gradients
                for param in pl_module.parameters():
                    if param.grad is not None:
                        param.grad.data.zero_()  # Set gradients to zero
