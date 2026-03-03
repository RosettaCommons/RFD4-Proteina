"""Callback for tracking loss vs timestep (t) during flow matching training."""

from collections import defaultdict

import lightning as L  # noqa: N812
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb
from beartype.typing import Any
from foundry.utils.ddp import RankedLogger
from scipy.signal import savgol_filter

logger = RankedLogger(__name__, rank_zero_only=True)


class TLossMonitoringCallback(L.Callback):
    """Track flow matching loss vs timestep (t) during training.

    This callback monitors how the model's loss varies across different timesteps
    during flow matching training, inspired by EDM paper analysis. It buckets
    timesteps and tracks running statistics to handle millions of examples efficiently.

    Features:
        - Buckets t values into configurable bins
        - Captures baseline from first N training steps
        - Normalizes current loss by baseline (optional)
        - Splits by design task for per-task analysis
        - Logs curves to WandB with smoothing
        - Persists state in checkpoints for resumption

    Args:
        n_buckets: Number of buckets for t values. Default: 50.
        t_min: Minimum t value. Default: 0.0.
        t_max: Maximum t value. Default: 1.0.
        log_every_n_steps: How often to log to WandB. Default: 10000.
        baseline_capture_steps: Steps to capture baseline statistics. Default: 1000.
        normalize_by_baseline: Whether to normalize curves by baseline. Default: True.
        smoothing_window: Window size for Savitzky-Golay smoothing. Default: 5.
        data_mode: Which data mode to track ('bb_ca' or 'local_latents'). Default: 'bb_ca'.
        sync_baseline: Whether to sync baseline stats across GPUs in DDP. Default: True.
        sync_snapshots: Whether to sync snapshot stats across GPUs in DDP. Default: False.
        freeze_during_baseline: If True, zero gradients before optimizer step during
            baseline capture to prevent weight updates. Default: False.

    Example:
        >>> callback = TLossMonitoringCallback(log_every_n_steps=10000)
        >>> trainer = L.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        n_buckets: int = 50,
        t_min: float = 0.0,
        t_max: float = 1.0,
        log_every_n_steps: int = 10000,
        baseline_capture_steps: int = 1000,
        normalize_by_baseline: bool = True,
        smoothing_window: int = 5,
        data_mode: str = "bb_ca",
        sync_baseline: bool = True,
        sync_snapshots: bool = False,
        freeze_during_baseline: bool = False,
    ):
        super().__init__()
        self.n_buckets = n_buckets
        self.t_min = t_min
        self.t_max = t_max
        self.log_every_n_steps = log_every_n_steps
        self.baseline_capture_steps = baseline_capture_steps
        self.normalize_by_baseline = normalize_by_baseline
        self.smoothing_window = smoothing_window
        self.data_mode = data_mode
        self.sync_baseline = sync_baseline
        self.sync_snapshots = sync_snapshots
        self.freeze_during_baseline = freeze_during_baseline

        # Bucket edges (linear scale for flow matching)
        self.bucket_edges = np.linspace(t_min, t_max, n_buckets + 1)
        self.bucket_centers = (self.bucket_edges[:-1] + self.bucket_edges[1:]) / 2

        # Per-task statistics: {task_name: {step: {"sum": array, "count": array}}}
        self.stats: dict[str, dict[int, dict[str, np.ndarray]]] = defaultdict(dict)

        # Current accumulator (reset at each snapshot)
        self.current_stats: dict[str, dict[str, np.ndarray]] = {}

        # Baseline statistics: {task_name: {"sum": array, "count": array}}
        self.baseline: dict[str, dict[str, np.ndarray]] = {}

        # Track if baseline capture is complete
        self.baseline_captured = False

    @property
    def state_key(self) -> str:
        """Unique key for checkpoint state based on data_mode."""
        return f"{self.__class__.__name__}_{self.data_mode}"

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Zero gradients during baseline capture to prevent weight updates.

        Also resets optimizer state after baseline capture to avoid corrupted
        momentum statistics from the zero-gradient steps.
        """
        if not self.freeze_during_baseline:
            return

        if trainer.global_step < self.baseline_capture_steps:
            # During baseline capture: zero gradients to prevent weight updates
            optimizer.zero_grad()
        elif trainer.global_step == self.baseline_capture_steps:
            # First step after baseline: reset optimizer state to start fresh
            optimizer.state.clear()
            logger.info(
                f"TLossMonitoringCallback ({self.data_mode}): Reset optimizer state "
                f"after baseline capture at step {trainer.global_step}"
            )

    def _get_empty_stats(self) -> dict[str, np.ndarray]:
        """Create empty statistics arrays."""
        return {
            "sum": np.zeros(self.n_buckets, dtype=np.float64),
            "count": np.zeros(self.n_buckets, dtype=np.int64),
        }

    def _accumulate(self, task: str, bucket_idx: int, loss_value: float) -> None:
        """Accumulate a loss value into current stats for a task."""
        if task not in self.current_stats:
            self.current_stats[task] = self._get_empty_stats()

        # Clamp bucket index to valid range
        if bucket_idx < 0 or bucket_idx >= self.n_buckets:
            raise ValueError(f"Bucket index {bucket_idx} is out of range [0, {self.n_buckets - 1}]")

        self.current_stats[task]["sum"][bucket_idx] += loss_value
        self.current_stats[task]["count"][bucket_idx] += 1

    def _accumulate_baseline(self, task: str, bucket_idx: int, loss_value: float) -> None:
        """Accumulate a loss value into baseline stats for a task."""
        if task not in self.baseline:
            self.baseline[task] = self._get_empty_stats()

        if bucket_idx < 0 or bucket_idx >= self.n_buckets:
            raise ValueError(f"Bucket index {bucket_idx} is out of range [0, {self.n_buckets - 1}]")

        self.baseline[task]["sum"][bucket_idx] += loss_value
        self.baseline[task]["count"][bucket_idx] += 1

    def _compute_curve(self, stats: dict[str, np.ndarray]) -> np.ndarray:
        """Compute mean loss curve from sum/count statistics."""
        with np.errstate(divide="ignore", invalid="ignore"):
            curve = stats["sum"] / stats["count"]
            # Replace NaN/inf with 0 for buckets with no samples
            curve = np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)
        return curve

    def _smooth(self, curve: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to a curve."""
        if self.smoothing_window <= 1:
            return curve

        # Need at least window_length points for smoothing
        if len(curve) < self.smoothing_window:
            return curve

        # Ensure window length is odd
        window = self.smoothing_window if self.smoothing_window % 2 == 1 else self.smoothing_window + 1

        # Polyorder must be less than window_length
        polyorder = min(3, window - 1)

        return savgol_filter(curve, window, polyorder)

    def _sync_stats(
        self, stats_dict: dict[str, dict[str, np.ndarray]], trainer: L.Trainer
    ) -> dict[str, dict[str, np.ndarray]]:
        """Reduce statistics across all ranks to rank 0.

        All ranks must call this method. After the call, only rank 0 has the
        aggregated statistics; other ranks have unchanged local stats.

        Args:
            stats_dict: Dictionary mapping task names to {"sum": array, "count": array}.
            trainer: Lightning trainer (used to get device).

        Returns:
            On rank 0: aggregated stats. On other ranks: original stats unchanged.
        """
        if not dist.is_initialized() or trainer.world_size == 1:
            return stats_dict

        device = trainer.strategy.root_device

        # First, gather all task names from all ranks to get the global set
        local_tasks = list(stats_dict.keys())
        all_tasks_gathered = [None] * trainer.world_size
        dist.all_gather_object(all_tasks_gathered, local_tasks)
        global_tasks = set()
        for task_list in all_tasks_gathered:
            global_tasks.update(task_list)

        # For each task, reduce sum and count to rank 0
        result = {}
        for task in global_tasks:
            # Get local stats or zeros if this rank doesn't have the task
            if task in stats_dict:
                local_sum = torch.tensor(stats_dict[task]["sum"], dtype=torch.float64, device=device)
                local_count = torch.tensor(stats_dict[task]["count"], dtype=torch.int64, device=device)
            else:
                local_sum = torch.zeros(self.n_buckets, dtype=torch.float64, device=device)
                local_count = torch.zeros(self.n_buckets, dtype=torch.int64, device=device)

            # Reduce to rank 0
            dist.reduce(local_sum, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(local_count, dst=0, op=dist.ReduceOp.SUM)

            if trainer.is_global_zero:
                result[task] = {
                    "sum": local_sum.cpu().numpy(),
                    "count": local_count.cpu().numpy(),
                }

        # Rank 0 gets aggregated result, others keep their local stats
        if trainer.is_global_zero:
            return result
        return stats_dict

    def _create_snapshot(self, step: int) -> None:
        """Create a snapshot of current statistics at the given step."""
        for task, stats in self.current_stats.items():
            # Initialize task in stats if needed
            if task not in self.stats:
                self.stats[task] = {}

            # Store snapshot
            self.stats[task][step] = {
                "sum": stats["sum"].copy(),
                "count": stats["count"].copy(),
            }

        # Reset current accumulator
        self.current_stats = {}

    def _log_to_wandb(self, step: int) -> None:
        """Log loss-vs-t curves to WandB as matplotlib figures.

        Logs baseline + current snapshot. WandB creates a step slider automatically.
        """
        if wandb.run is None:  # for when nolog=True
            return

        for task in set(self.current_stats.keys()) & set(self.baseline.keys()):
            # Get baseline curve
            baseline_curve = self._compute_curve(self.baseline[task])

            # Get current snapshot curve
            current_curve = self._compute_curve(self.current_stats[task])

            if self.normalize_by_baseline:
                # Only normalize where baseline has data (count > 0)
                baseline_has_data = self.baseline[task]["count"] > 0
                normalized_curve = np.ones_like(current_curve)
                normalized_curve[baseline_has_data] = (
                    current_curve[baseline_has_data] / baseline_curve[baseline_has_data]
                )
                current_curve = normalized_curve

            # Apply smoothing
            smoothed = self._smooth(current_curve)

            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            xs = self.bucket_centers

            # Plot baseline
            if self.normalize_by_baseline:
                ax.axhline(y=1.0, color="black", linestyle="--", label="baseline", alpha=0.7)
            else:
                ax.plot(xs, baseline_curve, color="black", linestyle="--", label="baseline", alpha=0.7)

            # Plot current snapshot
            ax.plot(xs, smoothed, color="blue", label=f"step {step:,}", linewidth=2)

            ax.set_xlabel("t")
            ax.set_ylabel("Loss (normalized)" if self.normalize_by_baseline else "Loss")
            ax.set_title(f"Loss vs t ({task})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

            # Set consistent y-axis range for normalized plots
            if self.normalize_by_baseline:
                ax.set_ylim(0, 1.5)

            # Log as image - WandB creates step slider automatically
            wandb.log({f"t_loss/{self.data_mode}/{task}": wandb.Image(fig)}, step=step)
            plt.close(fig)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Track loss vs t from each training batch."""

        # Extract t and loss from outputs
        t = outputs["t"][self.data_mode].cpu().numpy()  # (batch,)
        loss = outputs["losses_per_sample"][self.data_mode].cpu().numpy()  # (batch,)

        # Get task names from batch.
        task_dicts = batch.get("task_batch", [])
        task_names = [str(n) if n is not None else "unknown" for n in task_dicts.get("name", [])]

        # Ensure we have matching lengths
        if len(task_names) != len(t):
            # If mismatch, use "unknown" for all
            task_names = ["unknown"] * len(t)

        # Bucket and accumulate
        bucket_indices = np.digitize(t, self.bucket_edges) - 1

        for bi, l, task in zip(bucket_indices, loss, task_names):
            self._accumulate(task, bi, float(l))
            self._accumulate("all_tasks", bi, float(l))

        # Capture baseline during early training
        if trainer.global_step < self.baseline_capture_steps:
            for bi, l, task in zip(bucket_indices, loss, task_names):
                self._accumulate_baseline(task, bi, float(l))
                self._accumulate_baseline("all_tasks", bi, float(l))
        elif not self.baseline_captured:
            # Baseline capture just completed - sync across ranks if enabled
            # All ranks must participate in sync
            if self.sync_baseline:
                self.baseline = self._sync_stats(self.baseline, trainer)
            self.baseline_captured = True
            logger.info(
                f"TLossMonitoringCallback: Baseline captured with {len(self.baseline)} tasks "
                f"after {trainer.global_step} steps"
            )

        # Log snapshot at intervals (log first, then snapshot resets current_stats)
        if trainer.global_step > 0 and trainer.global_step % self.log_every_n_steps == 0:
            # Sync current_stats across ranks if enabled (all ranks must participate)
            if self.sync_snapshots:
                self.current_stats = self._sync_stats(self.current_stats, trainer)

            if trainer.is_global_zero:
                self._log_to_wandb(trainer.global_step)
                self._create_snapshot(trainer.global_step)

    def state_dict(self) -> dict:
        """Save callback state for checkpoint resumption.

        Returns:
            Dictionary containing persistent state.
        """
        return {
            "stats": {task: dict(steps) for task, steps in self.stats.items()},
            "current_stats": dict(self.current_stats),
            "baseline": dict(self.baseline),
            "baseline_captured": self.baseline_captured,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore callback state from checkpoint.

        Args:
            state_dict: State dictionary from a previous checkpoint.
        """
        # Restore stats
        self.stats = defaultdict(dict)
        for task, steps in state_dict.get("stats", {}).items():
            self.stats[task] = dict(steps)

        # Restore current stats
        self.current_stats = state_dict.get("current_stats", {})

        # Restore baseline
        self.baseline = state_dict.get("baseline", {})
        self.baseline_captured = state_dict.get("baseline_captured", False)

        logger.info(
            f"Restored TLossMonitoringCallback state: {len(self.baseline)} baseline tasks, "
            f"{sum(len(s) for s in self.stats.values())} total snapshots"
        )
