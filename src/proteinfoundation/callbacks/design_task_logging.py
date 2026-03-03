"""Callback for tracking and logging design task sampling statistics during training."""

from collections import defaultdict

import lightning as L
import wandb
from beartype.typing import Any
from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class DesignTaskStatsCallback(L.Callback):
    """Track and log design task sampling frequencies during training.

    This callback monitors which design tasks are sampled during training and logs
    statistics to W&B as both metrics and bar charts. Supports checkpoint resumption
    by implementing state_dict/load_state_dict.

    Features:
        - Tracks total counts per task across entire training
        - Logs running frequencies to wandb every `log_every_n` steps
        - Creates bar chart visualizations on wandb
        - Persists state in checkpoints for resumption

    Args:
        log_every_n: How often to log statistics to wandb. Default: 500.

    Example:
        >>> callback = DesignTaskStatsCallback(log_every_n=500)
        >>> trainer = L.Trainer(callbacks=[callback])
    """

    def __init__(self, log_every_n: int = 500):
        super().__init__()
        self.log_every_n = log_every_n

        # Persistent state (saved in checkpoints)
        self.task_counts = defaultdict(int)  # Total counts across all training
        self.total_examples = 0  # Total examples processed

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Track design task from batch and log statistics."""

        task_dicts = batch.get("task_batch", [])
        task_names = [str(n) for n in task_dicts.get("name", []) if n is not None]

        for task_name in task_names:
            self.task_counts[task_name] += 1
            self.total_examples += 1

        # Only log every N steps to avoid wandb bottleneck at scale
        if trainer.global_step % self.log_every_n == 0:
            self._log_batch_stats(trainer, pl_module)

    def _log_batch_stats(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log epoch-level statistics and create wandb bar chart."""
        if self.total_examples == 0:
            return

        # Compute frequencies
        total_count = sum(self.task_counts.values())
        task_stats = []

        for task_name, count in sorted(self.task_counts.items()):
            frequency = count / total_count if total_count > 0 else 0.0
            task_stats.append(
                {
                    "task": task_name,
                    "count": count,
                    "frequency": frequency,
                }
            )

        # Log as wandb table for bar chart visualization
        if trainer.is_global_zero and wandb.run is not None:
            # Create wandb table
            table = wandb.Table(
                columns=["task", "count", "frequency"],
                data=[[s["task"], s["count"], s["frequency"]] for s in task_stats],
            )

            # Log as custom bar chart
            wandb.log(
                {
                    "design_task_stats/training_task_frequency": wandb.plot.bar(
                        table, "task", "frequency", title="Design Task Sampling Frequencies"
                    ),
                }
            )

    def state_dict(self) -> dict:
        """Save callback state for checkpoint resumption.

        Returns:
            Dictionary containing persistent state that should be saved in checkpoints.
        """
        return {
            "task_counts": dict(self.task_counts),
            "total_examples": self.total_examples,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore callback state from checkpoint.

        Args:
            state_dict: State dictionary from a previous checkpoint.
        """
        self.task_counts = defaultdict(int, state_dict.get("task_counts", {}))
        self.total_examples = state_dict.get("total_examples", 0)

        logger.info(
            f"Restored DesignTaskStatsCallback state: {self.total_examples} examples, "
            f"{len(self.task_counts)} unique tasks"
        )
