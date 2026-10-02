"""Track observed dataset and design-task frequencies across training ranks."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch.distributed as dist
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save_dataset_task_heatmap(counts: Counter[tuple[str, str]], path: Path) -> list[list[Any]]:
    """Save an annotated heatmap with dataset and task marginal counts."""
    datasets = sorted({dataset for dataset, _ in counts})
    tasks = sorted({task for _, task in counts})
    matrix = np.array([[counts[dataset, task] for task in tasks] for dataset in datasets], dtype=np.int64)
    totals = matrix.sum(axis=1)
    task_totals = matrix.sum(axis=0)
    fractions = matrix / totals[:, None]
    overall = int(totals.sum())
    rows = [
        [
            dataset,
            task,
            int(matrix[i, j]),
            float(fractions[i, j]),
            int(totals[i]),
            float(totals[i] / overall),
            int(task_totals[j]),
            float(task_totals[j] / overall),
        ]
        for i, dataset in enumerate(datasets)
        for j, task in enumerate(tasks)
    ]
    display_counts = np.block([[matrix, totals[:, None]], [task_totals[None, :], np.array([[overall]])]])
    display_fractions = display_counts / overall
    display_fractions[:-1, :-1] = fractions
    colors = np.pad(fractions, ((0, 1), (0, 1)), constant_values=np.nan)
    figure = Figure(figsize=(max(9, 1.4 * (len(tasks) + 1)), max(4, 0.8 * (len(datasets) + 1) + 2.5)))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.set_facecolor("#edf1f5")
    plot = axes.imshow(np.ma.masked_invalid(colors), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axes.set_xticks(range(len(tasks) + 1), labels=[*tasks, "All tasks"], rotation=40, ha="right")
    axes.set_yticks(range(len(datasets) + 1), labels=[*datasets, "All datasets"])
    axes.axhline(len(datasets) - 0.5, color="white", linewidth=3)
    axes.axvline(len(tasks) - 0.5, color="white", linewidth=3)
    for i in range(len(datasets) + 1):
        for j in range(len(tasks) + 1):
            marginal = i == len(datasets) or j == len(tasks)
            axes.text(
                j,
                i,
                f"{display_counts[i, j]:,}\n{display_fractions[i, j]:.1%}",
                ha="center",
                va="center",
                color="white" if not marginal and display_fractions[i, j] > 0.55 else "black",
                fontsize=9,
                fontweight="bold" if marginal else "normal",
            )
    axes.set_title(f"Observed training examples by dataset and task (n={overall:,})")
    axes.set_xlabel("Cells: % within dataset · Totals: % of all examples")
    figure.colorbar(plot, ax=axes, label="Fraction within dataset")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return rows


class DatasetTaskStatsCallback(L.Callback):
    """Log cumulative dataset/task counts across all ranks every ``log_every_n`` steps."""

    def __init__(self, log_every_n: int = 500) -> None:
        super().__init__()
        if log_every_n < 1:
            raise ValueError("log_every_n must be positive")
        self.log_every_n = log_every_n
        self.counts: Counter[tuple[str, str]] = Counter()
        self.pending: Counter[tuple[str, str]] = Counter()
        self._last_logged_step = -1

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Count the delivered examples, including any successful fallback substitutions."""
        tasks = batch.get("task_batch", {}).get("name", [])
        datasets = batch.get("dataset_name_batch", ["unknown"] * len(tasks))
        for dataset, task in zip(datasets, tasks, strict=True):
            self.pending[dataset or "unknown", task or "unknown"] += 1
        step = trainer.global_step
        if step > 0 and step % self.log_every_n == 0 and step != self._last_logged_step:
            self._log_batch_stats(trainer)

    def _sync_counts(self) -> None:
        """Merge only new local counts so restored global totals are never multiplied."""
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, dict(self.pending))
            for counts in gathered:
                self.counts.update(counts)
        else:
            self.counts.update(self.pending)
        self.pending.clear()

    def _log_batch_stats(self, trainer: L.Trainer) -> None:
        self._sync_counts()
        self._last_logged_step = trainer.global_step
        if not trainer.is_global_zero or not self.counts:
            return
        output = Path(trainer.default_root_dir) / "dataset_stats"
        output.mkdir(parents=True, exist_ok=True)
        image_path = output / "dataset_task_heatmap.png"
        rows = save_dataset_task_heatmap(self.counts, image_path)
        columns = [
            "dataset",
            "task",
            "count",
            "fraction_within_dataset",
            "dataset_count",
            "dataset_fraction",
            "task_count",
            "task_fraction",
        ]
        (output / "dataset_task_counts.json").write_text(
            json.dumps({"step": trainer.global_step, "columns": columns, "data": rows}, indent=2)
        )
        if wandb.run is not None:
            wandb.log(
                {
                    "trainer/global_step": trainer.global_step,
                    "dataset_stats/dataset_task_heatmap": wandb.Image(str(image_path)),
                    "dataset_stats/dataset_task_counts": wandb.Table(columns=columns, data=rows),
                }
            )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Flush the final partial logging window of each epoch."""
        self._log_batch_stats(trainer)

    def on_save_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict) -> None:
        """Flush pending counts on all ranks before Lightning writes the checkpoint."""
        self._sync_counts()
        checkpoint["callbacks"][self.state_key] = self.state_dict()

    def state_dict(self) -> dict:
        """Serialize globally synchronized counts."""
        return {
            "dataset_task_counts": [[dataset, task, count] for (dataset, task), count in sorted(self.counts.items())]
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore joint counts without multiplying them by the number of ranks."""
        self.counts = Counter(
            {(dataset, task): count for dataset, task, count in state_dict.get("dataset_task_counts", [])}
        )
        self.pending.clear()
        self._last_logged_step = -1
