"""Coordinate full Python garbage collection across training ranks."""

import gc
import time
from typing import Any

import lightning as L


class CoordinatedGC(L.Callback):
    """Collect generation two at common training-batch boundaries.

    Automatic young-generation collection remains enabled. This callback owns
    process-wide GC thresholds during training, including those used by loader
    threads. Validation and cleanup restore the preceding process settings.
    Validation does not reset the batch counter or force an extra collection.
    All distributed ranks must run the callback and consume equal batch counts.

    Args:
        interval_batches: Completed training batches between full collections.
            Counts batches rather than optimizer steps, including under manual
            optimization or gradient accumulation. Defaults to 128.
    """

    def __init__(self, interval_batches: int = 128) -> None:
        if isinstance(interval_batches, bool) or not isinstance(interval_batches, int) or interval_batches < 1:
            raise ValueError("interval_batches must be a positive integer")
        self.interval_batches = interval_batches
        self.last_collection_seconds = 0.0
        self._batches_since_collection = 0
        self._training_started = False
        self._original_thresholds: tuple[int, int, int] | None = None

    def _collect(self, trainer: L.Trainer) -> None:
        start = time.perf_counter()
        trainer.strategy.barrier("coordinated_gc")
        gc.collect(2)
        self.last_collection_seconds = time.perf_counter() - start

    def _activate(self, trainer: L.Trainer, *, collect: bool = False) -> None:
        if self._original_thresholds is not None:
            return
        if not gc.isenabled():
            raise RuntimeError("CoordinatedGC requires automatic young-generation GC; enable GC before training")
        self._original_thresholds = gc.get_threshold()
        gc.set_threshold(*self._original_thresholds[:2], 1_000_000_000)
        try:
            if collect:
                self._collect(trainer)
                if self._batches_since_collection >= self.interval_batches:
                    self._batches_since_collection = 0
        except BaseException:
            self._restore()
            raise

    def _restore(self) -> None:
        if self._original_thresholds is not None:
            gc.set_threshold(*self._original_thresholds)
            gc.enable()
            self._original_thresholds = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._training_started = True
        self._activate(trainer, collect=True)

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        if self._batches_since_collection >= self.interval_batches:
            self._collect(trainer)
            self._batches_since_collection = 0

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        self._batches_since_collection += 1

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._restore()

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._training_started:
            self._activate(trainer)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._training_started = False
        self._restore()

    def on_exception(self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException) -> None:
        self._training_started = False
        self._restore()

    def teardown(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        self._training_started = False
        self._restore()

    def state_dict(self) -> dict[str, Any]:
        return {"batches_since_collection": self._batches_since_collection}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        count = state_dict.get("batches_since_collection", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("CoordinatedGC checkpoint batches_since_collection must be a nonnegative integer")
        self._batches_since_collection = count
