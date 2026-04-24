import contextlib
from collections.abc import Generator

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn


class SimpleEMA(Callback):
    """Exponential Moving Average callback with robust state management.

    Maintains a shadow copy of model weights using exponential moving average.
    Uses PyTorch's AveragedModel for reliability. Automatically swaps to EMA
    weights during validation and provides context managers for checkpoint saving.

    This implementation avoids the bugs present in weight-swapping approaches by:
    - Keeping EMA weights in a separate model instance
    - Using context managers to ensure correct weights during saves
    - Tracking state to handle nested operations (validation + checkpointing)

    Args:
        decay: EMA decay rate. Common values: 0.9999, 0.999. Higher values give
            more weight to historical parameters.
        update_every_n: Update EMA every N training steps. Defaults to 1 (every step).
            Use higher values (e.g., 10) to reduce computational overhead.

    Example:
        Basic usage::

            trainer = Trainer(callbacks=[SimpleEMA(decay=0.9999)])

        Update every 10 steps::

            trainer = Trainer(callbacks=[SimpleEMA(decay=0.9999, update_every_n=10)])
    """

    def __init__(self, decay: float = 0.9999, update_every_n: int = 1) -> None:
        if not (0 <= decay <= 1):
            raise ValueError(f"EMA decay must be between 0 and 1, got {decay}")
        if update_every_n < 1:
            raise ValueError(f"update_every_n must be >= 1, got {update_every_n}")

        self.decay = decay
        self.update_every_n = update_every_n
        self.ema_model: AveragedModel | None = None

        # Track which weights are currently loaded in the model
        self._ema_weights_active = False
        self._original_state_dict = None

    def _ensure_ema_initialized(self, pl_module: L.LightningModule) -> None:
        """Lazily initialize EMA model if not already initialized.

        Creates an AveragedModel wrapping the training model and optionally
        loads saved EMA state from a checkpoint.

        Args:
            pl_module: The LightningModule to create EMA for.
        """
        if self.ema_model is None:
            self.ema_model = AveragedModel(pl_module, avg_fn=get_ema_avg_fn(self.decay), use_buffers=True)

            # Load saved state if resuming from checkpoint
            if hasattr(self, "_loaded_ema_state"):
                self.ema_model.load_state_dict(self._loaded_ema_state)
                delattr(self, "_loaded_ema_state")
                rank_zero_info("Restored EMA state from checkpoint")
            else:
                rank_zero_info(
                    f"Initialized fresh EMA model (decay={self.decay}, update_every_n={self.update_every_n})"
                )

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Ensure EMA is initialized when training starts."""
        self._ensure_ema_initialized(pl_module)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        """Update EMA after training batch.

        Updates every update_every_n steps. Use update_every_n > 1 to reduce
        computational overhead at the cost of slightly less smooth averaging.
        """
        if trainer.global_step % self.update_every_n == 0:
            self.ema_model.update_parameters(pl_module)

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Swap to EMA weights for validation.

        Temporarily replaces the model's weights with EMA weights so that
        validation metrics reflect the averaged model's performance.
        """
        self._ensure_ema_initialized(pl_module)

        if not self._ema_weights_active:
            # Deep copy to avoid reference issues
            self._original_state_dict = {k: v.clone() for k, v in pl_module.state_dict().items()}
            pl_module.load_state_dict(self.ema_model.module.state_dict())
            self._ema_weights_active = True

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Restore training weights after validation.

        Swaps back to the original training weights so that gradient updates
        continue on the correct parameters.
        """
        if self._ema_weights_active:
            pl_module.load_state_dict(self._original_state_dict)
            self._original_state_dict = None
            self._ema_weights_active = False

    @contextlib.contextmanager
    def use_training_weights(self, pl_module: L.LightningModule) -> Generator[None, None, None]:
        """Context manager ensuring training weights are active.

        Temporarily switches to training weights if EMA is active,
        and updates state tracking accordingly. Useful when saving
        training checkpoints during validation.

        Args:
            pl_module: The LightningModule to manage.

        Yields:
            None - weights are swapped on entry, restored on exit.

        Example:
            Save training checkpoint even if in validation::

                with ema_callback.use_training_weights(pl_module):
                    torch.save(pl_module.state_dict(), "training.ckpt")
        """
        was_ema_active = self._ema_weights_active

        if was_ema_active:
            # Currently showing EMA - temporarily restore training
            # Deep copy to avoid reference issues
            ema_state = {k: v.clone() for k, v in pl_module.state_dict().items()}
            pl_module.load_state_dict(self._original_state_dict)
            self._ema_weights_active = False

        try:
            yield
        finally:
            if was_ema_active:
                # Restore EMA weights and flag
                pl_module.load_state_dict(ema_state)
                self._ema_weights_active = True

    @contextlib.contextmanager
    def use_ema_weights(self, pl_module: L.LightningModule) -> Generator[None, None, None]:
        """Context manager ensuring EMA weights are active.

        Temporarily switches to EMA weights if training is active,
        and updates state tracking accordingly. Useful when saving
        EMA checkpoints during training.

        Args:
            pl_module: The LightningModule to manage.

        Yields:
            None - weights are swapped on entry, restored on exit.

        Example:
            Save EMA checkpoint during training::

                with ema_callback.use_ema_weights(pl_module):
                    torch.save(pl_module.state_dict(), "ema.ckpt")
        """
        was_ema_active = self._ema_weights_active

        if not was_ema_active:
            # Currently showing training - temporarily load EMA
            # Deep copy to avoid reference issues
            training_state = {k: v.clone() for k, v in pl_module.state_dict().items()}
            self._original_state_dict = training_state
            pl_module.load_state_dict(self.ema_model.module.state_dict())
            self._ema_weights_active = True

        try:
            yield
        finally:
            if not was_ema_active:
                # Restore training weights and flag
                pl_module.load_state_dict(training_state)
                self._ema_weights_active = False
                self._original_state_dict = None

    def state_dict(self) -> dict:
        """Save EMA state for checkpointing.

        Returns:
            Dictionary containing EMA model state, decay, and update_every_n.
            This gets saved in checkpoint['callbacks']['SimpleEMA'].
        """
        return {
            "ema_state_dict": self.ema_model.state_dict() if self.ema_model else None,
            "decay": self.decay,
            "update_every_n": self.update_every_n,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint.

        Called by Lightning's checkpoint restoration. Stores the EMA state
        for lazy initialization since we don't have access to pl_module here.

        Note: We intentionally do NOT load decay and update_every_n from the
        checkpoint. These are set from the current config, allowing you to
        change EMA hyperparameters when resuming training.

        Args:
            state_dict: Dictionary saved by state_dict() method.
        """
        # Only load the EMA model weights, not the hyperparameters
        if state_dict.get("ema_state_dict"):
            self._loaded_ema_state = state_dict["ema_state_dict"]


class EMAModelCheckpoint(ModelCheckpoint):
    """Checkpoint callback with optional separate EMA checkpoint saving.

    Extends ModelCheckpoint to optionally save both training and EMA checkpoints.
    Robustly handles saving during validation by using context managers to ensure
    correct weights are saved regardless of current model state.

    Checkpoint structure:
        - Training checkpoint (``model.ckpt``): Contains training weights in state_dict
          and EMA state in callbacks section.
        - EMA checkpoint (``model-ema.ckpt``): Contains EMA weights in state_dict,
          ready for immediate inference.

    Args:
        save_ema_separately: If ``True``, saves both training and EMA checkpoints.
            If ``False``, only saves single checkpoint with EMA in callback state.
            Defaults to ``False``.
        **kwargs: Additional arguments passed to ModelCheckpoint.

    Example:
        Frequent last checkpoint (single file)::

            EMAModelCheckpoint(
                save_last=True,
                filename="last",
                save_ema_separately=False,  # Single file for resume
            )

        Validation checkpoints (dual files)::

            EMAModelCheckpoint(
                every_n_epochs=1,
                save_top_k=3,
                monitor="val/loss",
                save_ema_separately=True,  # Two files for evaluation
            )
    """

    def __init__(self, save_ema_separately: bool = False, **kwargs: bool | int | float | str | None) -> None:
        super().__init__(**kwargs)
        self.save_ema_separately = save_ema_separately

    @property
    def state_key(self) -> str:
        """Generate unique state key including filename and save_last.

        ModelCheckpoint's default state_key doesn't include filename or save_last,
        which can cause conflicts when using multiple checkpoint callbacks with
        the same monitor/mode/every_n_epochs but different filenames.
        """
        base_key = super().state_key
        # Include filename and save_last to differentiate callbacks
        return f"{base_key}_fn={self.filename}_last={self.save_last}"

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        """Save checkpoint(s) with correct weights regardless of current state.

        Always saves the training checkpoint with training weights, even if we're
        currently in validation with EMA weights active. Optionally saves a separate
        EMA checkpoint if save_ema_separately is True.

        Args:
            trainer: The Trainer instance.
            filepath: Path where checkpoint should be saved.
        """

        # '=' in filenames adds complications for Hydra
        filepath = filepath.replace("=", "_")

        ema_callback = self._find_ema_callback(trainer)

        if ema_callback is None:
            # No EMA - just save normally
            super()._save_checkpoint(trainer, filepath)
            return

        pl_module = trainer.lightning_module

        # Always save training checkpoint with training weights
        # (even if we're currently in validation with EMA weights active)
        with ema_callback.use_training_weights(pl_module):
            super()._save_checkpoint(trainer, filepath)

        # Optionally save separate EMA checkpoint
        if self.save_ema_separately:
            ema_filepath = filepath.replace(".ckpt", "-ema.ckpt")
            if self.verbose:
                rank_zero_info(f"Saving EMA checkpoint: {ema_filepath}")

            # Save with EMA weights active
            # (even if we're currently in training with training weights active)
            with ema_callback.use_ema_weights(pl_module):
                super()._save_checkpoint(trainer, ema_filepath)

    def _find_ema_callback(self, trainer: L.Trainer) -> SimpleEMA | None:
        """Find the SimpleEMA callback in trainer callbacks.

        Args:
            trainer: The Trainer instance.

        Returns:
            The SimpleEMA callback if found, None otherwise.
        """
        for callback in trainer.callbacks:
            if isinstance(callback, SimpleEMA):
                return callback
        return None
