import logging
import os
import warnings
from os import PathLike
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from biotite.structure import AtomArray
from lightning.fabric import seed_everything
from omegaconf import OmegaConf

from foundry.inference_engines.checkpoint_registry import REGISTERED_CHECKPOINTS
from foundry.utils.ddp import RankedLogger, set_accelerator_based_on_availability
from foundry.utils.logging import (
    configure_minimal_inference_logging,
    print_config_tree,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def merge(cfg, overrides: dict):
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


class BaseInferenceEngine:
    """
    Base inference engine.
    Separates model setup (expensive, once) from inference (can run multiple times).

    Supports deferred weight loading via ``load_weights=False``, which allows
    multiple engine instances to coexist while only one holds model weights in
    GPU memory at a time::

        engine_a = SomeEngine("ckpt_a.pt", load_weights=False)
        engine_b = SomeEngine("ckpt_b.pt", load_weights=False)

        engine_a.load_weights()
        results_a = engine_a.run(inputs_a)
        engine_a.unload_weights()

        engine_b.load_weights()
        results_b = engine_b.run(inputs_b)
    """

    def __init__(
        self,
        ckpt_path: PathLike,
        num_nodes: int = 1,
        devices_per_node: int = 1,
        # Config overrides
        transform_overrides={},
        inference_sampler_overrides={},
        trainer_overrides={},
        # Weight loading control
        load_weights: bool = True,
        # Debug
        verbose: bool = False,
        # Multi-seed support
        model_seeds: list[int] | None = None,
        # Backward compatibility (deprecated)
        seed: int | None = None,
    ):
        """Initialize inference engine and load model.

        Model config is loaded from checkpoint and overridden with parameters provided here.

        Args:
          ckpt_path: Path to model checkpoint.
          model_seeds: List of random seeds. If empty or None, uses external RNG state.
            Multiple seeds will run inference for each seed. Defaults to ``None``.
          seed: (Deprecated) Use model_seeds instead. Converted to model_seeds=[seed].
          num_nodes: Number of nodes for distributed inference. Defaults to ``1``.
          devices_per_node: Number of devices per node. Defaults to ``1``.
          load_weights: Whether to load model weights during initialization.
              If False, call ``load_weights()`` before first inference. Defaults to ``True``.
          verbose: If True, show detailed logging and config trees. Defaults to ``False``.
        """
        if not verbose:
            configure_minimal_inference_logging()

        # Set attrs
        self.initialized_ = False
        self._weights_loaded = False
        self._external_distributed_context = False
        self.trainer = None
        self.pipeline = None
        self.verbose = verbose
        self._load_weights = load_weights

        # Resolve checkpoint path
        if "." not in str(ckpt_path):
            # Assume registered model
            name = str(ckpt_path)
            assert (
                name in REGISTERED_CHECKPOINTS
            ), "Checkpoint provided not and not in registered checkpoints"
            ckpt = REGISTERED_CHECKPOINTS[name]

            ckpt_path = ckpt.get_default_path()
            ranked_logger.info(
                "Using checkpoint from default installation directory, got: {}".format(
                    str(ckpt_path)
                )
            )
            assert os.path.exists(
                ckpt_path
            ), "Invalid checkpoint: {}. And could not find checkpoint in default installation location: {}".format(
                name, ckpt_path
            )
        self.ckpt_path = Path(ckpt_path).resolve()

        # Handle backward compatibility: convert old seed parameter to model_seeds
        if seed is not None and model_seeds is not None:
            raise ValueError(
                "Cannot specify both 'seed' (deprecated) and 'model_seeds'. "
                "Use 'model_seeds' for multi-seed support."
            )
        if seed is not None:
            warnings.warn(
                "'seed' parameter is deprecated. Use 'model_seeds=[seed]' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            model_seeds = [seed]

        # Store model_seeds (None or list) - seeding happens before each forward pass
        self.model_seeds = model_seeds
        if model_seeds:
            ranked_logger.info(f"Model seeds configured: {model_seeds}")
        else:
            ranked_logger.info("No model seeds configured - using external RNG state")

        # Stored for later;
        self.transform_overrides = transform_overrides
        self.overrides: dict[str, Any] = {}

        base_overrides = {
            "trainer.seed": model_seeds[0] if model_seeds else None,
            "trainer.metrics": None,  # Must be None, not {} - empty dict doesn't override nested configs
            "trainer.loss": None,
            "trainer.num_nodes": num_nodes,
            "trainer.devices_per_node": devices_per_node,
        }
        for key, value in base_overrides.items():
            self._assign_override(key, value)

        for key, value in trainer_overrides.items():
            self._assign_override(f"trainer.{key}", value)

        for key, value in inference_sampler_overrides.items():
            self._assign_override(f"model.net.inference_sampler.{key}", value)

    ###################################################################################
    # Required subclasss methods
    ###################################################################################

    def initialize(self):
        if self.initialized_:
            return getattr(self, "cfg", None)

        # Detect external distributed context before any fabric/DDP setup
        self._external_distributed_context = torch.distributed.is_initialized()
        if self._external_distributed_context:
            ranked_logger.info(
                "Existing distributed context detected - overriding strategy to 'auto' "
                "to avoid nested DDP"
            )
            self._assign_override("trainer.strategy", "auto")

        # Load checkpoint and config
        ranked_logger.info(f"Loading checkpoint from {Path(self.ckpt_path).resolve()}...")
        checkpoint = torch.load(self.ckpt_path, "cpu", weights_only=False)
        cfg = self._override_checkpoint_config(checkpoint["train_cfg"])

        # Load pipeline first before trainer/model
        self._construct_pipeline(cfg)
        self._construct_trainer(cfg, checkpoint=checkpoint)

        ranked_logger.info("Model loaded and ready for inference.")
        self.initialized_ = True
        return cfg

    def run(
        self,
        inputs: (Dict[str, dict] | AtomArray | list[AtomArray] | PathLike | list[PathLike]),
        *_,
    ) -> dict[str, dict] | None:
        self.initialize()
        raise NotImplementedError("Subclasses must implement inference logic in `run` method.")

    ###################################################################################
    # Weight management
    ###################################################################################

    def load_weights(self):
        """Load model weights from checkpoint into GPU memory.

        If ``initialize()`` hasn't been called yet, calls it first to build the
        trainer and pipeline. Weights are then loaded into the existing model
        structure.
        """
        if self._weights_loaded:
            ranked_logger.debug("Weights already loaded")
            return

        if not self.initialized_:
            self.initialize()
            return

        # Load weights into existing model structure (after init with
        # load_weights=False, or after a previous unload_weights() call)
        self._load_model_weights(getattr(self, "_stored_checkpoint", None) or self.ckpt_path)

    def unload_weights(self):
        """Unload model weights from GPU to free memory.

        The engine remains usable — call ``load_weights()`` to reload.
        This is useful for freeing GPU memory between inference runs when
        multiple engine instances share the same device.
        """
        if not self._weights_loaded:
            ranked_logger.debug("Weights not loaded, nothing to unload")
            return

        ranked_logger.info("Unloading model weights from GPU...")

        if self.trainer is not None and self.trainer.state.get("model") is not None:
            del self.trainer.state["model"]
            self.trainer.state["model"] = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._weights_loaded = False
        ranked_logger.info("Model weights unloaded, GPU memory freed")

    ###################################################################################
    # Util methods
    ###################################################################################

    def _override_checkpoint_config(self, cfg):
        cfg = merge(cfg, self.overrides)
        cfg = set_accelerator_based_on_availability(cfg)
        return cfg

    def _construct_trainer(self, cfg, checkpoint=None):
        """Build trainer, model structure, and optionally load weights.

        When ``self._load_weights`` is False, the trainer and model graph are
        constructed but checkpoint weights are not loaded — call
        ``load_weights()`` later to populate them.

        When an external distributed context is detected (``torch.distributed``
        already initialized), ``fabric.launch()`` is skipped to avoid nested DDP.
        """
        # Store checkpoint for later use by load_weights()/unload_weights()
        self._stored_checkpoint = checkpoint

        ranked_logger.info("Instantiating trainer...")
        if self.verbose:
            print_config_tree(cfg.trainer, resolve=True, title="INFERENCE TRAINER CONFIGURATION")
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            _convert_="partial",
            _recursive_=False,
        )

        # Setup model
        ranked_logger.info("Setting up model...")
        if self._external_distributed_context:
            ranked_logger.info(
                "Distributed process group already initialized - skipping fabric.launch()"
            )
        else:
            trainer.fabric.launch()

        trainer.initialize_or_update_trainer_state(
            {"train_cfg": cfg}
        )  # config from training stores net params
        trainer.construct_model()

        if self._load_weights:
            self._load_model_weights(checkpoint or self.ckpt_path, trainer=trainer)
        else:
            ranked_logger.info(
                "Deferring weight loading (load_weights=False). "
                "Call load_weights() before inference."
            )

        self.trainer = trainer

    def _load_model_weights(self, checkpoint, trainer=None):
        """Load checkpoint weights into the trainer model and set to eval mode."""
        trainer = trainer or self.trainer
        ranked_logger.info("Loading model weights from checkpoint...")

        if trainer.state.get("model") is None:
            trainer.construct_model()

        trainer.load_checkpoint(checkpoint=checkpoint)

        # Ensure optimizer isn't loaded
        trainer.state["optimizer"] = None
        trainer.state["train_cfg"].model.optimizer = None
        trainer.setup_model_optimizers_and_schedulers()
        trainer.state["model"].eval()
        self._weights_loaded = True

    def _assign_override(self, dotted_key: str, value: Any) -> None:
        """Assign ``value`` into ``self.overrides`` using a dotted path."""
        target = self.overrides
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def _construct_pipeline(self, cfg):
        """
        Sets attr self.pipeline
        """
        # Construct pipeline
        ranked_logger.info("Building Transform pipeline...")
        first_val_dataset_key, first_val_dataset = next(iter(cfg.datasets.val.items()))
        ranked_logger.info(f"Using settings from validation dataset: {first_val_dataset_key}.")
        transform = first_val_dataset.dataset.transform
        transform = merge(transform, self.transform_overrides)

        if self.verbose:
            print_config_tree(
                transform,
                resolve=True,
                title="INFERENCE TRANSFORM PIPELINE",
            )

        self.pipeline = hydra.utils.instantiate(transform)

    def _seed_for_inference(self, seed: int | None) -> None:
        """Set random seed before inference forward pass.

        Called by subclasses before each forward pass when using multi-seed inference.

        Args:
            seed: Random seed to set. If None, no seeding is performed.
        """
        if seed is not None:
            ranked_logger.info(f"Seeding everything with seed={seed}...")
            seed_everything(seed, verbose=True)

    # aliases for run
    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    # for use as a context manager: e.g. `with BaseInferenceEngine(...) as engine:` to automatically cleanup
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.trainer = None
        self.pipeline = None
        self.initialized_ = False
        self._weights_loaded = False
