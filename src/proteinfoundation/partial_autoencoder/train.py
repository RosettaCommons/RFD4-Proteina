#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/latent_exec.sh" "$0" "$@"'

import json
import os
from pathlib import Path

import hydra
import lightning as L
import rootutils
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from foundry.utils.ddp import RankedLogger
from foundry.utils.env import load_dotenv
from foundry.utils.logging import print_config_tree, silence_warnings
from foundry.utils.weights import (
    WeightLoadingConfig,
    freeze_parameters_with_config,
    load_weights_with_policies,
)

# Setup root dir and environment variables (more info: https://github.com/ashleve/rootutils)
# NOTE: Sets the `PROJECT_ROOT` environment variable to the root directory of the project (where `.project-root` is located)  # noqa: E501
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Absolute, so the entrypoint works regardless of where it sits under src/.
_config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")

from foundry.utils.datasets import (  # noqa: E402
    assemble_distributed_loader,
    assemble_val_loader_dict,
    recursively_instantiate_datasets_and_samplers,
)
from foundry.utils.instantiators import instantiate_callbacks  # noqa: E402
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder  # noqa: E402
from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt  # noqa: E402
from proteinfoundation.utils.rank_seed_callback import RankSeedCallback  # noqa: E402
from proteinfoundation.utils.simple_ema import EMAModelCheckpoint  # noqa: E402
from proteinfoundation.utils.torch_utils import apply_torch_deterministic  # noqa: E402

logger = RankedLogger(__name__, rank_zero_only=True)

RELATIVE_CONFIG_PATH = "../../configs"
"""Path to configs relative to this file."""

load_dotenv()


class DelayedValidationCallback(Callback):
    """Delay validation until a given epoch by temporarily disabling val batches."""

    def __init__(self, start_epoch: int, limit_val_batches: int | float):
        super().__init__()
        self.start_epoch = start_epoch
        self.limit_val_batches = limit_val_batches

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.current_epoch < self.start_epoch:
            trainer.limit_val_batches = 0
        else:
            trainer.limit_val_batches = self.limit_val_batches


@rank_zero_only
def create_dir(ckpt_path_store: str, parents: bool = True, exist_ok: bool = True) -> None:
    """Create directory for storing checkpoints only with rank zero process."""
    Path(ckpt_path_store).mkdir(parents=parents, exist_ok=exist_ok)


def get_run_dirs(cfg_exp: DictConfig) -> tuple[str, str, str]:
    """Get root directory for run and directory to store checkpoints.

    Uses ``out_dir`` from config if set, otherwise defaults to ``./store/<run_name>``.

    Returns:
        run_name: Name of the run.
        root_run: Root directory for the run.
        ckpt_path_store: Directory to store checkpoints.
    """
    run_name = cfg_exp.run_name_
    logger.info(f"Job name: {run_name}")

    out_dir = cfg_exp.get("out_dir", None)
    if out_dir:
        root_run = out_dir
    else:
        root_run = os.path.join(".", "store", run_name)
    logger.info(f"Root run: {root_run}")

    ckpt_path_store = os.path.join(root_run, "checkpoints")
    logger.info(f"Checkpoints directory: {ckpt_path_store}")
    return run_name, root_run, ckpt_path_store


def get_training_precision(cfg_exp: DictConfig) -> str:
    """Get and set correct training precision.

    Supports three precision modes:
    - "32": Full FP32 precision
    - "bf16-mixed": BF16 mixed precision (default)
    - "fp8": FP8 precision via Transformer Engine (requires TE installed)

    The precision can be set via the `precision` config option. For backward
    compatibility, `force_precision_f32: True` is equivalent to `precision: "32"`.

    Returns:
        Lightning precision string ("32" or "bf16-mixed").
    """
    # Check for new precision config option first
    precision = cfg_exp.get("precision", None)

    # Legacy support for force_precision_f32
    if precision is None:
        if cfg_exp.get("force_precision_f32", False):
            precision = "32"
        else:
            precision = "bf16-mixed"

    if precision == "fp8":
        logger.info("Using FP8 precision via Transformer Engine (base: bf16-mixed)")
        torch.set_float32_matmul_precision("medium")
        # FP8 uses bf16-mixed as base precision; TE autocast handles FP8 in the model
        return "bf16-mixed"
    elif precision == "32":
        logger.info("Using full precision (FP32)")
        torch.set_float32_matmul_precision("high")
        return "32"
    else:
        logger.info(f"Using {precision} precision")
        torch.set_float32_matmul_precision("medium")
    return precision


def get_model_n_ckpt_resume(cfg_exp: DictConfig, ckpt_path_store: str) -> tuple[AutoEncoder, str | None]:
    """Load the model and the checkpoint to start training from.

    This could be just a set of parameters (pretrain_ckpt_path) or resuming
    training (last). It also handles LoRA layers if requested.
    """
    # Detect checkpoint BEFORE model init so we can seed first when starting fresh.
    # This ensures reproducible random weight initialization across runs.
    last_ckpt_name = fetch_last_ckpt(ckpt_path_store)
    last_ckpt_path = os.path.join(ckpt_path_store, last_ckpt_name) if last_ckpt_name is not None else None
    logger.info(f"Last checkpoint: {last_ckpt_path}")

    # Seed before model init for fresh runs so random weight initialization is reproducible.
    # (Resuming from a checkpoint loads deterministic weights, so no seed is needed here.)
    if last_ckpt_path is None:
        logger.info(f"Not starting from existing checkpoint; seeding with seed: {cfg_exp.seed}")
        L.seed_everything(cfg_exp.seed, workers=False, verbose=True)

    model = AutoEncoder(cfg_exp)

    # If LoRA is turned on, replace Linear with LoRA layers
    if cfg_exp.get("lora") and cfg_exp.lora.get("r"):
        try:
            import loralib as lora

            from proteinfoundation.utils.lora_utils import replace_lora_layers

            lora_r = cfg_exp.lora.r
            lora_alpha = cfg_exp.lora.lora_alpha
            lora_dropout = cfg_exp.lora.lora_dropout
            logger.info(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

            # Get exclude_keys from config (if specified)
            exclude_keys = tuple(cfg_exp.lora.get("exclude_keys", []))

            replace_lora_layers(model, lora_r, lora_alpha, lora_dropout, exclude_keys=exclude_keys)
            lora.mark_only_lora_as_trainable(model, bias=cfg_exp.lora.train_bias)
            logger.info("LoRA layers successfully applied")
        except ImportError:
            logger.info("WARNING: LoRA is enabled in config but loralib is not installed.")
            logger.info("To use LoRA, install loralib: pip install loralib")

    # If this is the first run for fine-tuning, load pre-trained checkpoint
    pretrain_ckpt_path = cfg_exp.get("pretrain_ckpt_path", None)
    if last_ckpt_path is None and pretrain_ckpt_path is not None:
        logger.info(f"Loading from pre-trained checkpoint path {pretrain_ckpt_path}")
        ckpt = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)

        weight_loading_cfg = cfg_exp.get("weight_loading", None)
        if weight_loading_cfg is not None:
            # Policy-based loading: instantiate WeightLoadingConfig via Hydra and apply
            wl_config: WeightLoadingConfig = hydra.utils.instantiate(weight_loading_cfg)
            logger.info(
                f"Using policy-based weight loading: default={wl_config.default_policy}, "
                f"fallback={wl_config.fallback_policy}, "
                f"per-param overrides={len(wl_config.param_policies)}"
            )
            updated_state = load_weights_with_policies(model, ckpt["state_dict"], wl_config)
            model.load_state_dict(updated_state, strict=True)
        else:
            # Legacy behaviour: silently drop missing/extra keys
            model.load_state_dict(ckpt["state_dict"], strict=False)

        # Optional parameter freezing (applied after weight loading)
        param_freezing_cfg = cfg_exp.get("parameter_freezing", None)
        if param_freezing_cfg is not None:
            pf_config = hydra.utils.instantiate(param_freezing_cfg)
            logger.info(
                f"Applying parameter freezing (freeze_by_default={pf_config.freeze_by_default}, "
                f"overrides={len(pf_config.param_policies)})"
            )
            freeze_parameters_with_config(model, pf_config)

    return model, last_ckpt_path


def setup_ckpt(cfg_exp: DictConfig, ckpt_path_store: str) -> list[EMAModelCheckpoint]:
    """Create checkpointing callbacks and directory to store checkpoints.

    Creates two checkpoint callbacks (epoch-based to ensure clean resume):
    - checkpoint_callback_last: Saves last.ckpt at epoch boundaries for resume
    - checkpoint_callback: Top-K checkpoints at epoch boundaries for evaluation
    """
    args_ckpt_last = {
        "dirpath": ckpt_path_store,
        "save_weights_only": False,
        "filename": "ignore",
        "every_n_epochs": cfg_exp.log.last_ckpt_every_n_epochs,
        "save_last": True,
        "save_ema_separately": False,  # Single file for resume
        "save_on_train_epoch_end": True,  # Save BEFORE validation to avoid losing progress if killed during val
    }
    args_ckpt = {
        "dirpath": ckpt_path_store,
        "save_last": False,
        "save_weights_only": False,
        "filename": "chk_{epoch:08d}_{step:012d}",
        "every_n_epochs": cfg_exp.log.checkpoint_every_n_epochs,
        "monitor": None,
        "save_top_k": -1,
        "save_ema_separately": True,  # Dual files for evaluation
        "save_on_train_epoch_end": True,  # Save at epoch end (works even if validation disabled)
    }
    checkpoint_callback = EMAModelCheckpoint(**args_ckpt)
    checkpoint_callback_last = EMAModelCheckpoint(**args_ckpt_last)

    create_dir(ckpt_path_store, parents=True, exist_ok=True)
    return [checkpoint_callback, checkpoint_callback_last]


@rank_zero_only
def store_n_log_configs(
    cfg_exp: DictConfig,
    cfg_data: DictConfig,
    run_name: str,
    ckpt_path_store: str,
    wandb_logger: WandbLogger | CSVLogger | None,
) -> None:
    """Store config files locally and log them to wandb run."""

    def store_n_log_config(cfg: DictConfig, cfg_path: str, wlogger: WandbLogger | CSVLogger | None) -> None:
        with open(cfg_path, "w") as f:
            cfg_aux = OmegaConf.to_container(cfg, resolve=True)
            json.dump(cfg_aux, f, indent=4, sort_keys=True)

        if wlogger is not None:
            artifact = wandb.Artifact(f"config_files_{run_name}", type="config")
            artifact.add_file(cfg_path)
            wlogger.experiment.log_artifact(artifact)

    cfg_exp_file = os.path.join(ckpt_path_store, f"exp_config_{run_name}.json")
    cfg_data_file = os.path.join(ckpt_path_store, f"data_config_{run_name}.json")

    store_n_log_config(cfg_exp, cfg_exp_file, wandb_logger)
    store_n_log_config(cfg_data, cfg_data_file, wandb_logger)


@hydra.main(
    version_base=None,
    config_path=_config_path,
    config_name="training_ae",
)
def main(cfg_exp: DictConfig) -> None:
    silence_warnings()

    torch._dynamo.config.inline_inbuilt_nn_modules = True
    torch._inductor.config.coordinate_descent_tuning = True

    nolog = cfg_exp.get("nolog", False)
    single = cfg_exp.get("single", False)
    plugins = [SLURMEnvironment(auto_requeue=True)] if not single else []
    if not single:
        logger.info("Running on cluster")
    else:
        logger.info("Running on single GPU")
    show_prog_bar = cfg_exp.get("show_prog_bar", False)
    if single:
        # Rewrite number of GPUs and nodes for local runs
        cfg_exp.hardware.ngpus_per_node_ = 1
        cfg_exp.hardware.nnodes_ = 1
        cfg_exp.run_name_ = cfg_exp.run_name_ + "_local"

    print_config_tree(cfg_exp, resolve=True)

    run_name, root_run, ckpt_path_store = get_run_dirs(cfg_exp)
    callbacks = instantiate_callbacks(cfg_exp.callbacks)

    # Create model, warm-up or last ckpt
    model, resume_ckpt_path = get_model_n_ckpt_resume(cfg_exp, ckpt_path_store)

    # Fully deterministic mode — MUST come after model init, because model.__init__
    # may call _configure_flash_attention_backends() which re-enables Flash SDP.
    if cfg_exp.torch_deterministic:
        apply_torch_deterministic(logger)

    # logger
    train_logger: WandbLogger | CSVLogger | None = None
    if cfg_exp.log.log_wandb and not nolog:
        train_logger = WandbLogger(
            project=cfg_exp.log.wandb_project,
            id=run_name,
            entity=cfg_exp.log.wandb_entity,
            tags=cfg_exp.log.tags,
        )
    elif nolog:
        train_logger = CSVLogger(save_dir=root_run, name="csv_logs")

    if train_logger is not None:
        cfg_dict = OmegaConf.to_container(cfg_exp, resolve=True)
        train_logger.log_hyperparams(cfg_dict)

    # checkpoints
    if cfg_exp.log.checkpoint and not nolog:
        ckpt_callbacks = setup_ckpt(cfg_exp, ckpt_path_store)
        callbacks += ckpt_callbacks
        store_n_log_configs(cfg_exp, cfg_exp.datasets, run_name, ckpt_path_store, train_logger)

    # Add per-rank seeding callback to ensure RNG diversity across GPUs
    callbacks.append(RankSeedCallback(base_seed=cfg_exp.seed))
    val_start_epoch = cfg_exp.opt.get("val_start_epoch", 0)
    if val_start_epoch and val_start_epoch > 0:
        callbacks.append(
            DelayedValidationCallback(
                start_epoch=val_start_epoch,
                limit_val_batches=cfg_exp.opt.get("limit_val_batches", None),
            )
        )

    # ==============================================================================
    # Dataset instantiation (rf3-style)
    # ==============================================================================

    # ... build the train dataset
    assert "train" in cfg_exp.datasets and cfg_exp.datasets.train, "No 'train' dataset configuration provided!"

    logger.info("Instantiating train datasets and samplers...")
    dataset_and_sampler = recursively_instantiate_datasets_and_samplers(cfg_exp.datasets.train)
    train_dataset, train_sampler = (
        dataset_and_sampler["dataset"],
        dataset_and_sampler["sampler"],
    )

    # ==============================================================================
    # Trainer instantiation
    # ==============================================================================

    # Determine distributed strategy
    dist_strategy = cfg_exp.opt.dist_strategy
    if hasattr(dist_strategy, "_target_"):
        dist_strategy = hydra.utils.instantiate(dist_strategy)

    # Gradient clipping must be handled manually when Muon is enabled (manual optimization).
    muon_patterns = list(cfg_exp.opt.get("muon_param_patterns") or [])
    gradient_kwargs = {}
    if not muon_patterns:
        gradient_kwargs = {
            "gradient_clip_algorithm": "norm",
            "gradient_clip_val": cfg_exp.opt.get("gradient_clip_val", 1.0),
        }

    trainer = L.Trainer(
        max_epochs=cfg_exp.opt.max_epochs,
        accelerator=cfg_exp.hardware.accelerator,
        devices=cfg_exp.hardware.ngpus_per_node_,  # Number of gpus per node, not total
        num_nodes=cfg_exp.hardware.nnodes_,
        callbacks=callbacks,
        logger=train_logger,
        log_every_n_steps=cfg_exp.log.log_every_n_steps,
        default_root_dir=root_run,
        check_val_every_n_epoch=cfg_exp.opt.val_every_n_epochs,
        val_check_interval=None,  # Use epoch-based validation only
        strategy=dist_strategy,
        enable_progress_bar=show_prog_bar,
        plugins=plugins,
        limit_train_batches=cfg_exp.opt.get("limit_train_batches", 1.0),
        limit_val_batches=cfg_exp.opt.get("limit_val_batches", 1.0),
        accumulate_grad_batches=cfg_exp.opt.accumulate_grad_batches,
        num_sanity_val_steps=cfg_exp.opt.get("num_sanity_val_steps", 1),
        precision=get_training_precision(cfg_exp),
        use_distributed_sampler=False,
        profiler=cfg_exp.log.get("profiler", None),
        **gradient_kwargs,
    )

    # ==============================================================================
    # Dataloader creation
    # ==============================================================================

    # Define epoch as a fixed number of steps (not full dataset traversal)
    # This ensures predictable epoch timing and clean checkpoint boundaries
    steps_per_epoch = cfg_exp.opt.steps_per_epoch
    batch_size = cfg_exp.dataloaders.train.dataloader_params.batch_size
    n_examples_per_epoch = steps_per_epoch * batch_size * trainer.world_size * cfg_exp.opt.accumulate_grad_batches
    logger.info(f"Epoch = {steps_per_epoch} steps = {n_examples_per_epoch} examples")

    logger.info(f"Creating train dataloader (rank={trainer.global_rank}, world_size={trainer.world_size})...")
    train_loader = assemble_distributed_loader(
        dataset=train_dataset,
        sampler=train_sampler,
        rank=trainer.global_rank,
        world_size=trainer.world_size,
        n_examples_per_epoch=n_examples_per_epoch,
        loader_cfg=hydra.utils.instantiate(cfg_exp.dataloaders.train),
    )

    # ... compose the validation loader(s) with correct distributed settings
    val_loaders = None
    if "val" in cfg_exp.datasets and cfg_exp.datasets.val:
        logger.info(f"Creating validation dataloaders (rank={trainer.global_rank}, world_size={trainer.world_size})...")
        val_loaders = assemble_val_loader_dict(
            cfg=cfg_exp.datasets.val,
            rank=trainer.global_rank,
            world_size=trainer.world_size,
            loader_cfg=hydra.utils.instantiate(cfg_exp.dataloaders.val),
        )
        # Convert dict to list for Lightning
        val_loaders = list(val_loaders.values()) if isinstance(val_loaders, dict) else val_loaders
    else:
        logger.warning("No validation datasets provided! Skipping validation...")

    # Train the model with dataloaders
    logger.info("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loaders, ckpt_path=resume_ckpt_path)
    # If resume_ckpt_path is None then it creates a new optimizer


if __name__ == "__main__":
    main()
