import hashlib
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from rfproteina.rewards.reward_model import RewardModel
    from rfproteina.search.base_strategy import SearchStrategy

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from atomworks.common import exists
from atomworks.ml.utils.nested_dict import flatten as flatten_nested_dict
from atomworks.ml.utils.token import get_token_count
from hydra.errors import InstantiationException
from jaxtyping import Bool, Float
from lightning_utilities import apply_to_collection
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from torch import Tensor

from rfproteina.flow_matching.product_space_flow_matcher import (
    ProductSpaceFlowMatcher,
)
from rfproteina.inference_utils.foldseek import (
    add_diversity_summary_metrics,
    apply_cluster_map_to_metrics,
    run_foldseek_distributed,
)
from rfproteina.inference_utils.insertion_mapping import build_predicted_insertion_mapping
from rfproteina.inference_utils.solve_rate_metrics import compute_solve_rate_metrics
from rfproteina.metrics.metric import MetricManager
from rfproteina.metrics.oracles.oracle_metric import example_output_dir
from rfproteina.metrics.utils.metrics_utils import (
    MetricsPool,
    apply_pre_oracle_filters,
    auto_pick_metrics_workers,
    compute_oracle_metrics_for_val_output,
    fold_designs_for_oracles,
    resolve_mm_cfgs_from_metrics,
)
from rfproteina.nn.local_latents_transformer import LocalLatentsTransformer
from rfproteina.partial_autoencoder.autoencoder import AutoEncoder
from rfproteina.rewards.reward_model import SEARCH_REWARD_KEY
from rfproteina.utils.coors import nm_to_ang, trans_nm_to_atom37
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.logging import print_df_as_table, safe_print
from rfproteina.utils.optim_utils import build_optimizers
from rfproteina.utils.structure_export import process_single_sample_to_atom_array
from rfproteina.utils.torch_compile import maybe_mark_cudagraph_step_begin

logger = RankedLogger(__name__, rank_zero_only=True)

# Lazy import Transformer Engine for FP8 support
TE_AVAILABLE = None
te_autocast = None
DelayedScaling = None
Format = None


def _load_transformer_engine() -> bool:
    """Load Transformer Engine on demand for FP8 support."""
    global TE_AVAILABLE, te_autocast, DelayedScaling, Format
    if TE_AVAILABLE is None:
        try:
            from transformer_engine.common.recipe import (
                DelayedScaling as DelayedScaling_mod,
            )
            from transformer_engine.common.recipe import (
                Format as Format_mod,
            )
            from transformer_engine.pytorch import autocast as te_autocast_mod
        except ImportError:
            TE_AVAILABLE = False
            te_autocast = None
            DelayedScaling = None
            Format = None
        else:
            TE_AVAILABLE = True
            te_autocast = te_autocast_mod
            DelayedScaling = DelayedScaling_mod
            Format = Format_mod
    return TE_AVAILABLE


_last_log_time: dict[str, float] = {}  # Track last log time per stage


def _input_path_for_batch_index(batch: dict, batch_idx: int) -> dict[str, str] | None:
    """Extract input traceability paths from a collated inference batch."""
    spec_batch = batch.get("specification_batch")
    if spec_batch is None:
        return None
    if isinstance(spec_batch, list):
        spec = spec_batch[batch_idx] if batch_idx < len(spec_batch) else None
    elif isinstance(spec_batch, dict):
        spec = {}
        for key, val in spec_batch.items():
            if isinstance(val, list) and batch_idx < len(val):
                spec[key] = val[batch_idx]
            elif not isinstance(val, list):
                spec[key] = val
        spec = spec or None
    else:
        spec = None
    if not spec:
        return None
    out: dict[str, str] = {}
    if "input_cif" in spec:
        out["input_cif_path"] = str(spec["input_cif"])
    if "input_json_path" in spec:
        out["input_json_path"] = str(spec["input_json_path"])
    return out or None


def _collect_val_metrics_tags(cfg_datasets_val: DictConfig) -> list[set[str] | None]:
    """Collect per-dataset metrics_tags from all val dataset configs.

    For each top-level dataset entry in ``cfg_datasets_val``, finds the first
    ``metrics_tags`` list by recursive search and returns it as a set.
    Datasets with no ``metrics_tags`` in their config contribute ``None``,
    which means "no tag filtering — all metrics run for this dataset."

    Args:
        cfg_datasets_val: The ``cfg_exp.datasets.val`` DictConfig node.

    Returns:
        One element per val dataset. ``None`` means the dataset defines no
        tag filtering; a ``set[str]`` holds that dataset's exact tags.
    """

    def _normalize_metrics_tags(tags: Any) -> list[str] | None:
        if OmegaConf.is_list(tags) or isinstance(tags, list):
            return [str(t) for t in tags]
        if isinstance(tags, str):
            stripped = tags.strip()
            if not stripped:
                return []
            return [t.strip() for t in stripped.split(",") if t.strip()]
        return None

    def _find_first_tags(obj: Any) -> list[str] | None:
        """Recursively return the first ``metrics_tags`` list found, or None."""
        if OmegaConf.is_dict(obj):
            if "metrics_tags" in obj:
                tags = _normalize_metrics_tags(obj["metrics_tags"])
                if tags is not None:
                    return tags
            for v in obj.values():
                result = _find_first_tags(v)
                if result is not None:
                    return result
        elif OmegaConf.is_list(obj):
            for item in obj:
                result = _find_first_tags(item)
                if result is not None:
                    return result
        return None

    dataset_tags: list[set[str] | None] = []
    if OmegaConf.is_dict(cfg_datasets_val):
        for dataset_cfg in cfg_datasets_val.values():
            if not dataset_cfg:
                continue
            tags = _find_first_tags(dataset_cfg)
            dataset_tags.append(set(tags) if tags is not None else None)

    return dataset_tags


def _log_batch_example_ids(
    example_ids: list[str],
    epoch: int,
    batch_idx: int,
    stage: str,
    log_frequency: int = 100,
    total_batches: int | None = None,
) -> None:
    """Log example IDs, timing, and epoch progress for debugging batch ordering.

    Logs every log_frequency batches to verify shuffling behavior across epochs.
    Also logs time elapsed since last log and steps per second.
    """
    if batch_idx % log_frequency != 0:
        return

    current_time = time.time()
    elapsed = current_time - _last_log_time.get(stage, current_time)
    steps_per_sec = log_frequency / elapsed if elapsed > 0 else 0.0
    _last_log_time[stage] = current_time

    ids_preview = f"{example_ids[:3]}, ... ({len(example_ids) - 3} more)" if len(example_ids) > 3 else example_ids

    # Show epoch progress if total batches is known
    progress_str = ""
    if total_batches is not None and total_batches > 0:
        pct = batch_idx / total_batches * 100
        progress_str = f", progress={pct:.1f}%"

    logger.info(
        f"[{stage}] epoch={epoch}, batch={batch_idx}{progress_str}, "
        f"steps={log_frequency}, elapsed={elapsed:.1f}s, steps/s={steps_per_sec:.2f}, "
        f"n_examples={len(example_ids)}, ids={ids_preview}"
    )


def _combined_val_tags(
    val_dataset_tags: "list[set[str] | None] | None",
    reward_metrics_tags: list[str],
) -> "list[set[str] | None] | None":
    """Merge val_dataset_tags with reward_metrics_tags for MetricManager setup.

    Ensures metrics needed by the reward model are set up even when no validation
    dataset uses their tags.
    """
    base = list(val_dataset_tags or [])
    if reward_metrics_tags:
        base.append(set(reward_metrics_tags))
    return base if base else None


def _clean_metrics_dict(metrics: dict) -> dict | None:
    """Convert a metrics dict to clean floats for gathering across ranks."""
    if metrics is None:
        return None

    # Standardize types
    cleaned = {}
    chain_of_interest_values = []
    non_metadata_metrics_detected = False
    for key, value in metrics.items():
        if key in (
            "dataset_name",
            "example_id",
            "task",
            "designed_structure_path",
            "designed_chain_path",
            "ckpt_path",
            "ckpt_name",
            "output_base_dir",
        ) or key.endswith(
            (
                "refolded_structure_path",
                "confidence_json_path",
            )
        ):
            cleaned[key] = str(value)
            continue
        elif key == "metrics_tags":
            cleaned[key] = value if isinstance(value, str) else (",".join(value) if value else "")
            continue
        elif key in ("input_cif_path", "input_json_path"):
            cleaned[key] = str(value) if value is not None else None
            continue
        elif key.endswith("chain_of_interest"):
            chain_of_interest_values.append(str(value))
            continue
        if isinstance(value, int | float):
            cleaned[key] = float(value)
            non_metadata_metrics_detected = True
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            cleaned[key] = value.item()
            non_metadata_metrics_detected = True

    # Add chain of interest column
    if len(set(chain_of_interest_values)) == 1:
        cleaned["chain_of_interest"] = chain_of_interest_values[0]
    elif len(set(chain_of_interest_values)) > 1:
        logger.warning(f"Multiple different chains of interest reported: {set(chain_of_interest_values)}")
        cleaned["chain_of_interest"] = None
    else:
        cleaned["chain_of_interest"] = None

    # Return None if only metadata (no actual metrics)
    return cleaned if non_metadata_metrics_detected else None


class Proteina(L.LightningModule):
    def __init__(
        self,
        cfg_exp: DictConfig,
        store_dir: str | None = None,
        autoencoder_ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg_exp = cfg_exp
        # Handle inf_cfg for validation/inference
        if "inf_cfg" in self.cfg_exp:
            # The config loads generation/inference_base.yaml directly into inf_cfg
            # No need to look for a .generation subkey
            inf_cfg = self.cfg_exp.inf_cfg
        else:
            inf_cfg = None

        # If True, these will be set in configure_inference
        self.is_inference = False
        self.inference_seed: int | None = None  # Set via generate.py, not during training
        self.ckpt_path: str | None = None  # Set via configure_inference
        self.ckpt_name: str | None = None  # Set via configure_inference
        # Whether to write per-design chain-of-interest CIFs (only used by foldseek clustering).
        # Overridden for inference via configure_inference; training validation always writes.
        self.save_chain_of_interest = True

        self.inf_cfg = inf_cfg  # Used for inference-like validation
        self.store_dir = store_dir if store_dir is not None else os.path.join(".", "runs", cfg_exp.run_name_)

        self._metrics_pool = MetricsPool(num_workers=0, mm_cfgs={})
        try:
            self.configure_metrics(cfg_exp.metrics)
        except InstantiationException as e:
            logger.warning(
                f"Failed to instantiate metric managers: {e}. "
                f"This can occur if the model was trained with Metrics that no longer exist. "
                f"Skipping — inference-time metrics will be configured via configure_inference()."
            )

        if autoencoder_ckpt_path is not None:
            # Allow adding new keys
            logger.info(f"Manually setting autoencoder_ckpt_path to {autoencoder_ckpt_path}")
            OmegaConf.set_struct(cfg_exp, False)
            # Update the configuration with the new key-value pair
            cfg_exp.autoencoder_ckpt_path = autoencoder_ckpt_path
            # Re-enable struct mode if needed
            OmegaConf.set_struct(cfg_exp, True)

        self.autoencoder, latent_dim = self.load_autoencoder(cfg_exp, freeze_params=True)

        # Add right latent dimensionality in the config file, needed to instantiate the flow matcher below
        if latent_dim is not None:
            self.latent_dim = latent_dim
        else:
            self.latent_dim = cfg_exp.product_flowmatcher.local_latents.get("dim", 8)

        if self.autoencoder is not None:
            try:
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
            except Exception:
                OmegaConf.set_struct(cfg_exp, False)
                # Update the configuration with the new key-value pair
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
                # Re-enable struct mode if needed
                OmegaConf.set_struct(cfg_exp, True)

        self.fm = ProductSpaceFlowMatcher(cfg_exp)
        logger.info(f"cfg_exp.nn: {cfg_exp.nn}")

        self.nn = LocalLatentsTransformer(**cfg_exp.nn, latent_dim=self.latent_dim)

        # Scaling laws stuff
        self.nflops = 0
        self.nsamples_processed = 0
        self.nparams = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)

        self.nn_ag = None

        # Muon and Adam are stepped by hand in training_step, which requires manual optimization.
        # Gradient clipping and accumulation are handled there too, not by the Trainer.
        self.automatic_optimization = False

        # Search integration
        self.search_strategy = None
        self.search_reward_model = None

        # Differentiable-metric gradient guidance applied during sampling
        self.guidance_model = None

        # FP8 training support via Transformer Engine
        self.use_fp8 = cfg_exp.get("precision") == "fp8"
        if self.use_fp8:
            if not _load_transformer_engine():
                raise ImportError(
                    "Transformer Engine is required for FP8 precision. "
                    "Install it via: pip install transformer_engine[pytorch]"
                )
            self.fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID)
            logger.info("FP8 training enabled via Transformer Engine")
        else:
            self.fp8_recipe = None

    def load_autoencoder(self, cfg_exp: DictConfig, freeze_params: bool = True) -> tuple[Any, int | None]:
        """Loads autoencoder, if required."""
        if "autoencoder_ckpt_path" in cfg_exp:
            # for new runs trained with refactored codebase
            ae_ckp_path = cfg_exp.autoencoder_ckpt_path
        elif "autoencoder_ckpt_path" in cfg_exp.product_flowmatcher.local_latents:
            # for old runs trained with old codebase
            ae_ckp_path = cfg_exp.product_flowmatcher.local_latents.autoencoder_ckpt_path
        else:
            raise ValueError("No autoencoder checkpoint path provided")

        if ae_ckp_path is None:
            self._use_unified_ae_encoding = False
            return None, None

        logger.info(f"Loading autoencoder from {ae_ckp_path}")
        autoencoder = AutoEncoder.load_from_checkpoint(ae_ckp_path, strict=False, weights_only=False)
        if freeze_params:
            for param in autoencoder.parameters():
                param.requires_grad = False
        # Optionally compile the frozen encoder's forward. It runs every training step to
        # produce the latent targets but otherwise executes eager; compiling it is
        # output-preserving and gives ~+10% training steps/s (encoder is ~1/4 of step compute).
        if cfg_exp.get("compile_autoencoder", False):
            logger.info("Compiling autoencoder encoder forward (compile_autoencoder=True).")
            autoencoder.encoder.forward = torch.compile(autoencoder.encoder.forward)

        n_ae_residue_types = autoencoder.decoder.logit_linear[-1].out_features
        self._use_unified_ae_encoding = n_ae_residue_types == 33
        logger.info(
            f"Autoencoder uses {'unified' if self._use_unified_ae_encoding else 'legacy'} "
            f"encoding (n_residue_types={n_ae_residue_types})"
        )

        return autoencoder, autoencoder.latent_dim

    def configure_optimizers(self) -> torch.optim.Optimizer | list[torch.optim.Optimizer]:
        return build_optimizers(self.named_parameters(), self.cfg_exp.opt)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Adds additional variables to checkpoint."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except KeyError:
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

    def _inactive_self_cond_placeholder(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Zero-filled ``x_sc`` / ``x_recycle`` dict for steps where the feature is inactive.

        Paired with a ``*_is_active`` 0/1 scalar so the batch shape is invariant
        to the Bernoulli / recycle outcome and torch.compile traces a single graph.
        """
        coords = batch["coords_nm"]
        b, n = coords.shape[:2]
        device, dtype = coords.device, coords.dtype
        return {
            "bb_ca": torch.zeros(b, n, 3, device=device, dtype=dtype),
            "local_latents": torch.zeros(b, n, self.latent_dim, device=device, dtype=dtype),
        }

    def call_nn(
        self,
        batch: dict[str, torch.Tensor],
        n_recycle: int,
    ) -> dict[str, torch.Tensor]:
        """
        Calls NN with recycling. Should this be here or in the NN? Possibly better here,
        in case we want to recycle using decoder for some approach, etc, and this is akin
        to self conditioning, also here.
        Also, if we want to recycle clean sample predictions... Then we'd need this here,
        as the nn does not know about relations between v, x1, ...
        """
        device = batch["coords_nm"].device

        # Pre-seed x_recycle inactive on the first call; the recycle loop overwrites.
        batch["x_recycle"] = self._inactive_self_cond_placeholder(batch)
        batch["x_recycle_is_active"] = torch.zeros((), device=device)

        # Pre-seed x_sc inactive only when not already populated
        if "x_sc" not in batch:
            batch["x_sc"] = self._inactive_self_cond_placeholder(batch)
            batch["x_sc_is_active"] = torch.zeros((), device=device)

        # First call
        nn_out = self.nn(batch)

        # Recycle n_recycle times detaching gradients and updating input
        # Note that recycling is supported by the codebase, but the models provided
        # with the La-Proteina paper do not use it, nor were trained with it.
        for _ in range(n_recycle):
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
            batch["x_recycle"] = {dm: x_1_pred[dm].detach() for dm in x_1_pred}
            batch["x_recycle_is_active"] = torch.ones((), device=device)
            nn_out = self.nn(batch)

        # Final prediction
        return nn_out

    def predict_for_sampling(
        self,
        batch: dict,
        mode: Literal["full", "ucond"],
        n_recycle: int,
    ) -> tuple[dict[str, torch.Tensor] | float | None]:
        """
        This function predicts clean samples for multiple models:
        x_pred, the 'original' model, if mode == full
        x_pred_ucond, the unconditional model, , if mode == ucond

        TODO: Need to update to include autoguidance again

        These predictions will later be used to sample with guidance and autoguidance.

        Args:
            batch: Dict
            mode: str

        Returns:
            x_pred (tensor) for the requested mode
        """
        if mode == "full":
            nn_out = self.call_nn(batch, n_recycle=n_recycle)
        elif mode == "ucond":
            assert "cath_code" in batch or "x_motif" in batch, "Only support CFG when cath_code or x_motif is provided"
            uncond_batch = batch.copy()
            if "cath_code" in uncond_batch:
                uncond_batch.pop("cath_code")
            if "x_motif" in uncond_batch:
                uncond_batch.pop("x_motif")
            nn_out = self.call_nn(uncond_batch, n_recycle=n_recycle)
        else:
            raise OSError(f"Wrong {mode} passed to `predict_for_sampling`")

        return nn_out

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batch dimension.
        """
        # Mark step boundary for torch.compile CUDA graphs (no-op when graphs
        # are off). Without this, EMA / checkpoint callbacks that take
        # references to graph outputs can raise once those buffers get
        # overwritten by the next compiled invocation.
        maybe_mark_cudagraph_step_begin()

        val_step = batch_idx == -1
        log_prefix = (
            f"validation_loss_{batch.pop('validation_dataset_name', 'unknown_dataset')}" if val_step else "train"
        )

        # Log example IDs for debugging batch ordering
        _log_batch_example_ids(
            example_ids=batch.get("example_id_batch", []),
            epoch=self.current_epoch,
            batch_idx=batch_idx,
            stage="train",
            total_batches=self.trainer.num_training_batches,
        )

        # Preprocess batch from atom37 collate format
        batch["coords_nm"] = batch["coords"] / 10.0  # Convert Angstroms to nm
        batch["coord_mask"] = batch["atom37_mask"]
        # residue_type already provided by atom37 collate

        # FP8 autocast context for Transformer Engine
        fp8_ctx = te_autocast(enabled=True, recipe=self.fp8_recipe) if self.use_fp8 else nullcontext()

        with fp8_ctx:
            # Add clean samples for all data modes / spaces we are working on
            batch = self.add_clean_samples(batch)

            # Corrupt the batch. On validation, override t with uniform draws so the
            # val loss is an unbiased estimator of E_{t~U[0,1]}[loss] instead of the
            # Beta-weighted training distribution.
            if val_step:
                batch_shape = batch["padding_mask"].shape[:1]
                device = batch["padding_mask"].device
                t_uniform = {dm: torch.rand(batch_shape, device=device) for dm in self.fm.data_modes}
                batch = self.fm.corrupt_batch(batch, t=t_uniform)
            else:
                batch = self.fm.corrupt_batch(batch)  # adds x_1, t, x_0, x_t
            bs, n = batch["padding_mask"].shape

            # Handle conditioning variables (synced across ranks via per-epoch schedule)
            batch = self.handle_self_cond(batch, batch_idx=batch_idx, val_step=val_step)

            # Number of recycling steps
            n_recycle = self.handle_recycling()

            nn_out = self.call_nn(batch, n_recycle=n_recycle)
            losses = self.fm.compute_loss(
                batch=batch,
                nn_out=nn_out,
            )  # Dict[str, Tensor w.batch shape [*]]

            self.log_losses(bs=bs, losses=losses, log_prefix=log_prefix, batch=batch, val_step=val_step)
            train_loss = sum([torch.mean(losses[k]) for k in losses if "_justlog" not in k])

            self.log(
                f"{log_prefix}/loss",
                train_loss,
                on_step=not val_step,  # Only log per-step for training, not validation
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=False,
                add_dataloader_idx=False,
            )

            if not val_step:
                self.log_train_loss_n_prog_bar(bs, train_loss)
                self.update_n_log_flops(bs, n)
                self.update_n_log_nsamples_processed(bs)
                self.log_nparams()

                should_step = (batch_idx + 1) % self.cfg_exp.opt.get("accumulate_grad_batches", 1) == 0

                # If we are not stepping, we wrap backward in no_sync to stay local
                if not should_step:
                    with self.trainer.strategy.no_sync(self):
                        self.manual_backward(train_loss)
                else:
                    self.manual_backward(train_loss)

                    # [muon, adam], or just adam if the exclude patterns left Muon nothing.
                    optimizers = self.optimizers()
                    if not isinstance(optimizers, list):
                        optimizers = [optimizers]

                    clip_val = self.cfg_exp.opt.gradient_clip_val
                    for opt in optimizers:
                        self.clip_gradients(opt, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")

                    manual_opt = self.trainer.fit_loop.epoch_loop.manual_optimization
                    for i, opt in enumerate(optimizers):
                        opt.step()
                        # Lightning bumps global_step on every manual step; undo all but the last
                        # so one training step is one global_step and every hook sees the same value.
                        if i < len(optimizers) - 1:
                            manual_opt.optim_step_progress.total.completed -= 1
                            manual_opt.optim_step_progress.current.completed -= 1

                    for opt in optimizers:
                        opt.zero_grad()

            # Return dict with loss and extra data for callbacks.
            # Lightning uses the reserved "loss" key for backpropagation - other keys are
            # passed through to callbacks via the `outputs` parameter in on_train_batch_end.
            return {
                "loss": train_loss,
                "losses_per_sample": {k: v.detach() for k, v in losses.items()},
                "t": {k: v.detach() for k, v in batch["t"].items()},
            }

    def add_clean_samples(self, batch: dict) -> dict:
        """
        Adds clean sample for all data modes / spaces we are working on. For instance, if we have two
        data modes, bb_ca and local_latents, it adds the clean data to the batch
        x_1 = {
            "bb_ca": Corresponding tensor with clean bb_ca coordinates, shape [b, n, 3]
            "local_latents": Corresponding tensor with clean local_latents, shape [b, n, d]
        }

        Args:
            batch: Batch to add clean samples to.

        Returns:
            Batch with clean sample added.
        """
        batch["x_1"] = {dm: self._get_clean_sample(batch, dm) for dm in self.fm.data_modes}
        return batch

    def _get_clean_sample(self, batch: dict, dm: str) -> torch.Tensor:
        """
        Gets clean sample for a given data mode.

        Args:
            batch: Batch to get clean sample from.
            dm: Data mode to get clean sample for.

        Returns:
            Clean sample for the given data mode.
            For bb_ca, we provide coords for all center atoms (even though some are not resolved
            and not that meaningful).
            For local_latents, we encode only tokens that the autoencoder can encode (standard AAs for now).
        """
        if dm == "bb_ca":
            return batch["coords_nm"][:, :, 1, :]  # [b, n, 3]
        elif dm == "local_latents":
            # Get latent dimension from config
            latent_dim = self.cfg_exp.product_flowmatcher.local_latents.get("dim", 8)

            if self.autoencoder is None:
                logger.warning("No autoencoder available, returning zero latents")
                b = batch["coords_nm"].shape[0]
                n_tokens = batch["coords_nm"].shape[1]
                device = batch["coords_nm"].device
                return torch.zeros(b, n_tokens, latent_dim, device=device)

            # Get can_be_encoded mask (token-level from atom37 collate)
            can_be_encoded = batch["can_be_encoded"]  # [b, n_tokens]
            padding_mask = batch["padding_mask"]  # [b, n_tokens]

            # Only encode tokens that can be encoded
            encodable_mask = can_be_encoded & padding_mask  # [b, n_tokens]

            # If no tokens can be encoded (e.g., ligand-only inputs), return zero latents
            if not encodable_mask.any():
                b = batch["coords_nm"].shape[0]
                n_tokens = batch["coords_nm"].shape[1]
                device = batch["coords_nm"].device
                return torch.zeros(b, n_tokens, latent_dim, device=device)

            # Legacy AEs were trained with 0-indexed protein-only residue types (ALA=0).
            # Unified AEs use the full pipeline encoding (<M>=0, ALA=1, ..., RNA/DNA=21+)
            # and can also encode nucleotides.
            if self._use_unified_ae_encoding:
                residue_type_for_ae = batch["residue_type"]
            else:
                residue_type_for_ae = batch["residue_type"] - 1
                # Legacy AE can only encode proteins -- mask out nucleotides
                token_type = batch["token_type"]  # 0=protein, 1=RNA, 2=DNA, 3=atomized, 4=unknown
                encodable_mask = encodable_mask & (token_type == 0)

            # Clamp for non-encodable tokens to avoid invalid indices for embeddings
            residue_type_for_ae = torch.where(
                encodable_mask, residue_type_for_ae, torch.zeros_like(residue_type_for_ae)
            )
            ae_batch = {
                **batch,
                "residue_type": residue_type_for_ae,
                "padding_mask": encodable_mask,
                "can_be_encoded": encodable_mask,
            }

            # Encode
            encoded = self.autoencoder.encode(ae_batch)
            latents = encoded["z_latent"]  # [b, n_tokens, latent_dim]

            # Zero out non-encodable tokens
            latents = latents * encodable_mask.unsqueeze(-1).float()

            return latents  # [b, n_tokens, latent_dim]
        else:
            raise ValueError(f"Loading clean samples from data mode {dm} not supported.")

    def handle_self_cond(self, batch: dict, batch_idx: int, val_step: bool) -> dict:
        enabled = self.cfg_exp.training.get("self_cond", False)
        use_sc = (not val_step) and enabled and (random.random() < 0.5)
        device = batch["coords_nm"].device

        if use_sc:
            n_recycle = self.cfg_exp.training.get("n_recycle", 0)
            with torch.no_grad():
                nn_out = self.call_nn(batch, n_recycle=n_recycle)
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
            batch["x_sc"] = {k: x_1_pred[k].detach() for k in x_1_pred}
            # Mark as active so we use these self-cond predictions in the next step
            batch["x_sc_is_active"] = torch.ones((), device=device)
        else:
            batch["x_sc"] = self._inactive_self_cond_placeholder(batch)
            # Mark as inactive so we can skip later
            batch["x_sc_is_active"] = torch.zeros((), device=device)

        return batch

    def handle_recycling(self) -> int:
        n_recycle = self.cfg_exp.training.get("n_recycle", 0)
        if n_recycle == 0:
            return 0
        return random.randint(0, n_recycle)  # 0 and n_recycle included

    def log_losses(
        self,
        bs: int,
        losses: dict[str, Float[torch.Tensor, "b"]],
        log_prefix: str,
        batch: dict,
        val_step: bool = False,
    ) -> None:
        for k in losses:
            log_name = k[: -len("_justlog")] if k.endswith("_justlog") else k

            self.log(
                f"{log_prefix}/loss_{log_name}",
                torch.mean(losses[k]),
                on_step=not val_step,  # Only log per-step for training, not validation
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        if val_step:
            self._log_binned_val_losses(losses=losses, log_prefix=log_prefix, t=batch["t"], n_bins=5)

    def _log_binned_val_losses(
        self,
        losses: dict[str, Float[torch.Tensor, "b"]],
        log_prefix: str,
        t: dict[str, Float[torch.Tensor, "b"]],
        n_bins: int = 5,
    ) -> None:
        """Log per-t-bin mean loss, matching each loss key to a data_mode by substring."""
        width = 1.0 / n_bins
        for k, loss in losses.items():
            name = k.removesuffix("_justlog")
            dm = next((d for d in t if d in name), None)
            if dm is None:
                continue
            bins = (t[dm] * n_bins).floor().long().clamp(max=n_bins - 1)
            # Log every bin on every rank (even empty) so ranks agree on keys and sync doesn't deadlock.
            for b in range(n_bins):
                mask = bins == b
                n = int(mask.sum())
                value = loss[mask].mean() if n > 0 else loss.new_zeros(())
                self.log(
                    f"{log_prefix}/loss_{name}_t_{b * width:.1f}-{(b + 1) * width:.1f}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=max(n, 1),
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

    def log_train_loss_n_prog_bar(self, b: int, train_loss: torch.Tensor) -> None:
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=b,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def log_nparams(self) -> None:
        self.log(
            "scaling/nparams",
            self.nparams * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )  # constant line

    def update_n_log_nsamples_processed(self, b: int) -> None:
        self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def update_n_log_flops(self, b: int, n: int) -> None:
        """
        Updates and logs flops, if available
        """
        try:
            nflops_step = self.nn.nflops_computer(b, n)  # nn should implement this function if we want to see nflops
        except Exception:
            nflops_step = None

        if nflops_step is not None:
            self.nflops = (
                self.nflops + nflops_step * self.trainer.world_size
            )  # Times number of processes so it logs sum across devices
            self.log(
                "scaling/nflops",
                self.nflops * 1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

    def on_validation_start(self) -> None:
        """Bring up the per-rank metrics pool."""
        self._metrics_pool.start(self.val_dataset_tags)

    def on_validation_end(self) -> None:
        self._metrics_pool.shutdown()

    def on_validation_epoch_start(self) -> None:
        """Re-initialize validation outputs at the start of each validation epoch."""
        self.validation_output_generation = []

        # Tracks whether oracle/refold/pass MMs were set up in this epoch
        # to avoid double-setup
        self._oracle_mms_setup_this_epoch = False

        reward_metrics_tags = (
            self.search_reward_model.reward_metrics_tags if self.search_reward_model is not None else []
        )
        combined_tags = _combined_val_tags(self.val_dataset_tags, reward_metrics_tags)

        if self.design_metric_manager is not None:
            self.design_metric_manager.on_validation_epoch_start(val_dataset_tags=combined_tags)

        if reward_metrics_tags:
            for mm in [
                self.oracle_metric_manager,
                self.refolding_metric_manager,
                self.pass_criteria_metric_manager,
                self.pre_oracle_filter_metric_manager,
            ]:
                if mm is not None:
                    mm.on_validation_epoch_start(val_dataset_tags=combined_tags)

            # We just set up the metric managers - ensure we don't set them up again later
            self._oracle_mms_setup_this_epoch = True

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """
        Dual validation step. Runs training task and inference task on each batch.

        Args:
            batch: batch from dataset
            batch_idx: batch index
            dataloader_idx: dataloader index. Necessary for determining which dataset we're validating on.

        Returns:
            Dictionary containing metrics_output for aggregation
        """
        # Get the name of the current dataset
        # Always use the actual dataset key to ensure unique names for multiple dataloaders
        # Mirror the filtering in assemble_val_loader_dict: skip None-config entries so that
        # dataloader_idx aligns with the actual dataloaders passed to Lightning.
        val_dataset_names = [k for k, v in self.cfg_exp.datasets.val.items() if v]
        n_val_datasets = len(val_dataset_names)
        actual_dataset_name = val_dataset_names[dataloader_idx]
        if self.inf_cfg is not None and self.inf_cfg.get("val_set_name_override") is not None:
            # Use override as prefix, but keep actual name for uniqueness
            dataset_name = f"{self.inf_cfg.val_set_name_override}"
        else:
            dataset_name = actual_dataset_name

        # Log dataset progress on first batch
        if batch_idx == 0:
            metrics_tags = batch.get("metrics_tags_batch", [[]])[0]
            tags_str = ", ".join(metrics_tags) if metrics_tags else "none"
            content = f"Dataset: {dataset_name}\nTags:    {tags_str}"
            panel = Panel(
                content,
                title=f"Validation [{dataloader_idx + 1}/{n_val_datasets}]",
                border_style="cyan",
                width=60,
            )
            safe_print(panel, logger=logger)

        # Log example IDs for debugging batch ordering
        _log_batch_example_ids(
            example_ids=batch.get("example_id_batch", []),
            epoch=self.current_epoch,
            batch_idx=batch_idx,
            stage=f"val/{dataset_name}",
        )

        # Loss-based validation (training task) - per-dataset opt-in via compute_loss config
        # HACK: We need to encode the dataset name when we have multiple dataloaders, to avoid the lightning
        # error about duplicated metrics. To avoid adding a kwarg to training_step, we add this data to the batch.
        # We will remove it in the training_step function.
        # NOTE: Use actual_dataset_name (without prefix) to look up config
        dataset_cfg = self.cfg_exp.datasets.val.get(actual_dataset_name) or {}

        # Run some validation sets only every N validation epochs
        run_every_n_val_epochs = dataset_cfg.get("run_every_n_val_epochs", 1)
        if run_every_n_val_epochs > 1:
            val_every_n_epochs = self.cfg_exp.opt.get("val_every_n_epochs", 1)
            val_epoch_idx = self.current_epoch // val_every_n_epochs
            if val_epoch_idx % run_every_n_val_epochs != 0:
                logger.info(
                    f"Skipping val dataset '{dataset_name}' this epoch "
                    f"(runs every {run_every_n_val_epochs} val epochs, val_epoch={val_epoch_idx})"
                )
                return {}

        # Loss-based validation: run forward pass and collect per-example losses for metrics
        per_example_losses: dict[str, torch.Tensor] | None = None
        if dataset_cfg.get("compute_loss", False) and self.cfg_exp.opt.get("compute_val_loss", True):
            batch["validation_dataset_name"] = dataset_name
            with torch.no_grad():
                step_outputs = self.training_step(batch, batch_idx=-1)
            per_example_losses = step_outputs["losses_per_sample"]

        metrics_output = {}
        if self.inf_cfg is None:
            return metrics_output

        if self.is_inference:
            output_suffix = f"_job{self.job_id}" if self.job_id is not None else ""
        else:
            output_suffix = f"_rank{self.global_rank}"

        # Log where we are saving validation structures
        # Flatten structure: step_X/generated/ instead of step_X/rank_Y/generated/
        output_base = Path(self.store_dir) / ("val_samples" if not self.is_inference else "inference_samples")
        if not self.is_inference:
            output_base = output_base / dataset_name / f"step_{self.global_step}"

        output_base.mkdir(parents=True, exist_ok=True)

        # Expose to search strategies so they can save oracle results during generation
        # (enables on_validation_epoch_end to skip re-folding already-computed results).
        self._current_output_base_dir = output_base
        self._current_output_suffix = output_suffix

        # Hash dataset_name to keep example_id short; full name is preserved in metrics_output["dataset_name"].
        # Computed before predict_step so beam search can build canonical IDs that match what
        # proteina will use here, avoiding a content-addressed directory mismatch.
        dataset_name_hash = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
        self._current_dataset_name_hash = dataset_name_hash

        # Generate designs for this inference batch
        generation_list = self.predict_step(batch, batch_idx)

        example_id_batch = batch.get("example_id_batch", [])
        n_batch = len(example_id_batch)
        for i in range(len(generation_list)):
            orig_batch_idx = i % n_batch if n_batch > 0 else 0
            raw_batch_id = example_id_batch[orig_batch_idx] if n_batch > 0 else "sample"

            # Use example_id pre-assigned by the search strategy if available.
            # Beam search pre-assigns canonical IDs that already include the dataset_name_hash
            # prefix so oracle outputs land in the same directory as the design CIF.
            base_id = generation_list[i].get("example_id", raw_batch_id)
            if base_id == raw_batch_id:
                # Not pre-assigned — apply the standard naming logic.
                if len(generation_list) > n_batch:
                    base_id = f"{base_id}_s{i}"
                example_id = f"{dataset_name_hash}__{base_id}{output_suffix}"
            else:
                # Pre-assigned by beam search with canonical prefix already included.
                example_id = f"{base_id}{output_suffix}"

            metrics_output: dict = {"example_id": example_id}

            # Search strategies score every sample they emit; record the reward they selected on.
            if "reward" in generation_list[i]:
                metrics_output[SEARCH_REWARD_KEY] = generation_list[i]["reward"]

            if per_example_losses is not None:
                for k, v in per_example_losses.items():
                    if "_justlog" not in k:
                        metrics_output[f"loss_{k}"] = v[orig_batch_idx].item()
                metrics_output["loss"] = sum(
                    v[orig_batch_idx].item() for k, v in per_example_losses.items() if "_justlog" not in k
                )

            # Generated CIF path: shard_{N}/{digest[:8]}/{example_id}_generated.cif
            # All files for this design (chain PDB, oracle outputs, trajectories) share
            # the same per-example subdirectory so related artifacts stay together.
            generated_output_path = example_output_dir(output_base, example_id) / f"{example_id}_generated.cif"

            # Add dataset_name and task as metadata
            metrics_output["dataset_name"] = dataset_name
            task_batch = batch.get("task_batch") or []
            if orig_batch_idx < len(task_batch) and task_batch[orig_batch_idx]:
                metrics_output["task"] = task_batch[orig_batch_idx]
            else:
                task_base = generation_list[i].get("raw_batch_id", raw_batch_id)
                metrics_output["task"] = "_".join(task_base.split("_")[:-1])

            # Add seed used for generation
            if self.inference_seed is not None:
                metrics_output["seed"] = self.inference_seed

            # Add full generation config as dot-separated columns
            if self.inf_cfg is not None:
                flat_cfg = flatten_nested_dict(OmegaConf.to_container(self.inf_cfg, resolve=True), fuse_keys=".")
                for key, value in flat_cfg.items():
                    metrics_output[f"generation.{key}"] = value

            # Add checkpoint identity
            if self.ckpt_path is not None:
                metrics_output["ckpt_path"] = self.ckpt_path
            if self.ckpt_name is not None:
                metrics_output["ckpt_name"] = self.ckpt_name

            # Add input specification or CIF path
            input_paths = _input_path_for_batch_index(batch, orig_batch_idx)
            if input_paths:
                metrics_output.update(input_paths)

            metrics_output = apply_to_collection(metrics_output, torch.Tensor, lambda x: x.detach())
            metrics_output["designed_structure_path"] = str(generated_output_path.absolute())
            metrics_output["output_base_dir"] = str(output_base)

            # Store metrics_tags for the oracle loop in on_validation_epoch_end
            metrics_tags_batch = batch.get("metrics_tags_batch", [])
            _tags = metrics_tags_batch[orig_batch_idx] if orig_batch_idx < len(metrics_tags_batch) else []
            metrics_output["metrics_tags"] = ",".join(_tags) if _tags else ""

            staged_payload = None
            if self.design_metric_manager is not None:
                staged_payload = self._build_staged_payload(
                    generation_list[i],
                    example_id,
                    list(_tags),
                    output_base,
                    generated_output_path,
                    dataset_name=dataset_name,
                    task=metrics_output["task"],
                )

            # Store for epoch-end aggregation
            self.validation_output_generation.append(metrics_output)
            if staged_payload is not None:
                self._metrics_pool.submit_generated_structure(example_id, staged_payload, self.design_metric_manager)

        return metrics_output

    def _gather_metrics(self, local_metrics: list[dict]) -> list[dict] | None:
        """Gather per-example metric dicts from all ranks to rank 0.

        Returns the merged list on rank 0, or None on rank > 0 (all ranks must call this).
        """
        if self.trainer.world_size > 1 and dist.is_initialized():
            gathered = [None] * self.trainer.world_size if self.global_rank == 0 else None
            torch.cuda.set_device(self.device)
            dist.gather_object(local_metrics, gathered, dst=0)
            if self.global_rank == 0:
                merged: list[dict] = []
                for rank_list in gathered:
                    merged.extend(rank_list)
                return merged
            return None  # rank > 0
        return local_metrics

    def on_validation_epoch_end(self) -> None:
        """Aggregate and log generation validation metrics from all ranks.

        Steps (in order):
          1. Drain per-structure metric futures into ``validation_output_generation`` (all ranks).
          1b. Pre-oracle filter: excise designs that fail design-phase criteria before oracle
              folding (all ranks).
          2. Gather per-design metrics to rank 0 (DDP).
          3. Run oracle fold + refolding + pass_criteria (all ranks).
          4. Gather refolding metrics to rank 0 (DDP).
          5. Distributed foldseek clustering across ranks (DDP scatter+gather;
             no-op for ``world_size == 1`` or ``FOLDSEEK_EXEC`` unset).
          6. Rank 0 merges design+refold metrics, applies cluster IDs, runs the
             in-process foldseek fallback for any groups not covered by step 5,
             and writes the CSV.
        """

        # 1. Drain per-structure metric futures + merge into validation_output_generation.
        staged_results_by_id = self._metrics_pool.drain_generated_structure()
        if staged_results_by_id:
            for entry in self.validation_output_generation:
                eid = entry.get("example_id")
                if eid in staged_results_by_id:
                    for k, v in staged_results_by_id[eid].items():
                        entry.setdefault(k, v)
        if self.design_metric_manager is not None:
            self.design_metric_manager.on_validation_epoch_end()

        # 1b. Pre-oracle filter: excise designs that fail design-phase criteria before
        # oracle folding. Runs inline (no workers) since it is pure dict look-ups.
        pre_oracle_filter_mm = self.pre_oracle_filter_metric_manager
        if pre_oracle_filter_mm is not None:
            _reward_tags_for_filter = (
                self.search_reward_model.reward_metrics_tags if self.search_reward_model is not None else []
            )
            if not _reward_tags_for_filter:
                pre_oracle_filter_mm.on_validation_epoch_start(val_dataset_tags=self.val_dataset_tags)
            self.validation_output_generation = apply_pre_oracle_filters(
                self.validation_output_generation,
                pre_oracle_filter_mm,
                delete_failing=self.cfg_exp.get("delete_failing_pre_oracle", True),
            )
            pre_oracle_filter_mm.on_validation_epoch_end()

        # 2. Gather per-design metrics to rank 0 (DDP collective).
        local_design_metrics = [
            c for m in self.validation_output_generation if (c := _clean_metrics_dict(m)) is not None
        ]
        all_design_metrics = self._gather_metrics(local_design_metrics)

        metrics_dir = Path(self.store_dir) / ("val_metrics" if not self.is_inference else "inference_metrics")

        # 3. Oracle (fold) + refolding + pass_criteria, per rank.
        oracle_mm = self.oracle_metric_manager
        refold_mm = self.refolding_metric_manager
        pass_mm = self.pass_criteria_metric_manager

        # Non-search runs deferred MM setup until now so weight-loading oracles
        # (e.g. RF3) don't sit on GPU during diffusion; daemon oracles (AF3) don't
        # care. Search runs already set up these managers in on_validation_epoch_start.
        if not self._oracle_mms_setup_this_epoch:
            # Set up the oracle metric managers, if not already set up by the search
            # strategy in on_validation_epoch_start.
            for mm in [oracle_mm, refold_mm, pass_mm]:
                if mm is not None:
                    mm.on_validation_epoch_start(val_dataset_tags=self.val_dataset_tags)
            self._oracle_mms_setup_this_epoch = True

        local_refolding_metrics: list[dict] = []
        example_id_to_design_metrics = {m["example_id"]: m for m in (local_design_metrics or []) if m}
        items = sorted(
            self.validation_output_generation,
            key=lambda it: example_id_to_design_metrics.get(it.get("example_id", ""), {}).get("token_count", 0),
        )

        # 3b. Oracle fold_batch — the actual structure prediction (AF3 or RF3).
        if oracle_mm is not None:
            # At inference, we don't want to crash the whole run for partially-incomplete outputs
            # We also don't want to load the value from the checkpoint's training config
            on_incomplete = "warn" if self.is_inference else self.cfg_exp.get("val_refold_coverage_mode", "raise")
            fold_designs_for_oracles(
                items,
                oracle_mm,
                on_incomplete=on_incomplete,
            )

        post_fold_payloads = [
            {
                "val_output": item,
                "design_metrics": example_id_to_design_metrics.get(item.get("example_id", ""), {}),
            }
            for item in items
        ]

        def _post_fold_inline(payload: dict) -> dict | None:
            return compute_oracle_metrics_for_val_output(
                payload["val_output"],
                oracle_mm,
                refold_mm,
                pass_mm,
                prior_design_metrics_by_id={
                    payload["val_output"].get("example_id", ""): payload.get("design_metrics") or {}
                },
            )

        # 3c. Post-fold metrics (refolding RMSDs, pass criteria) on workers.
        for result in self._metrics_pool.map_post_fold(post_fold_payloads, _post_fold_inline):
            if exists(result):
                cleaned = _clean_metrics_dict(result)
                if cleaned:
                    local_refolding_metrics.append(cleaned)
        for mm in [oracle_mm, refold_mm, pass_mm]:
            if exists(mm):
                mm.on_validation_epoch_end()

        # 4. Gather refolding metrics to rank 0.
        all_refolding_metrics = self._gather_metrics(local_refolding_metrics)

        # 5. Distributed foldseek (no-op for world_size==1 / FOLDSEEK_EXEC unset).
        foldseek_cluster_map = run_foldseek_distributed(
            all_design_metrics,
            world_size=self.trainer.world_size,
            global_rank=self.global_rank,
            is_inference=self.is_inference,
        )

        if self.global_rank != 0:
            return  # non-zero ranks exit after the foldseek-step collectives

        # 6. Rank 0 writes the per-example CSV + summary logs.
        if not all_design_metrics and not all_refolding_metrics:
            logger.warning("on_validation_epoch_end_generation: no metrics to log")
            return

        # Merge design and refolding metrics per example_id.
        example_id_to_design_metrics = {m["example_id"]: m for m in (all_design_metrics or []) if m}
        refold_by_id = {m["example_id"]: m for m in (all_refolding_metrics or []) if m}
        all_ids = list(example_id_to_design_metrics) + [
            eid for eid in refold_by_id if eid not in example_id_to_design_metrics
        ]
        combined_metrics = [
            {**example_id_to_design_metrics.get(eid, {}), **refold_by_id.get(eid, {})} for eid in all_ids
        ]

        if not combined_metrics:
            return

        # Merge distributed-foldseek cluster IDs into combined_metrics.
        apply_cluster_map_to_metrics(combined_metrics, foldseek_cluster_map)

        self._save_and_log_csv(combined_metrics, metrics_dir, csv_stem="per_example_metrics")

    def _output_prefix_suffix(self) -> tuple[str, str]:
        """Return (prefix, suffix) strings for output filenames."""
        if self.is_inference:
            prefix = "" if self.global_prefix is None else self.global_prefix + "_"
            suffix = f"_job{self.job_id}" if self.job_id is not None else ""
        else:
            prefix = ""
            suffix = f"_step{self.global_step}"
        return prefix, suffix

    def _save_and_log_csv(self, all_metrics: list[dict], metrics_dir: Path, csv_stem: str) -> None:
        """Save per-example CSV, aggregate, and log to WandB (rank 0 only)."""
        metrics_dir.mkdir(parents=True, exist_ok=True)
        output_prefix, output_suffix = self._output_prefix_suffix()

        per_example_df = pd.DataFrame(all_metrics).round(4)

        # Reorder columns: metadata first, then metrics.
        cols = per_example_df.columns.tolist()
        priority = [
            "example_id",
            "designed_structure_path",
            "task",
            "dataset_name",
            "chain_of_interest",
            "ckpt_path",
            "ckpt_name",
            "seed",
            "input_cif_path",
            "input_json_path",
        ]
        ordered_cols = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
        per_example_df = per_example_df[ordered_cols]

        # Solve-rate pass columns are exactly the PassCriterionMetric outputs (identified by metric
        # type, not column name), passed explicitly so nothing else can be mistaken for a criterion.
        pass_cols = (
            [f"{name}.pass" for name in self.pass_criteria_metric_manager.metrics]
            if self.pass_criteria_metric_manager is not None
            else None
        )

        # Foldseek cluster IDs were already populated by :py:func:`run_foldseek_distributed`
        # in :py:meth:`on_validation_epoch_end` (covers both single-process and DDP).
        if "foldseek_cluster_id" in per_example_df.columns and per_example_df["foldseek_cluster_id"].notna().any():
            try:
                per_example_df = add_diversity_summary_metrics(
                    per_example_df,
                    group_by=["dataset_name", "task"],
                    pass_cols=pass_cols,
                )
            except Exception as e:
                logger.warning(f"Foldseek diversity summary metrics failed, skipping cluster metrics: {e}")

        per_example_path = metrics_dir / f"{output_prefix}{csv_stem}{output_suffix}.csv"
        per_example_df.to_csv(per_example_path, index=False)
        logger.info(f"Saved {csv_stem} to {per_example_path}")

        agg_rows = []
        _non_metric_cols = {
            "example_id",
            "dataset_name",
            "task",
            "designed_structure_path",
            "chain_of_interest",
            "foldseek_cluster_id",
            "ckpt_path",
            "ckpt_name",
            "seed",
            "input_cif_path",
            "input_json_path",
            "metrics_tags",
            "output_base_dir",
        }
        # MetricManager names every column "{metric_name}.{output_key}", so a column's owning
        # metric is whatever registered under the part before the first dot.
        metrics_by_name = {
            name: m
            for mgr in (
                self.design_metric_manager,
                self.oracle_metric_manager,
                self.refolding_metric_manager,
                self.pass_criteria_metric_manager,
            )
            if mgr is not None
            for name, m in mgr.metrics.items()
        }
        dataset_col = (
            per_example_df["dataset_name"]
            if "dataset_name" in per_example_df.columns
            else pd.Series(["unknown"] * len(per_example_df))
        )
        for ds in dataset_col.unique():
            ds_df = per_example_df[dataset_col == ds]
            for col in ds_df.columns:
                if col in _non_metric_cols or col.startswith("generation."):
                    continue
                vals = pd.to_numeric(ds_df[col], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                name, _, suffix = col.partition(".")
                metric = metrics_by_name.get(name)
                if metric is not None:
                    reductions = metric.aggregate(suffix, vals)
                else:
                    # No owning metric (e.g. foldseek-derived column): fall back to mean.
                    reductions = {"mean": float(np.nanmean(vals))}
                for red_name, value in reductions.items():
                    agg_rows.append(
                        {"dataset": ds, "metric": col, "reduction": red_name, "value": value, "n": len(vals)}
                    )

            # Dataset-level solve-rate statistics (e.g., percentage of targets with at least n% passing designs)
            # n per row is that oracle's valid-target denominator (solve_rate_n_targets)
            solve = compute_solve_rate_metrics(ds_df, pass_cols=pass_cols)
            for metric_name, value in solve.items():
                prefix = metric_name[: metric_name.rindex("solve_rate_")]  # "af3." (or "" for a bare pass col)
                n_used = int(solve.get(f"{prefix}solve_rate_n_targets", 0))
                # Solve-rate rows are not per-example reductions; leave reduction empty so
                # their wandb keys stay unsuffixed (val_generation.{ds}/{metric}).
                agg_rows.append({"dataset": ds, "metric": metric_name, "reduction": "", "value": value, "n": n_used})

        if not agg_rows:
            return

        agg_df = pd.DataFrame(agg_rows).round(4)
        print_df_as_table(agg_df, title=csv_stem)
        agg_path = metrics_dir / f"{output_prefix}aggregated_metrics{output_suffix}.csv"
        agg_df.to_csv(agg_path, index=False)
        if wandb.run is not None:
            wandb.log(
                {
                    (
                        f"val_generation.{row['dataset']}/{row['metric']}.{row['reduction']}"
                        if row["reduction"]
                        else f"val_generation.{row['dataset']}/{row['metric']}"
                    ): row["value"]
                    for _, row in agg_df.iterrows()
                },
                step=self.global_step,
            )

    def configure_metrics(
        self,
        metrics_cfg: DictConfig | None = None,
    ) -> None:
        """Instantiate design, oracle, refolding, and pass-criteria MetricManagers from config.

        Note that we allow a metrics_cfg other than self.cfg_exp.metrics to be passed via (e.g.) generate.py
        This is necessary since by default the cfg_exp used to train a checkpoint is loaded with it.
        """

        if metrics_cfg is None:
            self.design_metric_manager = None
            self.oracle_metric_manager = None
            self.refolding_metric_manager = None
            self.pass_criteria_metric_manager = None
            self.pre_oracle_filter_metric_manager = None
            return

        cfg_key_to_attr = {
            "inline_metrics": "design_metric_manager",
            "refolding_oracles": "oracle_metric_manager",
            "refolding_metrics": "refolding_metric_manager",
            "pass_criteria": "pass_criteria_metric_manager",
            "pre_oracle_filters": "pre_oracle_filter_metric_manager",
        }
        for cfg_key, attr_name in cfg_key_to_attr.items():
            metric_manager_cfg = metrics_cfg.get(cfg_key)
            if exists(metric_manager_cfg):
                metric_manager = MetricManager.instantiate_from_hydra(metrics_cfg=metric_manager_cfg)
            else:
                metric_manager = None
            setattr(self, attr_name, metric_manager)

        if self.pre_oracle_filter_metric_manager is not None:
            for name, metric in self.pre_oracle_filter_metric_manager.metrics.items():
                if not metric.required_tags_any and not metric.required_tags_all:
                    raise ValueError(
                        f"pre_oracle_filters metric '{name}' has no required_tags_any or "
                        f"required_tags_all. Every pre-oracle filter must be tag-gated so it "
                        f"cannot fire unconditionally and block designs unexpectedly."
                    )

        if hasattr(self.cfg_exp, "datasets") and hasattr(self.cfg_exp.datasets, "val"):
            self.val_dataset_tags = _collect_val_metrics_tags(self.cfg_exp.datasets.val)
        else:
            self.val_dataset_tags = None

        # Warn if multiple oracles apply to the same dataset
        if self.val_dataset_tags is not None and self.oracle_metric_manager is not None:
            for dataset_tags in self.val_dataset_tags:
                if dataset_tags is None:
                    continue
                dataset_oracle_name = None
                for oracle_name, oracle_metric in self.oracle_metric_manager.metrics.items():
                    if oracle_metric.applies_to_tags(dataset_tags):
                        if dataset_oracle_name is None:
                            dataset_oracle_name = oracle_name
                        else:
                            logger.warning(
                                f"Multiple oracle metrics ({dataset_oracle_name}, {oracle_name}) apply to dataset"
                                f" with tags {dataset_tags}. "
                                "This is intentional for some inference configs but may indicate a mistake."
                            )

        # Per-rank worker pool for design + post-fold metrics. See `MetricsPool`.
        self._metrics_pool = MetricsPool(
            num_workers=auto_pick_metrics_workers(self.cfg_exp),
            mm_cfgs=resolve_mm_cfgs_from_metrics(metrics_cfg),
        )

    def configure_inference(
        self,
        inf_cfg: DictConfig,
        nn_ag: torch.nn.Module | None,
        store_dir: str | None = None,
        metrics_cfg: DictConfig | None = None,
        job_id: int | None = None,
        is_inference: bool = False,
        global_prefix: str | None = None,
        ckpt_path: str | None = None,
        ckpt_name: str | None = None,
        save_chain_of_interest: bool = False,
    ) -> None:
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided).

        Args:
            inf_cfg: Inference configuration.
            nn_ag: Autoguidance neural network (or None).
            store_dir: Directory to store outputs.
            metrics_cfg: Metrics configuration.
            job_id: Job ID used to disambiguate per-job output files.
            is_inference: Whether running in inference mode (via generate.py).
            global_prefix: Global prefix for inference output filenames.
            ckpt_path: Directory containing the checkpoint (recorded in output metrics).
            ckpt_name: Checkpoint filename (recorded in output metrics).
            save_chain_of_interest: Whether to write per-design chain-of-interest CIFs.
                These are only consumed by foldseek clustering, so they default to ``False``
                for inference; the launch script enables them when foldseek will run.
        """
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag
        if store_dir is not None:
            self.store_dir = store_dir

        self.configure_metrics(metrics_cfg)

        self.is_inference = is_inference
        self.job_id = job_id
        self.global_prefix = global_prefix
        self.ckpt_path = ckpt_path
        self.ckpt_name = ckpt_name
        self.save_chain_of_interest = save_chain_of_interest

    def configure_search(
        self,
        search_strategy: "SearchStrategy",
        search_reward_model: "RewardModel | None" = None,
    ) -> None:
        """Configure search strategy and reward model for reward-guided generation.

        Args:
            search_strategy: SearchStrategy instance (BeamSearch, Vanilla, etc.)
            search_reward_model: RewardModel instance for scoring candidates.
        """
        self.search_strategy = search_strategy
        self.search_reward_model = search_reward_model
        logger.info(f"Configured search strategy: {type(search_strategy).__name__}")
        if search_reward_model:
            logger.info(
                f"Configured search reward model: "
                f"reward_metrics_tags={search_reward_model.reward_metrics_tags}, "
                f"weights={list(search_reward_model.weights.keys())}"
            )

    def configure_guidance(self, guidance_model: Any) -> None:
        """Configure differentiable-metric gradient guidance applied during sampling.

        Args:
            guidance_model: :py:class:`~rfproteina.guidance.guidance_model.GuidanceModel`
                instance whose terms nudge the trajectory each sampling step.
        """
        self.guidance_model = guidance_model
        logger.info(f"Configured guidance with terms: {list(guidance_model.terms.keys())}")

    def decode_for_guidance(
        self,
        state: dict[str, torch.Tensor],
        extra_info: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Decode grad-enabled state leaves to an atom37 namespace for guidance metrics.

        Mirrors the generation decode (:py:meth:`_format_sample_local_latents`) so guidance
        metrics see the same coordinates the sampler emits, but on the grad tape: gradients
        flow from the returned coordinates back to ``state['local_latents']`` and
        ``state['bb_ca']``. Invoked by
        :py:class:`~rfproteina.guidance.guidance_model.GuidanceModel` only when an
        active term reads the ``decoded`` namespace.

        Args:
            state: Grad-enabled clean prediction with ``bb_ca`` and ``local_latents``.
            extra_info: Must contain ``padding_mask`` and ``can_be_encoded``.

        Returns:
            Dict with ``atom37_coords`` ([b, n, 37, 3], Angstrom, grad), ``residue_type``
            ([b, n], discrete), and ``padding_mask`` ([b, n]).
        """
        formatted = self._format_sample_local_latents(
            x=state,
            ret_mode="coors37_n_aatype",
            padding_mask=extra_info["padding_mask"],
            extra_info={"can_be_encoded": extra_info["can_be_encoded"]},
        )
        return {
            "atom37_coords": formatted["coors"],
            "residue_type": formatted["residue_type"],
            "padding_mask": formatted["padding_mask"],
        }

    def insertion_mapping_for_guidance(
        self,
        _state: dict[str, torch.Tensor],
        extra_info: dict[str, Any],
    ) -> list[dict[int, int]]:
        """Resolve the insertion head's predicted token mapping for guidance metrics.

        Mirrors the inference resolution (:py:func:`build_predicted_insertion_mapping`): per
        sample, slice the current step's insertion logits to valid tokens and run per-chain
        Hungarian assignment of unindexed → generated tokens. The mapping is discrete (token
        ids), so it carries no gradient -- it is contextual metadata, like ``residue_type``.
        Invoked by :py:class:`~rfproteina.guidance.guidance_model.GuidanceModel` only
        when an active term reads the ``predicted_insertion_mapping`` namespace.

        Args:
            _state: Grad-enabled clean prediction. Unused here (the mapping depends only on
                logits + static token metadata); present for a uniform provider signature
                shared with :py:meth:`decode_for_guidance`.
            extra_info: Must contain ``insertion_logits`` ([b, n, n] from the insertion head),
                ``padding_mask`` ([b, n]), and ``atom_array_batch`` (per-sample AtomArrays).

        Returns:
            One ``{unindexed_token_id: generated_token_id}`` dict per batch sample (empty when
            logits/atom arrays are unavailable or a sample has no unindexed tokens).
        """
        padding_mask = extra_info["padding_mask"]
        insertion_logits = extra_info.get("insertion_logits")
        atom_array_batch = extra_info.get("atom_array_batch")
        if insertion_logits is None or atom_array_batch is None:
            return [{} for _ in range(padding_mask.shape[0])]

        mappings: list[dict[int, int]] = []
        for i in range(insertion_logits.shape[0]):
            pad_i = padding_mask[i]  # [n]
            logits_i = insertion_logits[i][pad_i][:, pad_i]  # [n_valid, n_valid]
            try:
                mappings.append(build_predicted_insertion_mapping(logits_i, atom_array_batch[i]))
            except (AttributeError, AssertionError, ValueError, RuntimeError) as e:
                logger.warning(f"Guidance insertion mapping failed for sample {i}: {e}")
                mappings.append({})
        return mappings

    def preprocess_batch_for_generation(self, batch: dict) -> dict:
        """Preprocess batch from atom37 collate format for generation.

        This is used by both vanilla generation and search strategies.

        Args:
            batch: Raw batch from dataloader

        Returns:
            Preprocessed batch ready for generation
        """
        batch["coords_nm"] = batch["coords"] / 10.0  # Convert Angstroms to nm
        batch["coord_mask"] = batch["atom37_mask"]

        return batch

    def predict_step(self, batch: dict, batch_idx: int) -> list[dict]:
        """
        Makes predictions. Should call set_inf_cfg before calling this.

        Args:
            batch: Fully atomized data batch (same format as training).

        Returns:
            List of dictionaries, one per sample. Each dict contains:
                - "coors": coordinates tensor of shape [n, 37, 3]
                - "residue_type": aatype tensor of shape [n]
                - "padding_mask": mask tensor of shape [n]
                - "input_atom_array": input AtomArray object (before generation)
                - "generated_atom_array": output AtomArray object with generated coordinates
        """
        # Always use search_strategy interface for consistency
        # Default to VanillaStrategy if no search strategy is configured
        if self.search_strategy is None:
            from rfproteina.search import VanillaStrategy

            strategy = VanillaStrategy()
        else:
            strategy = self.search_strategy

        results = strategy.generate(self, batch, self.search_reward_model)

        return results

    def _process_single_sample_to_atom_array(
        self,
        coords: torch.Tensor,
        residue_type: torch.Tensor,
        padding_mask: torch.Tensor,
        input_atom_array: Any,
        sample_idx: int = 0,
        batch_idx: int = 0,
        bb_ca_only: bool = False,
        insertion_logits: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
    ) -> dict:
        """Convert one sample's tensor outputs into output AtomArrays.

        Steps:
        - Converts model output coordinate and residue tensors into an AtomArray
        - Removes padding and in-context conditioning tokens
        - Annotates unindexed insertion tokens based on insertion prediction head

        Args:
            coords: Generated coordinates [n_max_tokens, 37, 3].
            residue_type: Residue types [n_max_tokens].
            padding_mask: Padding mask [n_max_tokens].
            input_atom_array: Original input atom array.
            bb_ca_only: If True, subset ``input_atom_array`` to CA atoms only.
            insertion_logits: Optional [n_max_tokens, n_max_tokens] logits from the prediction head

        Returns:
            Dict with an annotated ``generated_atom_array`` and ``input_atom_array``.
        """
        return process_single_sample_to_atom_array(
            coords=coords,
            residue_type=residue_type,
            padding_mask=padding_mask,
            input_atom_array=input_atom_array,
            bb_ca_only=bb_ca_only,
            atom_mask=atom_mask,
            insertion_logits=insertion_logits,
        )

    def _build_staged_payload(
        self,
        sample: dict,
        example_id: str,
        metrics_tags: list[str],
        output_base: Path,
        generated_output_path: Path,
        dataset_name: str | None = None,
        task: str | None = None,
    ) -> dict:
        """Assemble the input dict for ``_stage_generated_structure``."""
        inf_args = self.inf_cfg.args if self.inf_cfg is not None else None
        bb_ca_only = bool(inf_args and inf_args.get("bb_ca_only_trajectories", False))
        save_traj_every = int(inf_args.get("save_trajectory_every", 0)) if inf_args else 0
        nsteps = inf_args.nsteps if inf_args is not None else None

        trajectories: dict[str, tuple] = {}
        for key in ("trajectory_x1", "trajectory_xt"):
            if key not in sample:
                continue
            traj = sample[key]
            if bb_ca_only:
                step_labels = np.arange(traj.stack_depth()) * save_traj_every
                if nsteps is not None:
                    step_labels[-1] = nsteps
                trajectories[key] = (traj, step_labels)
            else:
                trajectories[key] = (traj, None)  # CIF path; no step labels

        return {
            "example_id": example_id,
            "generated_atom_array": sample["generated_atom_array"],
            "token_count": get_token_count(sample["generated_atom_array"]),
            "input_atom_array": sample.get("input_atom_array"),
            "full_array_with_unindexed": sample.get("full_array_with_unindexed"),
            "predicted_insertion_mapping": sample.get("predicted_insertion_mapping") or {},
            "metrics_tags": metrics_tags,
            "output_base_dir": str(output_base),
            "generated_output_path": str(generated_output_path),
            "save_input": not self.is_inference,
            "save_chain_of_interest": self.save_chain_of_interest,
            "trajectories": trajectories,
            "dataset_name": dataset_name,
            "task": task,
        }

    def format_sample_prots_to_atom_arrays(
        self,
        sample_prots: dict,
        batch: dict,
        batch_idx: int = 0,
        bb_ca_only: bool = False,
        example_ids: list[str] | None = None,
        insertion_logits: torch.Tensor | None = None,
    ) -> list[dict]:
        """Convert per-sample tensors back to AtomArrays for an entire batch.

        Args:
            insertion_logits: Optional [nsamples, n, n] logits from the insertion prediction head.
        """
        generation_list: list[dict] = []
        n = sample_prots["coors"].shape[0]
        for i in range(n):
            orig_batch_idx = i % len(batch.get("atom_array_batch", [batch.get("atom_array")]))
            input_atom_array = (
                batch["atom_array_batch"][orig_batch_idx] if "atom_array_batch" in batch else batch["atom_array"]
            )

            sample_insertion_logits = insertion_logits[i] if insertion_logits is not None else None

            result = self._process_single_sample_to_atom_array(
                coords=sample_prots["coors"][i],
                residue_type=sample_prots["residue_type"][i],
                padding_mask=sample_prots["padding_mask"][i],
                input_atom_array=input_atom_array,
                bb_ca_only=bb_ca_only,
                insertion_logits=sample_insertion_logits,
            )
            if result is None:
                continue
            result["raw_batch_id"] = batch["example_id_batch"][orig_batch_idx]
            result["example_id"] = example_ids[i] if example_ids is not None else result["raw_batch_id"]
            generation_list.append(result)

        return generation_list

    def sample_formatting(
        self,
        x: dict[str, Tensor],
        extra_info: dict[str, Tensor],
        ret_mode: str,
    ) -> Any:
        """
        Given a batch of b samples x produced by the flow matcher, it returns the samples in the
        requested format (ret_mode).

        Supports `ret_modes` for:
            - `samples` returns the original sample from the flow matcher, a
            dictionary[str, Tensor]. for the data modalities, each with batch shape b.
            - `atom37` returns an Tensor of shape [b, n, 37, 3] just for coordinates.
            - `pdb_string` returns a list of dictionaries {"pdb_string": str, "nres": int},
            with one dictionary per sample.
            - `coors37_n_aatype` returns a dictionary with keys `coors` (atom37),
            `residue_type`, and `padding_mask`, and values with shapes [b, n, 37, 3] float,
            [b, n] int, [b, n] boolean, respectively.

        Args:
            x: sample.
            extra_info: a dict with additional things, including:
                - "padding_mask", whose value is boolean of shape [nsamples, n]
                - ...
            ret_mode: target format, for now only supports atom37.

        Returns:
            Sample x in the requested format.
        """
        data_modes = sorted(self.cfg_exp.product_flowmatcher)
        if data_modes == ["bb_ca"]:
            return self._format_sample_bb_ca(x=x, ret_mode=ret_mode, padding_mask=extra_info["padding_mask"])
        elif data_modes == ["bb_ca", "local_latents"]:
            return self._format_sample_local_latents(
                x=x,
                ret_mode=ret_mode,
                padding_mask=extra_info["padding_mask"],
                extra_info=extra_info,
            )
        else:
            raise NotImplementedError(f"Format {ret_mode} not implemented")

    def _format_sample_bb_ca(
        self,
        x: dict[str, torch.Tensor],
        ret_mode: str,
        padding_mask: Bool[torch.Tensor, "b n"],
    ) -> dict:
        if ret_mode == "coors37_n_aatype":
            coors = trans_nm_to_atom37(x["bb_ca"].float()) * padding_mask[..., None, None]  # [b, n, 37, 3]
            residue_type = torch.full(coors[..., 0, 0].shape, -1, device=coors.device, dtype=torch.long)  # [b, n]
            return {
                "coors": coors,  # [b, n, 37, 3]
                "residue_type": residue_type.long(),  # [b, n]
                "padding_mask": padding_mask,  # [b, n]
            }
        else:
            raise NotImplementedError(f"{ret_mode} format for data modes `[bb_ca]` not implemented")

    def _format_sample_local_latents(
        self,
        x: dict[str, torch.Tensor],
        ret_mode: str,
        padding_mask: Bool[torch.Tensor, "b n"],
        extra_info: dict[str, torch.Tensor],
    ) -> dict:
        """
        Given a batch of b samples consisting on `bb_ca` and `local_latents` this
        returns formatted samples.

        Note: This calls the decoder from the autoencoder, since it needs to go from
        local latent variables to the actual coordinates and sequence.

        Note: The self.autoencoder.decode function (used here) returns a dictoinary like
        {
            "coors_nm": [b, n, 37, 3], already masked
            "residue_type": [b, n], already masked, careful with 0s
            "residue_mask": [b, n]
            "atom_mask": [b, n, 37]
        }

        Args:
            x: sample.
            ret_mode: target format.
            padding_mask: mask tensor of shape [b, n]
            extra_info: dict with additional info including can_be_encoded

        Returns:
            Sample x in the requested format.
        """
        if ret_mode == "coors37_n_aatype":
            # Only pass generated tokens to decoder (like encoder during training)
            can_be_encoded = extra_info["can_be_encoded"]

            # Decode only generated tokens
            b, n = x["bb_ca"].shape[0], x["bb_ca"].shape[1]

            # Initialize output tensors with zeros for coordinates/residue mask...
            coors_nm_full = torch.zeros(b, n, 37, 3, device=x["bb_ca"].device, dtype=x["bb_ca"].dtype)
            residue_mask_full = torch.zeros(b, n, device=x["bb_ca"].device, dtype=torch.bool)

            # ... and -1 for residue types (so that unfilled tokens are -1 vs. 0 which conflicts with alanine)
            residue_type_full = torch.full((b, n), -1, device=x["bb_ca"].device, dtype=torch.long)

            # Process each batch element separately (since different samples may have different generated masks)
            for i in range(b):
                encoded_mask_i = can_be_encoded[i]  # [n]
                padding_mask_i = padding_mask[i]  # [n]
                if encoded_mask_i.any():
                    # Extract tokens with latents
                    latents_gen = x["local_latents"][i][encoded_mask_i].unsqueeze(0)  # [1, n_enc, d]
                    ca_gen = x["bb_ca"][i][encoded_mask_i].unsqueeze(0)  # [1, n_enc, 3]
                    mask_gen = torch.ones(1, encoded_mask_i.sum().item(), device=x["bb_ca"].device, dtype=torch.bool)

                    # Decode latents
                    output_decoder = self.autoencoder.decode(z_latent=latents_gen, ca_coors_nm=ca_gen, mask=mask_gen)

                    # Place decoded outputs back into full tensors
                    coors_nm_full[i, encoded_mask_i] = output_decoder["coors_nm"][0]
                    residue_type_full[i, encoded_mask_i] = output_decoder["residue_type"][0]
                    residue_mask_full[i, encoded_mask_i] = output_decoder["residue_mask"][0]

                if (no_latent_tokens := (~encoded_mask_i & padding_mask_i)).any():
                    # extract tokens with no latents (unfixed ligands, etc.)
                    coors_nm_full[i, no_latent_tokens, 1] = x["bb_ca"][i, no_latent_tokens]
                    residue_mask_full[i, no_latent_tokens] = 0
                    # (We use -1 residue mask to indicate these tokens were not encoded/decoded)
                    residue_type_full[i, no_latent_tokens] = -1

            # Legacy AEs (n_residue_types=20) were trained with a -1 offset applied to
            # Legacy AEs (n_residue_types=20) were trained with a -1 offset applied to
            # the pipeline's UNIFIED_ATOM37_ENCODING indices (ALA=1 -> 0, etc.), so the
            # decoder outputs in that shifted space and we must undo it here. Unified AEs
            # (n_residue_types=33) use the encoding indices directly -- no shift needed.
            if not self._use_unified_ae_encoding:
                residue_type_full = torch.where(residue_type_full >= 0, residue_type_full + 1, residue_type_full)

            return {
                "coors": nm_to_ang(coors_nm_full),  # [b, n, 37, 3] - padded, use mask to slice
                "residue_type": residue_type_full,  # [b, n] - padded, use mask to slice
                "padding_mask": padding_mask,  # [b, n] - True for non-padding tokens
            }
        else:
            raise NotImplementedError(f"{ret_mode} format for data modes `[bb_ca, latent_locals]` not implemented")
