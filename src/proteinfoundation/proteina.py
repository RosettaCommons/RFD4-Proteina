import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import biotite.structure as struc
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb

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
            from transformer_engine.pytorch import autocast as te_autocast_mod
            from transformer_engine.common.recipe import (
                DelayedScaling as DelayedScaling_mod,
                Format as Format_mod,
            )
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
from atomworks.io.utils.atom_array_plus import as_atom_array_plus, concatenate_atom_array_plus
from atomworks.ml.utils.condition import save_atom_array_with_conditions_to_cif
from atomworks.ml.conditions import C_IDX, C_CRD
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
from atomworks.ml.transforms.encoding import atom_array_from_encoding
from atomworks.ml.transforms.filters import remove_protein_terminal_oxygen
from atomworks.ml.utils.token import apply_and_spread_token_wise, apply_token_wise, get_token_count, get_token_starts
from jaxtyping import Bool, Float
from lightning_utilities import apply_to_collection
from foundry.metrics.metric import MetricManager
from foundry.utils.ddp import RankedLogger
from foundry.utils.logging import print_df_as_table, safe_print
from omegaconf import OmegaConf
from rich.panel import Panel
from torch import Tensor

from atomworks.ml.utils.nested_dict import flatten as flatten_nested_dict

from proteinfoundation.conditions import C_INTCRD, C_CRDGRP
from proteinfoundation.flow_matching.product_space_flow_matcher import (
    ProductSpaceFlowMatcher,
)
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.transforms.unindexing import process_unindexed_outputs
from proteinfoundation.transforms.utils.conditioning_utils import insert_internal_coord_conditioned_groups
from proteinfoundation.utils.coordinate_transfer import copy_generated_coordinates_to_input
from proteinfoundation.utils.coors_utils import nm_to_ang, trans_nm_to_atom37

logger = RankedLogger(__name__, rank_zero_only=True)


_last_log_time: dict[str, float] = {}  # Track last log time per stage


def _log_batch_example_ids(
    example_ids: list[str],
    epoch: int,
    batch_idx: int,
    stage: str,
    log_frequency: int = 100,
) -> None:
    """Log example IDs and timing for debugging batch ordering.

    Logs every log_frequency batches to verify shuffling behavior across epochs.
    Also logs time elapsed since last log and steps per second.
    """
    if batch_idx % log_frequency != 0:
        return

    import time

    current_time = time.time()
    elapsed = current_time - _last_log_time.get(stage, current_time)
    steps_per_sec = log_frequency / elapsed if elapsed > 0 else 0.0
    _last_log_time[stage] = current_time

    ids_preview = f"{example_ids[:3]}, ... ({len(example_ids) - 3} more)" if len(example_ids) > 3 else example_ids
    logger.info(
        f"[{stage}] epoch={epoch}, batch={batch_idx}, "
        f"steps={log_frequency}, elapsed={elapsed:.1f}s, steps/s={steps_per_sec:.2f}, "
        f"n_examples={len(example_ids)}, ids={ids_preview}"
    )
    


def _clean_metrics_dict(metrics: dict) -> dict | None:
    """Convert a metrics dict to clean floats for gathering across ranks."""
    if metrics is None:
        return None

    # Standardize types
    cleaned = {}
    chain_of_interest_values = []
    for key, value in metrics.items():
        if key in ("dataset_name", "example_id", "task"):
            cleaned[key] = str(value)
            continue
        elif key == "input_cif_path" or key.startswith("input_spec."):
            cleaned[key] = str(value) if value is not None else None
            continue
        elif key.endswith("chain_of_interest"):
            chain_of_interest_values.append(str(value))
            continue
        if isinstance(value, int | float):
            cleaned[key] = float(value)
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            cleaned[key] = value.item()

    # Add chain of interest column
    if len(set(chain_of_interest_values)) == 1:
        cleaned["chain_of_interest"] = chain_of_interest_values[0]
    elif len(set(chain_of_interest_values)) > 1:
        logger.warning(f"Multiple different chains of interest reported: {set(chain_of_interest_values)}")
        cleaned["chain_of_interest"] = None
    else:
        cleaned["chain_of_interest"] = None

    # Return None if only metadata (no actual metrics)
    metric_keys = [k for k in cleaned if k not in ("dataset_name", "example_id", "task", "chain_of_interest")]
    return cleaned if metric_keys else None


class Proteina(L.LightningModule):
    def __init__(self, cfg_exp, store_dir=None, autoencoder_ckpt_path=None):
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
        self.offline_validation = False
        self.is_inference = False
        self.inference_seed: int | None = None  # Set via generate.py, not during training

        self.inf_cfg = inf_cfg  # Used for inference-like validation
        self.store_dir = store_dir if store_dir is not None else os.path.join(".", "store", cfg_exp.run_name_)
        self.metrics = MetricManager.instantiate_from_hydra(metrics_cfg=cfg_exp.metrics) if cfg_exp.metrics else None

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
            except:
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

        # Search integration
        self.search_strategy = None
        self.search_reward_model = None

        # Refinement integration (may share reward model or use separate one)
        self.refinement_strategy = None
        self.refinement_reward_model = None

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

    def load_autoencoder(self, cfg_exp, freeze_params=True):
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
            return None, None

        logger.info(f"Loading autoencoder from {ae_ckp_path}")
        autoencoder = AutoEncoder.load_from_checkpoint(ae_ckp_path, strict=False, weights_only=False)
        if freeze_params:
            for param in autoencoder.parameters():
                param.requires_grad = False
        return autoencoder, autoencoder.latent_dim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.cfg_exp.opt.lr)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except:
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

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
        # First call
        nn_out = self.nn(batch)

        # Recycle n_recycle times detaching gradients and updating input
        # Note that recycling is supported by the codebase, but the models provided
        # with the La-Proteina paper do not use it, nor were trained with it.
        for _ in range(n_recycle):
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
            batch["x_recycle"] = {dm: x_1_pred[dm].detach() for dm in x_1_pred}
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

    def training_step(self, batch: dict, batch_idx: int):
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batch dimension.
        """
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
        )

        # Preprocess batch from atom37 collate format
        batch["coords_nm"] = batch["coords"] / 10.0  # Convert Angstroms to nm
        batch["coord_mask"] = batch["atom37_mask"]
        # residue_type already provided by atom37 collate

        # Add co_generate_targets flag from config
        batch["co_generate_targets"] = self.cfg_exp.get("co_generate_targets", True)

        # FP8 autocast context for Transformer Engine
        fp8_ctx = te_autocast(enabled=True, recipe=self.fp8_recipe) if self.use_fp8 else nullcontext()

        with fp8_ctx:
            # Add clean samples for all data modes / spaces we are working on
            batch = self.add_clean_samples(batch)

            # Corrupt the batch
            batch = self.fm.corrupt_batch(batch)  # adds x_1, t, x_0, x_t
            bs, n = batch["padding_mask"].shape

            # Handle conditioning variables
            batch = self.handle_self_cond(batch)  # self conditioning, adds ["x_sc"] to batch prob 0.5

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
                sync_dist=True,
                add_dataloader_idx=False,
            )

            if not val_step:  # Don't log these for val step
                self.log_train_loss_n_prog_bar(bs, train_loss)
                self.update_n_log_flops(bs, n)
                self.update_n_log_nsamples_processed(bs)
                self.log_nparams()

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
        batch["x_1"] = {dm: self._get_clean_sample(batch, dm) for dm in self.cfg_exp.product_flowmatcher}
        return batch

    def _get_clean_sample(self, batch: dict, dm: str) -> torch.Tensor:
        """
        Gets clean sample for a given data mode.

        Args:
            batch: Batch to get clean sample from.
            dm: Data mode to get clean sample for.

        Returns:
            Clean sample for the given data mode.
            For bb_ca, we provide coords for all center atoms (even though some are not resolved and not that meaningful).
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

            # Change this once we retrain AE with correct token encoding
            residue_type_for_ae = batch["residue_type"] - 1  # [b, n_tokens]
            # Clamp for non-encodable tokens to avoid -1 index (invalid for embeddings)
            residue_type_for_ae = torch.where(encodable_mask, residue_type_for_ae, torch.zeros_like(residue_type_for_ae))
            ae_batch = {
                "coords": batch["coords"],  # [b, n_tokens, 37, 3]
                "coords_nm": batch["coords_nm"],  # [b, n_tokens, 37, 3]
                "coord_mask": batch.get("coord_mask", torch.ones_like(batch["coords"][..., 0])),  # [b, n_tokens, 37]
                "residue_type": residue_type_for_ae,  # [b, n_tokens]
                "padding_mask": encodable_mask,  # [b, n_tokens] - only encode these
                "can_be_encoded": encodable_mask,  # Pass to feature factory for proper masking
            }

            # Encode
            encoded = self.autoencoder.encode(ae_batch)
            latents = encoded["z_latent"]  # [b, n_tokens, latent_dim]

            # Zero out non-encodable tokens
            latents = latents * encodable_mask.unsqueeze(-1).float()

            return latents  # [b, n_tokens, latent_dim]
        else:
            raise ValueError(f"Loading clean samples from data mode {dm} not supported.")

    def handle_self_cond(self, batch: dict) -> dict:
        n_recycle = self.cfg_exp.training.get("n_recycle", 0)
        if random.random() > 0.5 and self.cfg_exp.training.self_cond:
            nn_out = self.call_nn(batch, n_recycle=n_recycle)
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
            batch["x_sc"] = {k: x_1_pred[k].detach() for k in x_1_pred}

        return batch

    def handle_recycling(self):
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
    ):
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

    def log_train_loss_n_prog_bar(self, b: int, train_loss: torch.Tensor):
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

    def log_nparams(self):
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

    def update_n_log_nsamples_processed(self, b: int):
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

    def update_n_log_flops(self, b: int, n: int):
        """
        Updates and logs flops, if available
        """
        try:
            nflops_step = self.nn.nflops_computer(b, n)  # nn should implement this function if we want to see nflops
        except:
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

    def on_validation_epoch_start(self) -> None:
        """Re-initialize validation outputs at the start of each validation epoch."""
        # TODO: Rename to validation_output_step or something to make it clear it's from the training procedure
        # (I am not even sure we need to track it?)
        self.validation_output_data = []
        # TODO We should refactor to just be a dictionary indexed by example_id with "metrics" and "structures"
        self.validation_output_generation = []
        if self.metrics is not None:
            self.metrics.on_validation_epoch_start()

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
        val_dataset_names = list(self.cfg_exp.datasets.val.keys())
        n_val_datasets = len(val_dataset_names)
        actual_dataset_name = val_dataset_names[dataloader_idx]
        if self.inf_cfg is not None and self.inf_cfg.get("val_set_name_override") is not None:
            # Use override as prefix, but keep actual name for uniqueness
            dataset_name = f"{self.inf_cfg.val_set_name_override}/{actual_dataset_name}"
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
        dataset_cfg = self.cfg_exp.datasets.val.get(actual_dataset_name, {})
        if dataset_cfg.get("compute_loss", False):
            batch["validation_dataset_name"] = dataset_name
            self.validation_step_data(batch, -1)

        # Generation-based validation (inference task)
        if self.inf_cfg is not None:
            generation_list = self.predict_step(batch, batch_idx)

        metrics_output = {}
        if self.metrics is not None:
            # Compute prefixes and suffixes for file naming to avoid overwriting results
            # Note than any global_prefix will have also been passed to the Dataset, so will be part of the example_id
            if self.offline_validation:
                if self.job_id is None:
                    raise ValueError("offline_validation=True requires job_id to be set")
                output_suffix = f"_rank{self.job_id}"
            elif self.is_inference:
                output_suffix = ""
            else:
                output_suffix = f"_rank{self.global_rank}"

            # Log where we are saving validation structures
            # Flatten structure: step_X/generated/ instead of step_X/rank_Y/generated/
            output_base = Path(self.store_dir) / ("val_samples" if not self.is_inference else "inference_samples")
            if not self.is_inference:
                output_base = output_base / dataset_name / f"step_{self.global_step}"

            output_base.mkdir(parents=True, exist_ok=True)

            for i in range(len(generation_list)):  # metrics were not written for batches
                example_id_base = batch["example_id_batch"][i]
                # Include rank/job suffix for unique identification across parallel workers
                example_id = f"{example_id_base}{output_suffix}"
                prediction_metadata = generation_list[i]["prediction_metadata"]

                # Log unindexed insertion RMSD to console (this is the motif RMSD)
                if prediction_metadata and "insertion_rmsd" in prediction_metadata:
                    logger.info(
                        f"[{dataset_name}] {example_id}: insertion_rmsd={prediction_metadata['insertion_rmsd']:.3f}Å, "
                        f"insertion_mae={prediction_metadata.get('insertion_mae', 0.0):.3f}Å"
                    )

                metrics_output = self.metrics(
                    network_input=None,
                    network_output=None,
                    extra_info={
                        "example_id": example_id,
                        "metrics_tags": batch["metrics_tags_batch"][i],
                        "output_base_dir": output_base,
                    },
                    ground_truth_atom_array=[generation_list[i]["input_atom_array"]],
                    predicted_atom_array=[generation_list[i]["generated_atom_array"]],
                    prediction_metadata=[prediction_metadata],
                )

                # Optionally save an aligned version of the predicted structure
                aligned_aa = []
                for metric_name in metrics_output:
                    if metric_name.endswith("aligned_predicted_structure"):
                        aligned_aa.append(metrics_output[metric_name])
                metrics_output = {
                    k: v for k, v in metrics_output.items() if not k.endswith("aligned_predicted_structure")
                }

                # If the model failed to reproduce the motif, the aligned coords would be all NaNs
                if (len(aligned_aa) == 1) and not np.isnan(aligned_aa[0].coord).all():
                    predicted_structure_to_save = aligned_aa[0]
                else:
                    predicted_structure_to_save = generation_list[i]["generated_atom_array"]

                # Save CIF files incrementally with rank suffix in filename
                extra_annotations_to_save = [
                    "mapped_to_coordinate_conditioned_region",
                    "mapped_to_coordinate_conditioned_region_token_lvl",
                    "coords_with_insertion",
                    "src_coordinate_conditioned_region",
                    "src_component",
                ]

                # Skip input saving during inference to lighten the filesystem load
                if not self.is_inference:
                    save_atom_array_with_conditions_to_cif(
                        generation_list[i]["input_atom_array"],
                        output_base / "input" / f"{example_id}_input.cif",
                    )
                    generated_output_path = output_base / "generated" / f"{example_id}_generated.cif"
                else:
                    generated_output_path = output_base / f"{example_id}_generated.cif"

                save_atom_array_with_conditions_to_cif(
                    predicted_structure_to_save,
                    generated_output_path,
                    extra_annotations=extra_annotations_to_save,
                )

                if metrics_output is not None:
                    # Add dataset_name and task as metadata
                    metrics_output["dataset_name"] = dataset_name
                    metrics_output["task"] = "_".join(example_id_base.split("_")[:-1])

                    # Add seed used for generation
                    if self.inference_seed is not None:
                        metrics_output["seed"] = self.inference_seed

                    # Add input specification or CIF path
                    spec_batch = batch.get("specification_batch")
                    if spec_batch is not None:
                        spec = spec_batch[i]
                        if "input_cif" in spec:
                            metrics_output["input_cif_path"] = str(spec["input_cif"])
                        else:
                            flat_spec = flatten_nested_dict(spec, fuse_keys=".")
                            for key, value in flat_spec.items():
                                metrics_output[f"input_spec.{key}"] = str(value) if value is not None else None

                    # Add insertion RMSD metrics to output for epoch-end aggregation
                    if prediction_metadata and "insertion_rmsd" in prediction_metadata:
                        metrics_output["insertion_rmsd"] = prediction_metadata["insertion_rmsd"]
                        metrics_output["insertion_mae"] = prediction_metadata.get("insertion_mae", 0.0)
                        metrics_output["insertion_rmcd"] = prediction_metadata.get("insertion_rmcd", 0.0)
                        if "join_point_rmsd" in prediction_metadata:
                            metrics_output["join_point_rmsd"] = prediction_metadata["join_point_rmsd"]

                    metrics_output = apply_to_collection(metrics_output, torch.Tensor, lambda x: x.detach())
                    # Store for epoch-end aggregation
                    self.validation_output_generation.append(metrics_output)

        return metrics_output

    def validation_step_data(self, batch: dict, batch_idx: int) -> None:
        """Evaluates the training loss on validation data."""
        with torch.no_grad():
            outputs = self.training_step(batch, batch_idx=-1)
            self.validation_output_data.append(outputs["loss"].item())

    def on_validation_epoch_end(self) -> None:
        """Aggregate and log validation metrics at the end of the validation epoch.

        Processes both training loss metrics and generation-based metrics.
        Note: validation outputs are cleared at epoch start, not end.
        """
        self.on_validation_epoch_end_data()
        self.on_validation_epoch_end_generation()
        if self.metrics is not None:
            self.metrics.on_validation_epoch_end()

    def on_validation_epoch_end_data(self) -> None:
        """Process data validation metrics (training loss on val set).

        Note: Clearing happens in on_validation_epoch_start().
        """
        pass

    def on_validation_epoch_end_generation(self) -> None:
        """Aggregate and log generation validation metrics from all ranks."""
        # 1. Collect and clean local per-example metrics
        local_example_metrics = [
            cleaned for m in self.validation_output_generation if (cleaned := _clean_metrics_dict(m)) is not None
        ]

        # 2. Gather from all ranks to rank 0
        if self.trainer.world_size > 1 and dist.is_initialized():
            gathered_metrics = [None] * self.trainer.world_size if self.global_rank == 0 else None
            # Set CUDA device context for gather_object's internal tensor operations
            torch.cuda.set_device(self.device)
            dist.gather_object(local_example_metrics, gathered_metrics, dst=0)
            if self.global_rank == 0:
                all_example_metrics = []
                for rank_metrics in gathered_metrics:
                    all_example_metrics.extend(rank_metrics)
            else:
                return  # Non-rank-0 exits early after gather
        else:
            all_example_metrics = local_example_metrics

        if not all_example_metrics:
            logger.warning("on_validation_epoch_end_generation: no metrics to log")
            return

        # 3. Save per-example CSV (one row per example)
        metrics_dir = Path(self.store_dir) / ("val_metrics" if not self.is_inference else "inference_metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)

        per_example_df = pd.DataFrame(all_example_metrics)

        per_example_df["designed_structure_path"] = per_example_df.apply(
            lambda row: str(
                (
                    Path(self.store_dir)
                    / ("val_samples" if not self.is_inference else "inference_samples")
                    / row["dataset_name"]
                    / f"step_{self.global_step}"
                    / "generated"
                    / f"{row['example_id']}_generated.cif"
                ).resolve()
            ),
            axis=1,
        )

        # Reorder columns: example_id first, then metadata, then metrics
        cols = per_example_df.columns.tolist()
        priority = [
            "example_id",
            "designed_structure_path",
            "task",
            "dataset_name",
            "chain_of_interest",
            "seed",
            "input_cif_path",
        ]
        ordered_cols = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
        per_example_df = per_example_df[ordered_cols]

        # Add prefixes or suffixes to avoid file conflicts across jobs
        # To align with the rfd3 inference settings, we use a prefix in that case
        if self.offline_validation:
            if self.job_id is None:
                raise ValueError("offline_validation=True requires job_id to be set")
            output_prefix = ""
            output_suffix = f"_job{self.job_id}"
        elif self.is_inference:
            if self.global_prefix is None:
                output_prefix = ""
            else:
                output_prefix = self.global_prefix + "_"
            output_suffix = ""
        else:
            output_prefix = ""
            output_suffix = f"_step{self.global_step}"

        per_example_path = metrics_dir / f"{output_prefix}per_example_metrics{output_suffix}.csv"
        per_example_df.to_csv(per_example_path, index=False)
        logger.info(f"Saved per-example metrics to {per_example_path}")

        # 4. Compute aggregate metrics (nanmean by dataset)
        agg_rows = []
        for dataset_name in per_example_df["dataset_name"].unique():
            dataset_df = per_example_df[per_example_df["dataset_name"] == dataset_name]
            _non_metric_cols = {
                "example_id",
                "dataset_name",
                "task",
                "designed_structure_path",
                "chain_of_interest",
                "seed",
                "input_cif_path",
            }
            metric_cols = [
                c for c in dataset_df.columns if c not in _non_metric_cols and not c.startswith("input_spec.")
            ]
            for metric_name in metric_cols:
                values = dataset_df[metric_name].dropna()
                if len(values) > 0:
                    agg_rows.append(
                        {
                            "dataset": dataset_name,
                            "metric": metric_name,
                            "value": float(np.nanmean(values)),
                            "n": len(values),
                        }
                    )

        if not agg_rows:
            logger.warning("on_validation_epoch_end_generation: no metrics after aggregation")
            return

        agg_df = pd.DataFrame(agg_rows)
        print_df_as_table(agg_df, title="Validation Metrics")

        # Add prefixes or suffixes to avoid file conflicts across jobs
        agg_path = metrics_dir / f"{output_prefix}aggregated_metrics{output_suffix}.csv"
        agg_df.to_csv(agg_path, index=False)
        logger.info(f"Saved aggregate metrics to {agg_path}")

        # 5. Log to WandB: val_generation_{dataset}/{metric}
        if wandb.run is not None:
            wandb.log(
                {
                    f"val_generation_{row['dataset']}/{row['metric']}": row["value"]
                    for _, row in agg_df.iterrows()
                },
                step=self.global_step,
            )

    def configure_inference(
        self,
        inf_cfg,
        nn_ag,
        store_dir=None,
        metrics_cfg=None,
        offline_validation: bool = False,
        job_id: int | None = None,
        is_inference: bool = False,
        global_prefix: str | None = None,
    ):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided).

        Args:
            inf_cfg: Inference configuration.
            nn_ag: Autoguidance neural network (or None).
            store_dir: Directory to store outputs.
            metrics_cfg: Metrics configuration.
            offline_validation: Whether running in offline validation mode.
            job_id: Job ID for offline validation (required when offline_validation=True).
            is_inference: Whether running in inference mode.
            global_prefix: Global prefix for inference outputs.
        """
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag
        self.metrics = MetricManager.instantiate_from_hydra(metrics_cfg=metrics_cfg) if metrics_cfg else None
        if store_dir is not None:
            self.store_dir = store_dir

        if is_inference and offline_validation:
            raise ValueError("is_inference and offline_validation cannot both be True.")

        # Record inference type and configure relevant settings
        self.is_inference = is_inference
        self.offline_validation = offline_validation
        if self.offline_validation:
            self.job_id = job_id
        elif self.is_inference:
            self.global_prefix = global_prefix

    def configure_search(self, search_strategy, search_reward_model=None):
        """Configure search strategy and reward model for reward-guided generation.

        Args:
            search_strategy: SearchStrategy instance (BeamSearch, BestOfN, etc.)
            search_reward_model: RewardModel instance for scoring candidates (can be non-differentiable)
        """
        self.search_strategy = search_strategy
        self.search_reward_model = search_reward_model
        logger.info(f"Configured search strategy: {type(search_strategy).__name__}")
        if search_reward_model:
            logger.info(
                f"Configured search reward model: "
                f"{len(search_reward_model.metric_manager.metrics)} metrics, "
                f"differentiable={search_reward_model.differentiable}"
            )

    def configure_refinement(self, refinement_strategy, refinement_reward_model):
        """Configure refinement strategy for post-generation optimization.

        Refinement uses a separate reward model from search.
        Refinement reward model must have differentiable=true.

        Args:
            refinement_strategy: RefinementStrategy instance (BindCraft, GradientAscent, etc.)
            refinement_reward_model: RewardModel for refinement (REQUIRED, must be differentiable)
        """
        if refinement_reward_model is None:
            raise ValueError("Refinement requires a reward model. " "Add 'refinement_rewards:' section to your config.")

        # Validate differentiable
        if not refinement_reward_model.differentiable:
            raise ValueError(
                f"Refinement requires differentiable=true in refinement_rewards config, "
                f"but got differentiable={refinement_reward_model.differentiable}"
            )

        self.refinement_strategy = refinement_strategy
        self.refinement_reward_model = refinement_reward_model

        logger.info(f"Configured refinement strategy: {type(refinement_strategy).__name__}")
        logger.info(
            f"Configured refinement reward model: "
            f"{len(refinement_reward_model.metric_manager.metrics)} metrics, "
            f"differentiable={refinement_reward_model.differentiable}"
        )

    def _apply_refinement_to_results(
        self,
        results: list[dict],
        batch: dict,
    ) -> list[dict]:
        """Apply refinement to generation results using standalone interface.

        Extracts AtomArrays → refine → update results.

        Args:
            results: List of generation results (from strategy.generate())
            batch: Original batch dict

        Returns:
            Refined results in same format
        """
        # Extract AtomArrays from results
        atom_arrays = [result["generated_atom_array"] for result in results]

        # Apply refinement using standalone interface
        refined_atom_arrays = self.refinement_strategy.refine_atom_arrays(
            atom_arrays=atom_arrays,
            reward_model=self.refinement_reward_model,
            autoencoder=self.autoencoder,
        )

        # Update results with refined AtomArrays
        for i, result in enumerate(results):
            result["generated_atom_array"] = refined_atom_arrays[i]
            # TODO: Update coors, residue_type from refined AtomArray

        return results

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
        batch["co_generate_targets"] = self.inf_cfg.get("co_generate_targets", False)
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
            from proteinfoundation.search import VanillaStrategy

            strategy = VanillaStrategy()
        else:
            strategy = self.search_strategy

        results = strategy.generate(self, batch, self.search_reward_model)

        # Optional: Apply refinement after generation
        if self.refinement_strategy is not None:
            logger.info("Applying post-generation refinement...")
            results = self._apply_refinement_to_results(results, batch)
            logger.info("Refinement complete")

        return results

    def _process_single_sample_to_atom_array(
        self,
        coords: torch.Tensor,
        residue_type: torch.Tensor,
        padding_mask: torch.Tensor,
        input_atom_array,
        sample_idx: int = 0,
        batch_idx: int = 0,
    ) -> dict:
        """Process a single sample from sample_prots format to atom array result.

        This method contains all the detailed post-processing logic for converting
        generated coordinates to final atom arrays. It is shared between vanilla
        generation and search strategies.

        Args:
            coords: Generated coordinates [n_max_tokens, 37, 3]
            residue_type: Residue types [n_max_tokens]
            padding_mask: Padding mask [n_max_tokens]
            input_atom_array: Original input atom array
            sample_idx: Sample index for naming
            batch_idx: Batch index for naming

        Returns:
            Dictionary with 'generated_atom_array', 'input_atom_array', 'prediction_metadata', 'example_id'
        """
        # + ------- Remove padding and in-context conditioning tokens from the generated coordinates and residue types ------- +
        # (Sanity check - either all or none of the atoms should be indexed per token)
        has_idx_atom_lvl = C_IDX.mask(input_atom_array)  # [n_atoms_in_input]
        has_idx_any_token_level = apply_token_wise(input_atom_array, has_idx_atom_lvl, np.any)
        has_idx_all_token_level = apply_token_wise(input_atom_array, has_idx_atom_lvl, np.all)
        if not np.all(has_idx_any_token_level == has_idx_all_token_level):
            raise ValueError("Input AtomArray should have either all or none of the atoms indexed per token")

        # Get indices of non-padding tokens within this sample
        non_padding_indices = np.where(padding_mask.cpu().numpy())[0]

        # Set as padding any tokens that do not have all atoms indexed
        padding_mask[non_padding_indices[~has_idx_all_token_level]] = False  # [n_max_tokens_in_batch]

        # Subset coordinates and residue types to tokens that are (a) not padding and (b) not in-context conditioning (e.g., unindexed)
        coords = coords[padding_mask]  # [n_tokens_in_this_sample, 37, 3]
        residue_type = residue_type[padding_mask]  # [n_tokens_in_this_sample]
        assert ~torch.any(
            residue_type == UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]
        ), "There should be no mask tokens in the autoencoder output!"

        # Subset input AtomArray to indexed residues, but preserve unindexed region to compute the mapping later
        unindexed_input_atom_array = input_atom_array[~has_idx_atom_lvl].copy()
        input_atom_array = input_atom_array[has_idx_atom_lvl]
        indexed_coord_conditioned_count = C_CRD.mask(input_atom_array).sum() + C_INTCRD.mask(input_atom_array).sum()
        unindexeded_coord_conditioned_count = (
            C_CRD.mask(unindexed_input_atom_array).sum() + C_INTCRD.mask(unindexed_input_atom_array).sum()
        )

        assert (
            get_token_count(input_atom_array) == coords.shape[0]
        ), "Token counts in input AtomArray and generated coords do not match after filtering!"

        # + ------- Pre-process to make format compatible with the encoding ------- +
        # Convert all-zero coordinates to NaN's and build encoded mask
        all_zeros = (coords == 0.0).all(dim=-1, keepdim=True)
        coords = coords.masked_fill(all_zeros, float("nan"))
        encoded_mask = ~torch.isnan(coords[..., 0])  # [n_tokens_in_this_sample, 37]

        # Atomized tokens should be set to -1 and have only the CA atom slot (1) occupied
        token_starts = get_token_starts(input_atom_array)
        atomized_token_mask = torch.from_numpy(input_atom_array.atomize[token_starts]).to(residue_type.device)
        assert torch.all(residue_type[atomized_token_mask] == -1), "Atomized tokens should have -1 residue type"
        assert torch.all(coords[atomized_token_mask, 1, :].isfinite()), "Atomized tokens should have CA atom occupied"
        assert torch.all(
            coords[atomized_token_mask][:, torch.arange(coords.shape[1], device=coords.device) != 1, :].isnan()
        ), "Non-CA atoms should be NaN for atomized tokens"

        # Set to atomized token from the encoding definition
        residue_type[atomized_token_mask] = UNIFIED_ATOM37_ENCODING.token_to_idx["<A>"]

        # + ------- Encode and standardize ------- +
        generated_array = atom_array_from_encoding(
            encoded_coord=coords,
            encoded_mask=encoded_mask,
            encoded_seq=residue_type,
            encoding=UNIFIED_ATOM37_ENCODING,
            token_id=input_atom_array.token_id[token_starts],
            chain_type=input_atom_array.chain_type[token_starts],
            token_is_atom=input_atom_array.atomize[token_starts],
        )
        # (Remove terminal oxygens from both AtomArrays to ensure matching)
        generated_array = remove_protein_terminal_oxygen(generated_array)
        input_atom_array = remove_protein_terminal_oxygen(input_atom_array)
        assert get_token_count(generated_array) == get_token_count(
            input_atom_array
        ), "Token counts in input and generated AtomArrays do not match after removing terminal oxygens!"

        # + ------- Copy over model-generated coordinates ------- +
        updated_array = copy_generated_coordinates_to_input(
            generated_atom_array=generated_array, input_atom_array=input_atom_array
        )

        # Preserve the original indexed coordinates reshaped to match the output AtomArray
        coords_with_insertion = updated_array.coord.copy()
        coords_with_insertion[C_CRD.mask(updated_array)] = updated_array.original_input_coord[C_CRD.mask(updated_array)]
        updated_array.set_annotation("coords_with_insertion", coords_with_insertion)

        # Insert internal coordinate conditioned groups into the `coords_with_insertion` annotation
        C_INTCRD.set_mask(input_atom_array, C_INTCRD.mask(input_atom_array))
        C_INTCRD.set_mask(updated_array, C_INTCRD.mask(updated_array))
        updated_array = insert_internal_coord_conditioned_groups(
            updated_array, original_input_coords=updated_array.original_input_coord
        )

        valid_mapping_found = True
        if C_CRD.mask(updated_array).sum() + C_INTCRD.mask(updated_array).sum() != indexed_coord_conditioned_count:
            logger.warning(
                f"Original coord conditioning count ({indexed_coord_conditioned_count}) does not match "
                f"recovered count ({C_CRD.mask(updated_array).sum() + C_INTCRD.mask(updated_array).sum()}). "
                "The model has failed to reproduce the input indexed motif."
            )
            valid_mapping_found = False

        # + ------- Compute unindexed mapping and optionally insert unindexed regions ------- +
        # Compute the correct mapping from indexed to unindexed tokens
        # NOTE: This must be done after the output AtomArray has been constructed and the coordinates have been added
        if not has_idx_all_token_level.all():
            # Preserve this annotation through concatentation
            unindexed_input_atom_array.set_annotation(
                "coords_with_insertion", np.full_like(unindexed_input_atom_array.coord, np.nan)
            )
            # NOTE: We use coord_to_be_noised as the unindexed coordinates to match what the model sees
            unindexed_input_atom_array.coord = unindexed_input_atom_array.coord_to_be_noised.copy()
            full_array_with_unindexed = concatenate_atom_array_plus([updated_array, unindexed_input_atom_array])
            updated_array, metadata = process_unindexed_outputs(
                full_array_with_unindexed,
                insert_unindexed_regions=self.cfg_exp.inf_cfg.insert_unindexed_regions,
                inputs_in_same_frame=True,
            )
        else:
            updated_array.set_annotation(
                "mapped_to_coordinate_conditioned_region", np.zeros(updated_array.array_length(), dtype=bool)
            )
            updated_array.set_annotation(
                "src_coordinate_conditioned_region", np.full(updated_array.array_length(), ".", dtype=str)
            )
            metadata = {}
        
        # Include indexed coord conditioning in mapping mask/annotation
        # TODO: Should this mapping annotation be added as a Condition? It is semantically different but practically similar
        indexed_coord_conditioned_mask = C_CRD.mask(updated_array) | C_INTCRD.mask(updated_array)
        updated_array.mapped_to_coordinate_conditioned_region |= indexed_coord_conditioned_mask
        updated_array.src_coordinate_conditioned_region[
            indexed_coord_conditioned_mask
        ] = updated_array.src_component[indexed_coord_conditioned_mask]
        
        # Check unindexed coord conditioning count
        total_coord_conditioned_count = indexed_coord_conditioned_count + unindexeded_coord_conditioned_count
        if updated_array.mapped_to_coordinate_conditioned_region.sum() != total_coord_conditioned_count:
            logger.warning(
                f"Unindexed coord conditioning count ({unindexeded_coord_conditioned_count}) does not match "
                f"remapped count ({updated_array.mapped_to_coordinate_conditioned_region.sum() - indexed_coord_conditioned_count}). "
                "The model has failed to reproduce the input unindexed motif."
            )
            valid_mapping_found = False

        # If the model has failed to create valid reproductions of the motif, give NaN values for coords_with_insertion
        if not valid_mapping_found:
            updated_array.set_annotation("coords_with_insertion", np.full_like(updated_array.coord, np.nan))

        # This information is used when determining which backbone atoms to align on for the AME benchmark
        mapped_to_coordinate_conditioned_region_token_lvl = apply_and_spread_token_wise(
            updated_array, updated_array.mapped_to_coordinate_conditioned_region, np.any
        )
        updated_array.set_annotation(
            "mapped_to_coordinate_conditioned_region_token_lvl", mapped_to_coordinate_conditioned_region_token_lvl
        )

        return {
            "generated_atom_array": as_atom_array_plus(updated_array),
            "input_atom_array": as_atom_array_plus(input_atom_array),
            "prediction_metadata": metadata,  # Contains unindexed residue mapping and insertion statistics
        }

    def format_sample_prots_to_atom_arrays(self, sample_prots: dict, batch: dict, batch_idx: int = 0) -> list[dict]:
        """Convert sample_prots dict to list of atom array results.

        This helper is used by search strategies to convert their internal format
        (sample_prots dict with coors, residue_type, padding_mask) to the standard
        output format expected by generate.py.

        Uses the comprehensive _process_single_sample_to_atom_array() method to ensure
        all search strategies have the same detailed post-processing as vanilla generation.

        Args:
            sample_prots: Dictionary with keys 'coors', 'residue_type', 'padding_mask'
            batch: Original batch dict with input atom_arrays
            batch_idx: Batch index for naming

        Returns:
            List of dictionaries with 'generated_atom_array', 'input_atom_array', etc.
        """
        generation_list = []

        for i in range(sample_prots["coors"].shape[0]):
            # Get batch index for this sample (handle replicas from search strategies)
            orig_batch_idx = i % len(batch.get("atom_array_batch", [batch.get("atom_array")]))
            input_atom_array = (
                batch["atom_array_batch"][orig_batch_idx] if "atom_array_batch" in batch else batch["atom_array"]
            )

            # Use the comprehensive processing method
            result = self._process_single_sample_to_atom_array(
                coords=sample_prots["coors"][i],
                residue_type=sample_prots["residue_type"][i],
                padding_mask=sample_prots["padding_mask"][i],
                input_atom_array=input_atom_array,
                sample_idx=i,
                batch_idx=batch_idx,
            )
            # Record for output file saving
            result["example_id"] = batch["example_id_batch"][orig_batch_idx]
            generation_list.append(result)

        return generation_list

    def sample_formatting(
        self,
        x: dict[str, Tensor],
        extra_info: dict[str, Tensor],
        ret_mode: str,
    ):
        """
        Given a batch of b samples x produced by the flow matcher, it returns the samples in the requested format (ret_mode).

        Supports `ret_modes` for:
            - `samples` returns the original sample from the flow matcher, a dictionary[str, Tensor].
            for the data modalities, each with batch shape b.
            - `atom37` returns an Tensor of shape [b, n, 37, 3] just for coordinates.
            - `pdb_string` returns a list of dictionaries {"pdb_string": str, "nres": int}, with one dictionary per sample.
            - `coors37_n_aatype` returns a dictionary with keys `coors` (atom37), `residue_type`, and `padding_mask`, and
            values with shapes [b, n, 37, 3] float, [b, n] int, [b, n] boolean, respectively.

        Args:
            x: sample.
            extra_info: a dict with additional things, including:
                - "padding_mask", whose value is boolean of shape [nsamples, n]
                - ...
            ret_mode: target format, for now only supports atom37.

        Returns:
            Sample x in the requested format.
        """
        data_modes = sorted([dm for dm in self.cfg_exp.product_flowmatcher])
        if data_modes == ["bb_ca"]:
            return self._format_sample_bb_ca(
                x=x, ret_mode=ret_mode, padding_mask=extra_info["padding_mask"]
            )
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
    ):
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
    ):
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

            # TEMPORARY HACK: Shift all non-negative residue types by 1 to adjust for differences in AF2 vs. unified Atom37 encoding
            residue_type_full = torch.where(residue_type_full >= 0, residue_type_full + 1, residue_type_full)

            return {
                "coors": nm_to_ang(coors_nm_full),  # [b, n, 37, 3] - padded, use mask to slice
                "residue_type": residue_type_full,  # [b, n] - padded, use mask to slice
                "padding_mask": padding_mask,  # [b, n] - True for non-padding tokens
            }
        else:
            raise NotImplementedError(f"{ret_mode} format for data modes `[bb_ca, latent_locals]` not implemented")
