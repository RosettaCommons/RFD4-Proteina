import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from proteinfoundation.refinement.base_refinement import RefinementStrategy  # noqa: F401 - used in type annotations
    from proteinfoundation.rewards.reward_model import RewardModel
    from proteinfoundation.search.base_strategy import SearchStrategy

import biotite.structure.io.xtc as xtc
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from atomworks.common import exists
from atomworks.io.parser import parse as parse_structure
from atomworks.io.transforms.atom_array import add_charge_from_ccd_codes
from atomworks.io.utils.atom_array_plus import as_atom_array_plus, concatenate_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.conditions import C_CRD, C_IDX
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
from atomworks.ml.transforms.encoding import atom_array_from_encoding
from atomworks.ml.transforms.filters import remove_protein_terminal_oxygen
from atomworks.ml.utils.nested_dict import flatten as flatten_nested_dict
from atomworks.ml.utils.token import apply_and_spread_token_wise, apply_token_wise, get_token_count, get_token_starts
from jaxtyping import Bool, Float
from lightning_utilities import apply_to_collection
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from torch import Tensor

from foundry.metrics.metric import MetricManager
from foundry.utils.ddp import RankedLogger
from foundry.utils.logging import print_df_as_table, safe_print
from foundry.utils.standard_annotations import SA_CRDSRC, SA_CWI, SA_SRC
from proteinfoundation.conditions import C_INTCRD
from proteinfoundation.flow_matching.product_space_flow_matcher import (
    ProductSpaceFlowMatcher,
)
from proteinfoundation.inference_utils.aggregate_metrics import (
    add_diversity_summary_metrics,
    add_foldseek_diversity_to_per_example_df,
)
from proteinfoundation.metrics.metrics_utils import run_metrics_for_result, run_oracle_refolding_and_pass_criteria
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.transforms.unindexing import process_unindexed_outputs
from proteinfoundation.transforms.utils.conditioning_utils import insert_internal_coord_conditioned_groups
from proteinfoundation.utils.coordinate_transfer import copy_generated_coordinates_to_input
from proteinfoundation.utils.coors_utils import nm_to_ang, trans_nm_to_atom37
from proteinfoundation.utils.parse_presets import STANDARD_ANNOTATIONS_ONLY

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

    def _find_first_tags(obj: Any) -> list[str] | None:
        """Recursively return the first ``metrics_tags`` list found, or None."""
        if OmegaConf.is_dict(obj):
            if "metrics_tags" in obj:
                tags = obj["metrics_tags"]
                if OmegaConf.is_list(tags) or isinstance(tags, list):
                    return list(tags)
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
        elif key == "input_cif_path" or key.startswith("input_spec."):
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

        self.inf_cfg = inf_cfg  # Used for inference-like validation
        self.store_dir = store_dir if store_dir is not None else os.path.join(".", "runs", cfg_exp.run_name_)

        self.configure_metrics(cfg_exp.metrics)

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
            return None, None

        logger.info(f"Loading autoencoder from {ae_ckp_path}")
        autoencoder = AutoEncoder.load_from_checkpoint(ae_ckp_path, strict=False, weights_only=False)
        if freeze_params:
            for param in autoencoder.parameters():
                param.requires_grad = False
        return autoencoder, autoencoder.latent_dim

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.cfg_exp.opt.lr,
        )

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
        # Always populate x_recycle for torch compile stability
        if "x_recycle" not in batch:
            batch["x_recycle"] = {k: torch.zeros_like(v) for k, v in batch["x_t"].items()}

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
        torch.compiler.cudagraph_mark_step_begin()

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

            # Change this once we retrain AE with correct token encoding
            residue_type_for_ae = batch["residue_type"] - 1  # [b, n_tokens]
            # Clamp for non-encodable tokens to avoid -1 index (invalid for embeddings)
            residue_type_for_ae = torch.where(
                encodable_mask, residue_type_for_ae, torch.zeros_like(residue_type_for_ae)
            )
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

    def handle_self_cond(self, batch: dict, batch_idx: int, val_step: bool) -> dict:
        # Canonical x_sc zeros so _extra_fields cardinality is stable for
        # torch.compile (removes a Python-side conditional that would otherwise
        # break fullgraph). Numerically identical to "no self-cond" because the
        # x_sc feature modules multiply by their output.
        batch["x_sc"] = {k: torch.zeros_like(v) for k, v in batch["x_t"].items()}

        if val_step or not self.cfg_exp.training.get("self_cond", False):
            return batch
        if random.random() >= 0.5:
            return batch

        n_recycle = self.cfg_exp.training.get("n_recycle", 0)
        with torch.no_grad():
            nn_out = self.call_nn(batch, n_recycle=n_recycle)
        x_1_pred = self.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
        batch["x_sc"] = {k: x_1_pred[k].detach() for k in x_1_pred}

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

    def on_validation_epoch_start(self) -> None:
        """Re-initialize validation outputs at the start of each validation epoch."""
        self.validation_output_generation = []

        reward_metrics_tags = (
            self.search_reward_model.reward_metrics_tags if self.search_reward_model is not None else []
        )
        combined_tags = _combined_val_tags(self.val_dataset_tags, reward_metrics_tags)

        if self.design_metric_manager is not None:
            self.design_metric_manager.on_validation_epoch_start(val_dataset_tags=combined_tags)

        # Set up oracle/refolding/pass managers early only when a search strategy will
        # need them during validation_step (beam/bestofn scoring).  For vanilla runs,
        # defer to on_validation_epoch_end Phase B to save memory.
        if reward_metrics_tags:
            for mm in [
                self.oracle_metric_manager,
                self.refolding_metric_manager,
                self.pass_criteria_metric_manager,
            ]:
                if mm is not None:
                    mm.on_validation_epoch_start(val_dataset_tags=combined_tags)

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
        dataset_cfg = self.cfg_exp.datasets.val.get(actual_dataset_name, {})

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
        if dataset_cfg.get("compute_loss", False):
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

        # Generate designs for this inference batch
        generation_list = self.predict_step(batch, batch_idx)

        example_id_batch = batch.get("example_id_batch", [])
        n_batch = len(example_id_batch)
        for i in range(len(generation_list)):  # metrics were not written for batches
            orig_batch_idx = i % n_batch if n_batch > 0 else 0
            raw_batch_id = example_id_batch[orig_batch_idx] if n_batch > 0 else "sample"

            # Use example_id pre-assigned by the search strategy if available
            base_id = generation_list[i].get("example_id", raw_batch_id)
            if base_id == raw_batch_id and len(generation_list) > n_batch:
                base_id = f"{base_id}_s{i}"

            # Include rank/job suffix for unique identification across parallel workers
            example_id = f"{base_id}{output_suffix}"
            metrics_output["example_id"] = example_id
            prediction_metadata = generation_list[i]["prediction_metadata"]

            # Log unindexed insertion RMSD to console (this is the motif RMSD)
            if prediction_metadata and "insertion_rmsd" in prediction_metadata:
                logger.info(
                    f"[{dataset_name}] {example_id}: insertion_rmsd={prediction_metadata['insertion_rmsd']:.3f}Å, "
                    f"insertion_mae={prediction_metadata.get('insertion_mae', 0.0):.3f}Å"
                )

            if self.design_metric_manager is not None:
                metrics_output = run_metrics_for_result(
                    result=generation_list[i],
                    batch=batch,
                    batch_item_idx=orig_batch_idx,
                    metric_manager=self.design_metric_manager,
                    output_base_dir=output_base,
                    example_id=example_id,
                    save_input_structure=not self.is_inference,
                )

            if per_example_losses is not None:
                for k, v in per_example_losses.items():
                    if "_justlog" not in k:
                        metrics_output[f"loss_{k}"] = v[orig_batch_idx].item()
                metrics_output["loss"] = sum(
                    v[orig_batch_idx].item() for k, v in per_example_losses.items() if "_justlog" not in k
                )

            # Optionally save an aligned version of the predicted structure
            aligned_aa = []
            for metric_name in metrics_output:
                if metric_name.endswith("aligned_predicted_structure"):
                    aligned_aa.append(metrics_output[metric_name])
            metrics_output = {k: v for k, v in metrics_output.items() if not k.endswith("aligned_predicted_structure")}

            # If the model failed to reproduce the motif, the aligned coords would be all NaNs
            if (len(aligned_aa) == 1) and not np.isnan(aligned_aa[0].coord).all():
                predicted_structure_to_save = aligned_aa[0]
            else:
                predicted_structure_to_save = generation_list[i]["generated_atom_array"]

            # Save CIF files incrementally with rank suffix in filename
            if not self.is_inference:
                # Skip input saving during inference to lighten the filesystem load
                to_cif_file(
                    generation_list[i]["input_atom_array"],
                    output_base / "input" / f"{example_id}_input.cif",
                )
                generated_output_path = output_base / "generated" / f"{example_id}_generated.cif"
            else:
                generated_output_path = output_base / f"{example_id}_generated.cif"

            to_cif_file(
                predicted_structure_to_save,
                generated_output_path,
            )

            # Save trajectories if present.
            # bb_ca_only=True  → XTC (fixed CA-per-residue atom count required by XTC format)
            # bb_ca_only=False → CIF (all-atom trajectories have variable atom counts per step)
            bb_ca_only = self.inf_cfg is not None and self.inf_cfg.args.get("bb_ca_only_trajectories", False)
            for traj_key in ["trajectory_x1", "trajectory_xt"]:
                if traj_key not in generation_list[i]:
                    continue
                traj_dir = output_base / "trajectories"
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj = generation_list[i][traj_key]

                if bb_ca_only:
                    # Fixed atom count (one CA per residue) → XTC.
                    # Save a CA-only CIF as the topology reference for the XTC file.
                    predicted_bb_ca = predicted_structure_to_save[predicted_structure_to_save.atom_name == "CA"]
                    to_cif_file(
                        predicted_bb_ca,
                        traj_dir / f"{example_id}_{traj_key}_bb_ca.cif",
                    )

                    save_trajectory_every = self.inf_cfg is not None and self.inf_cfg.args.get(
                        "save_trajectory_every", 0
                    )
                    step_labels = np.arange(traj.stack_depth()) * save_trajectory_every
                    step_labels[-1] = self.inf_cfg.args.nsteps

                    xtc_file = xtc.XTCFile()
                    xtc_file.set_structure(traj, time=step_labels)
                    xtc_file.write(traj_dir / f"{example_id}_{traj_key}.xtc")
                else:
                    # All-atom: variable sidechain atom counts across steps → multi-model CIF.
                    to_cif_file(traj, traj_dir / f"{example_id}_{traj_key}.cif")

            # Add dataset_name and task as metadata
            metrics_output["dataset_name"] = dataset_name
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
            spec_batch = batch.get("specification_batch")
            if spec_batch is not None:
                spec = spec_batch[orig_batch_idx]
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
            metrics_output["designed_structure_path"] = str(generated_output_path.resolve())
            metrics_output["output_base_dir"] = str(output_base)
            # Store metrics_tags for the oracle loop in on_validation_epoch_end
            metrics_tags_batch = batch.get("metrics_tags_batch", [])
            _tags = metrics_tags_batch[orig_batch_idx] if orig_batch_idx < len(metrics_tags_batch) else []
            metrics_output["metrics_tags"] = ",".join(_tags) if _tags else ""
            # Store for epoch-end aggregation
            self.validation_output_generation.append(metrics_output)

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

        Four-phase flow:
          Phase A — DDP gather design metrics; rank 0 saves crash-safe design CSV.
          Phase B — All ranks run oracle + refolding + pass_criteria.
          Phase C — DDP gather refolding metrics; rank > 0 exits.
          Phase D — Rank 0 merges design + refolding, applies foldseek, saves one CSV, logs.
        """

        if self.design_metric_manager is not None:
            self.design_metric_manager.on_validation_epoch_end()

        # ------------------------------------------------------------------
        # Phase A: gather design metrics from all ranks
        # ------------------------------------------------------------------
        local_design_metrics = [
            c for m in self.validation_output_generation if (c := _clean_metrics_dict(m)) is not None
        ]

        all_design_metrics = self._gather_metrics(local_design_metrics)

        metrics_dir = Path(self.store_dir) / ("val_metrics" if not self.is_inference else "inference_metrics")

        # ------------------------------------------------------------------
        # Phase B: oracle + refolding + pass_criteria (all ranks, per-rank)
        # ------------------------------------------------------------------
        oracle_mm = self.oracle_metric_manager
        refold_mm = self.refolding_metric_manager
        pass_mm = self.pass_criteria_metric_manager

        # For vanilla (non-search) runs the oracle managers were not set up in
        # on_validation_epoch_start to avoid holding their weights in memory during
        # diffusion.  Set them up now before the oracle loop.
        reward_metrics_tags = (
            self.search_reward_model.reward_metrics_tags if self.search_reward_model is not None else []
        )
        if not reward_metrics_tags:
            for mm in [oracle_mm, refold_mm, pass_mm]:
                if mm is not None:
                    mm.on_validation_epoch_start(val_dataset_tags=self.val_dataset_tags)

        local_refolding_metrics: list[dict] = []

        example_id_to_design_metrics = {m["example_id"]: m for m in (local_design_metrics or []) if m}

        # Sort by n_tokens ascending to avoid unnecessary recompilation of oracles
        items = sorted(
            self.validation_output_generation,
            key=lambda it: example_id_to_design_metrics.get(it.get("example_id", ""), {}).get(
                "general_metrics.n_tokens", 0
            ),
        )

        for item in items:
            result = self._run_oracle_loop_item(item, oracle_mm, refold_mm, pass_mm, example_id_to_design_metrics)
            if exists(result):
                local_refolding_metrics.append(result)

        for mm in [oracle_mm, refold_mm, pass_mm]:
            if exists(mm):
                mm.on_validation_epoch_end()

        # ------------------------------------------------------------------
        # Phase C: gather refolding metrics from all ranks
        # ------------------------------------------------------------------
        all_refolding_metrics = self._gather_metrics(local_refolding_metrics)

        if self.global_rank != 0:
            return  # rank > 0 exits after Phase C gather

        # ------------------------------------------------------------------
        # Phase D: rank 0 merges design + refolding, applies foldseek, saves one CSV, logs
        # ------------------------------------------------------------------
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
        ]
        ordered_cols = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
        per_example_df = per_example_df[ordered_cols]

        # Foldseek clustering (skipped during inference; separate script handles that).
        if (
            not self.is_inference
            and os.getenv("FOLDSEEK_EXEC") is not None
            and "designed_structure_path" in per_example_df.columns
            and "chain_of_interest" in per_example_df.columns
        ):
            foldseek_tmp = str(metrics_dir / "foldseek_tmp")
            try:
                per_example_df = add_foldseek_diversity_to_per_example_df(
                    per_example_df,
                    group_by=["dataset_name", "task"],
                    tmp_path=foldseek_tmp,
                    timeout=self.cfg_exp.opt.get("foldseek_timeout_minutes", 25) * 60,
                )
            except Exception as e:
                logger.warning(f"Foldseek clustering failed, skipping cluster metrics: {e}")
            else:
                try:
                    per_example_df = add_diversity_summary_metrics(
                        per_example_df,
                        group_by=["dataset_name", "task"],
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
            "metrics_tags",
            "output_base_dir",
        }
        dataset_col = (
            per_example_df["dataset_name"]
            if "dataset_name" in per_example_df.columns
            else pd.Series(["unknown"] * len(per_example_df))
        )
        for ds in dataset_col.unique():
            ds_df = per_example_df[dataset_col == ds]
            for col in ds_df.columns:
                if col in _non_metric_cols or col.startswith("input_spec.") or col.startswith("generation."):
                    continue
                vals = pd.to_numeric(ds_df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    agg_rows.append({"dataset": ds, "metric": col, "value": float(np.nanmean(vals)), "n": len(vals)})

        if not agg_rows:
            return

        agg_df = pd.DataFrame(agg_rows).round(4)
        print_df_as_table(agg_df, title=csv_stem)
        agg_path = metrics_dir / f"{output_prefix}{csv_stem}_aggregated{output_suffix}.csv"
        agg_df.to_csv(agg_path, index=False)
        if wandb.run is not None:
            wandb.log(
                {f"val_generation_{row['dataset']}/{row['metric']}": row["value"] for _, row in agg_df.iterrows()},
                step=self.global_step,
            )

    def _run_oracle_loop_item(
        self,
        item: dict,
        oracle_mm: MetricManager | None,
        refold_mm: MetricManager | None,
        pass_mm: MetricManager | None,
        example_id_to_design_metrics: dict[str, dict],
    ) -> dict | None:
        """Run oracle + refolding + pass_criteria for one item from validation_output_generation."""
        example_id = item.get("example_id", "unknown")
        designed_path = item.get("designed_structure_path")
        output_base_dir = item.get("output_base_dir")
        metrics_tags_str = item.get("metrics_tags", "")
        metrics_tags = [t for t in metrics_tags_str.split(",") if t] if metrics_tags_str else []

        if not designed_path or not Path(designed_path).exists():
            logger.warning(f"Oracle loop: designed structure not found for {example_id}, skipping.")
            return None

        atom_array = parse_structure(Path(designed_path), config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]
        atom_array = add_charge_from_ccd_codes(atom_array)

        output_dir = Path(output_base_dir) if output_base_dir else None
        result = run_oracle_refolding_and_pass_criteria(
            atom_array=atom_array,
            example_id=example_id,
            output_dir=output_dir,
            oracle_mm=oracle_mm,
            refold_mm=refold_mm,
            pass_mm=pass_mm,
            metrics_tags=metrics_tags,
            prior_metrics=example_id_to_design_metrics.get(example_id),
        )
        if result is None:
            return None

        result["designed_structure_path"] = designed_path
        result["dataset_name"] = item.get("dataset_name")
        return result

    def configure_metrics(
        self,
        metrics_cfg: DictConfig | None = None,
    ) -> None:
        """Instantiate design, oracle, refolding, and pass-criteria MetricManagers from config."""

        if metrics_cfg is None:
            self.design_metric_manager = None
            self.oracle_metric_manager = None
            self.refolding_metric_manager = None
            self.pass_criteria_metric_manager = None
            return

        cfg_key_to_attr = {
            "inline_metrics": "design_metric_manager",
            "refolding_oracles": "oracle_metric_manager",
            "refolding_metrics": "refolding_metric_manager",
            "pass_criteria": "pass_criteria_metric_manager",
        }
        for cfg_key, attr_name in cfg_key_to_attr.items():
            metric_manager_cfg = metrics_cfg.get(cfg_key)
            if exists(metric_manager_cfg):
                metric_manager = MetricManager.instantiate_from_hydra(metrics_cfg=metric_manager_cfg)
            else:
                metric_manager = None
            setattr(self, attr_name, metric_manager)

        if hasattr(self.cfg_exp, "datasets") and hasattr(self.cfg_exp.datasets, "val"):
            self.val_dataset_tags = _collect_val_metrics_tags(self.cfg_exp.datasets.val)
        else:
            self.val_dataset_tags = None

        # Enforce that only one Oracle will run per dataset
        if self.val_dataset_tags is not None and self.oracle_metric_manager is not None:
            for dataset_tags in self.val_dataset_tags:
                dataset_oracle_name = None
                for oracle_name, oracle_metric in self.oracle_metric_manager.metrics.items():
                    if oracle_metric.applies_to_tags(dataset_tags):
                        if dataset_oracle_name is None:
                            dataset_oracle_name = oracle_name
                        else:
                            raise ValueError(
                                f"Multiple oracle metrics ({dataset_oracle_name}, {oracle_name}) apply "
                                f"to dataset with tags {dataset_tags}."
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

    def configure_search(
        self,
        search_strategy: "SearchStrategy",
        search_reward_model: "RewardModel | None" = None,
    ) -> None:
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
                f"reward_metrics_tags={search_reward_model.reward_metrics_tags}, "
                f"weights={list(search_reward_model.weights.keys())}, "
                f"differentiable={search_reward_model.differentiable}"
            )

    def configure_refinement(self, refinement_strategy: Any, refinement_reward_model: Any) -> None:
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
            f"weights={list(refinement_reward_model.weights.keys())}, "
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
        input_atom_array: Any,
        sample_idx: int = 0,
        batch_idx: int = 0,
        bb_ca_only: bool = False,
    ) -> dict:
        """Process a single sample from sample_prots format to atom array result.

        Args:
            coords: Generated coordinates [n_max_tokens, 37, 3]
            residue_type: Residue types [n_max_tokens]
            padding_mask: Padding mask [n_max_tokens]
            input_atom_array: Original input atom array
            sample_idx: Sample index for naming
            batch_idx: Batch index for naming
            bb_ca_only: If True, subset input_atom_array to CA atoms only.

        Returns:
            Dictionary with 'generated_atom_array', 'input_atom_array',
            'prediction_metadata', 'example_id'
        """

        # Optionally subset the input to only backbone CA atoms
        if bb_ca_only:
            input_atom_array = input_atom_array[input_atom_array.atom_name == "CA"]
            input_atom_array.atomize = np.full_like(input_atom_array.atomize, True)

        # + ------- Remove padding and in-context conditioning tokens from the generated
        # coordinates and residue types ------- +
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

        # Subset coordinates and residue types to tokens that are (a) not padding and
        # (b) not in-context conditioning (e.g., unindexed)
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
        SA_CWI.set_annotation(updated_array, coords_with_insertion)

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
            SA_CWI.set_annotation(unindexed_input_atom_array, SA_CWI.default_annotation(unindexed_input_atom_array))
            # NOTE: We use coord_to_be_noised as the unindexed coordinates to match what the model sees
            unindexed_input_atom_array.coord = unindexed_input_atom_array.coord_to_be_noised.copy()
            full_array_with_unindexed = concatenate_atom_array_plus([updated_array, unindexed_input_atom_array])
            updated_array, metadata = process_unindexed_outputs(
                full_array_with_unindexed,
                insert_unindexed_regions=self.cfg_exp.inf_cfg.insert_unindexed_regions,
                inputs_in_same_frame=True,
            )
        else:
            SA_CRDSRC.set_mask(updated_array, SA_CRDSRC.default_mask(updated_array))
            SA_CRDSRC.set_annotation(updated_array, SA_CRDSRC.default_annotation(updated_array))
            metadata = {}

        # Include indexed coord conditioning in mapping mask/annotation
        # TODO: Should this mapping annotation be added as a Condition? It is semantically
        # different but practically similar
        indexed_coord_conditioned_mask = C_CRD.mask(updated_array) | C_INTCRD.mask(updated_array)
        mapped_mask = SA_CRDSRC.mask(updated_array)
        mapped_mask |= indexed_coord_conditioned_mask
        SA_CRDSRC.set_mask(updated_array, mapped_mask)
        src_crd = SA_CRDSRC.annotation(updated_array)
        src_crd[indexed_coord_conditioned_mask] = SA_SRC.annotation(updated_array)[indexed_coord_conditioned_mask]
        SA_CRDSRC.set_annotation(updated_array, src_crd)

        # Check unindexed coord conditioning count
        total_coord_conditioned_count = indexed_coord_conditioned_count + unindexeded_coord_conditioned_count
        if SA_CRDSRC.mask(updated_array).sum() != total_coord_conditioned_count:
            logger.warning(
                f"Unindexed coord conditioning count ({unindexeded_coord_conditioned_count}) does not match "
                f"remapped count ({SA_CRDSRC.mask(updated_array).sum() - indexed_coord_conditioned_count}). "
                "The model has failed to reproduce the input unindexed motif."
            )
            valid_mapping_found = False

        # If the model has failed to create valid reproductions of the motif, give NaN values for coords_with_insertion
        if not valid_mapping_found:
            SA_CWI.set_annotation(updated_array, SA_CWI.default_annotation(updated_array))

        # Compute token-level mapped-to-coordinate-conditioned-region annotation from atom-level mask
        updated_array.set_annotation(
            "mapped_to_coordinate_conditioned_region_token_lvl",
            apply_and_spread_token_wise(updated_array, SA_CRDSRC.mask(updated_array), np.any),
        )

        return {
            "generated_atom_array": as_atom_array_plus(updated_array),
            "input_atom_array": as_atom_array_plus(input_atom_array),
            "prediction_metadata": metadata,  # Contains unindexed residue mapping and insertion statistics
        }

    def format_sample_prots_to_atom_arrays(
        self,
        sample_prots: dict,
        batch: dict,
        batch_idx: int = 0,
        bb_ca_only: bool = False,
        example_ids: list[str] | None = None,
    ) -> list[dict]:
        """Convert sample_prots dict to list of atom array results.

        Args:
            sample_prots: Dictionary with keys 'coors', 'residue_type', 'padding_mask'
            batch: Original batch dict with input atom_arrays
            batch_idx: Batch index for naming
            bb_ca_only: If True, subset input_atom_array to CA atoms only.
            example_ids: Pre-assigned example IDs, one per sample. When provided (e.g.
                by beam search to include checkpoint-aware suffixes), these override
                the default ``batch["example_id_batch"][orig_batch_idx]`` lookup.
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
                bb_ca_only=bb_ca_only,
            )
            # Record for output file saving; use caller-provided id if available
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

            # TEMPORARY HACK: Shift all non-negative residue types by 1 to adjust for differences
            # in AF2 vs. unified Atom37 encoding
            residue_type_full = torch.where(residue_type_full >= 0, residue_type_full + 1, residue_type_full)

            return {
                "coors": nm_to_ang(coors_nm_full),  # [b, n, 37, 3] - padded, use mask to slice
                "residue_type": residue_type_full,  # [b, n] - padded, use mask to slice
                "padding_mask": padding_mask,  # [b, n] - True for non-padding tokens
            }
        else:
            raise NotImplementedError(f"{ret_mode} format for data modes `[bb_ca, latent_locals]` not implemented")
