import contextlib
import os
from typing import Literal

import lightning as L
import torch
import wandb
from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.ml.conditions import C_SEQ
from jaxtyping import Bool, Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch import Tensor
from torch.utils.data import DistributedSampler

from rfproteina.encoding import GLYCAN_NAMES, NCAA_NAMES, UNIFIED_ATOM37_ENCODING
from rfproteina.model_masks import compute_ae_supervision_mask
from rfproteina.partial_autoencoder.decoder import DecoderTransformer
from rfproteina.partial_autoencoder.encoder import EncoderTransformer
from rfproteina.utils.atom_permutation import permute_gt_atom37
from rfproteina.utils.coors import nm_to_ang
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.optim_utils import backward_with_accumulation, build_optimizers
from rfproteina.utils.structure_export import (
    process_single_sample_to_atom_array,
    write_generated_structure_cif,
    write_input_structure_cif,
)

logger = RankedLogger(__name__, rank_zero_only=True)

VALIDATION_RESIDUE_GROUPS = {
    "aa": tuple(STANDARD_AA),
    "ncaa": NCAA_NAMES,
    "glycan": GLYCAN_NAMES,
    "dna": tuple(STANDARD_DNA),
    "rna": tuple(STANDARD_RNA),
}

_RESIDUE_GROUP_IDS = tuple(
    next(
        (i for i, residues in enumerate(VALIDATION_RESIDUE_GROUPS.values()) if name in residues),
        len(VALIDATION_RESIDUE_GROUPS),
    )
    for name in UNIFIED_ATOM37_ENCODING.tokens
)


def blended_token_weights(
    residue_type: Tensor, mask: Bool[Tensor, "b n"], balanced_fraction: float = 0.25
) -> Float[Tensor, "b n"]:
    """Blend token and equal-group averages per example, leaving atomized tokens unweighted."""
    if not 0 <= balanced_fraction <= 1:
        raise ValueError("loss.balanced_fraction must be between 0 and 1")
    residue_mask = mask & (residue_type != UNIFIED_ATOM37_ENCODING.token_to_idx["<A>"])
    groups = torch.tensor(_RESIDUE_GROUP_IDS, device=mask.device)[residue_type.long()]
    counts = torch.zeros(mask.shape[0], len(VALIDATION_RESIDUE_GROUPS) + 1, device=mask.device)
    counts.scatter_add_(1, groups, residue_mask.float())
    weights = residue_mask / counts.gather(1, groups).clamp(min=1)
    weights = weights * residue_mask.sum(-1, keepdim=True) / weights.sum(-1, keepdim=True).clamp(min=1)
    return torch.where(residue_mask, 1 - balanced_fraction + balanced_fraction * weights, 1.0) * mask


def optimization_loss(metrics: dict[str, Tensor], structure_type: str) -> Tensor:
    """Sum weighted objective terms, excluding reconstruction diagnostics."""
    structure_keys = {"l1": ("struct_l1",), "l2": ("struct_l2",), "l12": ("struct_l1", "struct_l2")}
    if structure_type not in structure_keys:
        raise ValueError(f"Unknown structure loss type: {structure_type}")
    keys = ("kl_w", "ce_seq", "atom_presence_bce") + structure_keys[structure_type]
    return sum(metrics[key].mean() for key in keys)


def optimization_loss(metrics: dict[str, Tensor], structure_type: str) -> Tensor:
    """Sum weighted KL, sequence, and configured structure terms, excluding diagnostics."""
    structure_keys = {"l1": ("struct_l1",), "l2": ("struct_l2",), "l12": ("struct_l1", "struct_l2")}
    if structure_type not in structure_keys:
        raise ValueError(f"Unknown structure loss type: {structure_type}")
    return sum(metrics[key].mean() for key in ("kl_w", "ce_seq", *structure_keys[structure_type]))


@rank_zero_only
def create_dir(dir: str) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


class AutoEncoder(L.LightningModule):
    def __init__(self, cfg_ae: object, store_dir: str | None = None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg_ae = cfg_ae

        self.store_dir = store_dir if store_dir is not None else "./tmp_ae"
        self.val_path_tmp = os.path.join(self.store_dir, "val_stuff")
        create_dir(self.val_path_tmp)

        n_residue_types = self.cfg_ae.nn_ae.get("n_residue_types")
        if n_residue_types != UNIFIED_ATOM37_ENCODING.n_tokens:
            raise ValueError(
                f"AE vocabulary requires {UNIFIED_ATOM37_ENCODING.n_tokens} classes, " f"got {n_residue_types}"
            )

        self.encoder = EncoderTransformer(**self.cfg_ae.nn_ae)
        self.decoder = DecoderTransformer(**self.cfg_ae.nn_ae)
        self.nsamples_processed = 0
        self.nparams_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        self.nparams_dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

        self.validation_data_samples = []
        self.validation_rec_samples = []

        self.latent_dim = self.cfg_ae.nn_ae["latent_z_dim"]

        # Muon and Adam are stepped by hand in training_step, which requires manual optimization.
        # Gradient clipping and accumulation are handled there too, not by the Trainer.
        self.automatic_optimization = False

    def log_histogram(self, id_log: str, v: Float[torch.Tensor, "r"]) -> None:
        """
        Logs histogram, v must be a flat tensor.
        """
        assert v.ndim == 1, f"Tensor v has shape {v.shape}, cannot log histogram"
        with contextlib.suppress(Exception):
            self.logger.experiment.log({id_log: wandb.Histogram(v.cpu().detach().numpy())})

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        return build_optimizers(self.named_parameters(), self.cfg_ae.opt)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Adds additional variables to checkpoint."""
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Loads additional variables from checkpoint."""
        if (self.cfg_ae.get("lora") or {}).get("r"):
            raise ValueError("LoRA adapts the RFD4-Proteína denoiser; autoencoder LoRA is not supported.")
        try:
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except KeyError:
            logger.info("Failed to load nsamples_processed from checkpoint")
            self.nsamples_processed = 0

    def encode(self, batch: dict) -> Float[torch.Tensor, "b n d"]:
        """
        Runs the encoder and returns only the latent variables.
        """
        return self.encoder(batch)  # z_latent, mean, log_scale

    def decode(self, batch: dict, required_atom_mask: Bool[Tensor, "b n 37"] | None = None) -> dict:
        """
        Runs the decoder on the regular batch and returns decoding information.

        The batch must contain ``z_latent``, ``ca_coors_nm``, and the mask features
        (``padding_mask``, ``is_in_context_conditioning``, ``supports_local_latent``) from which the decoder
        derives its attention and output masks (see :py:func:`compute_ae_masks`);
        ``token_type`` is optional.

        Checkpoints with an ``attend_to_*`` flag require the corresponding conditioning feature.
        ``required_atom_mask`` retains required legal slots with their predicted coordinates.
        """
        output_dec = self.decoder(batch)
        emit_mask = output_dec["emit_mask"]  # [b, n]
        residue_type = output_dec["aatype_max"]
        if C_SEQ.full_name in batch:
            sequence = batch[C_SEQ.full_name]
            conditioned = (sequence != UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]) & emit_mask
            residue_type = torch.where(conditioned, sequence, residue_type)
        legal_slots = self.decoder.residue_atom_mask[residue_type]
        predicted_mask = output_dec["atom_presence_logits"] > 0
        predicted_mask[..., 1] = True
        if required_atom_mask is not None:
            predicted_mask = predicted_mask | required_atom_mask
        atom_mask = legal_slots & predicted_mask & emit_mask[..., None]
        coors_nm = output_dec["coors_nm"] * emit_mask[..., None, None] * atom_mask[..., None]  # [b, n, 37, 3]
        return {
            "coors_nm": coors_nm,
            "residue_type": residue_type * emit_mask,
            "emit_mask": emit_mask,
            "atom_mask": atom_mask,
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batch dimension.
        """
        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"
        histogram_every_n = 5000
        batch = {**batch}
        batch["coord_mask"] = batch["atom37_mask"]
        coords_nm = batch["coords"] / 10.0  # Convert Angstroms to nanometers
        batch["coords_nm"] = coords_nm  # Consumed by the encoder/decoder feature factories
        ca_coors_nm = coords_nm[..., 1, :]  # [b, n, 3] - CA is at slot 1
        bs = batch["padding_mask"].shape[0]

        output_enc = self.encoder(batch)
        loss_mask = compute_ae_supervision_mask(batch)
        log_prefix_stats = "train_stats_latent" if "train" in log_prefix else "val_stats_latent"
        self.log_tensor_statistics(
            bs=bs,
            v=output_enc["mean"],
            log_prefix=log_prefix_stats + "_mean_LS",
            mask=loss_mask,
            histogram_every_n=histogram_every_n,
        )
        self.log_tensor_statistics(
            bs=bs,
            v=torch.exp(output_enc["log_scale"]),
            log_prefix=log_prefix_stats + "_scale_LS",
            mask=loss_mask,
            histogram_every_n=histogram_every_n,
        )
        self.log_tensor_statistics(
            bs=bs,
            v=output_enc["z_latent"],
            log_prefix=log_prefix_stats + "_z_LS",
            mask=loss_mask,
            histogram_every_n=histogram_every_n,
        )
        # Decode from the same batch (augmented with the latents + CA); the decoder derives
        # its masks from the batch and returns `emit_mask`, which drives the losses.
        batch["z_latent"] = output_enc["z_latent"]
        batch["ca_coors_nm"] = ca_coors_nm
        output_dec = self.decoder(batch)

        losses = {}
        log_every_n = self.trainer.log_every_n_steps
        token_metrics = {} if not val_step and log_every_n > 0 and self.global_step % log_every_n == 0 else None

        # KL loss with annealing weight
        f = min(1.0, self.global_step / self.cfg_ae.loss.kl.patience) if self.cfg_ae.loss.kl.anneal else 1.0
        self.log(
            "kl_weight",
            self.cfg_ae.loss.kl.weight * f,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        losses.update(
            self.compute_kl_penalty(
                mean=output_enc["mean"],
                log_scale=output_enc["log_scale"],
                mask=loss_mask,
                w=self.cfg_ae.loss.kl.weight * f,
                token_metrics=token_metrics,
            )
        )

        token_weights = None if val_step else batch.get("loss_weight")
        if not val_step and self.cfg_ae.loss.get("balanced_fraction", 0):
            token_weights = blended_token_weights(batch["residue_type"], loss_mask, self.cfg_ae.loss.balanced_fraction)

        # Structure loss
        losses.update(
            self.compute_struct_rec_loss(
                output_dec=output_dec,
                batch=batch,
                loss_mask=loss_mask,
                reduce_mode="sum",
                loss_ty=self.cfg_ae.loss.struct.type,
                weight=self.cfg_ae.loss.struct.weight,
                atom_permutation=self.cfg_ae.loss.struct.get("atom_permutation", False),
                atom_permutation_align=self.cfg_ae.loss.struct.get("atom_permutation_align", False),
                token_weights=token_weights,
                token_metrics=token_metrics,
            )
        )

        # Sequence loss
        losses.update(
            self.compute_seq_rec_loss(
                output_dec=output_dec,
                batch=batch,
                loss_mask=loss_mask,
                weight=self.cfg_ae.loss.seq.weight,
                token_weights=token_weights,
                token_metrics=token_metrics,
            )
        )

        losses.update(
            self.compute_atom_presence_loss(
                output_dec=output_dec,
                batch=batch,
                loss_mask=loss_mask,
                weight=self.cfg_ae.loss.atom_presence.weight,
                token_weights=token_weights,
                token_metrics=token_metrics,
            )
        )

        self.log_losses(bs, losses, log_prefix)
        train_loss = optimization_loss(losses, self.cfg_ae.loss.struct.type)
        if token_metrics is not None:
            self.log_token_type_metrics(batch, loss_mask, token_metrics)

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=False,
        )

        if not val_step:  # Don't log these for val step
            # Log Kl statistics
            componentwise_kl = self._per_component_kl(
                mean=output_enc["mean"],
                log_scale=output_enc["log_scale"],
                mask=loss_mask,
            )  # [b, n, d]
            self.log_tensor_statistics(
                bs=bs,
                v=componentwise_kl,
                log_prefix=log_prefix + "_kl_latent",
                mask=loss_mask,
                histogram_every_n=histogram_every_n,
            )
            self.log_tensor_statistics(
                bs=bs,
                v=(componentwise_kl > 0.1) * 1.0,
                log_prefix=log_prefix + "_kl_latent_active_thresh_0p1",
                mask=loss_mask,
                do_hist=False,
                stats_to_log=["mean"],
            )
            # KL per aa type
            if self.global_step % histogram_every_n == 2000:
                # Get number of residue types from decoder output
                n_residue_types = self.decoder.logit_linear[-1].out_features
                for i in range(n_residue_types):
                    self.log_tensor_statistics(
                        bs=bs,
                        v=(componentwise_kl > 0.1) * 1.0,
                        log_prefix=log_prefix_stats + "_kl_latent_active_thresh_0p1_per_restype",
                        mask=loss_mask * (batch["residue_type"] == i),
                        do_hist=False,
                        suffix=f"_restype_{i}",
                        stats_to_log=["mean"],
                    )

            self.log_train_loss_n_prog_bar(bs, train_loss)
            self.update_n_log_nsamples_processed(bs)
            self.log_nparams()

            should_step = backward_with_accumulation(
                self, train_loss, batch_idx, self.cfg_ae.opt.get("accumulate_grad_batches", 1)
            )
            if should_step:
                # [muon, adam], or just adam if the exclude patterns left Muon nothing.
                optimizers = self.optimizers()
                if not isinstance(optimizers, list):
                    optimizers = [optimizers]

                clip_val = self.cfg_ae.opt.get("gradient_clip_val", 1.0)
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

        if val_step:
            return train_loss, output_dec
        return train_loss

    @staticmethod
    def compute_struct_rec_loss(
        output_dec: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        loss_mask: Bool[Tensor, "b n"],
        reduce_mode: Literal["sum", "mean"] = "sum",
        loss_ty: str = "l2",
        weight: float = 1.0,
        atom_permutation_align: bool = False,
        atom_permutation: bool = False,
        token_weights: Float[Tensor, "b n"] | None = None,
        token_metrics: dict[str, Tensor] | None = None,
    ) -> dict[str, Float[Tensor, "b"]]:
        """
        Computes structural loss.

        Args:
            output_dec:
                {
                  "coors_nm": all atom coordinates, shape [b, n, 37, 3], in nm
                  "seq_logits": logits for the residue types, shape [b, n, n_residue_types]
                  "emit_mask": boolean [b, n]
                  "aatype_max": residue type by taking the most likely logit, shape [b, n]
                  "atom_mask": boolean [b, n, 37], predicted atom-presence mask
                }
            batch: data batch from dataloader
            reduce_mode: whether to apply mean over atoms or just sum, when reducing accross 37 atoms types
            loss_ty: whether to apply l1 loss, l2 loss, or both, l2cut (ignore atoms close to each other, to expand)
            token_weights: Optional reconstruction multipliers; normalization remains the supervised token count.
            token_metrics: Optional output dictionary for detached per-token diagnostics.
        Returns:
            Per-example weighted losses and diagnostics. Objective terms are selected by optimization_loss.
        """

        def reduce_37(
            err: Float[torch.Tensor, "b n 37 3"],
            mask: Bool[torch.Tensor, "b n"],
            atom_mask: Bool[torch.Tensor, "b n 37"],
            mode: Literal["sum", "mean"] = "sum",
        ) -> Float[torch.Tensor, "b"]:
            # Masked entries have zero numerators, so clamping empty-token and empty-atom
            # denominators preserves zero loss instead of producing NaN.
            nres = mask.sum(dim=-1).clamp(min=1)  # [b]
            nat = (atom_mask.sum(dim=-1) * mask).clamp(min=1)  # [b, n]
            err = torch.sum(err, dim=(-1, -2))  # [b, n]
            if mode == "mean":
                err = err / nat  # Take mean over existing atoms if mode == "mean"
            if token_weights is not None:
                err = err * token_weights
            err = err.sum(dim=-1) / nres  # [b]
            return err

        coors_nm_pred = output_dec["coors_nm"]  # [b, n, 37, 3]
        coors_nm_true = batch["coords_nm"]  # [b, n, 37, 3] (already converted in training_step)
        # Unresolved emissions provide context but are excluded from supervision.
        mask = loss_mask  # [b, n] boolean; tokens we supervise
        atom_mask_true = batch["atom37_mask"] * mask[..., None]  # [b, n, 37] boolean

        # Permute target atom37 coords onto the prediction (Tyr ring flips, ligand
        # automorphisms, …), using what the dataloader workers precomputed.
        extra_info = batch.get("extra_info") or {}
        precomputed = extra_info.get("atom_automorphisms")
        if atom_permutation and precomputed is None:
            raise ValueError(
                "loss.struct.atom_permutation is on but no precomputed permutations reached the loss; "
                "set transform.precompute_atom_permutation=true on every training dataset."
            )
        if atom_permutation:
            with torch.no_grad():
                coors_nm_true = permute_gt_atom37(
                    coors_nm_pred,
                    coors_nm_true,
                    atom_mask_true,
                    precomputed,
                    extra_info.get("atom37_layout"),
                    align=atom_permutation_align,
                )

        err = coors_nm_true - coors_nm_pred  # [b, n, 37, 3]
        err = err * mask[..., None, None] * atom_mask_true[..., None]  # [b, n, 37, 3]

        losses = {}

        # Compute RMSD in Å (without alignment)
        err_ang = nm_to_ang(err)  # [b, n, 37, 3]
        err_ang = torch.linalg.norm(err_ang, dim=-1) ** 2  # [b, n, 37]
        err_ang = err_ang * mask[..., None] * atom_mask_true  # [b, n, 37]
        # clamp(min=1): an element with no supervised atoms has a zero masked numerator, so
        # this reports an RMSD of 0 rather than a NaN that would poison any average over it.
        nat = atom_mask_true.sum((-1, -2)).clamp(min=1)  # [b]
        rmsd = torch.sqrt(torch.sum(err_ang, dim=(-1, -2)) / nat)  # [b]
        losses["rmsd_no_align_a37_ang"] = rmsd

        err_l1 = reduce_37(torch.abs(err), mask, atom_mask_true, mode=reduce_mode)
        err_l2 = reduce_37(err**2, mask, atom_mask_true, mode=reduce_mode)
        if token_metrics is not None:
            token_metrics["atom_count"] = atom_mask_true.sum(-1).detach()
            token_metrics["squared_error_ang"] = err_ang.sum(-1).detach()
            for name, error in (("struct_l1", err.abs()), ("struct_l2", err.square())):
                values = error.sum((-1, -2)).detach()
                if reduce_mode == "mean":
                    values = values / atom_mask_true.sum(-1).clamp(min=1)
                if token_weights is not None:
                    values = values * token_weights.detach()
                token_metrics[name] = values * weight

        if loss_ty not in {"l1", "l2", "l12"}:
            raise OSError(f"Loss type {loss_ty} not recognized")
        losses["struct_l1"] = err_l1 * weight
        losses["struct_l2"] = err_l2 * weight
        return losses

    def compute_seq_rec_loss(
        self,
        output_dec: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        loss_mask: Bool[Tensor, "b n"],
        weight: float = 1.0,
        token_weights: Float[Tensor, "b n"] | None = None,
        token_metrics: dict[str, Tensor] | None = None,
    ) -> dict[str, Float[Tensor, "b"]]:
        """
        Computes cross entropy loss on predicted sequence.

        Args:
            output_dec:
                {
                  "coors_nm": all atom coordinates, shape [b, n, 37, 3], in nm
                  "seq_logits": logits for the residue types, shape [b, n, n_residue_types]
                  "emit_mask": boolean [b, n]
                  "aatype_max": residue type by taking the most likely logit, shape [b, n]
                  "atom_mask": boolean [b, n, 37], predicted atom-presence mask
                }
            batch: data batch from dataloader
            token_weights: Optional reconstruction multipliers; recovery metrics remain unweighted.
            token_metrics: Optional output dictionary for detached per-token diagnostics.

        Returns:
            Per-example weighted losses and diagnostics. Objective terms are selected by optimization_loss.
        """
        assert "residue_type" in batch, "`residue_type` not in batch, failed in compute_seq_rec_loss"
        # See compute_struct_rec_loss: supervision is `loss_mask`, not `emit_mask`.
        mask = loss_mask  # [b, n]; tokens we supervise
        # clamp(min=1): elements with no supervised tokens have a zero numerator below,
        # so this yields 0 loss for them rather than a 0/0 NaN.
        nres = mask.sum(dim=-1).clamp(min=1)  # [b]
        logits_pred = output_dec["seq_logits"]  # [b, n, n_residue_types]
        target_aa = batch["residue_type"]  # [b, n]
        target_aa = target_aa * mask  # [b, n] gets rid of -1 for padding (issue with cross entropy loss below)

        n_residue_types = logits_pred.shape[-1]
        assert (
            n_residue_types == UNIFIED_ATOM37_ENCODING.n_tokens
        ), f"Expected {UNIFIED_ATOM37_ENCODING.n_tokens} residue types, got {n_residue_types}"

        # Compute cross entropy
        b, n = mask.shape[0], mask.shape[1]
        logits_pred_flat = logits_pred.view(b * n, n_residue_types)  # [b * n, n_residue_types]
        target_aa_flat = target_aa.view(b * n)  # [b * n]
        seq_loss_flat = torch.nn.functional.cross_entropy(
            input=logits_pred_flat,
            target=target_aa_flat,
            reduction="none",
        )  # [b * n]
        seq_loss = seq_loss_flat.view(b, n)  # [b, n]
        if token_weights is not None:
            seq_loss = seq_loss * token_weights
        if token_metrics is not None:
            token_metrics["ce_seq_before_loss_weight"] = seq_loss.detach()
            token_metrics["ce_seq"] = seq_loss.detach() * weight
            token_metrics["seq_rec_rate"] = (output_dec["aatype_max"] == target_aa).float().detach()
        seq_loss = seq_loss * mask  # [b, n]
        seq_loss = torch.sum(seq_loss, dim=-1) / nres  # [b]

        # Compute seq recovery rate
        pred_aa = output_dec["aatype_max"]  # [b, n]
        seq_rec = pred_aa == target_aa  # [b, n]
        seq_rec = seq_rec * mask  # [b, n]
        seq_rec_rate = seq_rec.sum(dim=-1) / nres  # [b]
        return {
            "ce_seq": seq_loss * weight,
            "ce_seq_before_loss_weight": seq_loss,
            "seq_rec_rate": seq_rec_rate,
        }

    def compute_atom_presence_loss(
        self,
        output_dec: dict[str, Tensor],
        batch: dict[str, Tensor],
        loss_mask: Bool[Tensor, "b n"],
        weight: float,
        token_weights: Tensor | None = None,
        token_metrics: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Supervise input atom inventory on resolved tokens, averaged over legal slots.

        Coordinate supervision remains independently masked by ground-truth presence.
        """
        logits = output_dec["atom_presence_logits"]
        slots = self.decoder.residue_atom_mask[batch["residue_type"].long()] & loss_mask[..., None]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["atom37_mask"].to(logits.dtype), reduction="none"
        )
        per_token = (bce * slots).sum(-1) / slots.sum(-1).clamp(min=1)
        per_token = per_token * weight
        if token_weights is not None:
            per_token = per_token * token_weights
        if token_metrics is not None:
            token_metrics["atom_presence_bce"] = per_token.detach()
        return {"atom_presence_bce": per_token.sum(-1) / loss_mask.sum(-1).clamp(min=1)}

    def _per_component_kl(
        self,
        mean: Float[Tensor, "b n d"],
        log_scale: Float[Tensor, "b n d"],
        mask: Bool[Tensor, "b n"],
    ) -> Float[Tensor, "b n d"]:
        """
        Computes KL penalty on the latent Gaussian distribution, per residue.

        Returns KL, per residue (masked) and latent dimension, shape [b n d].
        """
        scale = torch.exp(log_scale)  # [b, n, 3]
        kl_div = (scale**2 + mean**2 - 1.0 - 2.0 * log_scale) * 0.5  # [b, n, d]
        return kl_div * mask[..., None]  # [b, n, d]

    def compute_kl_penalty(
        self,
        mean: Float[Tensor, "b n d"],
        log_scale: Float[Tensor, "b n d"],
        mask: Bool[Tensor, "b n"],
        w: float,
        token_metrics: dict[str, Tensor] | None = None,
    ) -> dict[str, Float[Tensor, "b"]]:
        """Compute the KL penalty on the latent Gaussian distribution."""
        # clamp(min=1): a batch element with no supervised tokens has a zero masked KL
        # numerator, so this keeps its penalty at 0 instead of dividing 0 by 0.
        nres = torch.sum(mask, dim=-1).clamp(min=1)  # [b]
        kl_div = self._per_component_kl(
            mean=mean,
            log_scale=log_scale,
            mask=mask,
        )  # [b, n, d]
        if token_metrics is not None:
            token_metrics["kl_now"] = kl_div.detach().sum(-1)
            token_metrics["kl_w"] = token_metrics["kl_now"] * w
        kl_div = torch.sum(kl_div, dim=(-1, -2))  # [b]
        kl_div = kl_div / nres  # [b]
        loss = {
            "kl_w": kl_div * w,
            "kl_now": kl_div,
        }
        return loss

    @torch.no_grad()
    def log_token_type_metrics(
        self,
        batch: dict[str, Tensor],
        mask: Bool[Tensor, "b n"],
        metrics: dict[str, Tensor],
    ) -> None:
        """Log rank-local, token-averaged diagnostics for sampled training batches.

        RMSD averages per-residue RMSDs in Angstroms; losses retain training weights.
        Empty groups are omitted; sequence recovery and RMSD are unweighted.
        """
        groups = torch.tensor(_RESIDUE_GROUP_IDS, device=mask.device)[batch["residue_type"].long()]
        for group_id, name in enumerate((*VALIDATION_RESIDUE_GROUPS, "other")):
            selected = mask & (groups == group_id)
            values = {key: value[selected] for key, value in metrics.items()}
            if not values["seq_rec_rate"].numel():
                continue
            prefix = f"train_by_token_type/{name}"
            scalars = {
                key: value.mean() for key, value in values.items() if key not in {"atom_count", "squared_error_ang"}
            }
            occupied = values["atom_count"] > 0
            residue_rmsds = (values["squared_error_ang"][occupied] / values["atom_count"][occupied]).sqrt()
            if residue_rmsds.numel():
                scalars["rmsd_no_align_a37_ang"] = residue_rmsds.mean()
            scalars["token_count"] = selected.sum().float()
            scalars["loss"] = optimization_loss(scalars, self.cfg_ae.loss.struct.type)
            for key, value in scalars.items():
                self.log(f"{prefix}/{key}", value, on_step=True, on_epoch=False, sync_dist=False)

    def log_tensor_statistics(
        self,
        bs: int,
        v: Float[torch.Tensor, "b n d"],
        log_prefix: str,
        mask: Bool[torch.Tensor, "b n"],
        do_stats: bool = True,
        do_hist: bool = True,
        suffix: str = "",
        histogram_every_n: int = 3000,
        stats_to_log: list[str] = ["max", "min", "mean", "median", "std"],
    ) -> None:
        """Log local statistics, every 50 training steps; omit empty supervision."""
        if log_prefix.startswith("train") and self.global_step % 50:
            return
        vals = v.detach()[mask].flatten().float()
        if not vals.numel():
            return
        if do_stats:
            funs = {
                "max": torch.max,
                "min": torch.min,
                "mean": torch.mean,
                "median": torch.median,
                "std": lambda values: values.std() if values.numel() > 1 else values.new_zeros(()),
            }
            for k in stats_to_log:
                self.log(
                    f"{log_prefix}/{k}{suffix}",
                    funs[k](vals),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=False,
                )

        if vals.numel() and histogram_every_n and self.global_step % histogram_every_n == 0 and do_hist:
            self.log_histogram(id_log=f"{log_prefix}/histogram{suffix}", v=vals)

    def log_losses(self, bs: int, losses: dict[str, Float[torch.Tensor, "b"]], log_prefix: str) -> None:
        for k in losses:
            self.log(
                f"{log_prefix}/loss_{k}",
                torch.mean(losses[k]),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=False,
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
            sync_dist=False,
        )

    def log_nparams(self) -> None:
        self.log(
            "scaling/nparams_enc",
            self.nparams_enc * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=False,
        )  # constant line but ok, easy to compare # params

        self.log(
            "scaling/nparams_dec",
            self.nparams_dec * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=False,
        )  # constant line but ok, easy to compare # params

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
            sync_dist=False,
        )

    def on_validation_epoch_start(self) -> None:
        """Initialize exact per-panel sums for residue-group reconstruction metrics."""
        loaders = self.trainer.val_dataloaders
        self._val_loaders = list(loaders) if isinstance(loaders, list | tuple) else [loaders]
        self._val_panel_names = [name for name, config in self.cfg_ae.datasets.val.items() if config]
        if len(self._val_panel_names) != len(self._val_loaders):
            raise RuntimeError("Validation dataset names and dataloaders are misaligned")

        self._val_residue_groups = torch.tensor(_RESIDUE_GROUP_IDS, device=self.device)[:, None] == torch.arange(
            len(VALIDATION_RESIDUE_GROUPS), device=self.device
        )
        self._val_rmsd_totals = [
            torch.zeros(len(VALIDATION_RESIDUE_GROUPS), 2, device=self.device, dtype=torch.float64)
            for _ in self._val_loaders
        ]

    def _accumulate_validation_rmsd(
        self, batch: dict, output_dec: dict[str, Tensor], batch_idx: int, dataloader_idx: int
    ) -> None:
        """Accumulate permutation-aware per-residue RMSD sums and residue counts by group."""
        mask = compute_ae_supervision_mask(batch)
        loader = self._val_loaders[dataloader_idx]
        sampler = loader.sampler
        if isinstance(sampler, DistributedSampler) and not sampler.drop_last:
            positions = batch_idx * loader.batch_size + torch.arange(len(mask), device=mask.device)
            mask = mask & ((positions * sampler.num_replicas + sampler.rank) < len(loader.dataset))[:, None]

        atom_mask = batch["atom37_mask"].bool() & mask[..., None]
        target_nm = batch["coords"] / 10.0
        if self.cfg_ae.loss.struct.get("atom_permutation", False):
            extra_info = batch.get("extra_info") or {}
            target_nm = permute_gt_atom37(
                output_dec["coors_nm"],
                target_nm,
                atom_mask,
                extra_info.get("atom_automorphisms"),
                extra_info.get("atom37_layout"),
                align=self.cfg_ae.loss.struct.get("atom_permutation_align", False),
            )

        token_sse_nm2 = (((target_nm - output_dec["coors_nm"]) * atom_mask[..., None]) ** 2).sum((-1, -2))
        token_atom_count = atom_mask.sum(-1)
        residue_type = batch["residue_type"]
        valid_residue_type = (residue_type >= 0) & (residue_type < len(self._val_residue_groups))
        residue_type = residue_type.clamp(min=0, max=len(self._val_residue_groups) - 1)
        groups = self._val_residue_groups[residue_type] & valid_residue_type[..., None] & mask[..., None]
        groups = groups & (token_atom_count > 0)[..., None]
        token_rmsd_nm = (token_sse_nm2 / token_atom_count.clamp(min=1)).sqrt()
        numerators = groups.flatten(0, 1).double().T @ token_rmsd_nm.flatten().double()
        denominators = groups.sum((0, 1)).double()
        self._val_rmsd_totals[dataloader_idx] += torch.stack((numerators, denominators), dim=-1)

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute reconstruction loss and collect reconstructions from each validation loader."""
        with torch.no_grad():
            # Convert to expected format (coords_nm will be set in training_step)
            bs = batch["coords"].shape[0]
            _, rec_sample = self.training_step(batch, batch_idx=-1)
            self._accumulate_validation_rmsd(batch, rec_sample, batch_idx, dataloader_idx)
            coords_nm = batch["coords"] / 10.0  # Convert Angstroms to nanometers
            atom_array_batch = batch.get("atom_array_batch")
            for i in range(bs):
                if len(self.validation_data_samples) >= 10:
                    break
                self.validation_data_samples.append(
                    {
                        "coors_nm": coords_nm[i, ...].detach().cpu(),  # [n, 37, 3]
                        "aatype": batch["residue_type"][i, ...].detach().cpu(),  # [n]
                        "atom_mask": batch["atom37_mask"][i, ...].detach().cpu(),  # [n, 37]
                        "padding_mask": batch["padding_mask"][i, :].detach().cpu(),  # [n]
                        "atom_array": atom_array_batch[i] if atom_array_batch is not None else None,
                    }
                )
                self.validation_rec_samples.append(
                    {
                        "coors_nm": rec_sample["coors_nm"][i, ...].detach().cpu(),  # [n, 37, 3]
                        "aatype": rec_sample["aatype_max"][i, ...].detach().cpu(),  # [n]
                        "atom_mask": rec_sample["atom_mask"][i, ...].detach().cpu(),  # [n, 37]
                        "emit_mask": rec_sample["emit_mask"][i, ...].detach().cpu(),  # [n]
                    }
                )

    def on_validation_epoch_end(self) -> None:
        """Log per-panel residue-group metrics and write representative reconstruction CIFs."""
        totals_by_loader = [self.trainer.strategy.reduce(totals, reduce_op="sum") for totals in self._val_rmsd_totals]
        for panel, totals in zip(self._val_panel_names, totals_by_loader, strict=True):
            for group, row in zip(VALIDATION_RESIDUE_GROUPS, totals, strict=True):
                if row[1] > 0:
                    rmsd = 10 * row[0] / row[1]
                    self.log(
                        f"validation_rmsd/{panel}/{group}_ang",
                        rmsd,
                        add_dataloader_idx=False,
                        sync_dist=False,
                    )

        cif_suffix = ".cif.zst" if self.cfg_ae.get("compress_outputs", True) else ".cif"
        for count, (sample_data, sample_rec) in enumerate(
            zip(self.validation_data_samples, self.validation_rec_samples, strict=False), start=1
        ):
            if count > 10:
                break

            input_atom_array = sample_data.get("atom_array")
            if input_atom_array is None:
                logger.warning(f"Skipping validation CIF export for sample {count}: missing atom_array in batch")
                continue

            padding_mask = sample_data["padding_mask"].detach().cpu().bool()
            emit_mask = sample_rec["emit_mask"].detach().cpu().bool()
            no_latent_tokens = padding_mask & ~emit_mask
            coords_rec_ang = nm_to_ang(sample_rec["coors_nm"]).float().detach().cpu()
            aatype_data = sample_data["aatype"].detach().cpu()
            aatype_rec = sample_rec["aatype"].detach().cpu()
            atom_mask_data = sample_data["atom_mask"].detach().cpu().clone()
            atom_mask_rec = sample_rec["atom_mask"].detach().cpu().clone()

            # Atomized context has no latent output. Preserve its input CA as the
            # unencoded token representation used by flow-model export.
            coords_rec_ang[no_latent_tokens] = 0
            coords_rec_ang[..., 1, :] = torch.where(
                no_latent_tokens[..., None],
                nm_to_ang(sample_data["coors_nm"].detach().cpu()[..., 1, :]),
                coords_rec_ang[..., 1, :],
            )
            unencoded = torch.full_like(aatype_rec, -1)
            aatype_rec = torch.where(emit_mask, aatype_rec, unencoded)
            aatype_data = torch.where(emit_mask, aatype_data, unencoded)
            for atom_mask in (atom_mask_rec, atom_mask_data):
                atom_mask[no_latent_tokens] = False
                atom_mask[..., 1] |= no_latent_tokens

            f_base = f"epoch_{self.current_epoch}_id_{count}_rank_{self.global_rank}"

            try:
                write_input_structure_cif(
                    input_atom_array,
                    os.path.join(self.val_path_tmp, f_base + "_data_sample" + cif_suffix),
                )

                rec_result = process_single_sample_to_atom_array(
                    coords=coords_rec_ang,
                    residue_type=aatype_rec,
                    padding_mask=padding_mask,
                    input_atom_array=input_atom_array.copy(),
                    atom_mask=atom_mask_rec,
                )
                write_generated_structure_cif(
                    rec_result["generated_atom_array"],
                    os.path.join(self.val_path_tmp, f_base + "_rec_sample" + cif_suffix),
                )

                rec_true_seq_result = process_single_sample_to_atom_array(
                    coords=coords_rec_ang,
                    residue_type=aatype_data,
                    padding_mask=padding_mask,
                    input_atom_array=input_atom_array.copy(),
                    atom_mask=atom_mask_data,
                )
                write_generated_structure_cif(
                    rec_true_seq_result["generated_atom_array"],
                    os.path.join(self.val_path_tmp, f_base + "_rec_sample_w_true_seq" + cif_suffix),
                )
            except Exception as e:
                logger.warning(f"Failed to write validation CIFs for sample {count}: {e}")

        self.validation_data_samples = []
        self.validation_rec_samples = []

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Makes predictions. Given a data batch, encodes, and returns decoded batch.

        Args:
            batch: data batch.

        Returns:
            Dict representing the decoded batch, with all info from the encoder output.
        """
        # Copied, but the batch returned below still carries the injected keys for the metrics.
        batch = {**batch}
        coords_nm = batch["coords"] / 10.0  # Convert Angstroms to nanometers
        batch["coords_nm"] = coords_nm  # Consumed by the encoder/decoder feature factories
        ca_coors_nm = coords_nm[..., 1, :]  # [b, n, 3] - CA is at slot 1

        output_enc = self.encoder(batch)
        # {
        #   "z_latent": latent_sample, shape [b, n, d]
        #   "mean": mean of latent (diag) Gaussian dist, shape [b, n, d]
        #   "log_scale": log standard deviation of latent (diag) Gaussian dist, shape [b, n, d]
        #   "emit_mask": boolean [b, n]
        # }

        # Decode from the same batch; token_type is already present so the decoder's
        # per-token-type logit mask still fires. The decoder derives its masks from the batch.
        batch["z_latent"] = output_enc["z_latent"]
        batch["ca_coors_nm"] = ca_coors_nm
        output = self.decoder(batch)

        output.update(output_enc)
        return (batch, output)
