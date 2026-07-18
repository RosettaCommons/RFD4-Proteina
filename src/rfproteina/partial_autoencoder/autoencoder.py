import contextlib
import os
from typing import Literal

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Bool, Float, Int
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from sklearn.decomposition import PCA
from torch import Tensor

from rfproteina.partial_autoencoder.decoder import DecoderTransformer
from rfproteina.partial_autoencoder.decoder_ff import DecoderFFLocal
from rfproteina.partial_autoencoder.encoder import EncoderTransformer
from rfproteina.partial_autoencoder.latent_viz_utils import (
    plot_latent_projection_with_labels,
    project_latents_2d,
)
from rfproteina.utils.coors import nm_to_ang
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.optim_utils import build_optimizers
from rfproteina.utils.structure_export import (
    process_single_sample_to_atom_array,
    write_generated_structure_cif,
    write_input_structure_cif,
)
from rfproteina.utils.torch_compile import maybe_mark_cudagraph_step_begin

logger = RankedLogger(__name__, rank_zero_only=True)

COLORS_RT = [
    "#FF0000",  # Red
    "#008000",  # Green
    "#0000FF",  # Blue
    "#FFFF00",  # Yellow
    "#FFA500",  # Orange
    "#800080",  # Purple
    "#00FFFF",  # Cyan
    "#FF00FF",  # Magenta
    "#00FF00",  # Lime
    "#FFC0CB",  # Pink
    "#008080",  # Teal
    "#E6E6FA",  # Lavender
    "#A52A2A",  # Brown
    "#F5F5DC",  # Beige
    "#800000",  # Maroon
    "#808000",  # Olive
    "#FF7F50",  # Coral
    "#000080",  # Navy
    "#AAF0D1",  # Mint
    "#FFDB58",  # Mustard
]


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

        self.encoder = EncoderTransformer(**self.cfg_ae.nn_ae)

        decoder_type = self.cfg_ae.nn_ae.decoder.get("type", "transformer")
        if decoder_type == "transformer":
            self.decoder = DecoderTransformer(**self.cfg_ae.nn_ae)
        elif decoder_type == "ff_local":
            self.decoder = DecoderFFLocal(**self.cfg_ae.nn_ae)
        else:
            raise OSError(f"Invalid decoder_type {decoder_type}")

        # Optionally wrap encoder/decoder forward with torch.compile
        self.use_torch_compile = self.cfg_ae.nn_ae.get("use_torch_compile", False)
        if self.use_torch_compile:
            logger.info("Enabling torch.compile for AutoEncoder encoder and decoder.")
            self.encoder.forward = torch.compile(self.encoder.forward)
            self.decoder.forward = torch.compile(self.decoder.forward)

        self.nsamples_processed = 0
        self.nparams_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        self.nparams_dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

        self.validation_output = []
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

    def decode(
        self,
        z_latent: Float[torch.Tensor, "b n d"],
        ca_coors_nm: Float[torch.Tensor, "b n 3"],
        mask: Bool[torch.Tensor, "b n"],
        token_type: Int[torch.Tensor, "b n"] | None = None,
    ) -> dict:
        """
        Runs the decoder and returns a dictionary with all necessary decoding information.
        """
        input_decoder = {
            "z_latent": z_latent,
            "ca_coors_nm": ca_coors_nm,
            "residue_mask": mask,
            "padding_mask": mask,
        }
        if token_type is not None:
            input_decoder["token_type"] = token_type
        output_dec = self.decoder(input_decoder)
        mask = output_dec["residue_mask"]  # [b, n]
        atom_mask = output_dec["atom_mask"]  # [b, n, 37]
        coors_nm = output_dec["coors_nm"] * mask[..., None, None] * atom_mask[..., None]  # [b, n, 37, 3]
        return {
            "coors_nm": coors_nm,
            "residue_type": output_dec["aatype_max"] * mask,
            "residue_mask": mask,
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
        if self.use_torch_compile:
            maybe_mark_cudagraph_step_begin()

        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"
        histogram_every_n = 5000
        pca_every_n = self.cfg_ae.get("log", {}).get("projection_every_n", 5000)
        per_aatype_kl = True
        # Extract from new pipeline batch format
        mask = batch["padding_mask"]  # [b, n] boolean, True = valid token
        batch["coord_mask"] = batch["atom37_mask"]
        coords_nm = batch["coords"] / 10.0  # Convert Angstroms to nanometers
        batch["coords_nm"] = coords_nm  # For backward compatibility with encoder
        ca_coors_nm = coords_nm[..., 1, :]  # [b, n, 3] - CA is at slot 1
        ca_coors_nm = ca_coors_nm * mask[..., None]  # [b, n, 3]
        bs, _n = mask.shape[0], mask.shape[1]

        output_enc = self.encoder(batch)
        # {
        #   "z_latent": latent_sample, shape [b, n, d]
        #   "mean": mean of latent (diag) Gaussian dist, shape [b, n, d]
        #   "log_scale": log standard deviation of latent (diag) Gaussian dist, shape [b, n, d]
        # }
        log_prefix_stats = "train_stats_latent" if "train" in log_prefix else "val_stats_latent"
        self.log_tensor_statistics(
            bs=bs,
            v=output_enc["mean"],
            log_prefix=log_prefix_stats + "_mean_LS",
            mask=mask,
            histogram_every_n=histogram_every_n,
        )
        self.log_tensor_statistics(
            bs=bs,
            v=torch.exp(output_enc["log_scale"]),
            log_prefix=log_prefix_stats + "_scale_LS",
            mask=mask,
            histogram_every_n=histogram_every_n,
        )
        self.log_tensor_statistics(
            bs=bs,
            v=output_enc["z_latent"],
            log_prefix=log_prefix_stats + "_z_LS",
            mask=mask,
            histogram_every_n=histogram_every_n,
        )
        self.log_pca(
            v=output_enc["z_latent"],
            log_prefix=log_prefix_stats + "_z_PCA",
            mask=mask,
            every_n=pca_every_n,
        )
        self.log_pca_per_residue_type(
            v=output_enc["z_latent"],
            log_prefix=log_prefix_stats + "_z_PCA_pre_aatype",
            mask=mask,
            every_n=pca_every_n,
            res_ty=batch["residue_type"],
        )

        # t-SNE projection colored by residue type (configurable method)
        projection_method = self.cfg_ae.get("log", {}).get("projection_method", "tsne")
        projection_max_samples = self.cfg_ae.get("log", {}).get("projection_max_samples", 5000)
        if projection_method and projection_method != "pca":  # PCA already logged above
            self.log_projection_per_residue_type(
                v=output_enc["z_latent"],
                log_prefix=log_prefix_stats + f"_z_{projection_method.upper()}_by_aatype",
                mask=mask,
                every_n=pca_every_n,
                res_ty=batch["residue_type"],
                method=projection_method,
                max_samples=projection_max_samples,
            )

        input_decoder = {
            "z_latent": output_enc["z_latent"],
            "ca_coors_nm": ca_coors_nm,
            "residue_mask": mask,
            "padding_mask": mask,
        }
        output_dec = self.decoder(input_decoder)
        # {
        #   "coors_nm": all atom coordinates, shape [b, n, 37, 3], in nm
        #   "seq_logits": logits for the residue types, shape [b, n, 20]
        #   "residue_mask": boolean [b, n]
        #   "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
        #   "atom_mask": boolean [b, n, 37, 3], atom37 mask corresponding to aatype_max
        # }

        losses = {}  # Will be a Dict[str, tensor[b]]. If "_justlog" in name just for logging

        # KL loss with annealing weight
        f = min(1.0, self.global_step / self.cfg_ae.loss.kl.patience) if self.cfg_ae.loss.kl.anneal else 1.0
        self.log(
            "kl_weight",
            self.cfg_ae.loss.kl.weight * f,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        losses.update(
            self.compute_kl_penalty(
                mean=output_enc["mean"],
                log_scale=output_enc["log_scale"],
                mask=mask,
                w=self.cfg_ae.loss.kl.weight * f,
            )
        )

        # Structure loss
        losses.update(
            self.compute_struct_rec_loss(
                output_dec=output_dec,
                batch=batch,
                reduce_mode="sum",
                loss_ty=self.cfg_ae.loss.struct.type,
                weight=self.cfg_ae.loss.struct.weight,
            )
        )

        # Sequence loss
        losses.update(
            self.compute_seq_rec_loss(
                output_dec=output_dec,
                batch=batch,
                weight=self.cfg_ae.loss.seq.weight,
            )
        )

        # Log losses and training loss, losses with "_justlog" just for logging purposes
        self.log_losses(bs, losses, log_prefix)
        train_loss = sum([torch.mean(losses[k]) for k in losses if "_justlog" not in k])

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if not val_step:  # Don't log these for val step
            # Log Kl statistics
            componentwise_kl = self._per_component_kl(
                mean=output_enc["mean"],
                log_scale=output_enc["log_scale"],
                mask=mask,
            )  # [b, n, d]
            self.log_tensor_statistics(
                bs=bs,
                v=componentwise_kl,
                log_prefix=log_prefix + "_kl_latent",
                mask=mask,
                histogram_every_n=histogram_every_n,
            )
            self.log_tensor_statistics(
                bs=bs,
                v=(componentwise_kl > 0.1) * 1.0,
                log_prefix=log_prefix + "_kl_latent_active_thresh_0p1",
                mask=mask,
                do_hist=False,
                stats_to_log=["mean"],
            )
            # KL per aa type
            if per_aatype_kl and self.global_step % histogram_every_n == 2000:
                # Get number of residue types from decoder output
                n_residue_types = self.decoder.logit_linear[-1].out_features
                for i in range(n_residue_types):
                    self.log_tensor_statistics(
                        bs=bs,
                        v=(componentwise_kl > 0.1) * 1.0,
                        log_prefix=log_prefix_stats + "_kl_latent_active_thresh_0p1_per_restype",
                        mask=mask * (batch["residue_type"] == i),
                        do_hist=False,
                        suffix=f"_restype_{i}",
                        stats_to_log=["mean"],
                    )

            self.log_train_loss_n_prog_bar(bs, train_loss)
            self.update_n_log_nsamples_processed(bs)
            self.log_nparams()

            should_step = (batch_idx + 1) % self.cfg_ae.opt.get("accumulate_grad_batches", 1) == 0

            if not should_step:
                with self.trainer.strategy.no_sync(self):
                    self.manual_backward(train_loss)
            else:
                self.manual_backward(train_loss)

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

    def compute_struct_rec_loss(
        self,
        output_dec: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        reduce_mode: Literal["sum", "mean"] = "sum",
        loss_ty: str = "l2",
        weight: float = 1.0,
    ) -> dict[str, Float[Tensor, "b"]]:
        """
        Computes structural loss.

        Args:
            output_dec:
                {
                  "coors_nm": all atom coordinates, shape [b, n, 37, 3], in nm
                  "seq_logits": logits for the residue types, shape [b, n, 20]
                  "residue_mask": boolean [b, n]
                  "aatype_max": residue type by taking the most likely logit, shape [b, n],
                      with integer values {0, ..., 19}
                  "atom_mask": boolean [b, n, 37, 3], atom37 mask corresponding to aatype_max
                }
            batch: data batch from dataloader
            reduce_mode: whether to apply mean over atoms or just sum, when reducing accross 37 atoms types
            loss_ty: whether to apply l1 loss, l2 loss, or both, l2cut (ignore atoms close to each other, to expand)

        Returns:
            Dictionary [str, tensor[b]] with multiple per batch element losses. If the key has "_justlog" then this loss
            will not be used to compute the total loss, but will just be logged.
        """

        def reduce_37(
            err: Float[torch.Tensor, "b n 37 3"],
            mask: Bool[torch.Tensor, "b n"],
            atom_mask: Bool[torch.Tensor, "b n 37"],
            mode: Literal["sum", "mean"] = "sum",
        ) -> Float[torch.Tensor, "b"]:
            nres = mask.sum(dim=-1)  # [b]
            nat = atom_mask.sum(dim=-1) * mask  # [b, n]
            err = torch.sum(err, dim=(-1, -2))  # [b, n]
            if mode == "mean":
                err = err / nat  # Take mean over existing atoms if mode == "mean"
            err = err.sum(dim=-1) / nres  # [b]
            return err

        coors_nm_pred = output_dec["coors_nm"]  # [b, n, 37, 3]
        coors_nm_true = batch["coords_nm"]  # [b, n, 37, 3] (already converted in training_step)
        mask = output_dec["residue_mask"]  # [b, n] boolean
        atom_mask_true = batch["atom37_mask"] * mask[..., None]  # [b, n, 37] boolean

        err = coors_nm_true - coors_nm_pred  # [b, n, 37, 3]
        err = err * mask[..., None, None] * atom_mask_true[..., None]  # [b, n, 37, 3]

        losses = {}

        # Compute RMSD in Å (without alignment)
        err_ang = nm_to_ang(err)  # [b, n, 37, 3]
        err_ang = torch.linalg.norm(err_ang, dim=-1) ** 2  # [b, n, 37]
        err_ang = err_ang * mask[..., None] * atom_mask_true  # [b, n, 37]
        nat = atom_mask_true.sum((-1, -2))  # [b]
        rmsd = torch.sqrt(torch.sum(err_ang, dim=(-1, -2)) / nat)  # [b]
        losses["rmsd_no_align_a37_ang_justlog"] = rmsd

        err_l1 = reduce_37(torch.abs(err), mask, atom_mask_true, mode=reduce_mode)  # L1 loss
        err_l2 = reduce_37(err**2, mask, atom_mask_true, mode=reduce_mode)  # L2 loss

        if loss_ty == "l1":
            losses["struct_l1"] = err_l1 * weight
            losses["struct_l2_justlog"] = err_l2 * weight
        elif loss_ty == "l2":
            losses["struct_l1_justlog"] = err_l1 * weight
            losses["struct_l2"] = err_l2 * weight
        elif loss_ty == "l12":
            losses["struct_l1"] = err_l1 * weight
            losses["struct_l2"] = err_l2 * weight
        else:
            raise OSError(f"Loss type {loss_ty} not recognized")
        return losses

    def compute_seq_rec_loss(
        self,
        output_dec: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        weight: float = 1.0,
    ) -> dict[str, Float[Tensor, "b"]]:
        """
        Computes cross entropy loss on predicted sequence.

        Args:
            output_dec:
                {
                  "coors_nm": all atom coordinates, shape [b, n, 37, 3], in nm
                  "seq_logits": logits for the residue types, shape [b, n, 20]
                  "residue_mask": boolean [b, n]
                  "aatype_max": residue type by taking the most likely logit, shape [b, n],
                      with integer values {0, ..., 19}
                  "atom_mask": boolean [b, n, 37, 3], atom37 mask corresponding to aatype_max
                }
            batch: data batch from dataloader

        Returns:
            Dictionary [str, tensor[b]] with multiple per batch element losses. If the key has "_justlog" then this loss
            will not be used to compute the total loss, but will just be logged.
        """
        assert "residue_type" in batch, "`residue_type` not in batch, failed in compute_seq_rec_loss"
        mask = output_dec["residue_mask"]  # [b, n]
        nres = mask.sum(dim=-1)  # [b]
        logits_pred = output_dec["seq_logits"]  # [b, n, n_residue_types]
        target_aa = batch["residue_type"]  # [b, n]
        target_aa = target_aa * mask  # [b, n] gets rid of -1 for padding (issue with cross entropy loss below)

        # Support both legacy (20 AAs) and unified (33 types) encodings
        n_residue_types = logits_pred.shape[-1]
        assert n_residue_types in [20, 33], f"Expected 20 (legacy) or 33 (unified) residue types, got {n_residue_types}"

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
        seq_loss = seq_loss * mask  # [b, n]
        seq_loss = torch.sum(seq_loss, dim=-1) / nres  # [b]

        # Compute seq recovery rate
        pred_aa = output_dec["aatype_max"]  # [b, n]
        seq_rec = pred_aa == target_aa  # [b, n]
        seq_rec = seq_rec * mask  # [b, n]
        seq_rec_rate = seq_rec.sum(dim=-1) / nres  # [b]
        return {
            "ce_seq": seq_loss * weight,
            "ce_seq_now": seq_loss,
            "seq_rec_rate_justlog": seq_rec_rate,
        }

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
    ) -> dict[str, Float[Tensor, "b"]]:
        """
        Computes KL penalty on the latent Gaussian distribution.
        """
        nres = torch.sum(mask, dim=-1)  # [b]
        kl_div = self._per_component_kl(
            mean=mean,
            log_scale=log_scale,
            mask=mask,
        )  # [b, n, d]
        kl_div = torch.sum(kl_div, dim=(-1, -2))  # [b]
        kl_div = kl_div / nres  # [b]
        loss = {
            "kl_w": kl_div * w,
            "kl_now_justlog": kl_div,
        }
        return loss

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
        """
        Logs tensor statistics: mean, median, max, min, std, ...

        If res_type is not None, then it also logs histograms per residue type
        """
        vals = v.clone()[mask]  # [num of unmasked residues, d]
        vals = torch.flatten(vals)  # 1D vector
        if do_stats:
            funs = {
                "max": torch.max,
                "min": torch.min,
                "mean": torch.mean,
                "median": torch.median,
                "std": torch.std,
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
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

        if self.global_step % histogram_every_n == 0 and do_hist:
            self.log_histogram(id_log=f"{log_prefix}/histogram{suffix}", v=vals)

    def log_pca(
        self,
        v: Float[torch.Tensor, "b n d"],
        log_prefix: str,
        mask: Bool[torch.Tensor, "b n"],
        every_n: int = 3000,
    ) -> None:
        """
        Logs PCA components plot of latent variable z.
        """

        def _log_scatter(
            vals_x: np.ndarray,
            vals_y: np.ndarray,
            xlabel: str,
            ylabel: str,
            log_id: str,
        ) -> None:
            fig, ax = plt.subplots()
            ax.scatter(vals_x, vals_y)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{xlabel} - {ylabel} - Step: {self.trainer.global_step}")
            with contextlib.suppress(Exception):
                self.logger.experiment.log({log_id: wandb.Image(fig)})
            plt.close("all")

        if self.global_step % every_n != 0:
            return

        n_components = 4
        vals = v.clone()[mask].cpu().detach().float().numpy()  # [num of unmasked residues, d]
        vals_pca = PCA(n_components=n_components).fit_transform(vals)  # [num of unmasked residues, n_components]

        for i in range(n_components):
            for j in range(n_components):
                if j <= i:
                    continue
                _log_scatter(
                    vals_x=vals_pca[:, i],
                    vals_y=vals_pca[:, j],
                    xlabel=f"PC {i}",
                    ylabel=f"PC {j}",
                    log_id=f"{log_prefix}/{i}_{j}",
                )

    def log_pca_per_residue_type(
        self,
        v: Float[torch.Tensor, "b n d"],
        res_ty: Int[torch.Tensor, "b n"],
        log_prefix: str,
        mask: Bool[torch.Tensor, "b n"],
        every_n: int = 3000,
    ) -> None:
        """Log PCA projection colored by residue type (wrapper for backward compatibility).

        This method now delegates to log_projection_per_residue_type with method='pca'.
        """
        self.log_projection_per_residue_type(
            v=v,
            res_ty=res_ty,
            log_prefix=log_prefix,
            mask=mask,
            every_n=every_n,
            method="pca",
        )

    def log_projection_per_residue_type(
        self,
        v: Float[torch.Tensor, "b n d"],
        res_ty: Int[torch.Tensor, "b n"],
        log_prefix: str,
        mask: Bool[torch.Tensor, "b n"],
        every_n: int = 3000,
        method: str = "tsne",
        n_components: int = 2,
        max_samples: int = 5000,
    ) -> None:
        """Log dimensionality reduction projection colored by residue type.

        Uses shared visualization utilities for consistency with inference.
        Supports both legacy (20 AA) and unified (33 type) encodings automatically.

        Args:
          v: Latent vectors [batch, n_tokens, latent_dim].
          res_ty: Residue type indices [batch, n_tokens].
          log_prefix: Prefix for wandb logging.
          mask: Valid token mask [batch, n_tokens].
          every_n: Log every N steps.
          method: Projection method ('tsne', 'pca', or 'lda').
          n_components: Number of components (typically 2).
          max_samples: Max tokens to project (subsample if needed).
        """
        if self.global_step % every_n != 0:
            return

        # Extract valid tokens
        vals = v.clone()[mask].cpu().detach().float().numpy()  # [n_valid_tokens, latent_dim]
        labels = res_ty.clone()[mask].cpu().detach().numpy()  # [n_valid_tokens]

        if len(vals) == 0:
            logger.warning(f"No valid tokens for {log_prefix}, skipping")
            return

        # Subsample if needed
        if len(vals) > max_samples:
            indices = np.random.choice(len(vals), size=max_samples, replace=False)
            vals = vals[indices]
            labels = labels[indices]

        # Detect n_residue_types from unique labels
        max_label = int(np.max(labels))
        n_residue_types = 33 if max_label >= 20 else 20

        # Project to 2D using shared utility
        projection = project_latents_2d(
            latents=vals,
            labels=labels,
            method=method,
            perplexity=30,
            max_iter=1000,
            random_state=42,
        )

        # Create plot using shared utility (with cluster labels!)
        fig = plot_latent_projection_with_labels(
            projection=projection,
            labels=labels,
            n_residue_types=n_residue_types,
            method=method.upper(),
            title=f"Latent Space ({method.upper()}) - Step {self.trainer.global_step}",
            figsize=(10, 8),
            add_cluster_labels=True,  # ✅ Now adds cluster labels like inference!
        )

        # Log to wandb
        with contextlib.suppress(Exception):
            self.logger.experiment.log({log_prefix: wandb.Image(fig)})

        # Optional: Save locally for debugging
        save_local = self.cfg_ae.get("log", {}).get("save_projections_locally", False)
        if save_local:
            from pathlib import Path

            save_dir = Path(self.cfg_ae.get("log", {}).get("local_projection_dir", "./latent_projections"))
            save_dir.mkdir(exist_ok=True, parents=True)
            save_path = save_dir / f"{log_prefix.replace('/', '_')}_step{self.trainer.global_step}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved projection plot to {save_path}")

        plt.close("all")

    def log_losses(self, bs: int, losses: dict[str, Float[torch.Tensor, "b"]], log_prefix: str) -> None:
        for k in losses:
            log_name = k[: -len("_justlog")] if k.endswith("_justlog") else k
            self.log(
                f"{log_prefix}/loss_{log_name}",
                torch.mean(losses[k]),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
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
            "scaling/nparams_enc",
            self.nparams_enc * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )  # constant line but ok, easy to compare # params

        self.log(
            "scaling/nparams_dec",
            self.nparams_dec * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
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
            sync_dist=True,
        )

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """
        Evalaute validation loss.

        Args:
            batch: batch from dataset (see last argument)
            batch_idx: batch index (unused)
        """
        with torch.no_grad():
            # Convert to expected format (coords_nm will be set in training_step)
            bs = batch["coords"].shape[0]
            loss, rec_sample = self.training_step(batch, batch_idx=-1)
            self.validation_output.append(loss.item())
            coords_nm = batch["coords"] / 10.0  # Convert Angstroms to nanometers
            atom_array_batch = batch.get("atom_array_batch")
            for i in range(bs):
                self.validation_data_samples.append(
                    {
                        "coors_nm": coords_nm[i, ...],  # [n, 37, 3]
                        "aatype": batch["residue_type"][i, ...],  # [n]
                        "atom_mask": batch["atom37_mask"][i, ...],  # [n, 37]
                        "padding_mask": batch["padding_mask"][i, :],  # [n]
                        "atom_array": atom_array_batch[i] if atom_array_batch is not None else None,
                    }
                )
                self.validation_rec_samples.append(
                    {
                        "coors_nm": rec_sample["coors_nm"][i, ...],  # [n, 37, 3]
                        "aatype": rec_sample["aatype_max"][i, ...],  # [n]
                        "atom_mask": rec_sample["atom_mask"][i, ...],  # [n, 37]
                        # [n], should be the same as mask from orig sample
                        "padding_mask": rec_sample["residue_mask"][i, ...],
                    }
                )

    def on_validation_epoch_end(self) -> None:
        """Write validation structures as CIFs using the flow-model export path."""
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
            coords_rec_ang = nm_to_ang(sample_rec["coors_nm"]).float().detach().cpu()
            aatype_data = sample_data["aatype"].detach().cpu()
            aatype_rec = sample_rec["aatype"].detach().cpu()
            atom_mask_data = sample_data["atom_mask"].detach().cpu()
            atom_mask_rec = sample_rec["atom_mask"].detach().cpu()

            f_base = f"epoch_{self.current_epoch}_id_{count}_rank_{self.global_rank}"

            try:
                write_input_structure_cif(
                    input_atom_array,
                    os.path.join(self.val_path_tmp, f_base + "_data_sample.cif"),
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
                    os.path.join(self.val_path_tmp, f_base + "_rec_sample.cif"),
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
                    os.path.join(self.val_path_tmp, f_base + "_rec_sample_w_true_seq.cif"),
                )
            except Exception as e:
                logger.warning(f"Failed to write validation CIFs for sample {count}: {e}")

        self.validation_data_samples = []
        self.validation_rec_samples = []
        self.validation_output_data = []

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Makes predictions. Given a data batch, encodes, and returns decoded batch.

        Args:
            batch: data batch.

        Returns:
            Dict representing the decoded batch, with all info from the encoder output.
        """
        # Extract from new pipeline batch format
        mask = batch["padding_mask"]  # [b, n] boolean, True = valid token
        coords_nm = batch["coords"] / 10.0  # Convert Angstroms to nanometers
        batch["coords_nm"] = coords_nm  # For backward compatibility
        ca_coors_nm = coords_nm[..., 1, :]  # [b, n, 3] - CA is at slot 1
        ca_coors_nm = ca_coors_nm * mask[..., None]  # [b, n, 3]

        output_enc = self.encoder(batch)
        # {
        #   "z_latent": latent_sample, shape [b, n, d]
        #   "mean": mean of latent (diag) Gaussian dist, shape [b, n, d]
        #   "log_scale": log standard deviation of latent (diag) Gaussian dist, shape [b, n, d]
        # }

        input_decoder = {
            "z_latent": output_enc["z_latent"],
            "ca_coors_nm": ca_coors_nm,
            "residue_mask": mask,
            "padding_mask": mask,
            "token_type": batch["token_type"],
        }
        output = self.decoder(input_decoder)
        # {
        #   "coors_nm": all atom coordinates, shape [b, n, 37, 3], in nm
        #   "seq_logits": logits for the residue types, shape [b, n, 20]
        #   "residue_mask": boolean [b, n]
        #   "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
        #   "atom_mask": boolean [b, n, 37, 3], atom37 mask corresponding to aatype_max
        # }

        output.update(output_enc)
        return (batch, output)
