from collections.abc import Callable

import lightning as L
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from proteinfoundation.conditions import C_CRDGRP
from proteinfoundation.flow_matching.base_flow_matcher import BaseFlowMatcher
from proteinfoundation.flow_matching.rdn_flow_matcher import RDNFlowMatcher

FLOW_MATCHER_FACTORY = {
    "bb_ca": RDNFlowMatcher,
    "local_latents": RDNFlowMatcher,
}  # maps data modality to correspodning class


def extract_groups(batch: dict) -> dict[str, torch.Tensor | None]:
    """Extract token-level group tensors from a collated batch.

    Pulls all group keys that can be used by the ``"per_group"`` stochastic
    centering scope.

    Args:
        batch: Collated batch dictionary.

    Returns:
        Dict mapping each group name to its ``[b, n_tokens]`` tensor, or
        ``None`` if the key is not present in the batch.
    """
    crdgrp_atom37 = batch.get(C_CRDGRP.full_name)  # [b, n_tokens, 37] or None
    coord_group = crdgrp_atom37.long().max(dim=-1).values if crdgrp_atom37 is not None else None

    def _to_long(key: str) -> torch.Tensor | None:
        v = batch.get(key)
        return v.long() if v is not None else None

    return {
        "chain_iid": _to_long("chain_iid"),
        "pn_unit_iid": _to_long("pn_unit_iid"),
        "molecule_iid": _to_long("molecule_iid"),
        "coord_group": coord_group,
    }


def sample_noised_target_backbone(
    fm: "ProductSpaceFlowMatcher",
    batch: dict,
    target_backbone: Float[Tensor, "n_samples n 3"],
    t_bb: Float[Tensor, "n_samples"],
) -> Float[Tensor, "n_samples n 3"]:
    """Draw the target backbone from its forward marginal at ``t_bb``.

    Only the backbone matcher is evaluated. Sampling uses the same centering
    inputs and one-time centering behavior as training-time corruption.
    """
    padding_mask = batch["padding_mask"]
    coord_cond_mask = batch.get("coord_cond_mask")
    groups = extract_groups(batch)
    batch_shape = target_backbone.shape[:-2]
    n = target_backbone.shape[-2]
    backbone_matcher = fm.base_flow_matchers["bb_ca"]

    # ``interpolate`` applies x_0 stochastic centering. Using training=True here
    # prevents ``sample_noise`` from applying the same offset a second time.
    x_0 = backbone_matcher.sample_noise(
        n=n,
        shape=batch_shape,
        mask=padding_mask,
        device=target_backbone.device,
        training=True,
        coord_cond_mask=coord_cond_mask,
        groups=groups,
    )
    return backbone_matcher.interpolate(
        x_0=x_0,
        x_1=target_backbone,
        t=t_bb,
        mask=padding_mask,
        coord_cond_mask=coord_cond_mask,
        groups=groups,
    )


class ProductSpaceFlowMatcher(L.LightningModule):
    """
    Base class for flow matcher. Most of the methods in this class are abstract methods
    that should be implemented by classes that inheret from this one.
    """

    def __init__(self, cfg_exp: dict):
        super().__init__()
        self.cfg_exp = cfg_exp
        self.data_modes = list(self.cfg_exp.product_flowmatcher)
        self.base_flow_matchers = self.get_base_flow_matchers()

    def get_base_flow_matchers(self) -> dict[str, BaseFlowMatcher]:
        """Constructs all necessary flow matchers."""
        return {m: FLOW_MATCHER_FACTORY[m](**self.cfg_exp.product_flowmatcher[m]) for m in self.data_modes}

    def _apply_mask(self, x: dict[str, Tensor], mask: Bool[Tensor, "* n"] | None = None) -> dict[str, Tensor]:
        """
        Masks x.

        Args:
            x: sample to mask.
            mask (optional): binary mask, shape [*, n].

        Returns:
            sample x masked.
        """
        x = {
            data_mode: self.base_flow_matchers[data_mode]._apply_mask(
                x=x[data_mode],
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x

    def sample_noise(
        self,
        n: int,
        shape: tuple = (),
        device: torch.device | None = None,
        mask: Bool[Tensor, "* n"] | None = None,
        training: bool = True,
        coord_cond_mask: Bool[Tensor, "* n"] | None = None,
        groups: dict[str, Int[Tensor, "* n"] | None] | None = None,
    ) -> dict[str, Tensor]:
        """
        Samples reference distribution (possibly centered).

        Args:
            n: number of residues in a single sample (i.e. protein length), int
            shape: tuple (if empty then single sample)
            device (optional): torch device used
            mask (optional): Binary mask of shape [*, n]
            training (optional): whether in training mode
            coord_cond_mask (optional): Boolean mask of shape [*, n]. True for
                coordinate-conditioned tokens whose noise should not be offset.
            groups (optional): Dict of token-level group tensors keyed by name
                (``chain_iid``, ``pn_unit_iid``, ``molecule_iid``, ``coord_group``).

        Returns:
            Samples from refenrece with shape [*shape, n, ...]
        """
        x = {
            data_mode: self.base_flow_matchers[data_mode].sample_noise(
                n=n,
                shape=shape,
                device=device,
                mask=mask,
                training=training,
                coord_cond_mask=coord_cond_mask,
                groups=groups,
            )
            for data_mode in self.data_modes
        }
        return x

    def interpolate(
        self,
        x_0: dict[str, torch.Tensor],
        x_1: dict[str, torch.Tensor],
        t: dict[str, Float[Tensor, "*"]],
        mask: Bool[Tensor, "* n"] | None = None,
        coord_cond_mask: Bool[Tensor, "* n"] | None = None,
        groups: dict[str, Int[Tensor, "* n"] | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Interpolates between x_0 (base) and x_1 (data) using t.

        Args:
            x_0: Samples from reference with batch shape * (each value in the dict)
            x_1: Samples from target with batch shape *
            t: Interpolation times
            mask (optional): Binary mask, shape [*, n]
            coord_cond_mask (optional): Boolean mask of shape [*, n]. True for
                coordinate-conditioned tokens whose noise should not be offset.
            groups (optional): Dict of token-level group tensors keyed by name
                (``chain_iid``, ``pn_unit_iid``, ``molecule_iid``, ``coord_group``).

        Returns:
            x_t: Interpolated samples, same shape as x_0 and x_1
        """
        x_t = {
            data_mode: self.base_flow_matchers[data_mode].interpolate(
                x_0=x_0[data_mode],
                x_1=x_1[data_mode],
                t=t[data_mode],
                mask=mask,
                coord_cond_mask=coord_cond_mask,
                groups=groups,
            )
            for data_mode in self.data_modes
        }
        return x_t

    def process_batch(self, batch: dict) -> tuple[Tensor, Tensor, tuple, int, torch.dtype]:
        """
        Extracts clean sample, mask, batch size, protein length n, dtype and device
        from the batch coming from the dataloader. (Do we want dtype? Should default
        work?). Uses padding_mask to mask the x_1

        Args:
            batch: batch from dataloader.

        Returns:
            Tuple (x_1, padding_mask, batch_shape, n, dtype, device)

        WARNING: For dtype, it might become a dictionary as well, depending on
        type for sequence (discrete)
        """
        # For atomized batches, x_1 is at token level
        if "x_1" in batch and "bb_ca" in batch["x_1"]:
            coors_tensor = batch["x_1"]["bb_ca"]  # [b, n_tokens, 3]
        else:
            coors_tensor = batch["coords"]  # Fallback
        device = coors_tensor.device
        dtype = coors_tensor.dtype
        batch_shape = coors_tensor.shape[:-2]
        n = coors_tensor.shape[-2]
        padding_mask = batch["padding_mask"]  # [b, n_token] boolean
        x_1 = self._apply_mask(x=batch["x_1"], mask=padding_mask)
        return (x_1, padding_mask, batch_shape, n, dtype, device)

    def corrupt_batch(
        self,
        batch: dict,
        t: dict[torch.Tensor] | None = None,
    ) -> dict:
        """
        Runs forward process on x_1. Essentially samples reference and
        interpolates. If there's any coupling it should be included here.

        Augmentations in principle happen in the dataloader.

        All positions are corrupted.

        Args:
            batch: data batch
            t: t value to use for corruption. If None, will sample using `self.sample_t`.

        Returns:
            The same data batch with additional entries t, x_t, x_1, x_0, all of which are
            Dict[str, Tensor].
        """

        x_1, padding_mask, batch_shape, n, dtype, device = self.process_batch(batch)
        coord_cond_mask = batch.get("coord_cond_mask")  # [b, n_tokens] or None
        groups = extract_groups(batch)
        if t is None:
            t = self.sample_t(shape=batch_shape, device=device)
        x_0 = self.sample_noise(
            n=n,
            shape=batch_shape,
            mask=padding_mask,
            device=device,
            training=True,
            coord_cond_mask=coord_cond_mask,
            groups=groups,
        )
        x_t = self.interpolate(
            x_0=x_0,
            x_1=x_1,
            t=t,
            mask=padding_mask,
            coord_cond_mask=coord_cond_mask,
            groups=groups,
        )
        batch["x_0"] = x_0
        batch["x_1"] = x_1
        batch["x_t"] = x_t
        batch["t"] = t
        return batch

    def sample_t(self, shape: tuple, device: torch.device) -> dict[str, Float[Tensor, "*shape"]]:
        """
        Samples t for each data mode. Can use different distributions for each mode,
        and different modes can share the same t. This is all controlled in the config.

        Args:
            shape: shape of the sample
            device: device of the sample

        Returns:
            Sampled t for each modality (as a dictionary), each tensor has shape [*shape].
        """
        t = {
            data_mode: _sample_t(
                cfg_t_dist=self.cfg_exp.loss.t_distribution[data_mode],
                shape=shape,
                device=device,
            )
            for data_mode in self.data_modes
        }
        if self.cfg_exp.loss.t_distribution.shared_groups is None:
            return t

        # Apply grouping for data modalities we want the same t
        for t_share_modes in self.cfg_exp.loss.t_distribution.shared_groups:
            # each t_shared_modes is a list
            base = t_share_modes[0]  # some modality, the first one in the list
            shared_t = t[base]  # [*]
            for data_mode in t_share_modes[1:]:
                t[data_mode] = shared_t
        return t

    def compute_loss(self, batch: dict, nn_out: dict[str, dict[str, Tensor]]) -> dict[str, Float[Tensor, "*"]]:
        """
        Computes training loss, flow matching and some auxiliary loss.

        Args:
            batch: training batch with clean x_1, x_0, x_t, t
            nn_out: output of nn

        Returns:
            Flow matching loss and auxiliary loss, all in one Dict[str, Tensor w shape [*]]. Each loss
            is identified by its name (str) and is the loss per element in the batch (i.e. before mean
            reduction).
        """
        fm_loss = self.compute_fm_loss(batch, nn_out)
        aux_loss = self.compute_aux_loss(batch, nn_out)
        loss = {**fm_loss, **aux_loss}
        return loss

    def compute_fm_loss(
        self,
        batch: dict,
        nn_out: dict[str, dict[str, Tensor]],
    ) -> dict[str, Float[Tensor, "*"]]:
        """
        Computes flow matching loss for all modalities.
        We only compute loss for tokens that should be lossed, and we don't compute loss on latents
        where we couldn't encode them (would have started as zero)

        Args:
            batch: Input batch containing x_1, x_0, t, mask, etc
            nn_out: Output of nn

        Returns:
            Losses for each data modality, per element in the batch.
        """
        should_be_lossed_masks = {
            "bb_ca": batch["should_be_lossed"],
            "local_latents": batch["should_be_lossed"] & batch["can_be_encoded"],
        }
        # Optional per-token loss weight. None / absent → default-safe (no-op) inside compute_fm_loss.
        loss_weight = batch.get("loss_weight")
        loss = {
            data_mode: self.base_flow_matchers[data_mode].compute_fm_loss(
                x_0=batch["x_0"][data_mode],
                x_1=batch["x_1"][data_mode],
                x_t=batch["x_t"][data_mode],
                mask=should_be_lossed_masks[data_mode],
                t=batch["t"][data_mode],
                nn_out=nn_out[data_mode],
                weight=loss_weight,
            )
            for data_mode in self.data_modes
        }
        return loss

    def compute_aux_loss(
        self,
        batch: dict,
        nn_out: dict[str, dict[str, Tensor]],
    ) -> dict[str, Float[Tensor, "*"]]:
        """
        Computes auxiliary loss (if any). This is done here, and not in each specific flow matcher,
        since it may be a full atom loss, etc, meaning it may interact between modalities.

        Args:
            batch: Input batch containing x_1, x_0, t, mask, etc
            nn_out: Output of nn

        Returns:
            Losses for each data modality, per element in the batch.
        """
        losses = {}
        losses.update(self._compute_insertion_prediction_loss(batch, nn_out))
        return losses

    def _compute_insertion_prediction_loss(
        self,
        batch: dict,
        nn_out: dict[str, dict[str, Tensor]],
    ) -> dict[str, Float[Tensor, "b"]]:
        """Per-unindexed-token CE loss for the insertion-prediction head.

        Returns ``'insertion_prediction'`` (optimised) and two ``_justlog`` entries
        ``'insertion_ce_justlog'`` / ``'insertion_accuracy_justlog'`` (logged only).
        """
        logits: Float[Tensor, "b n n"] | None = nn_out.get("insertion", {}).get("logits")
        target: Int[Tensor, "b n"] | None = batch.get("insertion_source_idx")
        if logits is None or target is None:
            return {}

        b, n, _ = logits.shape

        # Tensor multiply (not Python branch) so the graph shape stays fixed under DDP + compile
        sample_has_unindexed_tokens = (target >= 0).any(dim=-1).float()  # [b]

        # True for tokens that are valid candidates to MAP TO (i.e. real, indexed)
        indexed_key_mask: Bool[Tensor, "b n"] = batch["padding_mask"] & (target < 0)
        logits_masked: Float[Tensor, "b n n"] = logits.masked_fill(~indexed_key_mask.unsqueeze(1), -1e9)

        # Flatten the (b, n) query axes into a single N=b*n, then unflatten the per-token CE back to (b, n)
        ce_per_token: Float[Tensor, "b n"] = F.cross_entropy(
            logits_masked.reshape(b * n, n),  # [b*n, n]
            target.long().reshape(b * n),  # [b*n]
            reduction="none",
            ignore_index=-1,
        ).reshape(b, n)

        is_unindexed: Bool[Tensor, "b n"] = target >= 0
        n_valid = is_unindexed.float().sum(dim=-1)  # [b]
        safe_n_valid = torch.where(n_valid > 0, n_valid, torch.ones_like(n_valid))
        loss = (ce_per_token.sum(dim=-1) / safe_n_valid) * sample_has_unindexed_tokens  # [b]

        # _justlog scalars broadcast to [b] so log_losses' mean reduction recovers them.
        n_samples = sample_has_unindexed_tokens.sum()
        ce_justlog = (loss.sum() / torch.where(n_samples > 0, n_samples, torch.ones_like(n_samples))) * torch.ones_like(
            loss
        )
        pred = logits_masked.argmax(dim=-1)  # [b, n]
        n_unindexed_total = is_unindexed.float().sum()
        accuracy = ((pred == target.long()) & is_unindexed).float().sum() / torch.where(
            n_unindexed_total > 0, n_unindexed_total, torch.ones_like(n_unindexed_total)
        )
        accuracy_justlog = accuracy * torch.ones_like(loss)

        return {
            "insertion_prediction": loss,
            "insertion_ce_justlog": ce_justlog,
            "insertion_accuracy_justlog": accuracy_justlog,
        }

    def simulation_step(
        self,
        x_t: dict[str, torch.Tensor],
        nn_out: dict[str, dict[str, torch.Tensor]],
        t: dict[str, Float[Tensor, "*"]],
        dt: dict[str, float],
        gt: dict[str, float],
        simulation_step_params: dict[str, dict],
        mask: Bool[Tensor, "* n"] | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""
        Single integration step of ODE \dot{x_t} = v(x_t, t) using Euler integration scheme.

        Args:
            x_t: Current values with batch shape *
            nn_out: Output of nn
            t: Current time, shape * for each value in the dict
            dt: Step-size, float
            gt: Noise injection schedule, float
            simulation_step_params contain extra simulation steps parameters
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Updated x_t after integration step, same shape as input
        """
        x_updated = {
            data_mode: self.base_flow_matchers[data_mode].simulation_step(
                x_t=x_t[data_mode],
                nn_out=nn_out[data_mode],
                t=t[data_mode],
                dt=dt[data_mode],
                gt=gt[data_mode],
                simulation_step_params=simulation_step_params[data_mode],
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x_updated

    def nn_out_to_clean_sample_prediction(
        self,
        batch: dict,
        nn_out: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Generates clean sample prediction from nn output.

        Args:
            batch: input to nn (needed because it contains t, x_t, and mask)
            nn_out: output of nn

        Returns:
            Clean sample prediction {data_mode: torch.Tensor} from nn output
        """
        nn_out = self.nn_out_add_clean_sample_prediction(batch=batch, nn_out=nn_out)
        return {data_mode: nn_out[data_mode]["x_1"] for data_mode in self.data_modes}

    def nn_out_add_clean_sample_prediction(
        self,
        batch: dict,
        nn_out: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Adds clean sample prediction to nn output.

        Args:
            batch: input to nn (needed because it contains t, x_t, and mask)
            nn_out: output of nn

        Returns:
            nn_out with clean sample prediction ("x_1" key) added to each data modality.
        """
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[data_mode].nn_out_add_clean_sample_prediction(
                x_t=batch["x_t"][data_mode],
                t=batch["t"][data_mode],
                mask=batch["padding_mask"],
                nn_out=nn_out[data_mode],
            )
        return nn_out

    def nn_out_add_simulation_tensor(
        self,
        batch: dict,
        nn_out: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Adds tensor used for simulation to nn output. This is the velocity "v" for flow matching,
        or "score" for diffusion models. One could work out their equivalence and leverage that,
        but keeping each one seems like an easier implementation.

        Args:
            batch: input to nn (needed because it contains t, x_t, and mask)
            nn_out: output of nn

        Returns:
            nn_out with simulation tensor prediction ("v", "score", ... keys) added to data modalities.
        """
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[data_mode].nn_out_add_simulation_tensor(
                x_t=batch["x_t"][data_mode],
                t=batch["t"][data_mode],
                mask=batch["padding_mask"],
                nn_out=nn_out[data_mode],
            )
        return nn_out

    def nn_out_add_guided_simulation_tensor(
        self,
        nn_out: dict[str, dict[str, torch.Tensor]],
        nn_out_ag: dict[str, dict[str, torch.Tensor]] | None,
        nn_out_ucond: dict[str, dict[str, torch.Tensor]] | None,
        guidance_w: float,
        ag_ratio: float,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Adds guided version of tensor used for simulation to nn output 'nn_out'. Supports classifier free
        guidance and autoguidance.

        Args:
            nn_out: nn output of full model
            nn_out_ag: nn output of model used for autoguidance
            nn_out_ucond: nn output of full but unconditional model
            guidance_w: Guidance weights, float
            ag_ratio: Autoguidance ratio, float

        Returns:
            nn_out with guided simulation tensor prediction ("v_guided", "score_guided", ... keys) added to
                data modalities with guidance enabled.
        """
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[data_mode].nn_out_add_guided_simulation_tensor(
                nn_out=nn_out[data_mode],
                nn_out_ag=nn_out_ag[data_mode] if nn_out_ag else None,
                nn_out_ucond=nn_out_ucond[data_mode] if nn_out_ucond else None,
                guidance_w=guidance_w,
                ag_ratio=ag_ratio,
            )
        return nn_out

    def get_clean_pred_n_guided_vector(
        self,
        batch: dict,
        predict_for_sampling: Callable,
        guidance_w: float,
        ag_ratio: float,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        This makes a prediction with the corresponding nn and conditioning, computes the
        corresponding clean sample prediction, and v/score, and also the guided v/score.

        When calling `predict_for_sampling` (which internally calls some nn), the output
        nn_out is like
        {
            "bb_ca": {"v": tensor}
            "frames": {"x1": tensor}
        }
        depending on the parameterization, which is indicated in the nn config file. Then,
        we call `nn_out_add_clean_sample_prediction` and `nn_out_add_simulation_tensor` to
        extend each data mode in nn_out with "v"/"score" (the simulation tensor) and "x_1".

        This function does this for the original nn, the unconditional input, and the
        autoguidance nn, and combines predictions accordingly, adding to nn_out an entry
        `v_guided`. Overall, it returns nn_out like
        {
            "bb_ca": {"v": tensor, "x_1": tensor, "v_guided": tensor}
            "frames": {"x1": tensor, "v": tensor, "v_guided": tensor}
        }
        where v could be replaced by the score in case of a diffusion.

        Args:
            batch: input to nn
            predict_for_sampling: function that takes as input batch and mode, where the latter indicated
            what model to evaluate (supports "full", "ag", "ucond").
            guidance_w: guidance weights
            ag_ratio: autoguidance ratio

        Returns:
            an nn_out dictionary with keys "x_1", "v" (or "score") and "v_guided" (or "score_guided").
        """

        def _add_clean_n_sim_tensor(batch: dict, mode: str) -> dict[str, dict[str, torch.Tensor]]:
            """
            We will use this for each model (full, ag, ucond).
            """
            nn_out_dict = predict_for_sampling(batch, mode=mode)
            nn_out_dict = self.nn_out_add_clean_sample_prediction(batch, nn_out_dict)
            nn_out_dict = self.nn_out_add_simulation_tensor(batch, nn_out_dict)
            return nn_out_dict

        nn_out = _add_clean_n_sim_tensor(batch, mode="full")
        nn_out_ag = None
        nn_out_ucond = None
        if guidance_w != 1.0:
            if ag_ratio > 0.0:  # Use auto-guidance
                nn_out_ag = _add_clean_n_sim_tensor(batch, mode="ag")
            if ag_ratio < 1.0:  # Use CFG
                nn_out_ucond = _add_clean_n_sim_tensor(batch, mode="ucond")

        nn_out = self.nn_out_add_guided_simulation_tensor(
            nn_out=nn_out,
            nn_out_ag=nn_out_ag,
            nn_out_ucond=nn_out_ucond,
            guidance_w=guidance_w,
            ag_ratio=ag_ratio,
        )  # Adds guided tensor used for simulation ("guided_v", "guided_score", or whatever the base flow matcher uses)
        return nn_out

    def full_simulation(
        self,
        batch: dict,
        predict_for_sampling: Callable,
        nsteps: int,
        nsamples: int,
        n: int,
        self_cond: bool,
        sampling_model_args: dict[str, dict],
        device: torch.device,
        save_trajectory_every: int = 0,
        dump_noisy_trajectories: bool = False,
        guidance_w: float = 1.0,
        ag_ratio: float = 0.0,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor | Bool[Tensor, "nsamples n"]]]:
        r"""
        Generates samples by simulating the

        dx_t = v(x_t, t) dt

        or

        dx_t = [v(x_t, t) + g(t) s(x_t, t)] dt + \sqrt{2g(t)} dw_t

        from t=0 up to t=1.

        Which equation is simulated depends on data modality and parameters chosen.

        Args:
            batch: Input batch. May contain:
                - 'nsamples': int, number of samples to generate (optional, will use nsamples param if not present)
                - 'nres' or 'n': int, protein length (optional, will use n param if not present)
                - 'mask': torch.BoolTensor, shape [nsamples, n], mask for valid residues
                    (optional, will be generated if not present)
                - Any conditioning fields required by the prediction function (e.g., cath_codes, motif info, etc.)

                The following fields will be added/updated during simulation:
                - 'x_t': torch.Tensor, current state (updated each step)
                - 't': dict of time values for each data mode (updated each step)
                - 'mask': torch.BoolTensor, shape [nsamples, n], mask for valid residues (added if not present)
                - 'x_sc': torch.Tensor, self-conditioning input (added if self_cond=True)
            predict_for_sampling: Function that calls the nn, receives the full batch dict
            nsteps: Number of simulation steps
            nsamples: Number of samples to generate
            n: Protein length
            self_cond: Whether to use self-conditioning
            sampling_model_args: Model-specific sampling arguments
            device: Device to run simulation on
            save_trajectory_every: How often to save full generation trajectory, default 0 (no saving)
            dump_noisy_trajectories: If True, also collect the noisy x_t at each trajectory step in
                addition to the clean x_1 prediction. Defaults to False.
            guidance_w: Guidance weight
            ag_ratio: Autoguidance ratio

        Returns:
            Tuple of (x, additional_info) where:
                - x: Generated samples, dict with keys for each data mode
                - additional_info: Dict with 'padding_mask', 'can_be_encoded', and optionally
                  'trajectory_steps' (list of dicts, each with 'step', 'x_1', and optionally 'x_t')
        """

        # Use mask from batch if present, otherwise generate it
        if "padding_mask" in batch and batch["padding_mask"] is not None:
            mask = batch["padding_mask"]
        else:
            mask = torch.ones(nsamples, n).long().bool().to(device)
        assert mask.shape == (nsamples, n)

        trajectory_steps: list[dict] = []

        # Get diffusion schedule using extracted method
        ts, gt = self.sample_schedule(nsteps=nsteps, sampling_model_args=sampling_model_args)

        with torch.no_grad():
            coord_cond_mask = batch.get("coord_cond_mask")  # [nsamples, n_tokens] or None
            groups = extract_groups(batch)
            x = self.sample_noise(
                n,
                shape=(nsamples,),
                device=device,
                mask=mask,
                training=False,
                coord_cond_mask=coord_cond_mask,
                groups=groups,
            )

            # After sampling noise, mask to generated tokens only
            is_encoded_mask = batch.get("can_be_encoded")  # [nsamples, n_tokens]
            if is_encoded_mask is not None:
                x["local_latents"] = x["local_latents"] * is_encoded_mask[..., None]  # zero out non-encoded latents
            x["local_latents"] = x["local_latents"] * batch["padding_mask"][..., None]
            x["bb_ca"] = x["bb_ca"] * batch["padding_mask"][..., None]  # zero out padded bb_ca tokens

            # Always-populated x_sc + is_active scalar; feature factory reads both unconditionally.
            batch["x_sc"] = {dm: torch.zeros_like(x[dm]) for dm in self.data_modes}
            batch["x_sc_is_active"] = torch.zeros((), device=device)

            x_1_pred = None
            nn_out = None
            for step in range(nsteps):
                t = {
                    data_mode: ts[data_mode][step] * torch.ones(nsamples, device=device)
                    for data_mode in self.data_modes
                }
                dt = {data_mode: ts[data_mode][step + 1] - ts[data_mode][step] for data_mode in self.data_modes}
                gt_step = {data_mode: gt[data_mode][step] for data_mode in self.data_modes}

                # Update the batch with current x, t, mask, etc.
                batch["x_t"] = x
                batch["t"] = t
                batch["padding_mask"] = mask

                # self conditioning
                if step > 0 and self_cond:
                    batch["x_sc"] = x_1_pred
                    batch["x_sc_is_active"] = torch.ones((), device=device)

                # get clean prediction and guided vector
                nn_out = self.get_clean_pred_n_guided_vector(
                    batch=batch,
                    predict_for_sampling=predict_for_sampling,
                    guidance_w=guidance_w,
                    ag_ratio=ag_ratio,
                )
                # Dict[data_mode,
                #   Dict[str, torch.Tensor]
                # ] where str is some prediction ("x_1", "v", "score", "guided_v", "guided_score", ...)
                # We just track all predictions this way

                x_1_pred = self.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
                # Dict[data_mode, torch.Tensor]

                simulation_step_params = {
                    data_mode: sampling_model_args[data_mode]["simulation_step_params"] for data_mode in self.data_modes
                }
                x = self.simulation_step(
                    x_t=x,
                    nn_out=nn_out,
                    t=t,
                    dt=dt,
                    gt=gt_step,
                    mask=mask,
                    simulation_step_params=simulation_step_params,
                )

                # Mask updates to encoded tokens and non-padded tokens only
                if is_encoded_mask is not None:
                    x["local_latents"] = x["local_latents"] * is_encoded_mask[..., None]
                x["local_latents"] = x["local_latents"] * batch["padding_mask"][..., None]
                x["bb_ca"] = x["bb_ca"] * batch["padding_mask"][..., None]

                # Collect trajectory snapshot (x_1 = clean prediction, x_t = noisy state)
                if save_trajectory_every > 0 and (step % save_trajectory_every == 0 or step == nsteps - 1):
                    step_snapshot: dict = {
                        "step": step,
                        "x_1": {dm: x_1_pred[dm].detach().cpu() for dm in x_1_pred},
                    }
                    if dump_noisy_trajectories:
                        step_snapshot["x_t"] = {dm: x[dm].detach().cpu() for dm in x}
                    trajectory_steps.append(step_snapshot)

            # Capture insertion logits from the final step for inference-time mapping
            final_insertion_logits = None
            if nn_out is not None and "insertion" in nn_out and "logits" in nn_out["insertion"]:
                final_insertion_logits = nn_out["insertion"]["logits"].detach()

            additional_info = {
                "padding_mask": mask,
                "can_be_encoded": is_encoded_mask,
                "trajectory_steps": trajectory_steps,
                "insertion_logits": final_insertion_logits,
            }
            return x, additional_info

    def sample_schedule(
        self,
        nsteps: int,
        sampling_model_args: dict[str, dict],
    ) -> tuple[dict[str, Float[Tensor, "nsteps_plus_1"]], dict[str, Float[Tensor, "nsteps"]]]:
        """Create diffusion schedule for sampling.

        Generates time points and noise injection schedules for each modality.
        Time points go from 0 to 1, noise injection controls stochasticity.

        Args:
            nsteps: Number of diffusion steps
            sampling_model_args: Per-modality sampling configuration with keys:
                - 'schedule': Dict with 'mode' and 'p' for time schedule
                - 'gt': Dict with 'mode', 'p', 'clamp_val' for noise schedule

        Returns:
            Tuple of (ts, gt) where:
                - ts: Time schedule per modality, dict of tensors [nsteps+1]
                      First element is 0.0, last is 1.0
                - gt: Noise injection schedule per modality, dict of tensors [nsteps]
                      Determines amount of stochasticity at each step
        """
        ts = {}
        for data_mode in self.data_modes:
            if sampling_model_args[data_mode]["simulation_step_params"]["sampling_mode"] == "vf_tsr":
                schedule_func = get_schedule_tsr_safe
            else:
                schedule_func = get_schedule
            ts[data_mode] = schedule_func(
                mode=sampling_model_args[data_mode]["schedule"]["mode"],
                nsteps=int(nsteps),
                p1=sampling_model_args[data_mode]["schedule"]["p"],
            )

        gt = {
            data_mode: get_gt(
                t=ts[data_mode][:-1],  # [nsteps], ts[-1]=1 not used to eval
                mode=sampling_model_args[data_mode]["gt"]["mode"],
                param=sampling_model_args[data_mode]["gt"]["p"],
                clamp_val=sampling_model_args[data_mode]["gt"]["clamp_val"],
            )
            for data_mode in self.data_modes
        }

        return ts, gt

    def partial_simulation(
        self,
        batch: dict,
        x: dict[str, Tensor],
        x_1_pred: dict[str, Tensor] | None,
        mask: Bool[Tensor, "nsamples n"],
        predict_for_sampling: Callable,
        nsteps: int,
        ts: dict[str, Tensor],
        gt: dict[str, Tensor],
        self_cond: bool,
        simulation_step_params: dict[str, dict],
        device: torch.device,
        save_trajectory_every: int = 0,
        dump_noisy_trajectories: bool = False,
        guidance_w: float = 1.0,
        ag_ratio: float = 0.0,
        resample_target_backbone: bool = False,
        target_backbone: Float[Tensor, "nsamples n 3"] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor | Bool[Tensor, "nsamples n"]]]:
        """Run partial diffusion simulation for nsteps.

        Used by beam search to evaluate partial trajectories at intermediate checkpoints.

        Args:
            batch: Input batch dictionary
            x: Current state dict {data_mode: tensor [nsamples, n, d]}
            x_1_pred: Previous clean prediction (for self-conditioning), can be None
            mask: Valid residue mask [nsamples, n]
            predict_for_sampling: Prediction function that takes batch and mode
            nsteps: Number of steps to take in the partial simulation
            ts: Time schedules per modality, dict of tensors [nsteps+1]
            gt: Noise injection schedules per modality, dict of tensors [nsteps]
            self_cond: Whether to use self-conditioning
            simulation_step_params: Simulation parameters per modality
            device: Device to run on
            save_trajectory_every: How often to save trajectory snapshots, 0 to disable.
            dump_noisy_trajectories: If True, also collect the noisy x_t alongside x_1.
            guidance_w: Guidance weight (1.0 = no guidance)
            ag_ratio: Autoguidance ratio (0.0 = CFG only, 1.0 = AG only)
            resample_target_backbone: If True, independently resample the target
                backbone from its forward marginal at every step while allowing
                the model to update only the sequence latents.
            target_backbone: Clean target backbone used for inverse folding.
                Defaults to ``batch['x_1']['bb_ca']`` when resampling is enabled.

        Returns:
            Tuple of (x, x_1_pred, extra_info) where:
                - x: Final state after partial simulation, dict of tensors
                - x_1_pred: Final clean prediction for self-conditioning, dict of tensors
                - extra_info: Dict with 'padding_mask', 'can_be_encoded', and optionally
                  'trajectory_steps' (list of dicts, each with 'step', 'x_1', and optionally 'x_t')
        """
        nsamples = x[self.data_modes[0]].shape[0]
        is_encoded_mask = batch.get("can_be_encoded")  # [nsamples, n_tokens]
        trajectory_steps: list[dict] = []

        if resample_target_backbone:
            if target_backbone is None:
                target_backbone = batch["x_1"]["bb_ca"]
            target_backbone = target_backbone.clone()

        nn_out = None
        with torch.no_grad():
            # Always-populated x_sc + is_active scalar; feature factory reads both unconditionally.
            batch["x_sc"] = {dm: torch.zeros_like(x[dm]) for dm in self.data_modes}
            batch["x_sc_is_active"] = torch.zeros((), device=device)
            if resample_target_backbone:
                x["bb_ca"] = sample_noised_target_backbone(
                    self,
                    batch,
                    target_backbone,
                    ts["bb_ca"][0] * torch.ones(nsamples, device=device),
                )
            for step in range(nsteps):
                t = {
                    data_mode: ts[data_mode][step] * torch.ones(nsamples, device=device)
                    for data_mode in self.data_modes
                }
                dt = {data_mode: ts[data_mode][step + 1] - ts[data_mode][step] for data_mode in self.data_modes}
                gt_step = {data_mode: gt[data_mode][step] for data_mode in self.data_modes}

                # Update batch with current state
                batch["x_t"] = x
                batch["t"] = t
                batch["padding_mask"] = mask

                # Add self-conditioning if enabled
                if self_cond and x_1_pred is not None:
                    batch["x_sc"] = {**x_1_pred, "bb_ca": target_backbone} if resample_target_backbone else x_1_pred
                    batch["x_sc_is_active"] = torch.ones((), device=device)

                # Get model prediction with guidance
                # This calls predict_for_sampling internally with proper mode handling
                nn_out = self.get_clean_pred_n_guided_vector(
                    batch=batch,
                    predict_for_sampling=predict_for_sampling,
                    guidance_w=guidance_w,
                    ag_ratio=ag_ratio,
                )

                # Track clean prediction for self-conditioning
                x_1_pred = self.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)

                # Take simulation step
                x = self.simulation_step(
                    x_t=x,
                    nn_out=nn_out,
                    t=t,
                    dt=dt,
                    gt=gt_step,
                    mask=mask,
                    simulation_step_params=simulation_step_params,
                )

                # Mask updates to encoded tokens and non-padded tokens only
                if is_encoded_mask is not None:
                    x["local_latents"] = x["local_latents"] * is_encoded_mask[..., None]
                x["local_latents"] = x["local_latents"] * batch["padding_mask"][..., None]
                x["bb_ca"] = x["bb_ca"] * batch["padding_mask"][..., None]

                if resample_target_backbone:
                    if step == nsteps - 1:
                        x["bb_ca"] = target_backbone.clone()
                    else:
                        x["bb_ca"] = sample_noised_target_backbone(
                            self,
                            batch,
                            target_backbone,
                            ts["bb_ca"][step + 1] * torch.ones(nsamples, device=device),
                        )

                # Collect trajectory snapshot (x_1 = clean prediction, x_t = noisy state)
                if save_trajectory_every > 0 and (step % save_trajectory_every == 0 or step == nsteps - 1):
                    step_snapshot: dict = {
                        "step": step,
                        "x_1": {dm: x_1_pred[dm].detach().cpu() for dm in x_1_pred},
                    }
                    if dump_noisy_trajectories:
                        step_snapshot["x_t"] = {dm: x[dm].detach().cpu() for dm in x}
                    trajectory_steps.append(step_snapshot)

        # Capture insertion logits from the final step for inference-time mapping
        final_insertion_logits = None
        if nn_out is not None and "insertion" in nn_out and "logits" in nn_out["insertion"]:
            final_insertion_logits = nn_out["insertion"]["logits"].detach()

        extra_info = {
            "padding_mask": mask,
            "can_be_encoded": is_encoded_mask,
            "trajectory_steps": trajectory_steps,
            "insertion_logits": final_insertion_logits,
        }
        return x, x_1_pred, extra_info


def get_gt(
    t: Float[Tensor, "nsteps"],
    mode: str,
    param: float,
    clamp_val: float | None = None,
    eps: float = 1e-2,
) -> Float[Tensor, "nsteps"]:
    r"""
    Computes gt (noise injection schedule) for different modes. This is in the SDE as

    dx_t = [v(x_t, t) + g(t) s(x_t, t)] dt + \sqrt{2g(t)} dw_t

    Args:
        t: times where we'll evaluate, covers [0, 1), shape [nsteps]
        mode: "1-t/t", "tan", "1/t", ...
        param: parameterized transformation
        clamp_val: value to clamp gt, no clamping if None
        eps: small value leave as it is

    Returns:
        Noise injection schedule, shape [nsteps]
    """

    def transform_gt(gt: Float[Tensor, "nsteps"], f_pow: float = 1.0) -> Float[Tensor, "nsteps"]:
        """Applies a transformation to the gt. f_pow=1.0 means no transformation."""
        if f_pow == 1.0:
            return gt

        # First we somewhat normalize between 0 and 1
        log_gt = torch.log(gt)
        mean_log_gt = torch.mean(log_gt)
        log_gt_centered = log_gt - mean_log_gt
        normalized = torch.nn.functional.sigmoid(log_gt_centered)
        # Transformation here
        normalized = normalized**f_pow
        # Undo normalization with the transformed variable
        log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
        log_gt_rec = log_gt_centered_rec + mean_log_gt
        gt_rec = torch.exp(log_gt_rec)
        return gt_rec

    t = torch.clamp(t, 0, 1 - 1e-5)  # For numerical reasons

    if mode == "1-t/t":
        num = 1.0 - t
        den = t
        gt = num / (den + eps)
    elif mode == "tan":
        num = torch.sin((1.0 - t) * torch.pi / 2.0)
        den = torch.cos((1.0 - t) * torch.pi / 2.0)
        gt = (torch.pi / 2.0) * num / (den + eps)
    elif mode == "1/t":
        num = 1.0
        den = t
        gt = num / (den + eps)
    else:
        raise NotImplementedError(f"gt not implemented {mode}")
    gt = transform_gt(gt, f_pow=param)
    gt = torch.clamp(gt, 0, clamp_val)  # If None no clamping
    return gt  # [s]


def get_schedule(
    mode: str, nsteps: int, *, p1: float | None = None, eps: float = 1e-5
) -> Float[Tensor, "nsteps_p_one"]:
    """
    Gets the partition of the unit interval with points where we'll evaluate the vector field / score.
    """
    if mode == "uniform":
        t = torch.linspace(0, 1, nsteps + 1)
        return t
    elif mode == "power":
        assert p1 is not None, "p1 cannot be none for the power schedule"
        t = torch.linspace(0, 1, nsteps + 1)
        t = t**p1
        return t
    elif mode == "log":
        assert p1 is not None, "p1 cannot be none for the log schedule"
        assert p1 > 0, f"p1 must be >0 for the log schedule, got {p1}"
        t = 1.0 - torch.logspace(-p1, 0, nsteps + 1).flip(0)
        t = t - torch.min(t)
        t = t / torch.max(t)
        return t
    else:
        raise OSError(f"Schedule mode not recognized {mode}")


def get_schedule_tsr_safe(
    mode: str, nsteps: int, *, p1: float | None = None, eps: float = 1e-3
) -> Float[Tensor, "nsteps_p_one"]:
    """Schedule safe for t=0 when TSR is used. Starts at eps instead of 0."""
    if mode == "uniform":
        t = torch.linspace(eps, 1, nsteps + 1)
        return t
    elif mode == "power":
        assert p1 is not None, "p1 cannot be none for the power schedule"
        t = torch.linspace(eps ** (1 / p1), 1, nsteps + 1)
        t = t**p1
        return t
    elif mode == "log":
        assert p1 is not None, "p1 cannot be none for the log schedule"
        assert p1 > 0, f"p1 must be >0 for the log schedule, got {p1}"
        t = 1.0 - torch.logspace(-p1, eps, nsteps + 1).flip(0)
        t = t - (torch.min(t) - eps)
        t = t / torch.max(t)
        return t
    else:
        raise OSError(f"Schedule mode not recognized {mode}. TSR only supports uniform, power, and log schedules.")


def _sample_t(cfg_t_dist: dict, shape: tuple, device: torch.device = torch.device) -> Float[Tensor, "*shape"]:
    """
    Samples time from different time distributions.

    Args:
        cfg_t_dist: dict specifying distribution to sample
        shape: shape of the returned tensor t
        device: torch.device

    Returns:
        Tensor with random samples t from the requested distribution and shape
    """
    if cfg_t_dist.name == "uniform":
        t_max = cfg_t_dist.p2
        return torch.rand(shape, device=device) * t_max
    elif cfg_t_dist.name == "logit-normal":
        mean = cfg_t_dist.p1
        std = cfg_t_dist.p2
        noise = torch.randn(shape, device=device) * std + mean
        return torch.nn.functional.sigmoid(noise)
    elif cfg_t_dist.name == "beta":
        p1 = cfg_t_dist.p1
        p2 = cfg_t_dist.p2
        dist = torch.distributions.beta.Beta(p1, p2)
        return dist.sample(shape).to(device)
    elif cfg_t_dist.name == "mix_unif_beta":
        p1 = cfg_t_dist.p1
        p2 = cfg_t_dist.p2
        p3 = cfg_t_dist.p3  # For mix weight of uniform
        assert 0.0 < p3 < 1.0, f"p3 value {p3} not in (0, 1)"
        dist = torch.distributions.beta.Beta(p1, p2)
        samples_beta = dist.sample(shape).to(device)
        samples_uniform = torch.rand(shape, device=device)
        u = torch.rand(shape, device=device)
        return torch.where(u < p3, samples_uniform, samples_beta)
    else:
        raise NotImplementedError(f"Sampling mode for t {cfg_t_dist.name} not implemented")
