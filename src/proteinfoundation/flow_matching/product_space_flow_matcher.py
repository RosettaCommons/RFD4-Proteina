from collections.abc import Callable

import lightning as L
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from proteinfoundation.flow_matching.base_flow_matcher import BaseFlowMatcher
from proteinfoundation.flow_matching.rdn_flow_matcher import RDNFlowMatcher

FLOW_MATCHER_FACTORY = {
    "bb_ca": RDNFlowMatcher,
    "local_latents": RDNFlowMatcher,
}  # maps data modality to correspodning class


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
    ) -> dict[str, Tensor]:
        """
        Samples reference distribution (possibly centered).

        Args:
            n: number of residues in a single sample (i.e. protein length), int
            shape: tuple (if empty then single sample)
            device (optional): torch device used
            mask (optional): Binary mask of shape [*, n]
            training (optional): whether in training mode

        Returns:
            Samples from refenrece with shape [*shaqpe, n, ...]
        """
        x = {
            data_mode: self.base_flow_matchers[data_mode].sample_noise(
                n=n,
                shape=shape,
                device=device,
                mask=mask,
                training=training,
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
    ) -> dict[str, torch.Tensor]:
        """
        Interpolates between x_0 (base) and x_1 (data) using t.

        Args:
            x_0: Samples from reference with batch shape * (each value in the dict)
            x_1: Samples from target with batch shape *
            t: Interpolation times
            mask (optional): Binary mask, shape [*, n]

        Returns:
            x_t: Interpolated samples, same shape as x_0 and x_1
        """
        x_t = {
            data_mode: self.base_flow_matchers[data_mode].interpolate(
                x_0=x_0[data_mode],
                x_1=x_1[data_mode],
                t=t[data_mode],
                mask=mask,
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
        if t is None:
            t = self.sample_t(shape=batch_shape, device=device)
        x_0 = self.sample_noise(n=n, shape=batch_shape, mask=padding_mask, device=device, training=True)
        x_t = self.interpolate(x_0=x_0, x_1=x_1, t=t, mask=padding_mask)
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
        loss = {
            data_mode: self.base_flow_matchers[data_mode].compute_fm_loss(
                x_0=batch["x_0"][data_mode],
                x_1=batch["x_1"][data_mode],
                x_t=batch["x_t"][data_mode],
                mask=should_be_lossed_masks[data_mode],
                t=batch["t"][data_mode],
                nn_out=nn_out[data_mode],
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

        # NOTE: We don't have auxiliary losses yet but if we do make sure to use the should_be_lossed_masks like below!
        # should_be_lossed_masks = {
        #     "bb_ca": batch["should_be_lossed"],
        #     "local_latents": batch["should_be_lossed"] & batch["can_be_encoded"],
        # }
        return losses

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
            guidance_w: Guidance weight
            ag_ratio: Autoguidance ratio

        Returns:
            Tuple of (x, additional_info) where:
                - x: Generated samples, dict with keys for each data mode
                - additional_info: Dict with 'padding_mask' and 'can_be_encoded'
        """

        # Use mask from batch if present, otherwise generate it
        if "padding_mask" in batch and batch["padding_mask"] is not None:
            mask = batch["padding_mask"]
        else:
            mask = torch.ones(nsamples, n).long().bool().to(device)
        assert mask.shape == (nsamples, n)

        if save_trajectory_every > 0:
            # Create a list of trajectory dictionaries, one for each sample in batch
            [{} for _ in range(nsamples)]

        # Get diffusion schedule using extracted method
        ts, gt = self.sample_schedule(nsteps=nsteps, sampling_model_args=sampling_model_args)

        with torch.no_grad():
            x = self.sample_noise(
                n,
                shape=(nsamples,),
                device=device,
                mask=mask,
                training=False,
            )

            # After sampling noise, mask to generated tokens only
            is_encoded_mask = batch.get("can_be_encoded")  # [nsamples, n_tokens]
            if is_encoded_mask is not None:
                x["local_latents"] = x["local_latents"] * is_encoded_mask[..., None]  # zero out non-encoded latents
            x["local_latents"] = x["local_latents"] * batch["padding_mask"][..., None]
            x["bb_ca"] = x["bb_ca"] * batch["padding_mask"][..., None]  # zero out padded bb_ca tokens

            x_1_pred = None
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

            additional_info = {
                "padding_mask": mask,
                "can_be_encoded": is_encoded_mask,
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
        ts = {
            data_mode: get_schedule(
                mode=sampling_model_args[data_mode]["schedule"]["mode"],
                nsteps=int(nsteps),
                p1=sampling_model_args[data_mode]["schedule"]["p"],
            )
            for data_mode in self.data_modes
        }  # each [nsteps + 1], first element is 0, last is 1

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
        guidance_w: float = 1.0,
        ag_ratio: float = 0.0,
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
            guidance_w: Guidance weight (1.0 = no guidance)
            ag_ratio: Autoguidance ratio (0.0 = CFG only, 1.0 = AG only)

        Returns:
            Tuple of (x, x_1_pred, extra_info) where:
                - x: Final state after partial simulation, dict of tensors
                - x_1_pred: Final clean prediction for self-conditioning, dict of tensors
                - extra_info: Dict with 'padding_mask' and 'can_be_encoded'
        """
        nsamples = x[self.data_modes[0]].shape[0]
        is_encoded_mask = batch.get("can_be_encoded")  # [nsamples, n_tokens]

        with torch.no_grad():
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
                    batch["x_sc"] = x_1_pred

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

        # Return extra_info in same format as full_simulation
        extra_info = {
            "padding_mask": mask,
            "can_be_encoded": is_encoded_mask,
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
