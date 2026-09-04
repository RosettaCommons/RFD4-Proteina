import logging
from typing import Literal

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from rfproteina.flow_matching.base_flow_matcher import BaseFlowMatcher
from rfproteina.utils.align import mean_w_mask

_logger = logging.getLogger(__name__)

VALID_GROUP_KEYS = ("chain_iid", "pn_unit_iid", "molecule_iid", "coord_group")


class RDNFlowMatcher(BaseFlowMatcher):
    """
    Flow matching on (R^d)^n, where n is for the number of elements
    per sample (e.g. number of residues) and d is the dimensionality
    for ach element.

    We include the option of using (R_0^d)^n (centering happening for each {1, ..., d} across n dimension).
    """

    def __init__(
        self,
        zero_com_noise: bool = False,
        guidance_enabled: bool = False,
        dim: int = 3,
        stochastic_centering_mode: Literal["none", "x_0", "x_1", "x_t"] = "none",
        stochastic_centering_scale: float | None = None,
        stochastic_centering_scope: Literal["all", "without_absolute_coords", "per_group"] = "without_absolute_coords",
        stochastic_centering_group_keys: list[str] | None = None,
        **kwargs,
    ):  # TODO: DR: kwargs added to avoid init error
        super().__init__(
            guidance_enabled=guidance_enabled,
            dim=dim,
        )
        self.zero_com_noise = zero_com_noise
        self.stochastic_centering_mode = stochastic_centering_mode
        self.stochastic_centering_scale = stochastic_centering_scale
        self.stochastic_centering_scope = stochastic_centering_scope
        self.stochastic_centering_group_keys = (
            list(stochastic_centering_group_keys) if stochastic_centering_group_keys else []
        )

        if self.stochastic_centering_scope == "per_group":
            if not self.stochastic_centering_group_keys:
                raise ValueError(
                    "stochastic_centering_scope='per_group' requires stochastic_centering_group_keys to be a non-empty."
                )
            invalid = [k for k in self.stochastic_centering_group_keys if k not in VALID_GROUP_KEYS]
            if invalid:
                raise ValueError(
                    f"Invalid stochastic_centering_group_keys: {invalid}. Must be a subset of {VALID_GROUP_KEYS}."
                )

    def _force_zero_com(
        self, x: Float[Tensor, "* n d"], mask: Bool[Tensor, "* n"] | None = None
    ) -> Float[Tensor, "* n d"]:
        """
        Centers tensor over n (residue) dimension.

        Args:
            x: Tensor of shape [*, n, d]
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Centered x = x - mean(x, dim=-2), shape [*, n, d].
        """
        if mask is None:
            x = x - torch.mean(x, dim=-2, keepdim=True)
        else:
            x = (x - mean_w_mask(x, mask, keepdim=True)) * mask[..., None]
        return x

    def _apply_mask(self, x: Float[Tensor, "* n d"], mask: Bool[Tensor, "* n"] | None = None) -> Float[Tensor, "* n d"]:
        """
        Applies mask to x. Sets masked elements to zero.

        Args:
            x: Tensor of shape [*, n, d]
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Masked x of shape [*, n, d]
        """
        if mask is None:
            return x
        return x * mask[..., None]  # [*, n, d]

    def _apply_stochastic_offset(
        self,
        x: Float[Tensor, "* n d"],
        mask: Bool[Tensor, "* n"] | None,
        gen_mask: Bool[Tensor, "* n"] | None,
        group_ids: Int[Tensor, "* n"] | None = None,
    ) -> Float[Tensor, "* n d"]:
        """Apply per-group random offsets to gen tokens.

        Each unique value in ``group_ids`` within ``gen_mask`` receives its own
        independent per-batch-element random translation. Tokens outside ``gen_mask``
        are never moved.

        Args:
            x: Coordinates, shape [*, n, d].
            mask: Padding mask, shape [*, n].
            gen_mask: Tokens eligible for offsetting, shape [*, n]. If ``None``,
                returns ``x`` unchanged (no padding mask provided).
            group_ids: Integer group label per token, shape [*, n]. If ``None``, all
                gen tokens are treated as a single group.

        Returns:
            x with stochastic offsets applied to gen tokens, shape [*, n, d].
        """
        if gen_mask is None:
            return x
        if group_ids is None:
            noise = torch.randn(*x.shape[:-2], x.shape[-1], device=x.device)  # [*, d]
            x = x + self.stochastic_centering_scale * noise.unsqueeze(-2) * gen_mask[..., None].float()
        else:
            for g in group_ids[gen_mask].unique():
                group_mask_f = (gen_mask & (group_ids == g))[..., None].float()  # [*, n, 1]
                noise = torch.randn(*x.shape[:-2], x.shape[-1], device=x.device)  # [*, d]
                x = x + self.stochastic_centering_scale * noise.unsqueeze(-2) * group_mask_f

        if mask is not None:
            x = x * mask[..., None]
        return x

    def _compose_group_ids(
        self,
        groups: dict[str, Int[Tensor, "* n"] | None] | None,
    ) -> Int[Tensor, "* n"]:
        """Combine the requested group-key tensors into a single integer id per token.

        Each unique combination of values across ``self.stochastic_centering_group_keys``
        gets mapped to a unique integer (computed globally across the batch via
        ``torch.unique`` on the stacked keys). Errors if any requested key is missing.
        """
        if groups is None:
            groups = {}
        keys = self.stochastic_centering_group_keys
        tensors = []
        for k in keys:
            t = groups.get(k)
            if t is None:
                raise ValueError(f"stochastic_centering_group_keys requested {k!r} but it is not present in the batch.")
            tensors.append(t.long())
        stacked = torch.stack(tensors, dim=-1)  # [*, n, k]
        flat = stacked.reshape(-1, stacked.shape[-1])  # [*n, k]
        _, inverse = torch.unique(flat, dim=0, return_inverse=True)
        return inverse.reshape(stacked.shape[:-1])  # [*, n]

    def add_stochastic_noise(
        self,
        x: Float[Tensor, "* n d"],
        mask: Bool[Tensor, "* n"] | None = None,
        coord_cond_mask: Bool[Tensor, "* n"] | None = None,
        groups: dict[str, Int[Tensor, "* n"] | None] | None = None,
    ) -> Float[Tensor, "* n d"]:
        """Add stochastic centering noise to x according to :attr:`stochastic_centering_scope`.

        Available scopes:

        * ``"all"`` - apply one shared offset to all non-padding tokens.
        * ``"without_absolute_coords"`` - offset all non-absolute-coordinate-conditioned
          tokens as a single group.
        * ``"per_group"`` - apply an independent offset per unique combination of the
          group keys listed in :attr:`stochastic_centering_group_keys` (a subset of
          ``{"chain_iid", "pn_unit_iid", "molecule_iid", "coord_group"}``).

        In all modes except ``"all"``, absolute-coordinate-conditioned tokens
        (``coord_cond_mask == True``) are left at the origin.

        Args:
            x: Coordinates, shape [*, n, d].
            mask: Padding mask, shape [*, n].
            coord_cond_mask: True for absolute-coordinate-conditioned tokens that should
                not be offset. Shape [*, n].
            groups: Dict of token-level group tensors keyed by name (``chain_iid``,
                ``pn_unit_iid``, ``molecule_iid``, ``coord_group``). Required for
                the ``"per_group"`` scope; ignored otherwise. Each value has shape [*, n].

        Returns:
            x with stochastic noise added, shape [*, n, d].
        """
        gen_mask = (mask & ~coord_cond_mask) if coord_cond_mask is not None else mask

        scope = self.stochastic_centering_scope

        if scope == "all":
            x = self._apply_stochastic_offset(x, mask, gen_mask=mask)
        elif scope == "without_absolute_coords":
            x = self._apply_stochastic_offset(x, mask, gen_mask=gen_mask)
        elif scope == "per_group":
            group_ids = self._compose_group_ids(groups)
            x = self._apply_stochastic_offset(x, mask, gen_mask=gen_mask, group_ids=group_ids)
        else:
            raise ValueError(f"Unknown stochastic_centering_scope: {scope!r}")

        return x

    def sample_noise(
        self,
        n: int,
        device: torch.device,
        shape: tuple = (),
        mask: Bool[Tensor, "* n"] | None = None,
        training: bool = True,
        coord_cond_mask: Bool[Tensor, "* n"] | None = None,
        groups: dict[str, Int[Tensor, "* n"] | None] | None = None,
    ) -> Float[Tensor, "* n d"]:
        """
        Samples reference distribution std Gaussian (possibly centered).

        Args:
            n: number of frames in a single sample, int
            device: torch device
            shape: tuple (if empty then single sample)
            mask (optional): Binary mask of shape [*, n]
            training (optional): whether in training mode
            coord_cond_mask (optional): Boolean mask of shape [*, n]. True for
                coordinate-conditioned tokens whose noise should not be offset.
            groups (optional): Dict of token-level group tensors keyed by name
                (``chain_iid``, ``pn_unit_iid``, ``molecule_iid``, ``coord_group``).
                Used by the ``"per_group"`` stochastic centering scope.

        Returns:
            Samples from refenrece [N(0, I_d)]^n shape [*shape, n, d]
        """
        noise = torch.randn(
            (*shape, n, self.dim),
            device=device,
        )
        noise = self._apply_mask(noise, mask)
        if self.zero_com_noise:
            noise = self._force_zero_com(noise, mask)

        # Apply stochastic centering at inference time if mode is x_0
        if self.stochastic_centering_mode == "x_0" and not training:
            noise = self.add_stochastic_noise(noise, mask, coord_cond_mask=coord_cond_mask, groups=groups)

        return noise

    def interpolate(
        self,
        x_0: Float[Tensor, "* n d"],
        x_1: Float[Tensor, "* n d"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* n"] | None = None,
        coord_cond_mask: Bool[Tensor, "* n"] | None = None,
        groups: dict[str, Int[Tensor, "* n"] | None] | None = None,
    ) -> Float[Tensor, "* n d"]:
        """
        Interpolates between rigids x_0 (base) and x_1 (data) using t.

        Args:
            x_0: Tensor sampled from reference, shape [*, n, d]
            x_1: Tensor sampled from target, shape [*, n, d]
            t: Interpolation times, shape [*]
            mask (optional): Binary mask, shape [*, n]
            coord_cond_mask (optional): Boolean mask of shape [*, n]. True for
                coordinate-conditioned tokens whose noise should not be offset.
            groups (optional): Dict of token-level group tensors keyed by name
                (``chain_iid``, ``pn_unit_iid``, ``molecule_iid``, ``coord_group``).
                Used by the ``"per_group"`` stochastic centering scope.

        Returns:
            x_t: Interpolated tensor, shape [*, n, d]
        """
        # x_0 masked, x_1 depends on dataloader
        # x_0 maybe centered (depends on input arg zero_com_noise), x_1 depends on dataloader
        x_0, x_1 = self._apply_mask(x_0, mask), self._apply_mask(x_1, mask)

        # Apply stochastic centering based on mode (during training)
        if self.stochastic_centering_mode == "x_0":
            x_0 = self.add_stochastic_noise(x_0, mask, coord_cond_mask=coord_cond_mask, groups=groups)
        elif self.stochastic_centering_mode == "x_1":
            x_1 = self.add_stochastic_noise(x_1, mask, coord_cond_mask=coord_cond_mask, groups=groups)

        t = t[..., None, None]
        # No mask since x_0 and x_1 are masked
        x_t = (1.0 - t) * x_0 + t * x_1

        # Apply stochastic centering to interpolated result if mode is x_t
        if self.stochastic_centering_mode == "x_t":
            x_t = self.add_stochastic_noise(x_t, mask, coord_cond_mask=coord_cond_mask, groups=groups)

        return x_t

    def nn_out_add_clean_sample_prediction(
        self,
        x_t: Float[Tensor, "* n d"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* n"],
        nn_out: dict[str, torch.Tensor],
    ) -> dict[str, Float[Tensor, "* n d"]]:
        """
        Computes predicted clean sample given nn output.

        Args:
            x_0: noise sample, shape [*, n, d]
            x_1: clean sample, shape [*, n, d]
            x_t: interpolated sample, shape [*, n, d]
            t: time sampled, shape [*]
            nn_out: output of neural network, Dict[str, torch.Tensor]

        Returns:
            The nn_out dictionary updated with clean sample prediction.
        """
        t = t[..., None, None]  # [*, 1, 1]
        if "x_1" in nn_out:
            pass
        elif "v" in nn_out:
            nn_out["x_1"] = x_t + (1.0 - t) * nn_out["v"]
        else:
            raise OSError(f"Cannot compute clean sample prediction from keys {list(nn_out.keys())}")
        nn_out["x_1"] = nn_out["x_1"] * mask[..., None]
        return nn_out

    def nn_out_add_simulation_tensor(
        self,
        x_t: Float[Tensor, "* n d"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* n"],
        nn_out: dict[str, torch.Tensor],
    ) -> dict[str, Float[Tensor, "* n d"]]:
        """
        Computes vector field v given nn output.

        Args:
            x_0: noise sample, shape [*, n, d]
            x_1: clean sample, shape [*, n, d]
            x_t: interpolated sample, shape [*, n, d]
            t: time sampled, shape [*]
            nn_out: output of neural network, Dict[str, torch.Tensor]

        Returns:
            The nn_out dictionary updated with v.
        """
        t = t[..., None, None]  # [*, 1, 1]
        if "v" in nn_out:
            pass
        elif "x_1" in nn_out:
            num = nn_out["x_1"] - x_t  # [*, n, d]
            den = 1.0 - t  # [*, 1, 1]
            nn_out["v"] = num / (den + 1e-5)  # [*, n, d]
        else:
            raise OSError(f"Cannot compute simulation tensor (v) from keys {list(nn_out.keys())}")
        nn_out["v"] = nn_out["v"] * mask[..., None]
        return nn_out

    def compute_fm_loss(
        self,
        x_0: Float[Tensor, "* n d"],
        x_1: Float[Tensor, "* n d"],
        x_t: Float[Tensor, "* n d"],
        mask: Bool[Tensor, "* n"],
        t: Float[Tensor, "*"],
        nn_out: dict[str, Float[Tensor, "* n d"]],
        weight: Float[Tensor, "* n"] | None = None,
    ) -> Float[Tensor, "*"]:
        """
        Computes flow matching loss per element in the batch.

        Args:
            x_0: noise sample, shape [*, n, d]
            x_1: clean sample, shape [*, n, d]
            x_t: interpolated sample, shape [*, n, d]
            mask: Binary mask for generated tokens with non-zero occupancy, shape [*, n]
            t: time sampled, shape [*]
            nn_out: neural network output, shape [*, n, d]
            weight: optional per-token loss weight, shape [*, n]. ``None`` (default)
                or all-ones is bitwise-identical to today's behavior. Multiplied into
                the per-token MSE before reduction; normalization stays atom-count
                based (``/ nres``), matching RF3's ``DiffusionLoss``.

        Returns:
            Loss per element in the batch, shape [*]
        """
        nn_out = self.nn_out_add_clean_sample_prediction(
            x_t=x_t,
            t=t,
            mask=mask,
            nn_out=nn_out,
        )
        nres = torch.sum(mask, dim=-1).clamp(min=1)  # [*] - avoid division by zero
        err = (x_1 - nn_out["x_1"]) * mask[..., None]  # [*, n, d]
        if weight is None:
            weighted_sq = err**2  # [*, n, d]
        else:
            w = weight.to(dtype=x_1.dtype) * mask  # [*, n], gated weights
            weighted_sq = (err**2) * w[..., None]  # [*, n, d]
        loss = torch.sum(weighted_sq, dim=(-1, -2)) / nres  # [*]
        total_loss_w = 1.0 / ((1.0 - t) ** 2 + 1e-5)
        loss = loss * total_loss_w  # [*]
        return loss

    def nn_out_add_guided_simulation_tensor(
        self,
        nn_out: dict[str, torch.Tensor],
        nn_out_ucond: dict[str, torch.Tensor] | None,
        guidance_w: float,
    ) -> dict[str, torch.Tensor]:
        """
        Computes the classifier-free-guided velocity given nn output. Note this assumes the nn_out
        dictionaries contain the simulation tensor (v, score, ...) for each corresponding
        base flow matcher.

        Args:
            nn_out: output of neural network from full model, Dict[str, Tensor]
            nn_out_ucond: output of neural network from unconditional model, Dict[str, torch.Tensor] or None
            guidance_w: guidance weight, float

        Returns:
            The nn_out dictionary updated with guided v.
        """
        assert "v" in nn_out, "`v` should be a key in the nn_out dict"
        if not self.guidance_enabled:
            return nn_out

        v = nn_out["v"]
        v_ucond = torch.zeros_like(v) if nn_out_ucond is None else nn_out_ucond["v"]

        nn_out["v_guided"] = guidance_w * v + (1 - guidance_w) * v_ucond
        return nn_out

    def simulation_step(
        self,
        x_t: Float[Tensor, "* n d"],
        nn_out: dict[str, Float[Tensor, "* n d"]],
        t: Float[Tensor, "*"],
        dt: float,
        gt: float,
        mask: Bool[Tensor, "* n"],
        simulation_step_params: dict,
    ) -> Float[Tensor, "* n d"]:
        r"""
        Single integration step of ODE

        eq. (1): d x_t = v(x_t, t) dt

        or SDE

        eq. (2): d x_t = [v(x_t, t) + g(t) s(x_t, t)] dt + \sqrt{2g(t)} dw_t

        using Euler integration scheme.

        For our interpolation scheme (i.e. stochastic interpolant) we can obtain
        the score as a function of the vector field from

        v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

        or equivalently,

        s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

        We add a few additional parameters to the SDE to control noise/score scale and
        perform stochastic and low temperature sampling:

        eq. (3): d x_t = [v(x_t, t) + g(t) * sc_score_scale * s(x_t, t) * sc_score_scale] dt + \
                        \sqrt{2 * g(t) * sc_noise_scale} dw_t.

        At the moment we do not scale the vector field v.

        Args:
            x_t: Current value, shape [*, n, d]
            nn_out: Dictionary with all available predictions, should include "v" and possibly guided "v_guided".
            May include "x_1", etc as well. All shape [*, n, d]
            t: Current time, shape [*]
            dt: Step-size, float
            gt: Noise injection, float
            mask: Binary mask of shape [*, n]
            simulation_step_params: parameters for the simulation step, keys
                sampling_mode, sc_scale_noise, sc_scale_score, t_lim_ode

        Returns:
            Updated values for x_t after an Euler integration step, shape [*, n, d].
        """
        sampling_mode = simulation_step_params["sampling_mode"]
        sc_scale_noise = simulation_step_params["sc_scale_noise"]
        sc_scale_score = simulation_step_params["sc_scale_score"]
        t_lim_ode = simulation_step_params["t_lim_ode"]
        t_lim_ode_below = simulation_step_params["t_lim_ode_below"]
        center_every_step = simulation_step_params["center_every_step"]
        tsr_k = simulation_step_params.get("tsr_k", 1.0)
        tsr_sigma = simulation_step_params.get("tsr_sigma", 1.0)

        assert sampling_mode in [
            "vf",
            "sc",
            "vf_ss",
            "vf_ss_sc_sn",
            "vf_tsr",
        ], f"Invalid sampling mode {sampling_mode}, should be `vf`, `sc`, `vf_ss`, `vf_ss_sc_sn`, or `vf_tsr`"
        assert sc_scale_noise >= 0, f"Scale noise for sampling should be >= 0, got {sc_scale_noise}"
        assert sc_scale_score >= 0, f"Scale score for sampling should be >= 0, got {sc_scale_score}"
        assert gt >= 0, f"gt for sampling should be >= 0, got {gt}"
        t_element = t.flatten()[0]
        assert torch.all(t_element == t), "Sampling only implemented for same time for all samples"

        if self.guidance_enabled:
            v = nn_out["v_guided"]
        else:
            v = nn_out["v"]

        sc_scale_score_def = 1.5  # used when very close to 1 for sc mode, since we don't want to add more noise, we switch to low temp ODE with this scale for the score  # noqa: E501
        sc_scale_noise_def = 0.3  # used when very close to 0 for vf_ss mode, since we can't use low temp ODE then, we switch to low temp SDE with this scale for the noise  # noqa: E501

        if sampling_mode == "vf":  # ODE mode
            delta_x = v * dt

        elif sampling_mode == "vf_ss":  # ODE with score scaling
            if t_element < t_lim_ode_below:  # Close to zero cannot do score_to_vf, switch to SDE with noise scaling
                score = vf_to_score(x_t, v, t)  # get score from v, [*, dim]
                eps = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)  # [*, dim]
                std_eps = torch.sqrt(2 * gt * sc_scale_noise_def * dt)
                delta_x = (v + gt * score) * dt + std_eps * eps
            else:
                score = vf_to_score(x_t, v, t)
                scaled_score = score * sc_scale_score
                v_scaled = score_to_vf(x_t, scaled_score, t)
                delta_x = v_scaled * dt

        elif sampling_mode == "sc":  # SDE with noise scaling
            if t_element > t_lim_ode:  # close to 1 stop SDE, switch to low temp ODE
                score = vf_to_score(x_t, v, t)
                scaled_score = score * sc_scale_score_def
                v_scaled = score_to_vf(x_t, scaled_score, t)
                delta_x = v_scaled * dt
            else:
                score = vf_to_score(x_t, v, t)  # get score from v, [*, dim]
                eps = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)  # [*, dim]
                std_eps = torch.sqrt(2 * gt * sc_scale_noise * dt)
                delta_x = (v + gt * score) * dt + std_eps * eps

        elif sampling_mode == "vf_ss_sc_sn":  # SDE with score scaling (in ODE) and noise scaling (in Langevin term)
            if t_element > t_lim_ode:  # close to 1 stop SDE, switch to plain low temp ODE
                score = vf_to_score(x_t, v, t)
                scaled_score = score * sc_scale_score_def
                v_scaled = score_to_vf(x_t, scaled_score, t)
                delta_x = v_scaled * dt
            elif t_element < t_lim_ode_below:  # close to 0 switch to SDE with noise scaling
                score = vf_to_score(x_t, v, t)  # get score from v, [*, dim]
                eps = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)  # [*, dim]
                std_eps = torch.sqrt(2 * gt * sc_scale_noise_def * dt)
                delta_x = (v + gt * score) * dt + std_eps * eps
            else:  # ODE scaled with SDE scaled
                score = vf_to_score(x_t, v, t)
                scaled_score = score * sc_scale_score
                v_scaled = score_to_vf(x_t, scaled_score, t)
                eps = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)  # [*, dim]
                std_eps = torch.sqrt(2 * gt * sc_scale_noise * dt)
                delta_x = (v_scaled + gt * score) * dt + std_eps * eps

        elif sampling_mode == "vf_tsr":  # ODE with time-scale-ratio scaling, optionally made stochastic
            tsr_ratio = compute_tsr_ratio(t_element, tsr_k, tsr_sigma)
            if t_element == 0.0:
                v_scaled = torch.zeros_like(v).to(v.device)
                delta_x = v_scaled * dt
            else:
                score = vf_to_score(x_t, v, t)
                scaled_score = score * tsr_ratio
                v_scaled = score_to_vf(x_t, scaled_score, t)
                if t_element > t_lim_ode or t_element < t_lim_ode_below:
                    delta_x = v_scaled * dt
                else:
                    eps = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)  # [*, dim]
                    std_eps = torch.sqrt(torch.as_tensor(2 * gt * sc_scale_noise * dt))
                    delta_x = (v_scaled + gt * score * sc_scale_score) * dt + std_eps * eps

        else:
            raise ValueError(f"Invalid sampling mode {sampling_mode}")

        x_next = x_t + delta_x

        # Mask and potentially center
        x_next = self._apply_mask(x_next, mask)
        if center_every_step:
            x_next = self._force_zero_com(x_next, mask)
        return x_next


def compute_tsr_ratio(
    t: Float[Tensor, "*"],
    tsr_k: float,
    tsr_sigma: float,
) -> Float[Tensor, "*"]:
    """Compute the time-scale-ratio (TSR) low-temperature scaling factor.

    Args:
      t: Time value(s) in [0, 1). Shape: [*].
      tsr_k: Controls how low the temperature goes (higher ``k`` = lower temperature).
      tsr_sigma: Controls how fast the low temperature is reached (higher ``sigma`` = faster).

    Returns:
      TSR scaling ratio, same shape as ``t``.
    """
    snr = t**2 / (1 - t) ** 2
    return (snr * tsr_sigma**2 + 1) / (snr * tsr_sigma**2 / tsr_k + 1)


def vf_to_score(
    x_t: Float[Tensor, "* n d"],
    v: Float[Tensor, "* n d"],
    t: Float[Tensor, "*"],
) -> Float[Tensor, "* n d"]:
    """
    Computes score of noisy density given the vector field learned by flow matching. With
    our interpolation scheme these are related by

    v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

    or equivalently,

    s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

    Args:
        x_t: Noisy sample, shape [*, dim]
        v: Vector field, shape [*, dim]
        t: Interpolation time, shape [*] (must be < 1)

    Returns:
        Score of intermediate density, shape [*, dim].
    """
    assert torch.all(t < 1.0), "vf_to_score requires t < 1 (strict)"
    num = t[..., None, None] * v - x_t  # [*, dim]
    den = (1.0 - t)[..., None, None]  # [*, 1]
    score = num / den
    return score  # [*, dim]


def score_to_vf(
    x_t: Float[Tensor, "* n d"],
    score: Float[Tensor, "* n d"],
    t: Float[Tensor, "*"],
) -> Float[Tensor, "* n d"]:
    """
    Computes vector field from score. See `vf_to_score` function for equations.

    Args:
        x_t: Noisy sample, shape [*, n, dim]
        score: Vector field, shape [*, n, dim]
        t: Interpolation time, shape [*] (must be > 0)

    Returns:
        Vector field, shape [*, n, dim].
    """
    assert torch.all(t > 0.0), "score_to_vf requires t > 0 (strict)"
    t = t[..., None, None]  # [*, 1, 1]
    return (x_t + (1.0 - t) * score) / t  # [*, n, dim]
