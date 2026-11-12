from collections.abc import Mapping, Sequence
import logging
from typing import Any, Literal

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from rfproteina.flow_matching.base_flow_matcher import BaseFlowMatcher
from rfproteina.symmetry import (
    SymmetryConfig,
    has_active_symmetry,
    project_invariant_symmetric,
    project_symmetric,
    resolve_batch_symmetry,
    sample_symmetric_ca_noise,
    symmetrize_decoded_atom37,
    zero_global_com,
)
from rfproteina.utils.align import mean_w_mask

_logger = logging.getLogger(__name__)

VALID_GROUP_KEYS = ("chain_iid", "pn_unit_iid", "molecule_iid", "coord_group")
_SYMMETRY_CONFIG_KEYS = ("group", "decode_mode", "input_is_asu", "tie_sequence")


def _is_dict_like(obj: Any) -> bool:
    """Return True for Mappings and custom dict-like batch containers (e.g. PaddedBatchedDataSample)."""
    if isinstance(obj, Mapping):
        return True
    return (
        hasattr(obj, "get")
        and hasattr(obj, "__contains__")
        and not isinstance(obj, (str, bytes, bytearray, Tensor, SymmetryConfig))
    )


def _infer_natural_batch_size(extracted: Any) -> int | None:
    """Infer natural row count from a sequence or collated dict-of-lists symmetry payload."""
    if isinstance(extracted, Sequence) and not isinstance(extracted, (str, bytes, bytearray, Tensor)):
        return len(extracted)
    if _is_dict_like(extracted):
        group_val = extracted.get("group")
        if isinstance(group_val, Tensor) and group_val.ndim >= 1:
            return int(group_val.shape[0])
        if isinstance(group_val, Sequence) and not isinstance(group_val, (str, bytes, bytearray)):
            return len(group_val)
        nested_sym = extracted.get("symmetry")
        if nested_sym is not None and nested_sym is not extracted:
            return _infer_natural_batch_size(nested_sym)
    return None


def normalize_symmetry_configs(
    symmetry_or_batch: Any,
    batch_size: int | None = None,
    fallback: Any = None,
) -> list[SymmetryConfig] | None:
    """Resolve per-row :class:`SymmetryConfig` list, or ``None`` when no active symmetry (> C1) exists.

    Handles ``None``, ``str``, :class:`SymmetryConfig`, single config ``Mapping``, collated
    dict-of-lists from ``batch_dataclass_recursive``, ``PaddedBatchedDataSample``, per-sample lists,
    and beam-search tiled batches where ``batch_size`` is an integer multiple of the natural batch length.
    """
    if symmetry_or_batch is None and fallback is None:
        return None

    target: Any = symmetry_or_batch
    inferred_n: int | None = None

    if _is_dict_like(symmetry_or_batch) and not any(k in symmetry_or_batch for k in _SYMMETRY_CONFIG_KEYS):
        # Fast exit for standard non-symmetric batches when no model-level fallback is set.
        sym_direct = symmetry_or_batch.get("symmetry")
        if sym_direct is None:
            sym_direct = symmetry_or_batch.get("symmetry_batch")
        extra_info = symmetry_or_batch.get("extra_info")
        spec_batch = symmetry_or_batch.get("specification_batch")
        spec_single = symmetry_or_batch.get("specification")

        if (
            fallback is None
            and sym_direct is None
            and extra_info is None
            and spec_batch is None
            and spec_single is None
        ):
            return None

        if sym_direct is not None:
            target = sym_direct
            inferred_n = _infer_natural_batch_size(sym_direct)
        elif _is_dict_like(extra_info) and "symmetry" in extra_info:
            target = extra_info["symmetry"]
            inferred_n = _infer_natural_batch_size(target)
        elif _is_dict_like(spec_batch) and "symmetry" in spec_batch:
            target = spec_batch["symmetry"]
            inferred_n = _infer_natural_batch_size(target)
        elif spec_batch is None and _is_dict_like(spec_single) and "symmetry" in spec_single:
            target = spec_single["symmetry"]
            inferred_n = _infer_natural_batch_size(target)
        else:
            inferred_n = _infer_natural_batch_size(extra_info) or _infer_natural_batch_size(spec_batch)
    else:
        inferred_n = _infer_natural_batch_size(target)

    resolve_n = inferred_n if inferred_n is not None and inferred_n > 0 else batch_size
    cfgs = resolve_batch_symmetry(target, fallback=fallback, batch_size=resolve_n)

    if batch_size is not None and len(cfgs) != batch_size:
        if batch_size == 0:
            cfgs = []
        elif len(cfgs) == 1:
            cfgs = cfgs * batch_size
        elif len(cfgs) > 1 and batch_size % len(cfgs) == 0:
            cfgs = (cfgs * (batch_size // len(cfgs)))[:batch_size]
        else:
            raise ValueError(f"Batch symmetry length {len(cfgs)} is incompatible with batch_size {batch_size}")

    if not cfgs or not has_active_symmetry(cfgs):
        return None
    return cfgs


def _row_has_abs_coord_cond(
    coord_cond_mask: Tensor | None,
    emit_mask: Tensor | None,
    batch_size: int,
) -> list[bool]:
    """Return per-row boolean flags indicating whether fixed absolute coordinates are present."""
    if coord_cond_mask is None or batch_size == 0:
        return [False] * batch_size
    eff = (coord_cond_mask.bool() & emit_mask.bool()) if emit_mask is not None else coord_cond_mask.bool()
    if eff.ndim == 1:
        val = bool(eff.any().item())
        return [val] * batch_size
    row_any = eff.reshape(batch_size, -1).any(dim=-1)
    if not bool(row_any.any().item()):
        return [False] * batch_size
    return [bool(v) for v in row_any.tolist()]


def project_symmetric_3d_batched(
    x: Tensor,
    sym_cfgs: list[SymmetryConfig] | None,
    emit_mask: Tensor | None = None,
    *,
    zero_com: bool = False,
    coord_cond_mask: Tensor | None = None,
) -> Tensor:
    """Project 3D Cartesian coordinates or velocities onto the per-row point-group subspace."""
    if sym_cfgs is None:
        return x

    unbatched = x.ndim == 2
    batch_size = 1 if unbatched else x.shape[0]
    is_uniform = all(c == sym_cfgs[0] for c in sym_cfgs[1:])

    if is_uniform:
        out = project_symmetric(x, sym_cfgs[0], mask=emit_mask)
    elif unbatched:
        out = project_symmetric(x, sym_cfgs[0], mask=emit_mask)
    else:
        out = project_symmetric(x, sym_cfgs, mask=emit_mask)

    if zero_com:
        abs_flags = _row_has_abs_coord_cond(coord_cond_mask, emit_mask, batch_size)
        if not any(abs_flags):
            out = zero_global_com(out, mask=emit_mask)
        elif not all(abs_flags) and not unbatched:
            rows = []
            for b_idx in range(batch_size):
                row_m = None if emit_mask is None else emit_mask[b_idx]
                if not abs_flags[b_idx]:
                    rows.append(zero_global_com(out[b_idx], mask=row_m))
                else:
                    rows.append(out[b_idx])
            out = torch.stack(rows, dim=0)

    return out


def tie_invariant_batched(
    h: Tensor,
    sym_cfgs: list[SymmetryConfig] | None,
    emit_mask: Tensor | None = None,
    *,
    ignore_tie_sequence_flag: bool = False,
) -> Tensor:
    """Average and re-tile SE(3)-invariant per-token tensors across symmetric subunits."""
    if sym_cfgs is None:
        return h
    if not ignore_tie_sequence_flag and not any(c.is_active and c.tie_sequence for c in sym_cfgs):
        return h

    is_uniform = all(c == sym_cfgs[0] for c in sym_cfgs[1:])
    if is_uniform:
        cfg0 = sym_cfgs[0]
        eff_group: int | SymmetryConfig = (
            cfg0.spec if ignore_tie_sequence_flag else (cfg0 if cfg0.tie_sequence else 1)
        )
        return project_invariant_symmetric(h, eff_group, mask=emit_mask)

    if h.ndim < 2 or h.shape[0] != len(sym_cfgs):
        raise ValueError(f"Expected leading batch dim {len(sym_cfgs)} for h shape {tuple(h.shape)}")

    eff_groups = [c.spec if ignore_tie_sequence_flag else c for c in sym_cfgs]
    return project_invariant_symmetric(h, eff_groups, mask=emit_mask)


def symmetrize_atom37_batched(
    coors_nm: Tensor,
    sym_cfgs: list[SymmetryConfig] | None,
    *,
    ca_coors_nm: Tensor | None = None,
    atom_mask: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """Symmetrize decoded ``[..., N, 37, 3]`` atom37 coordinates across symmetric subunits."""
    if sym_cfgs is None:
        return coors_nm

    unbatched = coors_nm.ndim == 3
    c_batch = coors_nm.unsqueeze(0) if unbatched else coors_nm
    ca_batch = None if ca_coors_nm is None else (ca_coors_nm.unsqueeze(0) if unbatched else ca_coors_nm)
    am_batch = None if atom_mask is None else (atom_mask.unsqueeze(0) if unbatched else atom_mask)
    m_batch = None if mask is None else (mask.unsqueeze(0) if unbatched else mask)

    batch_size = int(c_batch.shape[0])
    if len(sym_cfgs) != batch_size:
        raise ValueError(f"Expected {batch_size} SymmetryConfig entries, got {len(sym_cfgs)}")

    rows: list[Tensor] = []
    for b_idx, cfg in enumerate(sym_cfgs):
        c_row = c_batch[b_idx]
        ca_row = None if ca_batch is None else ca_batch[b_idx]
        am_row = None if am_batch is None else am_batch[b_idx]
        m_row = None if m_batch is None else m_batch[b_idx].to(device=coors_nm.device, dtype=torch.bool)

        if m_row is None or bool(m_row.all().item()):
            sym_row = symmetrize_decoded_atom37(
                c_row.unsqueeze(0),
                cfg,
                ca_coors_nm=None if ca_row is None else ca_row.unsqueeze(0),
                atom_mask=None if am_row is None else am_row.unsqueeze(0),
                mask=None,
                mode=cfg.decode_mode,
            ).squeeze(0)
            rows.append(sym_row)
        else:
            idx = torch.nonzero(m_row, as_tuple=False).squeeze(-1)
            if int(idx.numel()) == 0:
                rows.append(c_row)
                continue
            c_sub = c_row.index_select(0, idx).unsqueeze(0)
            ca_sub = None if ca_row is None else ca_row.index_select(0, idx).unsqueeze(0)
            am_sub = None if am_row is None else am_row.index_select(0, idx).unsqueeze(0)
            sym_sub = symmetrize_decoded_atom37(
                c_sub,
                cfg,
                ca_coors_nm=ca_sub,
                atom_mask=am_sub,
                mask=None,
                mode=cfg.decode_mode,
            ).squeeze(0)
            rows.append(c_row.scatter(0, idx.view(-1, 1, 1).expand_as(sym_sub), sym_sub))

    out = torch.stack(rows, dim=0)
    return out.squeeze(0) if unbatched else out


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
        self, x: Float[Tensor, "* n d"], emit_mask: Bool[Tensor, "* n"] | None = None
    ) -> Float[Tensor, "* n d"]:
        """
        Centers tensor over n (residue) dimension.

        Args:
            x: Tensor of shape [*, n, d]
            emit_mask: Tokens emitted by this modality, shape [*, n].

        Returns:
            Centered x = x - mean(x, dim=-2), shape [*, n, d].
        """
        if emit_mask is None:
            x = x - torch.mean(x, dim=-2, keepdim=True)
        else:
            x = (x - mean_w_mask(x, emit_mask, keepdim=True)) * emit_mask[..., None]
        return x

    def _apply_emit_mask(
        self, x: Float[Tensor, "* n d"], emit_mask: Bool[Tensor, "* n"] | None = None
    ) -> Float[Tensor, "* n d"]:
        """Set values outside the modality's emitted tokens to zero.

        Args:
            x: Tensor of shape [*, n, d]
            emit_mask: Tokens emitted by this modality, shape [*, n].

        Returns:
            Masked x of shape [*, n, d]
        """
        if emit_mask is None:
            return x
        return x * emit_mask[..., None]  # [*, n, d]

    def _apply_stochastic_offset(
        self,
        x: Float[Tensor, "* n d"],
        emit_mask: Bool[Tensor, "* n"] | None,
        offset_mask: Bool[Tensor, "* n"] | None,
        group_ids: Int[Tensor, "* n"] | None = None,
    ) -> Float[Tensor, "* n d"]:
        """Apply per-group random offsets to gen tokens.

        Each unique value in ``group_ids`` within ``offset_mask`` receives its own
        independent per-batch-element random translation. Tokens outside ``offset_mask``
        are never moved.

        Args:
            x: Coordinates, shape [*, n, d].
            emit_mask: Tokens emitted by this modality, shape [*, n].
            offset_mask: Tokens eligible for offsetting, shape [*, n]. If ``None``,
                returns ``x`` unchanged.
            group_ids: Integer group label per token, shape [*, n]. If ``None``, all
                gen tokens are treated as a single group.

        Returns:
            x with stochastic offsets applied to gen tokens, shape [*, n, d].
        """
        if offset_mask is None:
            return x
        if group_ids is None:
            noise = torch.randn(*x.shape[:-2], x.shape[-1], device=x.device)  # [*, d]
            x = x + self.stochastic_centering_scale * noise.unsqueeze(-2) * offset_mask[..., None].float()
        else:
            for g in group_ids[offset_mask].unique():
                group_mask_f = (offset_mask & (group_ids == g))[..., None].float()  # [*, n, 1]
                noise = torch.randn(*x.shape[:-2], x.shape[-1], device=x.device)  # [*, d]
                x = x + self.stochastic_centering_scale * noise.unsqueeze(-2) * group_mask_f

        if emit_mask is not None:
            x = x * emit_mask[..., None]
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
        emit_mask: Bool[Tensor, "* n"] | None = None,
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
            emit_mask: Tokens emitted by this modality, shape [*, n].
            coord_cond_mask: True for absolute-coordinate-conditioned tokens that should
                not be offset. Shape [*, n].
            groups: Dict of token-level group tensors keyed by name (``chain_iid``,
                ``pn_unit_iid``, ``molecule_iid``, ``coord_group``). Required for
                the ``"per_group"`` scope; ignored otherwise. Each value has shape [*, n].

        Returns:
            x with stochastic noise added, shape [*, n, d].
        """
        offset_mask = (emit_mask & ~coord_cond_mask) if coord_cond_mask is not None else emit_mask

        scope = self.stochastic_centering_scope

        if scope == "all":
            x = self._apply_stochastic_offset(x, emit_mask, offset_mask=emit_mask)
        elif scope == "without_absolute_coords":
            x = self._apply_stochastic_offset(x, emit_mask, offset_mask=offset_mask)
        elif scope == "per_group":
            group_ids = self._compose_group_ids(groups)
            x = self._apply_stochastic_offset(x, emit_mask, offset_mask=offset_mask, group_ids=group_ids)
        else:
            raise ValueError(f"Unknown stochastic_centering_scope: {scope!r}")

        return x

    def sample_noise(
        self,
        n: int,
        device: torch.device,
        shape: tuple = (),
        emit_mask: Bool[Tensor, "* n"] | None = None,
        training: bool = True,
        coord_cond_mask: Bool[Tensor, "* n"] | None = None,
        groups: dict[str, Int[Tensor, "* n"] | None] | None = None,
        symmetry: Any = None,
        apply_radius: bool = True,
    ) -> Float[Tensor, "* n d"]:
        """
        Samples reference distribution std Gaussian (possibly centered).

        Args:
            n: number of frames in a single sample, int
            device: torch device
            shape: tuple (if empty then single sample)
            emit_mask: Tokens emitted by this modality, shape [*, n].
            training (optional): whether in training mode
            coord_cond_mask (optional): Boolean mask of shape [*, n]. True for
                coordinate-conditioned tokens whose noise should not be offset.
            groups (optional): Dict of token-level group tensors keyed by name
                (``chain_iid``, ``pn_unit_iid``, ``molecule_iid``, ``coord_group``).
                Used by the ``"per_group"`` stochastic centering scope.
            symmetry (optional): Point-group symmetry specification (single or per-row).
            apply_radius (optional): If True, shift ASU initial noise by ``radius_nm * u_G``.

        Returns:
            Samples from refenrece [N(0, I_d)]^n shape [*shape, n, d]
        """
        batch_size = int(shape[0]) if shape else 1
        sym_cfgs = normalize_symmetry_configs(symmetry, batch_size=batch_size) if self.dim == 3 else None

        if sym_cfgs is None:
            noise = torch.randn(
                (*shape, n, self.dim),
                device=device,
            )
            noise = self._apply_emit_mask(noise, emit_mask)
            if self.zero_com_noise:
                noise = self._force_zero_com(noise, emit_mask)

            # Apply stochastic centering at inference time if mode is x_0
            if self.stochastic_centering_mode == "x_0" and not training:
                noise = self.add_stochastic_noise(noise, emit_mask, coord_cond_mask=coord_cond_mask, groups=groups)

            return noise

        abs_flags = _row_has_abs_coord_cond(coord_cond_mask, emit_mask, batch_size)
        is_uniform = all(c == sym_cfgs[0] for c in sym_cfgs[1:]) and len(set(abs_flags)) == 1

        if is_uniform:
            cfg0 = sym_cfgs[0]
            rad_nm = cfg0.radius_nm if apply_radius else 0.0
            noise = sample_symmetric_ca_noise(
                (*shape, n, 3),
                cfg0,
                mask=emit_mask,
                radius_nm=rad_nm,
                zero_com=bool(self.zero_com_noise and not abs_flags[0]),
                device=device,
                dtype=torch.float32,
            )
            return self._apply_emit_mask(noise, emit_mask)

        # Heterogeneous batch (e.g. mixed symmetry groups or mixed C1 / symmetric rows)
        has_inactive = any(not c.is_active for c in sym_cfgs)
        if has_inactive:
            noise = torch.randn((*shape, n, 3), device=device)
            noise = self._apply_emit_mask(noise, emit_mask)
            if self.zero_com_noise:
                noise = self._force_zero_com(noise, emit_mask)
            if self.stochastic_centering_mode == "x_0" and not training:
                noise = self.add_stochastic_noise(noise, emit_mask, coord_cond_mask=coord_cond_mask, groups=groups)
        else:
            noise = torch.zeros((*shape, n, 3), device=device, dtype=torch.float32)

        for b_idx, cfg in enumerate(sym_cfgs):
            if not cfg.is_active:
                continue
            row_mask = None if emit_mask is None else (emit_mask if not shape else emit_mask[b_idx])
            row_noise = sample_symmetric_ca_noise(
                (n, 3) if shape else (*shape, n, 3),
                cfg,
                mask=row_mask,
                radius_nm=cfg.radius_nm if apply_radius else 0.0,
                zero_com=bool(self.zero_com_noise and not abs_flags[b_idx]),
                device=device,
                dtype=torch.float32,
            )
            if shape:
                noise[b_idx] = row_noise
            else:
                noise = row_noise

        return self._apply_emit_mask(noise, emit_mask)

    def interpolate(
        self,
        x_0: Float[Tensor, "* n d"],
        x_1: Float[Tensor, "* n d"],
        t: Float[Tensor, "*"],
        emit_mask: Bool[Tensor, "* n"] | None = None,
        coord_cond_mask: Bool[Tensor, "* n"] | None = None,
        groups: dict[str, Int[Tensor, "* n"] | None] | None = None,
        symmetry: Any = None,
    ) -> Float[Tensor, "* n d"]:
        """
        Interpolates between rigids x_0 (base) and x_1 (data) using t.

        Args:
            x_0: Tensor sampled from reference, shape [*, n, d]
            x_1: Tensor sampled from target, shape [*, n, d]
            t: Interpolation times, shape [*]
            emit_mask: Tokens emitted by this modality, shape [*, n].
            coord_cond_mask (optional): Boolean mask of shape [*, n]. True for
                coordinate-conditioned tokens whose noise should not be offset.
            groups (optional): Dict of token-level group tensors keyed by name
                (``chain_iid``, ``pn_unit_iid``, ``molecule_iid``, ``coord_group``).
                Used by the ``"per_group"`` stochastic centering scope.
            symmetry (optional): Point-group symmetry specification (single or per-row).

        Returns:
            x_t: Interpolated tensor, shape [*, n, d]
        """
        # x_0 is already zero outside emitted tokens; x_1 depends on the dataloader.
        # x_0 maybe centered (depends on input arg zero_com_noise), x_1 depends on dataloader
        x_0, x_1 = self._apply_emit_mask(x_0, emit_mask), self._apply_emit_mask(x_1, emit_mask)

        batch_size = int(x_0.shape[0]) if x_0.ndim >= 3 else 1
        sym_cfgs = normalize_symmetry_configs(symmetry, batch_size=batch_size) if self.dim == 3 else None

        if sym_cfgs is None:
            # Apply stochastic centering based on mode (during training)
            if self.stochastic_centering_mode == "x_0":
                x_0 = self.add_stochastic_noise(x_0, emit_mask, coord_cond_mask=coord_cond_mask, groups=groups)
            elif self.stochastic_centering_mode == "x_1":
                x_1 = self.add_stochastic_noise(x_1, emit_mask, coord_cond_mask=coord_cond_mask, groups=groups)

            t_exp = t[..., None, None]
            # No additional masking is needed because both endpoints already obey the emit mask.
            x_t = (1.0 - t_exp) * x_0 + t_exp * x_1

            # Apply stochastic centering to interpolated result if mode is x_t
            if self.stochastic_centering_mode == "x_t":
                x_t = self.add_stochastic_noise(x_t, emit_mask, coord_cond_mask=coord_cond_mask, groups=groups)

            return x_t

        # Never apply per-chain stochastic centering offsets on active symmetric rows.
        if all(c.is_active for c in sym_cfgs):
            t_exp = t[..., None, None]
            x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
            x_t = project_symmetric_3d_batched(x_t, sym_cfgs, emit_mask)
            return self._apply_emit_mask(x_t, emit_mask)

        # Mixed batch with both C1 and active symmetric rows
        x_0_std, x_1_std = x_0.clone(), x_1.clone()
        if self.stochastic_centering_mode == "x_0":
            x_0_std = self.add_stochastic_noise(x_0_std, emit_mask, coord_cond_mask=coord_cond_mask, groups=groups)
        elif self.stochastic_centering_mode == "x_1":
            x_1_std = self.add_stochastic_noise(x_1_std, emit_mask, coord_cond_mask=coord_cond_mask, groups=groups)
        t_exp = t[..., None, None]
        x_t_std = (1.0 - t_exp) * x_0_std + t_exp * x_1_std
        if self.stochastic_centering_mode == "x_t":
            x_t_std = self.add_stochastic_noise(x_t_std, emit_mask, coord_cond_mask=coord_cond_mask, groups=groups)

        x_t_raw = (1.0 - t_exp) * x_0 + t_exp * x_1
        out = x_t_std.clone()
        for b_idx, cfg in enumerate(sym_cfgs):
            if cfg.is_active:
                row_m = None if emit_mask is None else emit_mask[b_idx]
                out[b_idx] = project_symmetric(x_t_raw[b_idx], cfg, mask=row_m)
        return self._apply_emit_mask(out, emit_mask)

    def nn_out_add_clean_sample_prediction(
        self,
        x_t: Float[Tensor, "* n d"],
        t: Float[Tensor, "*"],
        emit_mask: Bool[Tensor, "* n"],
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
        nn_out["x_1"] = nn_out["x_1"] * emit_mask[..., None]
        return nn_out

    def nn_out_add_simulation_tensor(
        self,
        x_t: Float[Tensor, "* n d"],
        t: Float[Tensor, "*"],
        emit_mask: Bool[Tensor, "* n"],
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
        nn_out["v"] = nn_out["v"] * emit_mask[..., None]
        return nn_out

    def compute_fm_loss(
        self,
        x_0: Float[Tensor, "* n d"],
        x_1: Float[Tensor, "* n d"],
        x_t: Float[Tensor, "* n d"],
        emit_mask: Bool[Tensor, "* n"],
        supervision_mask: Bool[Tensor, "* n"],
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
            emit_mask: Tokens emitted by this modality, shape [*, n].
            supervision_mask: Generated, resolved tokens included in the loss, shape [*, n].
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
            emit_mask=emit_mask,
            nn_out=nn_out,
        )
        nres = torch.sum(supervision_mask, dim=-1).clamp(min=1)  # [*] - avoid division by zero
        err = (x_1 - nn_out["x_1"]) * supervision_mask[..., None]  # [*, n, d]
        if weight is None:
            weighted_sq = err**2  # [*, n, d]
        else:
            w = weight.to(dtype=x_1.dtype) * supervision_mask  # [*, n], gated weights
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
        emit_mask: Bool[Tensor, "* n"],
        simulation_step_params: dict,
        coord_cond_mask: Bool[Tensor, "* n"] | None = None,
        symmetry: Any = None,
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

        eq. (3): d x_t = [v(x_t, t) + g(t) * s(x_t, t)] dt + \
                        \sqrt{2 * g(t) * sc_noise_scale} dw_t.

        At the moment we do not scale the vector field v.

        Args:
            x_t: Current value, shape [*, n, d]
            nn_out: Dictionary with all available predictions, should include "v" and possibly guided "v_guided".
            May include "x_1", etc as well. All shape [*, n, d]
            t: Current time, shape [*]
            dt: Step-size, float
            gt: Noise injection, float
            emit_mask: Tokens emitted by this modality, shape [*, n].
            simulation_step_params: parameters for the simulation step, keys
                sampling_mode, sc_scale_noise, t_lim_ode
            coord_cond_mask (optional): Fixed-coordinate conditioning mask, shape [*, n].
            symmetry (optional): Point-group symmetry specification (single or per-row).

        Returns:
            Updated values for x_t after an Euler integration step, shape [*, n, d].
        """
        sampling_mode = simulation_step_params["sampling_mode"]
        sc_scale_noise = simulation_step_params["sc_scale_noise"]
        t_lim_ode = simulation_step_params["t_lim_ode"]
        t_lim_ode_below = simulation_step_params["t_lim_ode_below"]
        center_every_step = simulation_step_params["center_every_step"]
        tsr_k = simulation_step_params.get("tsr_k", 1.0)
        tsr_sigma = simulation_step_params.get("tsr_sigma", 1.0)

        assert sampling_mode in [
            "vf",
            "sc",
            "vf_tsr",
        ], f"Invalid sampling mode {sampling_mode}, should be `vf`, `sc`, or `vf_tsr`"
        assert sc_scale_noise >= 0, f"Scale noise for sampling should be >= 0, got {sc_scale_noise}"
        assert gt >= 0, f"gt for sampling should be >= 0, got {gt}"
        t_element = t.flatten()[0]
        assert torch.all(t_element == t), "Sampling only implemented for same time for all samples"

        if self.guidance_enabled:
            v = nn_out["v_guided"]
        else:
            v = nn_out["v"]

        batch_size = int(x_t.shape[0]) if x_t.ndim >= 3 else 1
        sym_cfgs = normalize_symmetry_configs(symmetry, batch_size=batch_size) if self.dim == 3 else None

        if sym_cfgs is not None:
            x_t = project_symmetric_3d_batched(x_t, sym_cfgs, emit_mask)
            v = project_symmetric_3d_batched(v, sym_cfgs, emit_mask)

        def _sample_step_eps() -> Tensor:
            if sym_cfgs is None:
                return torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)
            abs_flags = _row_has_abs_coord_cond(coord_cond_mask, emit_mask, batch_size)
            is_uniform = all(c == sym_cfgs[0] for c in sym_cfgs[1:]) and len(set(abs_flags)) == 1
            if is_uniform:
                return sample_symmetric_ca_noise(
                    x_t.shape,
                    sym_cfgs[0],
                    mask=emit_mask,
                    radius_nm=0.0,
                    zero_com=bool(self.zero_com_noise and not abs_flags[0]),
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
            eps_out = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)
            n_tok = x_t.shape[-2]
            for b_idx, cfg in enumerate(sym_cfgs):
                if not cfg.is_active:
                    continue
                row_m = None if emit_mask is None else (emit_mask if x_t.ndim == 2 else emit_mask[b_idx])
                row_eps = sample_symmetric_ca_noise(
                    (n_tok, 3) if x_t.ndim >= 3 else x_t.shape,
                    cfg,
                    mask=row_m,
                    radius_nm=0.0,
                    zero_com=bool(self.zero_com_noise and not abs_flags[b_idx]),
                    device=x_t.device,
                    dtype=x_t.dtype,
                )
                if x_t.ndim >= 3:
                    eps_out[b_idx] = row_eps
                else:
                    eps_out = row_eps
            return eps_out

        endpoint_score_scale = 1.5  # Preserve regular SDE's low-temperature ODE endpoint.

        if sampling_mode == "vf":  # ODE mode
            delta_x = v * dt

        elif sampling_mode == "sc":  # SDE with noise scaling
            if t_element > t_lim_ode:  # close to 1 stop SDE, switch to low temp ODE
                score = vf_to_score(x_t, v, t)
                scaled_score = score * endpoint_score_scale
                v_scaled = score_to_vf(x_t, scaled_score, t)
                delta_x = v_scaled * dt
            else:
                score = vf_to_score(x_t, v, t)  # get score from v, [*, dim]
                eps = _sample_step_eps()
                std_eps = torch.sqrt(torch.as_tensor(2 * gt * sc_scale_noise * dt, device=x_t.device, dtype=x_t.dtype))
                delta_x = (v + gt * score) * dt + std_eps * eps

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
                    eps = _sample_step_eps()
                    std_eps = torch.sqrt(
                        torch.as_tensor(2 * gt * sc_scale_noise * dt, device=x_t.device, dtype=x_t.dtype)
                    )
                    delta_x = (v_scaled + gt * scaled_score) * dt + std_eps * eps

        else:
            raise ValueError(f"Invalid sampling mode {sampling_mode}")

        x_next = x_t + delta_x

        # Apply the emit mask and potentially center.
        x_next = self._apply_emit_mask(x_next, emit_mask)
        if sym_cfgs is not None:
            x_next = project_symmetric_3d_batched(
                x_next,
                sym_cfgs,
                emit_mask,
                zero_com=bool(center_every_step),
                coord_cond_mask=coord_cond_mask,
            )
            return self._apply_emit_mask(x_next, emit_mask)

        if center_every_step:
            x_next = self._force_zero_com(x_next, emit_mask)
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
