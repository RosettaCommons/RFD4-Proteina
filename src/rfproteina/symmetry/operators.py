"""Equivariant and invariant tensor operators for 3D point-group oligomer symmetry."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from torch import Tensor

from rfproteina.symmetry.point_groups import (
    VALID_DECODE_MODES,
    SymmetryConfig,
    SymmetrySpec,
    parse_symmetry,
)

SymmetryLike = Tensor | SymmetrySpec | SymmetryConfig | str | None
BatchSymmetryLike = SymmetryLike | Sequence[SymmetryLike]


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promote low-precision floats to float32 for accurate orthogonal matrix arithmetic."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def _resolve_rotations(
    group: SymmetryLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Resolve a group specification or raw ``[K, 3, 3]`` tensor onto ``(device, dtype)``."""
    if isinstance(group, Tensor):
        if group.ndim != 3 or group.shape[-2:] != (3, 3) or group.shape[0] < 1:
            raise ValueError(f"Rotation tensor must have shape [K, 3, 3] with K >= 1, got {tuple(group.shape)}")
        return group.to(device=device, dtype=dtype)
    spec = parse_symmetry(group)
    return spec.rotations(device=device, dtype=dtype)


def _resolve_radial_dir(
    group: SymmetryLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Resolve a collision-free unit radial vector ``[3]`` for ASU offset placement."""
    if isinstance(group, Tensor):
        rot_f64 = group.detach().to(device="cpu", dtype=torch.float64)
        if rot_f64.shape[0] <= 1:
            return torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        candidates = [
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            torch.tensor([math.cos(0.37), math.sin(0.37), 0.5], dtype=torch.float64),
            torch.tensor([1.0, 2.0, 3.5], dtype=torch.float64),
            torch.tensor([0.31, 0.73, 1.19], dtype=torch.float64),
        ]
        best_u = candidates[0]
        best_sep = -1.0
        for cand in candidates:
            u_cand = cand / torch.linalg.norm(cand)
            disp = torch.einsum("kij,j->ki", rot_f64[1:], u_cand) - u_cand.unsqueeze(0)
            sep = float(torch.linalg.norm(disp, dim=-1).min().item())
            if sep > best_sep:
                best_sep = sep
                best_u = u_cand
        return best_u.to(device=device, dtype=dtype)
    spec = parse_symmetry(group)
    return spec.radial_dir(device=device, dtype=dtype)


def _resolve_order(group: int | SymmetryLike) -> int:
    """Resolve a group order ``K >= 1`` from an integer or ``SymmetryLike`` object."""
    if isinstance(group, int) and not isinstance(group, bool):
        if group < 1:
            raise ValueError(f"Symmetry order must be >= 1, got {group}")
        return group
    if isinstance(group, Tensor):
        if group.ndim != 3 or group.shape[-2:] != (3, 3) or group.shape[0] < 1:
            raise ValueError(f"Rotation tensor must have shape [K, 3, 3], got {tuple(group.shape)}")
        return int(group.shape[0])
    return parse_symmetry(group).order


def _is_per_batch_sequence(group: Any) -> bool:
    """Return True if ``group`` is a non-string Python sequence of per-batch symmetry specs."""
    return isinstance(group, Sequence) and not isinstance(group, (str, bytes, bytearray, Tensor))


def _is_dict_like(obj: Any) -> bool:
    """Return True for ``Mapping`` and dict-like batch containers (e.g. ``PaddedBatchedDataSample``)."""
    if isinstance(obj, Mapping):
        return True
    if isinstance(obj, (str, bytes, bytearray, Tensor, SymmetryConfig, SymmetrySpec)):
        return False
    return hasattr(obj, "get") and hasattr(obj, "__contains__") and hasattr(obj, "__getitem__")


def resolve_batch_symmetry(
    batch_or_val: Any,
    fallback: Any = None,
    *,
    batch_size: int | None = None,
) -> list[SymmetryConfig]:
    """Extract per-example :class:`SymmetryConfig` objects from a batch dict, sequence, or config.

    Supports plain dicts, ``PaddedBatchedDataSample``, dict-of-lists from ``batch_dataclass_recursive``,
    and heterogeneous mixed batches. Checks ``batch["symmetry"]``, ``batch["extra_info"]["symmetry"]``,
    and ``batch["specification_batch"]`` before falling back to ``fallback``.
    """
    default_cfg = SymmetryConfig.from_any(fallback)

    def _from_item(item: Any) -> SymmetryConfig:
        if item is None:
            return default_cfg
        cfg = SymmetryConfig.from_any(item, default_radius=default_cfg.radius if default_cfg.radius > 0.0 else None)
        if cfg.group is None and default_cfg.is_active:
            return default_cfg
        return cfg

    eff_batch_size = batch_size
    extracted: Any = None
    if _is_dict_like(batch_or_val):
        is_direct_sym_dict = (
            "group" in batch_or_val
            or "decode_mode" in batch_or_val
            or "input_is_asu" in batch_or_val
            or "tie_sequence" in batch_or_val
        )
        if not is_direct_sym_dict and eff_batch_size is None:
            pm = batch_or_val.get("padding_mask") if "padding_mask" in batch_or_val else None
            if isinstance(pm, Tensor) and pm.ndim >= 1:
                eff_batch_size = int(pm.shape[0])
            else:
                for id_key in ("example_id_batch", "example_id"):
                    id_val = batch_or_val.get(id_key) if id_key in batch_or_val else None
                    if _is_per_batch_sequence(id_val):
                        eff_batch_size = len(id_val)
                        break

        if is_direct_sym_dict:
            extracted = batch_or_val
        elif "symmetry" in batch_or_val and batch_or_val.get("symmetry") is not None:
            extracted = batch_or_val["symmetry"]
        elif _is_dict_like(batch_or_val.get("extra_info")) and "symmetry" in batch_or_val["extra_info"]:
            extracted = batch_or_val["extra_info"]["symmetry"]
        elif _is_per_batch_sequence(batch_or_val.get("extra_info")):
            extracted = [
                ei.get("symmetry") if _is_dict_like(ei) else None for ei in batch_or_val["extra_info"]
            ]
        elif (
            "specification_batch" in batch_or_val and batch_or_val.get("specification_batch") is not None
        ) or ("specification" in batch_or_val and batch_or_val.get("specification") is not None):
            spec_b = batch_or_val.get("specification_batch") or batch_or_val.get("specification")
            if _is_per_batch_sequence(spec_b):
                extracted = [
                    sp.get("symmetry") if _is_dict_like(sp) and "symmetry" in sp else None for sp in spec_b
                ]
            elif _is_dict_like(spec_b) and "symmetry" in spec_b:
                extracted = spec_b["symmetry"]
    else:
        extracted = batch_or_val

    if _is_per_batch_sequence(extracted):
        items = list(extracted)
        if eff_batch_size is not None and len(items) != eff_batch_size:
            if len(items) == 1:
                items = items * eff_batch_size
            else:
                raise ValueError(f"Batch symmetry list length {len(items)} != batch_size {eff_batch_size}")
        return [_from_item(it) for it in items]

    if _is_dict_like(extracted):
        # Handle collated dict-of-lists (e.g. {"group": ["C3", "D2"], "radius": [11.0, 14.0], ...})
        n_rows: int | None = None
        for probe_key in ("group", "radius", "decode_mode", "input_is_asu", "tie_sequence", "order"):
            probe_val = extracted.get(probe_key) if probe_key in extracted else None
            if _is_per_batch_sequence(probe_val):
                n_rows = len(probe_val)
                break
            if isinstance(probe_val, Tensor) and probe_val.ndim >= 1:
                n_rows = int(probe_val.shape[0])
                break

        if n_rows is not None:
            if eff_batch_size is not None and n_rows != eff_batch_size:
                if n_rows == 1:
                    single_row: dict[str, Any] = {}
                    for key, value in extracted.items():
                        if isinstance(value, Tensor) and value.ndim >= 1 and value.shape[0] == 1:
                            single_row[key] = value[0].item() if value[0].numel() == 1 else value[0]
                        elif _is_per_batch_sequence(value) and len(value) == 1:
                            single_row[key] = value[0]
                        else:
                            single_row[key] = value
                    cfg0 = _from_item(single_row)
                    return [cfg0 for _ in range(eff_batch_size)]
                raise ValueError(f"Batch symmetry dict-of-lists length {n_rows} != batch_size {eff_batch_size}")

            out_cfgs: list[SymmetryConfig] = []
            for idx in range(n_rows):
                row_dict: dict[str, Any] = {}
                for key, value in extracted.items():
                    if isinstance(value, Tensor) and value.ndim >= 1 and value.shape[0] == n_rows:
                        row_dict[key] = value[idx].item() if value[idx].numel() == 1 else value[idx]
                    elif _is_per_batch_sequence(value) and len(value) == n_rows:
                        row_dict[key] = value[idx]
                    else:
                        row_dict[key] = value
                out_cfgs.append(_from_item(row_dict))
            return out_cfgs

    single = _from_item(extracted)
    count = 1 if eff_batch_size is None else eff_batch_size
    return [single for _ in range(count)]


def has_active_symmetry(configs: SymmetryLike | Sequence[SymmetryLike]) -> bool:
    """Return True if at least one configuration specifies nontrivial symmetry (``order > 1``)."""
    if _is_per_batch_sequence(configs):
        return any(has_active_symmetry(c) for c in configs)
    return _resolve_order(configs) > 1


def _resolve_cartesian_seq_axis(x: Tensor, *, mask: Tensor | None, seq_dim: int | None) -> int:
    """Resolve the sequence/residue axis of a 3D Cartesian tensor ``x`` (``[..., N, 3]`` or ``[..., N, A, 3]``)."""
    if x.ndim < 2 or x.shape[-1] != 3:
        raise ValueError(f"Expected Cartesian tensor with trailing dim 3 and ndim >= 2, got {tuple(x.shape)}")
    if seq_dim is not None:
        axis = seq_dim if seq_dim >= 0 else x.ndim + seq_dim
    elif mask is not None:
        axis = mask.ndim - 1
    elif x.ndim >= 4 and x.shape[-2] == 37:
        axis = x.ndim - 3
    else:
        axis = x.ndim - 2

    if not 0 <= axis < x.ndim - 1:
        raise ValueError(f"Invalid seq_dim={seq_dim} (resolved axis={axis}) for tensor shape {tuple(x.shape)}")
    return axis


def _uniform_prefix_length(mask_b: Tensor) -> int | None:
    """If ``mask_b`` has an identical contiguous prefix ``[:c]`` across all batch rows, return ``c``; else ``None``."""
    if mask_b.numel() == 0:
        return 0
    counts = mask_b.sum(dim=-1)
    c0 = int(counts.reshape(-1)[0].item())
    if not bool((counts == c0).all().item()):
        return None
    if c0 == 0:
        return 0
    if bool(mask_b[..., :c0].all().item()):
        return c0
    return None


def _rotate_asu_unmasked(x_asu: Tensor, rot: Tensor, *, seq_axis: int) -> Tensor:
    """Expand unmasked ASU Cartesian tensor ``x_asu`` along ``seq_axis`` using ``rot`` ``[K, 3, 3]``."""
    order = rot.shape[0]
    orig_dtype = x_asu.dtype
    work_dtype = _compute_dtype(orig_dtype)
    x_work = x_asu.to(dtype=work_dtype)
    rot_work = rot.to(device=x_asu.device, dtype=work_dtype)

    leading = x_work.shape[:seq_axis]
    l_asu = x_work.shape[seq_axis]
    middle = x_work.shape[seq_axis + 1 : -1]  # () or (A,)

    # Move [seq_axis, *middle] after leading, flatten middle if present: [*leading, L_asu * M, 3]
    m_prod = math.prod(middle) if middle else 1
    x_flat = x_work.reshape(*leading, l_asu * m_prod, 3)
    expanded = torch.einsum("kij,...mj->...kmi", rot_work, x_flat)  # [*leading, K, L_asu * M, 3]
    out = expanded.reshape(*leading, order * l_asu, *middle, 3)
    return out.to(dtype=orig_dtype)


def _reduce_asu_unmasked(
    x_full: Tensor,
    rot: Tensor,
    *,
    seq_axis: int,
    preserve_variance: bool = False,
) -> Tensor:
    """Inverse-rotate and reduce unmasked full oligomer tensor ``x_full`` back to ASU along ``seq_axis``."""
    order = rot.shape[0]
    n_total = x_full.shape[seq_axis]
    if n_total % order != 0:
        raise ValueError(f"Sequence length {n_total} along axis {seq_axis} is not divisible by group order {order}")
    l_asu = n_total // order

    orig_dtype = x_full.dtype
    work_dtype = _compute_dtype(orig_dtype)
    x_work = x_full.to(dtype=work_dtype)
    rot_work = rot.to(device=x_full.device, dtype=work_dtype)

    leading = x_work.shape[:seq_axis]
    middle = x_work.shape[seq_axis + 1 : -1]
    m_prod = math.prod(middle) if middle else 1

    x_grouped = x_work.reshape(*leading, order, l_asu * m_prod, 3)
    summed = torch.einsum("kij,...kmi->...mj", rot_work, x_grouped)
    denom = math.sqrt(float(order)) if preserve_variance else float(order)
    out = (summed / denom).reshape(*leading, l_asu, *middle, 3)
    return out.to(dtype=orig_dtype)


def expand_asu(
    x_asu: Tensor,
    group: SymmetryLike,
    *,
    mask: Tensor | None = None,
    seq_dim: int | None = None,
) -> Tensor:
    """Expand ASU 3D coordinates or velocities ``[..., L_asu, (A,) 3]`` to ``[..., K * L_asu, (A,) 3]``.

    Subunit ``g`` occupies slice ``[..., g * L_asu : (g + 1) * L_asu, ...]`` with ``X_g = R_g X_asu``.
    If ``mask`` has ASU shape ``[..., L_asu]``, masked positions are zeroed before expansion; if ``mask``
    has padded full-oligomer shape ``[..., N_max]`` (where each row has ``K * L_asu_b`` valid tokens),
    expanded tokens are placed into the valid mask positions and padded tokens are zeroed.
    """
    axis = _resolve_cartesian_seq_axis(x_asu, mask=None if mask is not None else None, seq_dim=seq_dim)
    if mask is not None and mask.ndim == axis + 1 and mask.shape[axis] != x_asu.shape[axis]:
        # Full-oligomer target mask [..., N_max]
        rot = _resolve_rotations(group, device=x_asu.device, dtype=_compute_dtype(x_asu.dtype))
        order = rot.shape[0]
        mask_b = mask.to(device=x_asu.device, dtype=torch.bool)
        n_max = mask_b.shape[-1]
        prefix = _uniform_prefix_length(mask_b)
        if prefix is not None:
            if prefix == 0:
                out_shape = (*x_asu.shape[:axis], n_max, *x_asu.shape[axis + 1 :])
                return torch.zeros(out_shape, device=x_asu.device, dtype=x_asu.dtype)
            if prefix % order != 0:
                raise ValueError(f"Valid token count {prefix} is not divisible by symmetry order {order}")
            l_valid = prefix // order
            expanded_valid = _rotate_asu_unmasked(
                x_asu.narrow(axis, 0, l_valid),
                rot,
                seq_axis=axis,
            )
            if prefix == n_max:
                return expanded_valid
            pad_shape = (*x_asu.shape[:axis], n_max - prefix, *x_asu.shape[axis + 1 :])
            zeros = torch.zeros(pad_shape, device=x_asu.device, dtype=x_asu.dtype)
            return torch.cat([expanded_valid, zeros], dim=axis)

        leading = x_asu.shape[:axis]
        trailing = x_asu.shape[axis + 1 :]
        b_flat = math.prod(leading) if leading else 1
        x_flat = x_asu.reshape(b_flat, x_asu.shape[axis], *trailing)
        m_flat = mask_b.reshape(b_flat, n_max)
        out_flat = torch.zeros((b_flat, n_max, *trailing), device=x_asu.device, dtype=x_asu.dtype)
        for r in range(b_flat):
            idx = torch.nonzero(m_flat[r], as_tuple=False).squeeze(-1)
            n_val = int(idx.numel())
            if n_val == 0:
                continue
            if n_val % order != 0:
                raise ValueError(f"Row {r} valid token count {n_val} is not divisible by symmetry order {order}")
            l_val = n_val // order
            exp_r = _rotate_asu_unmasked(x_flat[r, :l_val], rot, seq_axis=0)
            out_flat[r].index_copy_(0, idx, exp_r)
        return out_flat.reshape(*leading, n_max, *trailing)

    rot = _resolve_rotations(group, device=x_asu.device, dtype=_compute_dtype(x_asu.dtype))
    x_in = x_asu
    if mask is not None:
        mask_b = mask.to(device=x_asu.device, dtype=torch.bool)
        if mask_b.shape != x_asu.shape[: axis + 1]:
            raise ValueError(f"Mask shape {tuple(mask_b.shape)} incompatible with x_asu shape {tuple(x_asu.shape)}")
        expand_dims = (1,) * (x_asu.ndim - mask_b.ndim)
        x_in = x_asu * mask_b.view(*mask_b.shape, *expand_dims)
    return _rotate_asu_unmasked(x_in, rot, seq_axis=axis)


def reduce_to_asu(
    x_full: Tensor,
    group: SymmetryLike,
    *,
    mask: Tensor | None = None,
    seq_dim: int | None = None,
    preserve_variance: bool = False,
) -> Tensor:
    """Pull all ``K`` subunits back to the canonical ASU frame and average (``(1/K) sum_g R_g^T X_g``).

    When ``preserve_variance=True``, divides by ``sqrt(K)`` instead of ``K`` so i.i.d. unit-variance
    Gaussian noise retains unit variance on the ASU.
    """
    axis = _resolve_cartesian_seq_axis(x_full, mask=mask, seq_dim=seq_dim)
    rot = _resolve_rotations(group, device=x_full.device, dtype=_compute_dtype(x_full.dtype))
    order = rot.shape[0]

    if mask is None:
        return _reduce_asu_unmasked(x_full, rot, seq_axis=axis, preserve_variance=preserve_variance)

    mask_b = mask.to(device=x_full.device, dtype=torch.bool)
    if mask_b.shape != x_full.shape[: axis + 1]:
        raise ValueError(
            f"Mask shape {tuple(mask_b.shape)} does not match x_full prefix {tuple(x_full.shape[:axis+1])}"
        )

    prefix = _uniform_prefix_length(mask_b)
    if prefix is not None:
        if prefix == 0:
            out_shape = (*x_full.shape[:axis], 0, *x_full.shape[axis + 1 :])
            return torch.zeros(out_shape, device=x_full.device, dtype=x_full.dtype)
        return _reduce_asu_unmasked(
            x_full.narrow(axis, 0, prefix),
            rot,
            seq_axis=axis,
            preserve_variance=preserve_variance,
        )

    leading = x_full.shape[:axis]
    trailing = x_full.shape[axis + 1 :]
    b_flat = math.prod(leading) if leading else 1
    x_flat = x_full.reshape(b_flat, x_full.shape[axis], *trailing)
    m_flat = mask_b.reshape(b_flat, x_full.shape[axis])

    l_asu_list: list[int] = []
    per_row_asu: list[Tensor | None] = []
    for r in range(b_flat):
        idx = torch.nonzero(m_flat[r], as_tuple=False).squeeze(-1)
        n_val = int(idx.numel())
        if n_val == 0:
            l_asu_list.append(0)
            per_row_asu.append(None)
            continue
        if n_val % order != 0:
            raise ValueError(f"Row {r} valid token count {n_val} is not divisible by symmetry order {order}")
        l_asu_list.append(n_val // order)
        x_r = x_flat[r].index_select(0, idx)
        per_row_asu.append(_reduce_asu_unmasked(x_r, rot, seq_axis=0, preserve_variance=preserve_variance))

    l_max = max(l_asu_list, default=0)
    out_flat = torch.zeros((b_flat, l_max, *trailing), device=x_full.device, dtype=x_full.dtype)
    for r, row_asu in enumerate(per_row_asu):
        if row_asu is not None:
            out_flat[r, : row_asu.shape[0]] = row_asu
    return out_flat.reshape(*leading, l_max, *trailing)


def project_symmetric(
    x_full: Tensor,
    group: BatchSymmetryLike,
    *,
    mask: Tensor | None = None,
    seq_dim: int | None = None,
    preserve_variance: bool = False,
) -> Tensor:
    """Project full-oligomer 3D coordinates, velocities, or noise onto the G-equivariant subspace.

    Applies ``P_G(X) = expand_asu(reduce_to_asu(X))``. Supports padded batch masks ``[B, N_max]``
    where ``N_max >= K * L_asu_b`` as well as heterogeneous per-row symmetry lists.
    """
    axis = _resolve_cartesian_seq_axis(x_full, mask=mask, seq_dim=seq_dim)

    if _is_per_batch_sequence(group):
        if axis == 0 or x_full.shape[0] != len(group):
            raise ValueError(
                f"Per-batch symmetry length {len(group)} requires batch dim 0 match on {tuple(x_full.shape)}"
            )
        rows = []
        for b_idx, g_item in enumerate(group):
            row_mask = None if mask is None else mask[b_idx]
            rows.append(
                project_symmetric(
                    x_full[b_idx],
                    g_item,
                    mask=row_mask,
                    seq_dim=axis - 1,
                    preserve_variance=preserve_variance,
                )
            )
        return torch.stack(rows, dim=0)

    rot = _resolve_rotations(group, device=x_full.device, dtype=_compute_dtype(x_full.dtype))
    order = rot.shape[0]

    if mask is None:
        if order == 1:
            return x_full.clone()
        asu = _reduce_asu_unmasked(x_full, rot, seq_axis=axis, preserve_variance=preserve_variance)
        return _rotate_asu_unmasked(asu, rot, seq_axis=axis)

    mask_b = mask.to(device=x_full.device, dtype=torch.bool)
    if mask_b.shape != x_full.shape[: axis + 1]:
        raise ValueError(
            f"Mask shape {tuple(mask_b.shape)} does not match x_full prefix {tuple(x_full.shape[:axis+1])}"
        )

    if order == 1:
        expand_dims = (1,) * (x_full.ndim - mask_b.ndim)
        return x_full * mask_b.view(*mask_b.shape, *expand_dims)

    n_max = x_full.shape[axis]
    prefix = _uniform_prefix_length(mask_b)
    if prefix is not None:
        if prefix == 0:
            return torch.zeros_like(x_full)
        valid_slice = x_full.narrow(axis, 0, prefix)
        asu = _reduce_asu_unmasked(valid_slice, rot, seq_axis=axis, preserve_variance=preserve_variance)
        proj_valid = _rotate_asu_unmasked(asu, rot, seq_axis=axis)
        if prefix == n_max:
            return proj_valid
        pad_shape = (*x_full.shape[:axis], n_max - prefix, *x_full.shape[axis + 1 :])
        zeros = torch.zeros(pad_shape, device=x_full.device, dtype=x_full.dtype)
        return torch.cat([proj_valid, zeros], dim=axis)

    leading = x_full.shape[:axis]
    trailing = x_full.shape[axis + 1 :]
    b_flat = math.prod(leading) if leading else 1
    x_flat = x_full.reshape(b_flat, n_max, *trailing)
    m_flat = mask_b.reshape(b_flat, n_max)
    out_flat = torch.zeros_like(x_flat)

    for r in range(b_flat):
        idx = torch.nonzero(m_flat[r], as_tuple=False).squeeze(-1)
        n_val = int(idx.numel())
        if n_val == 0:
            continue
        if n_val % order != 0:
            raise ValueError(f"Row {r} valid token count {n_val} is not divisible by symmetry order {order}")
        x_r = x_flat[r].index_select(0, idx)
        asu_r = _reduce_asu_unmasked(x_r, rot, seq_axis=0, preserve_variance=preserve_variance)
        proj_r = _rotate_asu_unmasked(asu_r, rot, seq_axis=0)
        out_flat[r].index_copy_(0, idx, proj_r)

    return out_flat.reshape_as(x_full)


def expand_invariant_asu(
    h_asu: Tensor,
    group: int | SymmetryLike,
    *,
    mask: Tensor | None = None,
    seq_dim: int = -2,
) -> Tensor:
    """Tile SE(3)-invariant per-token features or logits ``[..., L_asu, D]`` across ``K`` subunits."""
    order = _resolve_order(group)
    axis = seq_dim if seq_dim >= 0 else h_asu.ndim + seq_dim
    if not 0 <= axis < h_asu.ndim:
        raise ValueError(f"Invalid seq_dim={seq_dim} for h_asu shape {tuple(h_asu.shape)}")

    h_in = h_asu
    if mask is not None and mask.shape == h_asu.shape[: axis + 1]:
        mask_b = mask.to(device=h_asu.device, dtype=torch.bool)
        expand_dims = (1,) * (h_asu.ndim - mask_b.ndim)
        h_in = h_asu * mask_b.view(*mask_b.shape, *expand_dims)

    leading = h_in.shape[:axis]
    l_asu = h_in.shape[axis]
    trailing = h_in.shape[axis + 1 :]
    tiled = h_in.unsqueeze(axis).expand(*leading, order, l_asu, *trailing).reshape(*leading, order * l_asu, *trailing)

    if mask is not None and mask.shape != h_asu.shape[: axis + 1]:
        mask_b = mask.to(device=h_asu.device, dtype=torch.bool)
        n_max = mask_b.shape[-1]
        prefix = _uniform_prefix_length(mask_b)
        if prefix is not None:
            if prefix == 0:
                return torch.zeros((*leading, n_max, *trailing), device=h_asu.device, dtype=h_asu.dtype)
            if prefix % order != 0:
                raise ValueError(f"Valid token count {prefix} is not divisible by symmetry order {order}")
            l_val = prefix // order
            valid_tiled = (
                h_asu.narrow(axis, 0, l_val)
                .unsqueeze(axis)
                .expand(*leading, order, l_val, *trailing)
                .reshape(*leading, prefix, *trailing)
            )
            if prefix == n_max:
                return valid_tiled.clone()
            zeros = torch.zeros((*leading, n_max - prefix, *trailing), device=h_asu.device, dtype=h_asu.dtype)
            return torch.cat([valid_tiled, zeros], dim=axis)
    return tiled.clone()


def _project_invariant_unmasked(
    h_full: Tensor,
    order: int,
    *,
    seq_axis: int,
    preserve_variance: bool = False,
) -> Tensor:
    """Average and re-tile an unmasked SE(3)-invariant tensor across ``order`` subunits."""
    n_total = h_full.shape[seq_axis]
    if n_total % order != 0:
        raise ValueError(f"Sequence length {n_total} along axis {seq_axis} is not divisible by group order {order}")
    if order == 1:
        return h_full.clone()

    l_asu = n_total // order
    leading = h_full.shape[:seq_axis]
    trailing = h_full.shape[seq_axis + 1 :]
    grouped = h_full.reshape(*leading, order, l_asu, *trailing)

    if h_full.dtype == torch.bool:
        asu = grouped.all(dim=seq_axis)
    elif not h_full.is_floating_point():
        asu = grouped.select(seq_axis, 0)
    else:
        grouped_f64 = grouped.to(dtype=torch.float64)
        denom = math.sqrt(float(order)) if preserve_variance else float(order)
        asu_f64 = grouped_f64.sum(dim=seq_axis) / denom
        if h_full.dtype != torch.float64:
            finfo = torch.finfo(h_full.dtype)
            finite_clamped = asu_f64.clamp(min=float(finfo.min), max=float(finfo.max))
            asu_f64 = torch.where(torch.isfinite(asu_f64), finite_clamped, asu_f64)
        asu = asu_f64.to(dtype=h_full.dtype)

    expanded = asu.unsqueeze(seq_axis).expand(*leading, order, l_asu, *trailing)
    return expanded.reshape(*leading, n_total, *trailing).clone()


def project_invariant_symmetric(
    h_full: Tensor,
    group: int | BatchSymmetryLike,
    *,
    mask: Tensor | None = None,
    seq_dim: int = -2,
    preserve_variance: bool = False,
) -> Tensor:
    """Average SE(3)-invariant features or logits across ``K`` subunits and re-tile to ``K * L_asu``.

    Safely handles padded batch masks ``[B, N_max]`` where ``N_max >= K * L_asu_b``, keeping padded
    positions zeroed. Also preserves ``-inf`` entries when disallowed token classes are masked.
    """
    axis = seq_dim if seq_dim >= 0 else h_full.ndim + seq_dim
    if not 0 <= axis < h_full.ndim:
        raise ValueError(f"Invalid seq_dim={seq_dim} for h_full shape {tuple(h_full.shape)}")

    if _is_per_batch_sequence(group):
        if axis == 0 or h_full.shape[0] != len(group):
            raise ValueError(
                f"Per-batch symmetry length {len(group)} requires batch dim 0 match on {tuple(h_full.shape)}"
            )
        rows = []
        for b_idx, g_item in enumerate(group):
            if isinstance(g_item, SymmetryConfig) and not g_item.tie_sequence:
                g_eff: int | SymmetryLike = 1
            else:
                g_eff = g_item
            row_mask = None if mask is None else mask[b_idx]
            rows.append(
                project_invariant_symmetric(
                    h_full[b_idx],
                    g_eff,
                    mask=row_mask,
                    seq_dim=axis - 1,
                    preserve_variance=preserve_variance,
                )
            )
        return torch.stack(rows, dim=0)

    order = _resolve_order(group)
    if mask is None:
        return _project_invariant_unmasked(h_full, order, seq_axis=axis, preserve_variance=preserve_variance)

    mask_b = mask.to(device=h_full.device, dtype=torch.bool)
    if mask_b.shape != h_full.shape[: axis + 1]:
        raise ValueError(
            f"Mask shape {tuple(mask_b.shape)} does not match h_full prefix {tuple(h_full.shape[:axis+1])}"
        )

    if order == 1:
        expand_dims = (1,) * (h_full.ndim - mask_b.ndim)
        return h_full * mask_b.view(*mask_b.shape, *expand_dims)

    n_max = h_full.shape[axis]
    prefix = _uniform_prefix_length(mask_b)
    if prefix is not None:
        if prefix == 0:
            return torch.zeros_like(h_full)
        proj_valid = _project_invariant_unmasked(
            h_full.narrow(axis, 0, prefix),
            order,
            seq_axis=axis,
            preserve_variance=preserve_variance,
        )
        if prefix == n_max:
            return proj_valid
        pad_shape = (*h_full.shape[:axis], n_max - prefix, *h_full.shape[axis + 1 :])
        zeros = torch.zeros(pad_shape, device=h_full.device, dtype=h_full.dtype)
        return torch.cat([proj_valid, zeros], dim=axis)

    leading = h_full.shape[:axis]
    trailing = h_full.shape[axis + 1 :]
    b_flat = math.prod(leading) if leading else 1
    h_flat = h_full.reshape(b_flat, n_max, *trailing)
    m_flat = mask_b.reshape(b_flat, n_max)
    out_flat = torch.zeros_like(h_flat)

    for r in range(b_flat):
        idx = torch.nonzero(m_flat[r], as_tuple=False).squeeze(-1)
        n_val = int(idx.numel())
        if n_val == 0:
            continue
        if n_val % order != 0:
            raise ValueError(f"Row {r} valid token count {n_val} is not divisible by symmetry order {order}")
        h_r = h_flat[r].index_select(0, idx)
        proj_r = _project_invariant_unmasked(h_r, order, seq_axis=0, preserve_variance=preserve_variance)
        out_flat[r].index_copy_(0, idx, proj_r)

    return out_flat.reshape_as(h_full)


def zero_global_com(
    x: Tensor,
    mask: Tensor | None = None,
    *,
    seq_dim: int = -2,
) -> Tensor:
    """Subtract global center of mass across all valid tokens while keeping padded tokens zeroed.

    Because ``mean_global = Pi_G @ mean_asu`` lies in ``Fix(G)`` for every point group ``G``,
    subtracting ``mean_global`` commutes with all ``R_g in G`` (removing z-drift for ``C_n`` and
    numerical float drift for ``D_n, T, O, I`` while preserving exact G-symmetry).
    """
    axis = seq_dim if seq_dim >= 0 else x.ndim + seq_dim
    if not 0 <= axis < x.ndim:
        raise ValueError(f"Invalid seq_dim={seq_dim} for tensor shape {tuple(x.shape)}")

    work_dtype = _compute_dtype(x.dtype)
    x_work = x.to(dtype=work_dtype)

    if mask is None:
        if x.shape[axis] == 0:
            return x.clone()
        com = x_work.mean(dim=axis, keepdim=True)
        return (x_work - com).to(dtype=x.dtype)

    mask_b = mask.to(device=x.device, dtype=torch.bool)
    if mask_b.shape != x.shape[: axis + 1]:
        raise ValueError(f"Mask shape {tuple(mask_b.shape)} does not match x prefix {tuple(x.shape[:axis+1])}")

    expand_dims = (1,) * (x.ndim - mask_b.ndim)
    mask_f = mask_b.view(*mask_b.shape, *expand_dims).to(dtype=work_dtype)
    denom = mask_f.sum(dim=axis, keepdim=True).clamp(min=1.0)
    com = (x_work * mask_f).sum(dim=axis, keepdim=True) / denom
    centered = (x_work - com) * mask_f
    return centered.to(dtype=x.dtype)


def sample_symmetric_ca_noise(
    shape: int | Sequence[int] | torch.Size,
    group: BatchSymmetryLike,
    *,
    mask: Tensor | None = None,
    radius_nm: float | None = None,
    zero_com: bool = True,
    scale: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample G-equivariant Gaussian CA noise ``[..., N_max, 3]`` with unit per-residue variance.

    Accepts ``shape`` as ``(..., N_max, 3)``, ``(..., N_max)``, or integer ``N_max``.
    Samples ``eps_asu ~ N(0, scale^2 I_3)`` on ``L_asu`` tokens, optionally adds ``radius_nm * u_G``
    along a collision-free canonical ray ``u_G``, expands via ``R_g eps_asu``, and optionally zeroes
    the global oligomer center of mass.
    """
    if isinstance(shape, int) and not isinstance(shape, bool):
        raw_shape = (shape,)
    else:
        raw_shape = tuple(int(d) for d in shape)

    if not raw_shape:
        raise ValueError(f"Expected non-empty shape for CA noise, got {shape!r}")

    if mask is not None:
        m_shape = tuple(int(d) for d in mask.shape)
        if raw_shape == m_shape:
            shape_tuple = (*raw_shape, 3)
        elif raw_shape == (*m_shape, 3):
            shape_tuple = raw_shape
        else:
            raise ValueError(f"Shape {raw_shape} is incompatible with mask shape {m_shape}")
    elif len(raw_shape) == 1 or raw_shape[-1] != 3:
        shape_tuple = (*raw_shape, 3)
    else:
        shape_tuple = raw_shape

    target_device = mask.device if (device is None and mask is not None) else torch.device(device or "cpu")

    if _is_per_batch_sequence(group):
        if len(shape_tuple) < 3 or shape_tuple[0] != len(group):
            raise ValueError(f"Per-batch symmetry sequence length {len(group)} mismatches shape {shape_tuple}")
        rows = []
        for b_idx, g_item in enumerate(group):
            row_mask = None if mask is None else mask[b_idx]
            rows.append(
                sample_symmetric_ca_noise(
                    shape_tuple[1:],
                    g_item,
                    mask=row_mask,
                    radius_nm=radius_nm,
                    zero_com=zero_com,
                    scale=scale,
                    device=target_device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        return torch.stack(rows, dim=0)

    eff_radius_nm = (
        (group.radius_nm if isinstance(group, SymmetryConfig) else 0.0) if radius_nm is None else float(radius_nm)
    )
    work_dtype = _compute_dtype(dtype)
    rot = _resolve_rotations(group, device=target_device, dtype=work_dtype)
    order = rot.shape[0]
    u_dir = _resolve_radial_dir(group, device=target_device, dtype=work_dtype)

    leading = shape_tuple[:-2]
    n_max = shape_tuple[-2]

    if mask is None:
        if n_max % order != 0:
            raise ValueError(f"Sequence length {n_max} is not divisible by symmetry order {order}")
        l_asu = n_max // order
        eps_asu = torch.randn((*leading, l_asu, 3), device=target_device, dtype=work_dtype, generator=generator) * scale
        if eff_radius_nm != 0.0 and order > 1:
            eps_asu = eps_asu + (eff_radius_nm * u_dir)
        out = _rotate_asu_unmasked(eps_asu, rot, seq_axis=len(leading))
        if zero_com:
            out = zero_global_com(out, mask=None, seq_dim=-2)
        return out.to(dtype=dtype)

    mask_b = mask.to(device=target_device, dtype=torch.bool)
    if mask_b.shape != (*leading, n_max):
        raise ValueError(f"Mask shape {tuple(mask_b.shape)} mismatches target shape {shape_tuple[:-1]}")

    prefix = _uniform_prefix_length(mask_b)
    if prefix is not None:
        if prefix == 0:
            return torch.zeros(shape_tuple, device=target_device, dtype=dtype)
        if prefix % order != 0:
            raise ValueError(f"Valid token count {prefix} is not divisible by symmetry order {order}")
        l_asu = prefix // order
        eps_asu = torch.randn((*leading, l_asu, 3), device=target_device, dtype=work_dtype, generator=generator) * scale
        if eff_radius_nm != 0.0 and order > 1:
            eps_asu = eps_asu + (eff_radius_nm * u_dir)
        valid_out = _rotate_asu_unmasked(eps_asu, rot, seq_axis=len(leading))
        if prefix == n_max:
            full_out = valid_out
        else:
            zeros = torch.zeros((*leading, n_max - prefix, 3), device=target_device, dtype=work_dtype)
            full_out = torch.cat([valid_out, zeros], dim=-2)
        if zero_com:
            full_out = zero_global_com(full_out, mask=mask_b, seq_dim=-2)
        return full_out.to(dtype=dtype)

    b_flat = math.prod(leading) if leading else 1
    m_flat = mask_b.reshape(b_flat, n_max)
    out_flat = torch.zeros((b_flat, n_max, 3), device=target_device, dtype=work_dtype)
    for r in range(b_flat):
        idx = torch.nonzero(m_flat[r], as_tuple=False).squeeze(-1)
        n_val = int(idx.numel())
        if n_val == 0:
            continue
        if n_val % order != 0:
            raise ValueError(f"Row {r} valid token count {n_val} is not divisible by symmetry order {order}")
        l_asu = n_val // order
        eps_r = torch.randn((l_asu, 3), device=target_device, dtype=work_dtype, generator=generator) * scale
        if eff_radius_nm != 0.0 and order > 1:
            eps_r = eps_r + (eff_radius_nm * u_dir)
        exp_r = _rotate_asu_unmasked(eps_r, rot, seq_axis=0)
        out_flat[r].index_copy_(0, idx, exp_r)

    full_out = out_flat.reshape(*leading, n_max, 3)
    if zero_com:
        full_out = zero_global_com(full_out, mask=mask_b, seq_dim=-2)
    return full_out.to(dtype=dtype)


def _symmetrize_atom37_unmasked(
    coors_nm: Tensor,
    rot: Tensor,
    *,
    ca_coors_nm: Tensor | None,
    atom_mask: Tensor | None,
    mode: str,
) -> Tensor:
    """Symmetrize an unmasked ``[..., K * L_asu, 37, 3]`` decoded atom37 coordinate tensor."""
    order = rot.shape[0]
    orig_dtype = coors_nm.dtype
    work_dtype = _compute_dtype(orig_dtype)
    coors_w = coors_nm.to(dtype=work_dtype)
    rot_w = rot.to(device=coors_nm.device, dtype=work_dtype)

    leading = coors_w.shape[:-3]
    n_total = coors_w.shape[-3]
    n_atoms = coors_w.shape[-2]
    if n_total % order != 0:
        raise ValueError(f"Decoded length {n_total} is not divisible by symmetry order {order}")
    l_asu = n_total // order

    if order == 1:
        y_chosen = coors_w
        if ca_coors_nm is not None:
            ca_w = ca_coors_nm.to(device=coors_nm.device, dtype=work_dtype)
            delta = ca_w - y_chosen[..., 1, :]
            y_shifted = y_chosen + delta.unsqueeze(-2)
            y_chosen = torch.cat([y_shifted[..., :1, :], ca_w.unsqueeze(-2), y_shifted[..., 2:, :]], dim=-2)
        return y_chosen.to(dtype=orig_dtype)

    # Pull all K subunits back into the canonical ASU frame: [*leading, K, L_asu, 37, 3]
    grouped = coors_w.reshape(*leading, order, l_asu, n_atoms, 3)
    y_all = torch.einsum("kij,...klai->...klaj", rot_w, grouped)

    if mode == "asu_first":
        y_chosen = y_all.select(-4, 0)
    elif mode == "chordal_mean":
        y_chosen = y_all.mean(dim=-4)
    elif mode == "asu_medoid":
        # Exact identity: argmin_g sum_h ||Y_g - Y_h||_F^2 == argmin_g ||Y_g - mean_h(Y_h)||_F^2
        if atom_mask is not None:
            am_bool = atom_mask.to(device=coors_nm.device, dtype=torch.bool)
            am_grouped = am_bool.reshape(*leading, order, l_asu, n_atoms)
            occ = am_grouped.any(dim=-3)  # [*leading, L_asu, 37]
            y_occ = y_all * occ.unsqueeze(-3).unsqueeze(-1)
        else:
            y_occ = y_all
        y_mean = y_occ.mean(dim=-4, keepdim=True)
        sq_dist_to_centroid = (y_occ - y_mean).square().sum(dim=(-3, -2, -1))  # [*leading, K]
        g_star = sq_dist_to_centroid.argmin(dim=-1)  # [*leading]
        if not leading:
            y_chosen = y_all[int(g_star.item())]
        else:
            gather_idx = g_star.view(*leading, 1, 1, 1, 1).expand(*leading, 1, l_asu, n_atoms, 3)
            y_chosen = torch.gather(y_all, dim=-4, index=gather_idx).squeeze(-4)
    else:
        raise ValueError(f"Unsupported decode symmetrization mode {mode!r}; expected one of {VALID_DECODE_MODES}")

    if ca_coors_nm is not None:
        ca_w = ca_coors_nm.to(device=coors_nm.device, dtype=work_dtype)
        ca_asu = _reduce_asu_unmasked(ca_w, rot_w, seq_axis=len(leading), preserve_variance=False)
        # Rigidly translate all 37 slots of residue l by (ca_asu[l] - y_chosen[l, 1]) to preserve bond geometry
        delta = ca_asu - y_chosen[..., 1, :]
        y_shifted = y_chosen + delta.unsqueeze(-2)
        y_chosen = torch.cat([y_shifted[..., :1, :], ca_asu.unsqueeze(-2), y_shifted[..., 2:, :]], dim=-2)

    expanded = _rotate_asu_unmasked(y_chosen, rot_w, seq_axis=len(leading))
    return expanded.to(dtype=orig_dtype)


def symmetrize_decoded_atom37(
    coors_nm: Tensor,
    group: BatchSymmetryLike,
    *,
    ca_coors_nm: Tensor | None = None,
    atom_mask: Tensor | None = None,
    mask: Tensor | None = None,
    mode: str | None = None,
) -> Tensor:
    """Symmetrize decoded atom37 coordinates ``[..., N_max, 37, 3]`` across ``K`` subunits.

    Supports modes ``"asu_medoid"`` (consensus single-conformer medoid avoiding rotamer-averaging
    bond shrinkage), ``"asu_first"``, and ``"chordal_mean"``. When ``ca_coors_nm`` is provided,
    every residue's 37-atom block is rigidly translated so slot 1 (``CA``) equals the exact
    symmetric CA coordinate without distorting internal bond lengths or angles.
    """
    if coors_nm.ndim < 3 or coors_nm.shape[-1] != 3:
        raise ValueError(f"Expected atom37 tensor [..., N, 37, 3], got {tuple(coors_nm.shape)}")

    if _is_per_batch_sequence(group):
        if coors_nm.ndim < 4 or coors_nm.shape[0] != len(group):
            raise ValueError(
                f"Per-batch symmetry sequence length {len(group)} mismatches coors_nm shape {tuple(coors_nm.shape)}"
            )
        rows = []
        for b_idx, g_item in enumerate(group):
            rows.append(
                symmetrize_decoded_atom37(
                    coors_nm[b_idx],
                    g_item,
                    ca_coors_nm=None if ca_coors_nm is None else ca_coors_nm[b_idx],
                    atom_mask=None if atom_mask is None else atom_mask[b_idx],
                    mask=None if mask is None else mask[b_idx],
                    mode=mode,
                )
            )
        return torch.stack(rows, dim=0)

    eff_mode = (
        (group.decode_mode if isinstance(group, SymmetryConfig) else "asu_medoid")
        if mode is None
        else str(mode).strip().lower()
    )
    if eff_mode not in VALID_DECODE_MODES:
        raise ValueError(f"Invalid mode {eff_mode!r}; expected one of {VALID_DECODE_MODES}")

    rot = _resolve_rotations(group, device=coors_nm.device, dtype=_compute_dtype(coors_nm.dtype))
    order = rot.shape[0]

    if mask is None:
        return _symmetrize_atom37_unmasked(
            coors_nm,
            rot,
            ca_coors_nm=ca_coors_nm,
            atom_mask=atom_mask,
            mode=eff_mode,
        )

    leading = coors_nm.shape[:-3]
    n_max = coors_nm.shape[-3]
    n_atoms = coors_nm.shape[-2]
    mask_b = mask.to(device=coors_nm.device, dtype=torch.bool)
    if mask_b.shape != (*leading, n_max):
        raise ValueError(f"Mask shape {tuple(mask_b.shape)} mismatches coors_nm prefix {(*leading, n_max)}")

    prefix = _uniform_prefix_length(mask_b)
    if prefix is not None:
        if prefix == 0:
            return torch.zeros_like(coors_nm)
        sym_valid = _symmetrize_atom37_unmasked(
            coors_nm[..., :prefix, :, :],
            rot,
            ca_coors_nm=None if ca_coors_nm is None else ca_coors_nm[..., :prefix, :],
            atom_mask=None if atom_mask is None else atom_mask[..., :prefix, :],
            mode=eff_mode,
        )
        if prefix == n_max:
            return sym_valid
        zeros = torch.zeros((*leading, n_max - prefix, n_atoms, 3), device=coors_nm.device, dtype=coors_nm.dtype)
        return torch.cat([sym_valid, zeros], dim=-3)

    b_flat = math.prod(leading) if leading else 1
    c_flat = coors_nm.reshape(b_flat, n_max, n_atoms, 3)
    ca_flat = None if ca_coors_nm is None else ca_coors_nm.reshape(b_flat, n_max, 3)
    am_flat = None if atom_mask is None else atom_mask.reshape(b_flat, n_max, n_atoms)
    m_flat = mask_b.reshape(b_flat, n_max)
    out_rows: list[Tensor] = []
    for r in range(b_flat):
        idx = torch.nonzero(m_flat[r], as_tuple=False).squeeze(-1)
        n_val = int(idx.numel())
        if n_val == 0:
            out_rows.append(torch.zeros_like(c_flat[r]))
            continue
        if n_val % order != 0:
            raise ValueError(f"Row {r} valid token count {n_val} is not divisible by symmetry order {order}")
        sym_r = _symmetrize_atom37_unmasked(
            c_flat[r].index_select(0, idx),
            rot,
            ca_coors_nm=None if ca_flat is None else ca_flat[r].index_select(0, idx),
            atom_mask=None if am_flat is None else am_flat[r].index_select(0, idx),
            mode=eff_mode,
        )
        out_rows.append(torch.zeros_like(c_flat[r]).index_copy(0, idx, sym_r))

    return torch.stack(out_rows, dim=0).reshape_as(coors_nm)


def expand_frames_asu(q_asu: Tensor, group: SymmetryLike) -> Tensor:
    """Expand ASU SO(3) rotation frames ``[..., L_asu, 3, 3]`` across ``K`` subunits via ``R_g @ Q_asu``."""
    if q_asu.ndim < 3 or q_asu.shape[-2:] != (3, 3):
        raise ValueError(f"Expected frame rotation tensor [..., L_asu, 3, 3], got {tuple(q_asu.shape)}")
    work_dtype = _compute_dtype(q_asu.dtype)
    rot = _resolve_rotations(group, device=q_asu.device, dtype=work_dtype)
    order = rot.shape[0]
    leading = q_asu.shape[:-3]
    l_asu = q_asu.shape[-3]
    expanded = torch.einsum("kij,...ljm->...klim", rot, q_asu.to(dtype=work_dtype))
    return expanded.reshape(*leading, order * l_asu, 3, 3).to(dtype=q_asu.dtype)


def project_frames_symmetric(q_full: Tensor, group: SymmetryLike) -> Tensor:
    """Project noisy/predicted frames ``[..., K * L_asu, 3, 3]`` onto exact G-equivariant SO(3) frames."""
    if q_full.ndim < 3 or q_full.shape[-2:] != (3, 3):
        raise ValueError(f"Expected frame rotation tensor [..., N, 3, 3], got {tuple(q_full.shape)}")
    work_dtype = torch.float64 if q_full.dtype == torch.float64 else torch.float32
    rot = _resolve_rotations(group, device=q_full.device, dtype=work_dtype)
    order = rot.shape[0]
    n_total = q_full.shape[-3]
    if n_total % order != 0:
        raise ValueError(f"Frame count {n_total} is not divisible by symmetry order {order}")
    l_asu = n_total // order
    leading = q_full.shape[:-3]

    grouped = q_full.to(dtype=work_dtype).reshape(*leading, order, l_asu, 3, 3)
    chordal_mean = torch.einsum("kij,...klim->...ljm", rot, grouped) / float(order)
    u, _, vh = torch.linalg.svd(chordal_mean)
    uv = torch.matmul(u, vh)
    det_sign = torch.sign(torch.linalg.det(uv))
    det_sign = torch.where(det_sign == 0, torch.ones_like(det_sign), det_sign)
    corr = torch.ones((*leading, l_asu, 3), device=q_full.device, dtype=work_dtype)
    corr[..., 2] = det_sign
    q_asu = torch.matmul(u * corr.unsqueeze(-2), vh)
    expanded = torch.einsum("kij,...ljm->...klim", rot, q_asu)
    return expanded.reshape(*leading, n_total, 3, 3).to(dtype=q_full.dtype)


def _symmetry_deviation_single(
    x_valid: Tensor,
    rot: Tensor,
    *,
    atom_mask_valid: Tensor | None,
    reference: Literal["first", "mean"],
) -> tuple[Tensor, Tensor]:
    """Compute scalar (rmsd, max_abs_dev) for a single unbatched valid oligomer ``[K * L_asu, (A,) 3]``."""
    order = rot.shape[0]
    n_total = x_valid.shape[0]
    if n_total == 0 or order <= 1:
        zero = torch.zeros((), device=x_valid.device, dtype=torch.float64)
        return zero.to(dtype=x_valid.dtype), zero.to(dtype=x_valid.dtype)
    if n_total % order != 0:
        raise ValueError(f"Valid token count {n_total} is not divisible by symmetry order {order}")

    x_f64 = x_valid.to(dtype=torch.float64)
    rot_f64 = rot.to(device=x_valid.device, dtype=torch.float64)
    l_asu = n_total // order
    has_atoms = x_f64.ndim == 3

    if reference == "first":
        asu_ref = x_f64[:l_asu]
        x_ideal = _rotate_asu_unmasked(asu_ref, rot_f64, seq_axis=0)
        # Evaluate across non-identity subunits g = 1 .. K - 1
        diff = x_f64[l_asu:] - x_ideal[l_asu:]
        am_eval = (
            None if atom_mask_valid is None else atom_mask_valid[l_asu:].to(device=x_valid.device, dtype=torch.bool)
        )
    elif reference == "mean":
        asu_ref = _reduce_asu_unmasked(x_f64, rot_f64, seq_axis=0, preserve_variance=False)
        x_ideal = _rotate_asu_unmasked(asu_ref, rot_f64, seq_axis=0)
        diff = x_f64 - x_ideal
        am_eval = None if atom_mask_valid is None else atom_mask_valid.to(device=x_valid.device, dtype=torch.bool)
    else:
        raise ValueError(f"Invalid reference {reference!r}; expected 'first' or 'mean'")

    if has_atoms and am_eval is not None:
        if not bool(am_eval.any().item()):
            zero = torch.zeros((), device=x_valid.device, dtype=x_valid.dtype)
            return zero, zero
        diff_occ = diff[am_eval]  # [M_occ, 3]
    else:
        diff_occ = diff.reshape(-1, 3)

    sq_atom_dist = diff_occ.square().sum(dim=-1)
    rmsd = torch.sqrt(sq_atom_dist.mean())
    max_dev = diff_occ.abs().max()
    return rmsd.to(dtype=x_valid.dtype), max_dev.to(dtype=x_valid.dtype)


def compute_symmetry_rmsd(
    x_full: Tensor,
    group: BatchSymmetryLike,
    *,
    mask: Tensor | None = None,
    atom_mask: Tensor | None = None,
    seq_dim: int | None = None,
    reference: Literal["first", "mean"] = "first",
    return_max_dev: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute unaligned pure-R_g point-group symmetry RMSD (and optional L_inf max deviation).

    Measures direct coordinate deviation ``X_g - R_g X_0`` (when ``reference='first'``) or
    ``X_g - P_G(X)_g`` (when ``reference='mean'``) without Kabsch superposition, verifying both
    inter-subunit symmetry and exact alignment to the canonical point-group axes/origin.
    """
    axis = _resolve_cartesian_seq_axis(x_full, mask=mask, seq_dim=seq_dim)

    if _is_per_batch_sequence(group):
        if axis == 0 or x_full.shape[0] != len(group):
            raise ValueError(
                f"Per-batch symmetry sequence length {len(group)} mismatches x_full shape {tuple(x_full.shape)}"
            )
        rmsds: list[Tensor] = []
        max_devs: list[Tensor] = []
        for b_idx, g_item in enumerate(group):
            res = compute_symmetry_rmsd(
                x_full[b_idx],
                g_item,
                mask=None if mask is None else mask[b_idx],
                atom_mask=None if atom_mask is None else atom_mask[b_idx],
                seq_dim=axis - 1,
                reference=reference,
                return_max_dev=True,
            )
            assert isinstance(res, tuple)
            rmsds.append(res[0])
            max_devs.append(res[1])
        r_stack = torch.stack(rmsds, dim=0)
        m_stack = torch.stack(max_devs, dim=0)
        return (r_stack, m_stack) if return_max_dev else r_stack

    rot = _resolve_rotations(group, device=x_full.device, dtype=torch.float64)
    leading = x_full.shape[:axis]
    n_max = x_full.shape[axis]
    trailing = x_full.shape[axis + 1 :]

    if not leading:
        if mask is not None:
            idx = torch.nonzero(mask.to(device=x_full.device, dtype=torch.bool), as_tuple=False).squeeze(-1)
            x_val = x_full.index_select(0, idx)
            am_val = None if atom_mask is None else atom_mask.index_select(0, idx)
        else:
            x_val = x_full
            am_val = atom_mask
        rmsd_s, max_s = _symmetry_deviation_single(x_val, rot, atom_mask_valid=am_val, reference=reference)
        return (rmsd_s, max_s) if return_max_dev else rmsd_s

    b_flat = math.prod(leading)
    x_flat = x_full.reshape(b_flat, n_max, *trailing)
    m_flat = None if mask is None else mask.to(device=x_full.device, dtype=torch.bool).reshape(b_flat, n_max)
    am_flat = None if atom_mask is None else atom_mask.reshape(b_flat, n_max, *atom_mask.shape[axis + 1 :])

    rmsd_list: list[Tensor] = []
    max_list: list[Tensor] = []
    for r in range(b_flat):
        if m_flat is not None:
            idx = torch.nonzero(m_flat[r], as_tuple=False).squeeze(-1)
            x_r = x_flat[r].index_select(0, idx)
            am_r = None if am_flat is None else am_flat[r].index_select(0, idx)
        else:
            x_r = x_flat[r]
            am_r = None if am_flat is None else am_flat[r]
        r_val, m_val = _symmetry_deviation_single(x_r, rot, atom_mask_valid=am_r, reference=reference)
        rmsd_list.append(r_val)
        max_list.append(m_val)

    out_rmsd = torch.stack(rmsd_list, dim=0).reshape(*leading)
    out_max = torch.stack(max_list, dim=0).reshape(*leading)
    return (out_rmsd, out_max) if return_max_dev else out_rmsd
