"""AtomArray transforms and batch config resolution for point-group symmetry design."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from atomworks.io.transforms.atom_array import add_id_and_entity_annotations, add_iid_annotations
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import (
    AtomArrayPlus,
    as_atom_array_plus,
    concatenate_atom_array_plus,
)
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.standard_annotations import S_ATM, S_SEGMIN
from atomworks.ml.conditions import C_CRD, C_IDX
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_count
from biotite.structure import AtomArray

from rfproteina.conditions import C_CENTROID, C_CRDGRP, C_IDXGRP, C_INTCRD
from rfproteina.symmetry.point_groups import SymmetryConfig, get_symmetry_chain_ids
from rfproteina.transforms.utils.conditioning_utils import separate_coordinate_groups
from rfproteina.utils.constants import LOCAL_LATENT_RESIDUES

SYM_ENTITY_GROUP_ANNOTATION = "sym_entity_group"
SYM_SUBUNIT_ANNOTATION = "symmetry_subunit"
SYM_GROUP_NAME_ANNOTATION = "symmetry_group"


def merge_symmetry_config(
    default_cfg: SymmetryConfig | Mapping[str, Any] | str | None,
    override: Any = None,
    *,
    sym_radius: float | None = None,
) -> SymmetryConfig:
    """Merge an optional per-example or CLI override onto a base :class:`SymmetryConfig`."""
    base = SymmetryConfig.from_any(default_cfg)
    merged = {
        "group": base.group,
        "radius": base.radius,
        "input_is_asu": base.input_is_asu,
        "tie_sequence": base.tie_sequence,
        "decode_mode": base.decode_mode,
    }

    if override is not None:
        if isinstance(override, SymmetryConfig):
            merged.update(
                {
                    "group": override.group,
                    "radius": override.radius,
                    "input_is_asu": override.input_is_asu,
                    "tie_sequence": override.tie_sequence,
                    "decode_mode": override.decode_mode,
                }
            )
        elif isinstance(override, str) or override is False:
            parsed_override = SymmetryConfig.from_any(override, default_radius=merged["radius"])
            merged["group"] = parsed_override.group
        elif isinstance(override, Mapping):
            if "symmetry" in override and "group" not in override:
                nested = override["symmetry"]
                if isinstance(nested, Mapping):
                    merged = merge_symmetry_config(
                        merged,
                        nested,
                        sym_radius=override.get("sym_radius", sym_radius),
                    ).to_dict()
                    merged.pop("order", None)
                else:
                    merged["group"] = SymmetryConfig.from_any(nested).group
            else:
                if "group" in override:
                    merged["group"] = override["group"]
                if "radius" in override and override["radius"] is not None:
                    merged["radius"] = float(override["radius"])
                elif "sym_radius" in override and override["sym_radius"] is not None:
                    merged["radius"] = float(override["sym_radius"])
                elif "radius_nm" in override and override["radius_nm"] is not None:
                    merged["radius"] = float(override["radius_nm"]) * 10.0
                if "input_is_asu" in override and override["input_is_asu"] is not None:
                    merged["input_is_asu"] = bool(override["input_is_asu"])
                if "tie_sequence" in override and override["tie_sequence"] is not None:
                    merged["tie_sequence"] = bool(override["tie_sequence"])
                if "decode_mode" in override and override["decode_mode"] is not None:
                    merged["decode_mode"] = str(override["decode_mode"])
        else:
            parsed_override = SymmetryConfig.from_any(override, default_radius=merged["radius"])
            merged = {
                "group": parsed_override.group,
                "radius": parsed_override.radius,
                "input_is_asu": parsed_override.input_is_asu,
                "tie_sequence": parsed_override.tie_sequence,
                "decode_mode": parsed_override.decode_mode,
            }

    if sym_radius is not None:
        merged["radius"] = float(sym_radius)

    return SymmetryConfig.from_any(merged)


def resolve_batch_symmetry_configs(
    batch: Mapping[str, Any],
    default: SymmetryConfig | Mapping[str, Any] | str | None = None,
) -> list[SymmetryConfig]:
    """Resolve one :class:`SymmetryConfig` per batch item from collated batch metadata or defaults.

    Handles all shapes emitted by ``batch_dataclass_recursive`` in ``collate_batch``:
      - ``batch["symmetry"]`` (direct ``SymmetryConfig``, ``str``, ``dict``, or ``list``)
      - ``batch["extra_info"]["symmetry"]`` as a dict-of-lists (when every sample carries symmetry)
        or a list-of-dicts/Nones (when only a subset of samples in the batch carry symmetry)
      - ``batch["specification_batch"]`` / ``batch["specification"]`` fallback
      - ``default`` global generation config fallback
    """
    base_cfg = SymmetryConfig.from_any(default)

    if "padding_mask" in batch and hasattr(batch["padding_mask"], "shape"):
        batch_size = int(batch["padding_mask"].shape[0])
    elif "example_id_batch" in batch and isinstance(batch["example_id_batch"], list | tuple):
        batch_size = len(batch["example_id_batch"])
    else:
        batch_size = 1

    raw = batch.get("symmetry") if hasattr(batch, "get") else None
    if raw is None and hasattr(batch, "get"):
        extra = batch.get("extra_info")
        if isinstance(extra, Mapping):
            raw = extra.get("symmetry")
        elif isinstance(extra, list | tuple):
            raw = [item.get("symmetry") if isinstance(item, Mapping) else None for item in extra]

    if raw is None and hasattr(batch, "get"):
        specs = batch.get("specification_batch") or batch.get("specification")
        if isinstance(specs, Mapping) and "symmetry" in specs:
            raw = specs["symmetry"]
        elif isinstance(specs, list | tuple):
            extracted = [s.get("symmetry") if isinstance(s, Mapping) else None for s in specs]
            if any(x is not None for x in extracted):
                raw = extracted

    if raw is None:
        return [base_cfg for _ in range(batch_size)]

    if isinstance(raw, SymmetryConfig | str) or raw is False:
        resolved = merge_symmetry_config(base_cfg, raw)
        return [resolved for _ in range(batch_size)]

    if isinstance(raw, list | tuple):
        items = [merge_symmetry_config(base_cfg, item) for item in raw]
        if len(items) == batch_size:
            return items
        if len(items) == 1:
            return items * batch_size
        return (items + [base_cfg] * batch_size)[:batch_size]

    if isinstance(raw, Mapping):
        list_valued = any(isinstance(v, list | tuple) for v in raw.values())
        if list_valued:
            configs: list[SymmetryConfig] = []
            for idx in range(batch_size):
                row_dict = {
                    k: (v[idx] if isinstance(v, list | tuple) and idx < len(v) else v)
                    for k, v in raw.items()
                }
                configs.append(merge_symmetry_config(base_cfg, row_dict))
            return configs
        resolved = merge_symmetry_config(base_cfg, raw)
        return [resolved for _ in range(batch_size)]

    resolved = merge_symmetry_config(base_cfg, raw)
    return [resolved for _ in range(batch_size)]


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    """Return unique values in order of first appearance."""
    if values.size == 0:
        return values
    _, first_idx = np.unique(values, return_index=True)
    return values[np.sort(first_idx)]


def _rotate_finite_xyz(values: np.ndarray, rot_f64: np.ndarray) -> np.ndarray:
    """Apply a 3x3 rotation matrix ``rot_f64`` to finite ``[..., 3]`` row vectors."""
    arr_f64 = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr_f64).all(axis=-1)
    if not finite.any():
        return values.copy()
    out = arr_f64.copy()
    out[finite] = np.einsum("ij,nj->ni", rot_f64, arr_f64[finite])
    return out.astype(values.dtype, copy=False)


def _count_tokens_pre_tokenization(aa: AtomArray, *, is_inference: bool = True) -> int:
    """Count final model tokens either from ``token_id`` or prior to ``AddGlobalTokenIdAnnotation``."""
    categories = set(aa.get_annotation_categories())
    if "token_id" in categories:
        return int(get_token_count(aa))
    if aa.array_length() == 0:
        return 0

    work = aa
    if is_inference:
        unindexed_expseg = S_SEGMIN.mask(work, default="generate") & ~C_IDX.mask(work, default="generate")
        if unindexed_expseg.any():
            work = work[~unindexed_expseg]
            if work.array_length() == 0:
                return 0

    work_cats = set(work.get_annotation_categories())
    if "element" in work_cats:
        elem = np.char.upper(np.char.strip(work.element.astype(str)))
        non_h = ~np.isin(elem, ("H", "D"))
        if non_h.any() and not non_h.all():
            work = work[non_h]
            if work.array_length() == 0:
                return 0

    explicit_atomize = (
        np.asarray(work.atomize, dtype=bool)
        if "atomize" in set(work.get_annotation_categories())
        else S_ATM.mask(work, default="generate")
    )
    default_atomize = ~np.isin(work.res_name, list(LOCAL_LATENT_RESIDUES))
    is_atomized = explicit_atomize | default_atomize

    res_starts = get_residue_starts(work, add_exclusive_stop=False)
    n_polymer_tokens = int(np.sum(~is_atomized[res_starts]))
    n_atomized_tokens = int(np.sum(is_atomized))
    return n_polymer_tokens + n_atomized_tokens


def _populate_bondless_ids(expanded: AtomArrayPlus, order: int, n_asu_chains: int) -> AtomArrayPlus:
    """Populate chain/molecule/entity annotations when a synthetic test AtomArray lacks ``BondList``."""
    chain_ids = expanded.chain_id.astype(str)
    unique_cids = _ordered_unique(chain_ids)
    max_iid_len = max(6, max((len(str(c)) + 2 for c in unique_cids), default=6))

    chain_iids = np.array([f"{cid}_1" for cid in chain_ids], dtype=f"<U{max_iid_len}")
    mol_id = np.empty(expanded.array_length(), dtype=np.int32)
    for idx, cid in enumerate(unique_cids):
        mol_id[chain_ids == cid] = idx

    expanded.set_annotation("chain_iid", chain_iids)
    expanded.set_annotation("pn_unit_id", expanded.chain_id.copy())
    expanded.set_annotation("pn_unit_iid", chain_iids.copy())
    expanded.set_annotation("molecule_id", mol_id)
    expanded.set_annotation("molecule_iid", mol_id.copy())
    if SYM_ENTITY_GROUP_ANNOTATION in expanded.get_annotation_categories():
        expanded.set_annotation(
            "molecule_entity",
            expanded.get_annotation(SYM_ENTITY_GROUP_ANNOTATION).astype(np.int32),
        )
    else:
        expanded.set_annotation("molecule_entity", (mol_id % max(1, n_asu_chains)).astype(np.int32))
    return expanded


def expand_asu_atom_array(
    asu_in: AtomArray, sym_cfg: SymmetryConfig | Mapping[str, Any] | str | None
) -> AtomArrayPlus:
    """Replicate an ASU ``AtomArray`` into ``K = sym_cfg.order`` point-group symmetric subunits.

    Produces ``K`` contiguous ASU blocks ``[g * L_asu : (g + 1) * L_asu]`` (supporting both single-chain
    homomers and multi-chain hetero-oligomer ASUs), assigns mmCIF base-26 chain IDs (``A..Z, AA..BH, ...``),
    rotates ``coord`` and coordinate-valued conditions (``C_CRD``, ``C_CENTROID``, ``C_INTCRD``) by ``R_g``,
    assigns disjoint positive ``C_CRDGRP`` IDs per subunit while keeping absolute ``C_CRDGRP == 0`` shared,
    refreshes chain/molecule/entity instance IDs, and stamps ``sym_entity_group`` for homomeric tying.
    """
    sym_cfg = SymmetryConfig.from_any(sym_cfg)
    asu = as_atom_array_plus(asu_in.copy())
    if not sym_cfg.is_active or asu.array_length() == 0:
        return asu

    spec = sym_cfg.spec
    order = spec.order
    rotations = spec.rotations_f64.cpu().numpy()  # [K, 3, 3] float64, R[0] = I_3

    asu_cats = set(asu.get_annotation_categories())
    chain_col = asu.chain_iid if "chain_iid" in asu_cats else asu.chain_id
    ordered_asu_chains = _ordered_unique(chain_col)
    n_asu_chains = len(ordered_asu_chains)

    asu_chain_idx = np.empty(asu.array_length(), dtype=np.int32)
    for m_idx, ch_val in enumerate(ordered_asu_chains):
        asu_chain_idx[chain_col == ch_val] = m_idx

    all_chain_ids = get_symmetry_chain_ids(order, chains_per_asu=n_asu_chains)
    max_cid_len = max(4, max(len(cid) for cid in all_chain_ids))
    group_label_dtype = f"<U{max(4, len(spec.name))}"

    has_idxgrp = C_IDXGRP.has_annotation(asu)
    if has_idxgrp:
        base_idxgrp = C_IDXGRP.annotation(asu, default="generate").astype(np.int64)
        idxgrp_span = (int(base_idxgrp.max()) - int(base_idxgrp.min()) + 1) if base_idxgrp.size else 1
    else:
        base_idxgrp = None
        idxgrp_span = 1

    copies: list[AtomArrayPlus] = []
    for g in range(order):
        copy_g = as_atom_array_plus(asu.copy())
        rot_g = rotations[g]

        new_cids = np.empty(copy_g.array_length(), dtype=f"<U{max_cid_len}")
        for m_idx in range(n_asu_chains):
            new_cids[asu_chain_idx == m_idx] = all_chain_ids[g * n_asu_chains + m_idx]
        copy_g.set_annotation("chain_id", new_cids)
        copy_g.set_annotation(SYM_ENTITY_GROUP_ANNOTATION, asu_chain_idx.copy())
        copy_g.set_annotation(SYM_SUBUNIT_ANNOTATION, np.full(copy_g.array_length(), g, dtype=np.int32))
        copy_g.set_annotation(
            SYM_GROUP_NAME_ANNOTATION,
            np.full(copy_g.array_length(), spec.name, dtype=group_label_dtype),
        )

        copy_g.coord = _rotate_finite_xyz(copy_g.coord, rot_g)

        if C_CRD.has_annotation(copy_g):
            crd_vals = C_CRD.annotation(copy_g, default="generate")
            C_CRD.set_annotation(copy_g, _rotate_finite_xyz(crd_vals, rot_g).astype(np.float32))

        if C_CENTROID.has_annotation(copy_g):
            cent_vals = C_CENTROID.annotation(copy_g, default="generate")
            C_CENTROID.set_annotation(copy_g, _rotate_finite_xyz(cent_vals, rot_g).astype(np.float32))

        if C_INTCRD.has_annotation(copy_g):
            intcrd_vals = C_INTCRD.annotation(copy_g, default="generate")
            C_INTCRD.set_annotation(copy_g, _rotate_finite_xyz(intcrd_vals, rot_g).astype(np.float32))

        if has_idxgrp and base_idxgrp is not None and g > 0:
            shifted_idxgrp = (base_idxgrp + g * idxgrp_span).astype(
                asu.get_annotation(C_IDXGRP.full_name).dtype
            )
            C_IDXGRP.set_annotation(copy_g, shifted_idxgrp)

        copies.append(copy_g)

    copies = [as_atom_array_plus(c) for c in separate_coordinate_groups(copies)]
    expanded = concatenate_atom_array_plus(
        copies,
        fill_missing_standard_annotations=True,
        on_annotation_mismatch_policy="raise",
    )

    expanded.set_annotation("transformation_id", np.full(expanded.array_length(), "1", dtype="<U4"))
    if expanded.bonds is not None:
        ensure_annotations(expanded, "is_polymer")
        expanded = add_id_and_entity_annotations(expanded, overwrite=True)
        expanded = add_iid_annotations(expanded, overwrite=True)
    else:
        expanded = _populate_bondless_ids(expanded, order=order, n_asu_chains=n_asu_chains)

    if C_IDXGRP.has_annotation(expanded):
        default_groups = C_IDXGRP.default_annotation(expanded)
        current_groups = expanded.get_annotation(C_IDXGRP.full_name)
        _, synced_groups = np.unique(np.stack([default_groups, current_groups], axis=1), axis=0, return_inverse=True)
        C_IDXGRP.set_annotation(expanded, synced_groups)

    return as_atom_array_plus(expanded)


class ExpandSymmetricASUTransform(Transform):
    """Expand an asymmetric unit into ``K`` point-group symmetric subunits during inference/sampling.

    Placed in ``sampling_transforms`` immediately after ``RealizeExpandableSegments`` so any
    variable-length expandable segments are sampled once on the ASU and replicated with identical
    lengths across all ``K`` symmetric copies.
    """

    def __init__(self, symmetry: SymmetryConfig | Mapping[str, Any] | str | None = None):
        super().__init__()
        self.default_symmetry = SymmetryConfig.from_any(symmetry)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        extra_info = data.get("extra_info")
        per_example_sym = None
        if isinstance(extra_info, Mapping):
            per_example_sym = extra_info.get("symmetry")
        if per_example_sym is None and isinstance(data.get("specification"), Mapping):
            per_example_sym = data["specification"].get("symmetry")
        if per_example_sym is None and "symmetry" in data:
            per_example_sym = data.get("symmetry")

        sym_cfg = merge_symmetry_config(self.default_symmetry, per_example_sym)

        if sym_cfg.group is None:
            return data

        sym_dict = sym_cfg.to_dict()
        extra_out = dict(extra_info) if isinstance(extra_info, Mapping) else {}
        extra_out["symmetry"] = sym_dict
        data["extra_info"] = extra_out
        if isinstance(data.get("specification"), Mapping):
            spec_out = dict(data["specification"])
            spec_out["symmetry"] = sym_dict
            data["specification"] = spec_out

        if not sym_cfg.is_active:
            return data

        aa = as_atom_array_plus(data["atom_array"])
        order = sym_cfg.order
        spec = sym_cfg.spec

        if sym_cfg.input_is_asu:
            data["atom_array"] = expand_asu_atom_array(aa, sym_cfg)
            return data

        # Pre-expanded multi-chain oligomer mode (input_is_asu=False)
        is_inf = bool(data.get("is_inference", True))
        n_tokens = _count_tokens_pre_tokenization(aa, is_inference=is_inf)
        if n_tokens <= 0 or n_tokens % order != 0:
            raise ValueError(
                f"Symmetry {spec.name} with input_is_asu=False requires positive token count divisible by "
                f"order K={order}, got {n_tokens} token(s)."
            )

        aa = as_atom_array_plus(aa.copy())
        aa_cats = set(aa.get_annotation_categories())
        chain_col = aa.chain_iid if "chain_iid" in aa_cats else aa.chain_id
        ordered_chains = _ordered_unique(chain_col)

        if len(ordered_chains) >= order and len(ordered_chains) % order == 0:
            chains_per_asu = len(ordered_chains) // order
            ent_grp = np.empty(aa.array_length(), dtype=np.int32)
            sub_idx = np.empty(aa.array_length(), dtype=np.int32)
            for c_idx, ch_val in enumerate(ordered_chains):
                mask = chain_col == ch_val
                ent_grp[mask] = c_idx % chains_per_asu
                sub_idx[mask] = c_idx // chains_per_asu
            aa.set_annotation(SYM_ENTITY_GROUP_ANNOTATION, ent_grp)
            aa.set_annotation(SYM_SUBUNIT_ANNOTATION, sub_idx)
        else:
            aa.set_annotation(SYM_ENTITY_GROUP_ANNOTATION, np.zeros(aa.array_length(), dtype=np.int32))

        aa.set_annotation(
            SYM_GROUP_NAME_ANNOTATION,
            np.full(aa.array_length(), spec.name, dtype=f"<U{max(4, len(spec.name))}"),
        )
        data["atom_array"] = aa
        return data
