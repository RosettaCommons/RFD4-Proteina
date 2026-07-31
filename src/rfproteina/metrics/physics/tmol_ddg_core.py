"""Shared TMol helpers for fast block-pair ddG metrics."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import tmol.database
import torch
from biotite.structure import AtomArray, get_residue_starts
from tmol.database import ParameterDatabase
from tmol.io.pose_stack_from_biotite import (
    BiotitePoseBuildContext,
    pose_stack_from_biotite,
)
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.move_map import CartesianMoveMap, MoveMap
from tmol.pack.packer_task import PackerPalette
from tmol.pose.pose_stack import PoseStack
from tmol.relax.fast_relax import _default_cart_min_fn, fast_relax
from tmol.score import beta2016_score_function
from tmol.score.score_function import ScoreFunction

from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.mask_spec import resolve_mask

logger = RankedLogger(__name__, rank_zero_only=True)


def create_score_function(
    device: torch.device,
    *,
    param_db: ParameterDatabase | None,
    use_soft_rep: bool = False,
) -> Any:
    """Create TMol beta2016, or ``beta_soft`` when ``use_soft_rep`` is set.

    ``param_db`` must be the ligand-extended database from a BiotitePoseBuildContext.
    """
    if param_db is None:
        raise ValueError("param_db is required; use the PoseBuildContext parameter database")

    if not use_soft_rep:
        return beta2016_score_function(device, param_db=param_db)

    weights_path = Path(tmol.database.__file__).parent / "score_functions" / "beta_soft.sfxn"
    return ScoreFunction.from_sfxn_file(str(weights_path), param_db, device)


def _residue_keys(atom_array: AtomArray, atom_sel: np.ndarray) -> set[tuple[str, int, str]]:
    """``(chain_id, res_id, ins_code)`` keys for residues with an atom in ``atom_sel``.

    Keys must be unique globally: pdb_info cannot disambiguate duplicates.
    """
    has_ins_code = "ins_code" in atom_array.get_annotation_categories()
    residue_bounds = get_residue_starts(atom_array, add_exclusive_stop=True)
    all_keys, selected = [], set()
    for lo, hi in pairwise(residue_bounds):
        ins_code = str(atom_array.ins_code[lo]) if has_ins_code else ""
        key = (str(atom_array.chain_id[lo]), int(atom_array.res_id[lo]), ins_code)
        all_keys.append(key)
        if atom_sel[lo:hi].any():
            selected.add(key)
    if len(all_keys) != len(set(all_keys)):
        raise ValueError(
            "AtomArray has residues sharing a (chain_id, res_id, ins_code) key; "
            "TMol's pdb_info cannot disambiguate them for block mapping."
        )
    return selected


def atom_sel_to_block_mask(
    pose_stack: PoseStack,
    atom_array: AtomArray,
    atom_sel: np.ndarray,
    device: torch.device,
    *,
    raise_on_dropped_residues: bool = True,
) -> torch.Tensor:
    """Map an atom selection to a TMol block mask via residue identity in pdb_info.

    Positional indices can diverge when TMol drops residues; unmatched selected residues raise by default.
    """
    selected_keys = _residue_keys(atom_array, atom_sel)
    pdb_info = pose_stack.pdb_info
    valid_blocks = (pose_stack.block_type_ind[0] >= 0).nonzero().flatten().tolist()
    block_mask_np = np.zeros((1, pose_stack.max_n_blocks), dtype=bool)
    matched_keys = set()
    for bi in valid_blocks:
        key = (
            str(pdb_info.chain_labels[0, bi]),
            int(pdb_info.residue_labels[0, bi]),
            str(pdb_info.residue_insertion_codes[0, bi]),
        )
        if key in selected_keys:
            block_mask_np[0, bi] = True
            matched_keys.add(key)

    dropped = selected_keys - matched_keys
    if dropped:
        msg = f"{len(dropped)} selected residue(s) dropped from the PoseStack by TMol: {sorted(dropped)}"
        if raise_on_dropped_residues:
            raise ValueError(msg)
        logger.warning(msg)

    if not block_mask_np.any():
        raise ValueError(
            "Atom selection matched atoms, but none survived as PoseStack blocks (TMol may have dropped them)"
        )

    return torch.as_tensor(block_mask_np, dtype=torch.bool, device=device)


def region_query_to_block_mask(
    pose_stack: PoseStack,
    atom_array: AtomArray,
    query: str,
    device: torch.device,
    *,
    raise_on_dropped_residues: bool = True,
) -> torch.Tensor:
    """Convert an AtomWorks query string region to a TMol block mask."""
    atom_sel = resolve_mask(atom_array, query)

    if not atom_sel.any():
        raise ValueError(f"Region query selected zero atoms: {query!r}")

    return atom_sel_to_block_mask(
        pose_stack, atom_array, atom_sel, device, raise_on_dropped_residues=raise_on_dropped_residues
    )


def build_pose_and_masks(
    atom_array: AtomArray,
    device: torch.device,
    *,
    no_opt_h: bool,
    region_a: str | None = None,
    region_b: str | None = None,
    raise_on_dropped_residues: bool = True,
    context: BiotitePoseBuildContext | None = None,
) -> tuple[PoseStack, torch.Tensor, torch.Tensor]:
    """Build a PoseStack and binder/target (or region) block masks.

    Pass a reusable BiotitePoseBuildContext to avoid redoing ligand prep; omit regions to
    infer binder as the chain of interest and target as everything else.
    """
    if context is not None:
        pose_stack = pose_stack_from_biotite(atom_array, device, context=context, no_optH=no_opt_h)
    else:
        pose_stack = pose_stack_from_biotite(atom_array, device, prepare_ligands=True, no_optH=no_opt_h)

    if region_a is not None and region_b is not None:
        mask_a = region_query_to_block_mask(
            pose_stack, atom_array, region_a, device, raise_on_dropped_residues=raise_on_dropped_residues
        )
        mask_b = region_query_to_block_mask(
            pose_stack, atom_array, region_b, device, raise_on_dropped_residues=raise_on_dropped_residues
        )
    else:
        coi, _ = get_chain_of_interest(atom_array)
        if coi is None:
            raise ValueError("Cannot infer binder chain — no protein chains in structure.")
        has_chain_iid = "chain_iid" in atom_array.get_annotation_categories()
        chain_ids = atom_array.chain_iid if has_chain_iid else atom_array.chain_id
        binder_mask_atoms = chain_ids == coi
        target_mask_atoms = ~binder_mask_atoms
        if not target_mask_atoms.any():
            raise ValueError(
                "Cannot infer binder/target interface split — only one chain. Set explicit "
                "region_a/region_b in the metric config."
            )
        mask_a = atom_sel_to_block_mask(
            pose_stack, atom_array, binder_mask_atoms, device, raise_on_dropped_residues=raise_on_dropped_residues
        )
        mask_b = atom_sel_to_block_mask(
            pose_stack, atom_array, target_mask_atoms, device, raise_on_dropped_residues=raise_on_dropped_residues
        )

    return pose_stack, mask_a, mask_b


def relax_pose_stack(
    pose_stack: PoseStack,
    sfxn: Any,
    *,
    cartesian: bool = True,
    num_repeats: int = 2,
) -> PoseStack:
    """Relax a PoseStack with TMol FastRelax (Cartesian or torsion-space)."""
    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)
    palette = PackerPalette()

    if cartesian:
        move_map = CartesianMoveMap()  # coord_mask=None -> every atom is free to move.
        min_fn = _default_cart_min_fn
    else:
        move_map = MoveMap.from_pose_stack(pose_stack)
        move_map.move_all_jumps = True
        move_map.move_all_named_torsions = True
        min_fn = None  # FastRelax's default is the kinematic (torsion-space) minimizer.

    return fast_relax(pose_stack, sfxn, palette, move_map, fold_forest, min_fn=min_fn, num_repeats=num_repeats)


def score_block_pair(
    pose_stack: PoseStack,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    sfxn: Any,
    *,
    term_names: list[str] | None = None,
    term_cutoffs: dict[str, tuple[str, float]] | None = None,
) -> dict[str, float]:
    """Score a block-pair interface; returns ``interface_ddg`` plus optional term outputs."""
    scorer = sfxn.render_block_pair_scoring_module(pose_stack)
    # [n_terms, n_poses, n_blocks, n_blocks]
    block_pair_scores = scorer(pose_stack.coords, sum_terms=False)
    if block_pair_scores.shape[1] != 1:
        raise ValueError("Expected a single pose for interface scoring")
    score_type_names = [st.name for st in sfxn.all_score_types()]

    # Scorer fills only one triangle of the block-pair matrix; OR both directions.
    cross_mask = ((mask_a.unsqueeze(2) & mask_b.unsqueeze(1)) | (mask_b.unsqueeze(2) & mask_a.unsqueeze(1)))[0]

    out = {"interface_ddg": float((block_pair_scores[:, 0] * cross_mask).sum().item())}
    for term_name in term_names or []:
        pair_vals = block_pair_scores[score_type_names.index(term_name), 0][cross_mask]
        out[term_name] = float(pair_vals.sum().item())
    for key, (term_name, cutoff) in (term_cutoffs or {}).items():
        pair_vals = block_pair_scores[score_type_names.index(term_name), 0][cross_mask]
        out[key] = float((pair_vals <= cutoff).sum().item())
    return out
