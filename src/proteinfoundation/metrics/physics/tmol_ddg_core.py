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
from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.move_map import CartesianMoveMap, MoveMap
from tmol.pack.packer_task import PackerPalette
from tmol.pose.pose_stack import PoseStack
from tmol.relax.fast_relax import _default_cart_min_fn, fast_relax
from tmol.score import beta2016_score_function
from tmol.score.score_function import ScoreFunction

from proteinfoundation.metrics.utils.atom_selection import atom_selection_from_spec
from proteinfoundation.utils.chain_utils import get_chain_of_interest
from proteinfoundation.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def create_score_function(device: torch.device, *, use_soft_rep: bool = False) -> Any:
    """Create TMol's beta2016 score function, or its native ``beta_soft`` variant.

    ``beta_soft`` (``tmol/database/score_functions/beta_soft.yaml``) is TMol's own
    softened-repulsion score function — a smoother LJ-repulsion potential plus rebalanced
    reference weights, not just a smaller ``fa_ljrep`` weight — recommended for scoring
    AI-generated structures that may still have minor clashes.
    """
    if not use_soft_rep:
        return beta2016_score_function(device)

    weights_path = Path(tmol.database.__file__).parent / "score_functions" / "beta_soft.yaml"
    return ScoreFunction.from_weights_file(str(weights_path), ParameterDatabase.get_default(), device)


def _residue_keys(atom_array: AtomArray, atom_sel: np.ndarray) -> set[tuple[str, int, str]]:
    """``(chain_id, res_id, ins_code)`` keys for residues with an atom in ``atom_sel``.

    Author identity is all TMol's :attr:`PoseStack.pdb_info` retains, so two residues
    sharing a key would make the block-to-residue mapping ambiguous — hence a global
    uniqueness check across every residue, not just the selected ones.
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
    """Map a boolean atom selection to a TMol block mask, by residue identity.

    TMol may silently drop residues while building the PoseStack (waters, unsupported
    ligands, residues missing backbone atoms), so block index and AtomArray residue
    index can diverge. Matching by ``(chain_id, res_id, ins_code)`` via
    :attr:`PoseStack.pdb_info` — which TMol keeps in sync with kept blocks — is robust
    to that, unlike a positional mapping. A selected residue with no surviving block
    was dropped: raise when ``raise_on_dropped_residues`` (default), else warn.
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


def atom_spec_to_block_mask(
    pose_stack: PoseStack,
    atom_array: AtomArray,
    region_spec: dict[str, Any] | list[dict[str, Any]],
    device: torch.device,
    *,
    raise_on_dropped_residues: bool = True,
) -> torch.Tensor:
    """Convert atom-level region specs to TMol block masks."""
    specs = region_spec if isinstance(region_spec, list) else [region_spec]
    atom_sel = np.zeros(atom_array.array_length(), dtype=bool)
    for spec in specs:
        atom_sel |= atom_selection_from_spec(atom_array, spec)

    if not atom_sel.any():
        raise ValueError(f"Region spec selected zero atoms: {region_spec}")

    return atom_sel_to_block_mask(
        pose_stack, atom_array, atom_sel, device, raise_on_dropped_residues=raise_on_dropped_residues
    )


def build_pose_and_masks(
    atom_array: AtomArray,
    device: torch.device,
    *,
    no_opt_h: bool,
    region_a: dict[str, Any] | list[dict[str, Any]] | None = None,
    region_b: dict[str, Any] | list[dict[str, Any]] | None = None,
    raise_on_dropped_residues: bool = True,
) -> tuple[PoseStack, torch.Tensor, torch.Tensor]:
    """Build a PoseStack and binder/target (or region) block masks.

    No ``ParameterDatabase`` is passed to TMol: with ``prepare_ligands=True``, TMol
    derives ligand chemistry (SMILES -> RDKit/OpenBabel typing) from the AtomArray
    itself and defaults to ``ParameterDatabase.get_default()`` (a cached singleton)
    otherwise. When ``region_a``/``region_b`` are omitted, the binder is the chain of
    interest (see :func:`~proteinfoundation.utils.chain_utils.get_chain_of_interest`)
    and the target is every other atom. ``raise_on_dropped_residues`` forwards to
    :func:`atom_sel_to_block_mask`.
    """
    pose_stack = pose_stack_from_biotite(atom_array, device, prepare_ligands=True, no_optH=no_opt_h)

    if region_a is not None and region_b is not None:
        mask_a = atom_spec_to_block_mask(
            pose_stack, atom_array, region_a, device, raise_on_dropped_residues=raise_on_dropped_residues
        )
        mask_b = atom_spec_to_block_mask(
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
    """Relax an entire PoseStack via TMol's FastRelax (repeated rotamer packing +
    ramped-fa_rep minimization), Cartesian or torsion-space.

    ``task_operations=None`` uses FastRelax's own default packer task (Dunbrack
    rotamers + fixed-chi + include-current, restricted to repacking).
    """
    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)
    palette = PackerPalette(pose_stack.packed_block_types.restype_set)

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
    """Score a block-pair interface in one TMol pass.

    Always returns ``interface_ddg`` (the sum of every score term over the cross-region
    block pairs). ``term_names`` additionally gives each named term's own summed
    cross-region energy, keyed by that term name (e.g. ``["hbond"]`` -> ``{"hbond":
    ...}``). ``term_cutoffs`` maps an output key to ``(term_name, energy_cutoff)``,
    adding the count of cross-region block-pairs at or under ``energy_cutoff`` for that
    term, keyed by the given key.
    """
    scorer = sfxn.render_block_pair_scoring_module(pose_stack)
    # [n_terms, n_poses, n_blocks, n_blocks]
    block_pair_scores = scorer(pose_stack.coords, sum_terms=False)
    if block_pair_scores.shape[1] != 1:
        raise ValueError("Expected a single pose for interface scoring")
    score_type_names = [st.name for st in sfxn.all_score_types()]

    # TMol's block-pair scorer only populates one triangular half of the (block, block)
    # matrix (whichever of (i, j) / (j, i) it computed), so both cross directions must
    # be OR'd — a one-sided mask silently reads all zeros whenever region_a's block
    # indices land after region_b's (the common case, e.g. a ligand added after chains).
    cross_mask = ((mask_a.unsqueeze(2) & mask_b.unsqueeze(1)) | (mask_b.unsqueeze(2) & mask_a.unsqueeze(1)))[0]

    out = {"interface_ddg": float((block_pair_scores[:, 0] * cross_mask).sum().item())}
    for term_name in term_names or []:
        pair_vals = block_pair_scores[score_type_names.index(term_name), 0][cross_mask]
        out[term_name] = float(pair_vals.sum().item())
    for key, (term_name, cutoff) in (term_cutoffs or {}).items():
        pair_vals = block_pair_scores[score_type_names.index(term_name), 0][cross_mask]
        out[key] = float((pair_vals <= cutoff).sum().item())
    return out
