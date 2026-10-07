"""Shared TMol helpers for fast block-pair ddG metrics."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import tmol.database
import torch
from atomworks.constants import STANDARD_AA
from atomworks.io.utils.atom_array import chain_identifier
from atomworks.io.utils.ccd import get_chem_comp_type
from atomworks.io.utils.standard_annotations import STANDARD_ANNOTATIONS
from biotite.structure import AtomArray, get_residue_starts
from scipy.spatial import cKDTree
from tmol.database import ParameterDatabase
from tmol.io import PoseBuildContext, pose_stack_from_biotite
from tmol.kinematics import CartesianMoveMap, FoldForest, MoveMap
from tmol.optimization import run_kin_min
from tmol.pack import PackerPalette, PackerTask, pack_rotamers
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.pose import PoseStack
from tmol.relax import fast_relax
from tmol.score import ScoreFunction, beta2016_score_function

from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.mask_spec import resolve_mask

logger = RankedLogger(__name__, rank_zero_only=True)


def prepare_tmol_atom_array(atom_array: AtomArray) -> AtomArray:
    """Copy chemistry for TMol, retaining the original array for region selections.

    Canonicalize registered annotation aliases for hydrogen placement. Generic
    peptide labels on standard amino acids take their specific CCD component type;
    achiral GLY uses its CCD peptide type. Other explicit stereochemical labels
    and nonstandard components stay unchanged.
    """
    prepared = atom_array.copy()
    categories = set(prepared.get_annotation_categories())
    for annotation in STANDARD_ANNOTATIONS:
        if annotation.n_body != 1:
            continue
        aliases = categories.intersection(annotation.aliases) - {annotation.full_name}
        if not aliases:
            continue
        value = annotation.annotation(prepared, default="raise")
        annotation.set_annotation(prepared, value)
        for alias in aliases:
            prepared.del_annotation(alias)

    if "chem_comp_type" in categories:
        types = prepared.chem_comp_type.astype(object)
        normalized = np.char.upper(np.char.strip(types.astype(str)))
        generic = normalized == "PEPTIDE LINKING"
        for name in set(prepared.res_name[generic]).intersection(STANDARD_AA):
            types[generic & (prepared.res_name == name)] = get_chem_comp_type(name, mode="raise")
        glycine = (prepared.res_name == "GLY") & (normalized == "L-PEPTIDE LINKING")
        if np.any(glycine):
            types[glycine] = get_chem_comp_type("GLY", mode="raise")
        prepared.set_annotation("chem_comp_type", types.astype(str))
    return prepared


def create_score_function(
    device: torch.device,
    *,
    param_db: ParameterDatabase | None,
    use_soft_rep: bool = False,
) -> Any:
    """Create TMol beta2016, or ``beta_soft`` when ``use_soft_rep`` is set.

    ``param_db`` must be the ligand-extended database from a PoseBuildContext.
    """
    if param_db is None:
        raise ValueError("param_db is required; use the PoseBuildContext parameter database")

    if not use_soft_rep:
        return beta2016_score_function(device, param_db=param_db)

    weights_path = Path(tmol.database.__file__).parent / "score_functions" / "beta_soft.sfxn"
    return ScoreFunction.from_sfxn_file(str(weights_path), param_db, device)


def _residue_keys(atom_array: AtomArray, atom_sel: np.ndarray) -> set[tuple[str, int, str]]:
    """``(chain identifier, res_id, ins_code)`` keys for residues in ``atom_sel``.

    Keys must be unique globally: pdb_info cannot disambiguate duplicates.
    """
    effective_chains = chain_identifier(atom_array)
    if not np.array_equal(effective_chains, atom_array.chain_id):
        # Match TMol's instance-aware boundaries without changing selection indices.
        atom_array = atom_array.copy()
        atom_array.chain_id = effective_chains.copy()
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
            "AtomArray has residues sharing a (chain identifier, res_id, ins_code) key; "
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
    context: PoseBuildContext | None = None,
) -> tuple[PoseStack, torch.Tensor, torch.Tensor]:
    """Build a PoseStack and binder/target (or region) block masks.

    Pass a reusable PoseBuildContext to avoid redoing ligand prep; omit regions to
    infer binder as the chain of interest and target as everything else.
    """
    chemistry_array = prepare_tmol_atom_array(atom_array)
    if context is not None:
        pose_stack = pose_stack_from_biotite(chemistry_array, device, context=context, no_optH=no_opt_h)
    else:
        pose_stack = pose_stack_from_biotite(chemistry_array, device, prepare_ligands=True, no_optH=no_opt_h)

    if region_a is not None and region_b is not None:
        mask_a = region_query_to_block_mask(
            pose_stack, atom_array, region_a, device, raise_on_dropped_residues=raise_on_dropped_residues
        )
        mask_b = region_query_to_block_mask(
            pose_stack, atom_array, region_b, device, raise_on_dropped_residues=raise_on_dropped_residues
        )
    else:
        coi, coi_unambiguous = get_chain_of_interest(atom_array)
        if coi is None or not coi_unambiguous:
            raise ValueError("Cannot unambiguously infer the binder chain; set region_a/region_b explicitly.")
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


def _torsion_minimize(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    *,
    fold_forest: FoldForest,
    move_map: MoveMap,
    verbose: bool,
) -> PoseStack:
    """Adapt the public torsion minimizer to FastRelax's callback signature."""
    return run_kin_min(pose_stack, sfxn, fold_forest, move_map, verbose=verbose)


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
        min_fn = None
    else:
        move_map = MoveMap.from_pose_stack(pose_stack)
        move_map.move_all_jumps = True
        move_map.move_all_named_torsions = True
        min_fn = _torsion_minimize

    return fast_relax(pose_stack, sfxn, palette, move_map, fold_forest, min_fn=min_fn, num_repeats=num_repeats)


def shell_block_mask(
    pose_stack: PoseStack,
    atom_array: AtomArray,
    seed_query: str,
    radius: float,
    device: torch.device,
) -> torch.Tensor:
    """Block mask for residues with any atom within ``radius`` of the ``seed_query`` selection.

    The seed residue is always included (distance 0). NaN-coordinate atoms are ignored.
    """
    coord = atom_array.coord
    ok = np.isfinite(coord).all(axis=-1)
    seed = resolve_mask(atom_array, seed_query) & ok
    if not seed.any():
        raise ValueError(f"Mutation selection matched no atoms with coordinates: {seed_query!r}")
    near = np.zeros(atom_array.array_length(), dtype=bool)
    near[ok] = cKDTree(coord[seed]).query(coord[ok])[0] <= radius
    shell = np.zeros_like(near)
    for lo, hi in pairwise(get_residue_starts(atom_array, add_exclusive_stop=True)):
        if near[lo:hi].any():
            shell[lo:hi] = True
    return atom_sel_to_block_mask(pose_stack, atom_array, shell, device, raise_on_dropped_residues=False)


def _configure_shell_task(
    task: Any, shell_mask: torch.Tensor, mutate_to: str | None, mutation_mask: torch.Tensor | None
) -> None:
    """Restrict a PackerTask to repack the shell (freezing the rest), optionally mutating a site.

    With ``mutate_to`` set, the ``mutation_mask`` block is opened to only that name3 (design);
    everything in the shell repacks at its own identity, everything outside is frozen.
    """
    task.restrict_to_repacking()
    if mutate_to is not None:
        considered = task.per_block_considered_block_types
        pose, block, slot = (mutation_mask.unsqueeze(-1) & (considered != -1)).nonzero(as_tuple=True)
        match = torch.tensor([bt.name3 == mutate_to for bt in task.pbt.active_block_types], device=task.device)
        task.per_block_is_block_type_allowed[pose, block, slot] = match[considered[pose, block, slot]]
        if not task.per_block_is_block_type_allowed[mutation_mask].any():
            raise ValueError(f"name3 {mutate_to!r} is not an available residue type at the mutation site")
    task.disable_packing_by_block_mask(~shell_mask)
    dun = create_dunbrack_sampler_from_database(ParameterDatabase.get_default(), task.device)
    task.add_conformer_sampler(dun)
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(IncludeCurrentSampler())


def pack_shell(
    pose_stack: PoseStack,
    sfxn: Any,
    shell_mask: torch.Tensor,
    *,
    mutate_to: str | None = None,
    mutation_mask: torch.Tensor | None = None,
) -> PoseStack:
    """Repack the ``shell_mask`` blocks in one pass, optionally mutating a site to ``mutate_to``.

    Separate from :func:`relax_shell` because a mutation changes the atom count, which
    ``fast_relax``'s accept/reject step cannot span.
    """
    task = PackerTask(pose_stack, PackerPalette())
    _configure_shell_task(task, shell_mask, mutate_to, mutation_mask)
    return pack_rotamers(pose_stack, sfxn, task)


def relax_shell(pose_stack: PoseStack, sfxn: Any, shell_mask: torch.Tensor, *, num_repeats: int = 2) -> PoseStack:
    """Repack + torsion-minimize only ``shell_mask`` blocks at their current identities, backbone fixed."""
    move_map = MoveMap.from_pose_stack(pose_stack)
    move_map.set_move_all_sc_tors_for_blocks(shell_mask)
    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)
    return fast_relax(
        pose_stack,
        sfxn,
        PackerPalette(),
        move_map,
        fold_forest,
        task_operations=[lambda task: _configure_shell_task(task, shell_mask, None, None)],
        min_fn=_torsion_minimize,
        num_repeats=num_repeats,
    )


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
