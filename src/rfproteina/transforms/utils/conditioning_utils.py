import numpy as np
import torch
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import AnnotationList2D
from atomworks.ml.conditions import C_CRD, C_DIS
from atomworks.ml.conditions.base import ConditionBase
from atomworks.ml.utils.geometry import random_rigid_augmentation
from biotite.structure import superimpose
from biotite.structure.atoms import AtomArray
from biotite.structure.residues import get_residue_starts, spread_residue_wise

from rfproteina.conditions import C_CRDGRP, C_IDX, C_IDXGRP, C_INTCRD
from rfproteina.utils.common import true_with_prob
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.standard_annotations import SA_CWI

global_logger = RankedLogger(__name__, rank_zero_only=False)


def sample_island_tokens(
    array_length: int,
    island_len_min: int = 5,
    island_len_max: int = 30,
    n_islands_min: int = 1,
    n_islands_max: int = 30,
    max_length: int | None = None,
) -> np.ndarray:
    """
    Generate a boolean mask of length `array_length` with random contiguous islands (True segments)
    while optionally constraining the total number of True values.

    Args:
        array_length (int): Total length of the boolean array.
        island_len_min (int): Minimum island length (inclusive).
        island_len_max (int): Maximum island length (inclusive).
        n_islands (int): Number of islands to attempt to generate.
        max_length (int, optional): Maximum allowed total number of True values in the output.
                                    If None, no constraint is applied.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Boolean array of length `array_length` with island positions set to True.
    """
    n_islands = np.random.randint(n_islands_min, n_islands_max + 1)

    mask = np.zeros(array_length, dtype=bool)
    for _ in range(n_islands):
        current_total = mask.sum()
        if max_length is not None:
            if current_total >= max_length:
                break
            remaining = max_length - current_total
        else:
            remaining = None  # not used

        # Randomly select a candidate island length.
        candidate_length = np.random.randint(island_len_min, island_len_max + 1)
        candidate_length = min(candidate_length, array_length)  # Fit into array

        # Choose a random starting index ensuring the island fits.
        high_start = array_length - candidate_length
        start = np.random.randint(0, high_start + 1)

        # Evaluate the segment that would be activated.
        segment = mask[start : start + candidate_length]
        new_trues = np.sum(~segment)

        # If we have a maximum True budget and adding all new positions would exceed it, adjust the island.
        if max_length is not None and new_trues > remaining:
            # We try to trim the island so that it adds at most `remaining` new True values.
            count_new = 0
            adjusted_length = 0
            for i in range(candidate_length):
                if not mask[start + i]:
                    count_new += 1
                adjusted_length += 1
                # Once we've added as many new trues as allowed, break.
                if count_new >= remaining:
                    break
            # Only add the island if its adjusted length meets the minimum requirement.
            if adjusted_length < island_len_min:
                continue  # Skip this island and try the next one.
            mask[start : start + adjusted_length] = True
        else:
            # No max constraint or this candidate island fits within the remaining budget.
            mask[start : start + candidate_length] = True

    assert mask.sum() <= array_length, "Generated mask exceeds array length."
    return mask


#################################################################################
# Utility functions
#################################################################################


def replace_2d_distance_conditioning_with_3d_coordinate_conditioning(
    atom_array: AtomArray,
    mask: np.ndarray,
) -> AtomArray:
    """
    Removes 2D distance conditioning from masked atoms in the atom array and instead adds
    absolute coordinate conditioning

    Args:
        atom_array: AtomArrayPlus
        mask: Bool[Array, "L"]

    Returns:
        AtomArrayPlus
    """
    assert len(mask) == atom_array.array_length(), "Mask must be the same length as the atom array"
    if not mask.any():
        return atom_array

    # remove 2D distance conditioning
    conditioning_inds = np.where(mask)[0]
    annot_2d = C_DIS.annotation(atom_array, default="generate")
    pairs, values = annot_2d.pairs, annot_2d.values
    pairs_to_discard_mask = np.any(np.isin(pairs, conditioning_inds), axis=1)
    pairs = pairs[~pairs_to_discard_mask]
    values = values[~pairs_to_discard_mask]
    C_DIS.set_annotation(
        atom_array,
        AnnotationList2D(n_atoms=atom_array.array_length(), pairs=pairs, values=values),
    )

    nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
    mask = mask & ~nan_coord_mask
    C_CRD.set_mask(atom_array, mask)
    return atom_array


def sample_unindexed_groups(
    atom_array: AtomArray,
    p_merge_groups: float = 0.0,
    p_split_group: float = 0.0,
    p_increase_group_id_gap: float = 0.1,
) -> None:
    """Sample the unindexed groups in an AtomArray."""

    ensure_annotations(atom_array, "within_chain_source_res_idx")

    # NOTE: At this stage, we have not yet called UnindexFlaggedTokens, so the source residue for a tip-atom unindexing
    # case would contains both indexed and unindexed atoms. We therefore first subset to the unindexed atoms before
    # calling get_residue_starts
    unindexed_atom_array = atom_array[~C_IDX.mask(atom_array, default="raise")]
    unindexed_residue_starts = get_residue_starts(unindexed_atom_array)
    if unindexed_atom_array.array_length() == 0:
        return C_IDXGRP.default_annotation(atom_array)

    # Get ground-truth of which residues are adjacent and on the same chain
    unindexed_contiguity_breaks = (
        np.diff(unindexed_atom_array.within_chain_source_res_idx[unindexed_residue_starts]) != 1
    )

    # Get chain annotation
    chain_annot_name = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
    chain_annot_residue_level = unindexed_atom_array.get_annotation(chain_annot_name)[unindexed_residue_starts]

    unindexed_chain_breaks = chain_annot_residue_level[1:] != chain_annot_residue_level[:-1]

    # Convert these to group ids, starting at 1 for the unindexed groups.
    unindexed_group_ids = np.cumsum(np.concatenate([[True], unindexed_contiguity_breaks | unindexed_chain_breaks]))

    # Merging groups allows for the use-case of two unindexed components with a fixed residue gap between them
    if true_with_prob(p_merge_groups) and unindexed_group_ids.max() > 1:
        # Sample the gap between the group ids to merge. This corresponds the number of unindexed
        # groups that belong between the final positions of the two (non-contiguous) merged groups
        # It is currently parametrized with a geometric distribution, as this "interleaving" becomes
        # less likely in practice the more groups there are in between
        max_group_id_gap = unindexed_group_ids.max() - unindexed_group_ids.min()
        geometric_parameter = 1 - p_increase_group_id_gap
        sampled_group_id_gap = min(np.random.geometric(geometric_parameter), max_group_id_gap)

        # Identify valid pairings for this gap size
        valid_smaller_group_ids = unindexed_group_ids[
            np.isin(unindexed_group_ids + sampled_group_id_gap, unindexed_group_ids)
        ]

        # Ensure that the two groups to be merged are on the same chain
        chain_by_group_id = {}
        for group_id in np.unique(unindexed_group_ids):
            chain_by_group_id[group_id] = chain_annot_residue_level[unindexed_group_ids == group_id][0]

        valid_smaller_group_ids = [
            gid
            for gid in valid_smaller_group_ids
            if chain_by_group_id[gid] == chain_by_group_id[gid + sampled_group_id_gap]
        ]

        if len(valid_smaller_group_ids) > 0:
            # Given the gap, uniformly sample the pair of groups to merge
            smaller_group_id = np.random.choice(valid_smaller_group_ids)
            larger_group_id = smaller_group_id + sampled_group_id_gap

            unindexed_group_ids[unindexed_group_ids == larger_group_id] = smaller_group_id
            if larger_group_id < unindexed_group_ids.max():
                unindexed_group_ids[unindexed_group_ids > larger_group_id] -= 1

    # Construct the corresponding atom-level annotation
    unindexed_group_ids_atom_level = spread_residue_wise(unindexed_atom_array, unindexed_group_ids)

    # Splitting a group avoids incorrectly teaching the model that different unindexed groups are never adjacent
    if true_with_prob(p_split_group):
        # To be split, an unindexed group must contain at least two residue starts
        all_unindexed_group_ids = np.unique(unindexed_group_ids)
        unindexed_residue_starts_mask = np.zeros(unindexed_atom_array.array_length(), dtype=bool)
        unindexed_residue_starts_mask[unindexed_residue_starts] = True
        candidate_group_ids_to_split = [
            gid
            for gid in all_unindexed_group_ids
            if ((unindexed_group_ids_atom_level == gid) & unindexed_residue_starts_mask).sum() > 1
        ]

        # We only split between residues
        if len(candidate_group_ids_to_split) > 0:
            group_to_split = np.random.choice(candidate_group_ids_to_split)
            group_to_split_mask = unindexed_group_ids_atom_level == group_to_split
            candidate_split_points = np.where(group_to_split_mask & unindexed_residue_starts_mask)[0]
            split_point = np.random.choice(
                candidate_split_points[1:]
            )  # Splitting at first residue start is meaningless
            group_to_split_indices = np.where(group_to_split_mask)[0]
            indices_to_reassign = group_to_split_indices[group_to_split_indices >= split_point]
            unindexed_group_ids_atom_level[indices_to_reassign] = unindexed_group_ids_atom_level.max() + 1

    # Get the default annotation for this AtomArray, which is correct for atoms with absolute indices
    all_index_group_ids = C_IDXGRP.default_annotation(atom_array)

    # Add the maximum indexed group id to each of the unindexed group ids (which start at 1)
    # Note that this may create gaps in the group ids, which is fine since the absolute values should never matter
    absolute_indexed_atom_mask = C_IDX.mask(atom_array, default="raise")
    if absolute_indexed_atom_mask.any():
        max_absolute_indexed_group_id = all_index_group_ids[absolute_indexed_atom_mask].max()
    else:
        max_absolute_indexed_group_id = -1
    all_index_group_ids[~absolute_indexed_atom_mask] = unindexed_group_ids_atom_level + max_absolute_indexed_group_id

    return all_index_group_ids


def set_specified_conditions_to_default(
    atom_array: AtomArray,
    conditions: list[ConditionBase] | dict[ConditionBase, dict[str, bool]],
) -> AtomArray:
    """Sets the annotations and/or masks of the specified Conditions to their default values.

    This is done by first removing all the Conditions and then iteratively re-adding the defaults, to avoid any affect
    on the computation of defaults due to the presence of non-default Conditions to be removed.

    Args:
        atom_array (AtomArray): The AtomArray to modify.
        conditions (list[ConditionBase] | dict[ConditionBase, dict[str, bool]]): The Conditions to
            reset. If a list, all specified Conditions will have their existing masks/annotations
            reset to default. If a dict, the keys are the Conditions to reset, and the values are
            dicts with keys "mask" and "annotation" indicating whether to add the corresponding
            defaults, whether present or not.
    """

    condition_dict = {}
    for condition in conditions:
        if not issubclass(condition, ConditionBase):
            raise ValueError(f"Condition {condition} must be a subclass of ConditionBase.")

    # Convert inputs to a standardized format
    if isinstance(conditions, dict):
        for val_dict in conditions.values():
            if not isinstance(val_dict, dict) or set(val_dict.keys()) != {
                "mask",
                "annotation",
            }:
                raise ValueError(f"Condition dict value {val_dict} must be a dict with keys 'mask' and 'annotation'.")
        condition_dict = conditions
    elif isinstance(conditions, list):
        for condition in conditions:
            condition_dict[condition] = {"mask": False, "annotation": False}
            if condition.get_mask_name() in atom_array.get_annotation_categories():
                condition_dict[condition]["mask"] = True
            if condition.get_full_name() in atom_array.get_annotation_categories():
                condition_dict[condition]["annotation"] = True

    # Remove all conditioning, so that we can compute all of the default annotations independently
    for cond in condition_dict:
        if condition_dict[cond]["mask"]:
            atom_array.del_annotation(cond.get_mask_name())
        if condition_dict[cond]["annotation"]:
            atom_array.del_annotation(cond.get_full_name())

    # Now that we have an AtomArray with no Condition annotations, compute the default values
    # NOTE: We assume that order of applying defaults doesn't matter -- otherwise, the Condition default is not robust.
    for cond in condition_dict:
        if condition_dict[cond]["mask"]:
            cond.set_mask(atom_array, cond.default_mask(atom_array))
        if condition_dict[cond]["annotation"]:
            cond.set_annotation(atom_array, cond.default_annotation(atom_array))

    return atom_array


def convert_to_internal_coords(atom_array: AtomArray, mask: np.ndarray) -> AtomArray:
    """Convert the given masked region of the atom array from global to internal coordinate conditioning.

    Atoms in ``mask`` with NaN coordinates are dropped from the mask.
    """
    mask = mask & np.isfinite(atom_array.coord).all(axis=1)

    # Absolute coord conditioning mask is False for the new internal coord region
    coord_condition_mask = C_CRD.mask(atom_array, default="generate")
    new_coord_condition_mask = coord_condition_mask & ~mask
    C_CRD.set_mask(atom_array, new_coord_condition_mask)

    # This annotation is rarely used, but if present it should have NaNs for internally-conditioned regions
    if C_CRD.full_name in atom_array.get_annotation_categories():
        absolute_coord_annotation = C_CRD.annotation(atom_array, default="raise")
        absolute_coord_annotation[mask, :] = np.nan
        C_CRD.set_annotation(atom_array, absolute_coord_annotation)

    # Create a new coord group for the new internal coord region
    coord_groups = C_CRDGRP.annotation(atom_array, default="generate")
    coord_groups[mask] = coord_groups.max() + 1 if coord_groups.max() > 0 else 1
    C_CRDGRP.set_annotation(atom_array, coord_groups)

    # Set internal coord conditioning mask to True for the new internal coord region
    internal_coords_mask = C_INTCRD.mask(atom_array, default="generate")
    internal_coords_mask = internal_coords_mask | mask
    C_INTCRD.set_mask(atom_array, internal_coords_mask)

    # Provide the centered and randomly-rotated ground-truth coords for the new internal coord region
    internal_coord_annotation = C_INTCRD.annotation(atom_array, default="generate")
    global_coords = atom_array.coord[mask, :]
    centered_coords = global_coords - global_coords.mean(axis=0, keepdims=True)
    internal_coords = random_rigid_augmentation(torch.from_numpy(centered_coords)[None], batch_size=1, s=0)[0].numpy()
    internal_coord_annotation[mask, :] = internal_coords
    C_INTCRD.set_annotation(atom_array, internal_coord_annotation)

    return atom_array


def insert_internal_coord_conditioned_groups(atom_array: AtomArray, original_input_coords: np.ndarray) -> AtomArray:
    """Update coord_with_insertion AtomArray annotation with internal coordinate conditioning.

    Aligns each internally-conditioned group to the model-generated coordinates before insertion.
    """
    intcrd_mask = C_INTCRD.mask(atom_array)
    if not intcrd_mask.any():
        return atom_array

    if not SA_CWI.has_annotation(atom_array):
        SA_CWI.set_annotation(atom_array, SA_CWI.default_annotation(atom_array))

    # Find unique internal coordinate groups (exclude absolute coords and unknown coords)
    coord_groups = C_CRDGRP.annotation(atom_array, default="generate")
    unique_groups = np.unique(coord_groups[intcrd_mask])
    unique_groups = unique_groups[
        (unique_groups != C_CRDGRP.KNOWN_ABSOLUTE_COORDS) & (unique_groups != C_CRDGRP.UNKNOWN_COORDS)
    ]

    # Align and insert each internal coordinate group separately
    for group_id in unique_groups:
        # Get mask for atoms in this coordinate group with internal coordinate conditioning
        group_intcrd_mask = intcrd_mask & (coord_groups == group_id)

        # Extract coordinates for alignment
        original_coords = original_input_coords[group_intcrd_mask]  # [n_atoms, 3]
        generated_coords = atom_array.coord[group_intcrd_mask]  # [n_atoms, 3]

        # Check for valid coordinates (no NaNs)
        if np.isnan(original_coords).any() or np.isnan(generated_coords).any():
            raise ValueError(f"NaN values detected in input or generated coords for coord group {group_id}.")

        # Superimpose original internal coordinates onto model-generated coordinates
        # This aligns the internal coordinate conditioning to match the model output
        aligned_coords, _ = superimpose(
            fixed=generated_coords,
            mobile=original_coords,
        )

        # Update coords_with_insertion with aligned coordinates
        cwi = SA_CWI.annotation(atom_array)
        cwi[group_intcrd_mask] = aligned_coords
        SA_CWI.set_annotation(atom_array, cwi)

    return atom_array


def align_atom_array_to_input_frame(atom_array: AtomArray, crdgrp_id: int | None = None) -> AtomArray:
    """Apply a rigid transform so the chosen coord group's atoms sit at their input coords.

    Kabsch-aligns ``atom_array.coord[anchor]`` onto ``SA_CWI.annotation[anchor]``,
    and applies that transform to every atom in the array. This is purely cosmetic:
    it makes saved CIFs open consistently across designs (motif at its input position,
    scaffold rotated to match).

    Anchor selection:
      - ``crdgrp_id`` if explicitly provided
      - Else the ``KNOWN_ABSOLUTE_COORDS`` bucket (C_CRD atoms; all share one frame)
      - Else the largest single C_INTCRD group (deterministic fallback)
      - Else no-op if no anchor found or fewer than 3 atoms

    Args:
        atom_array: AtomArray with ``SA_CWI`` populated for conditioned atoms.
        crdgrp_id: Optional ``C_CRDGRP`` value to use as the anchor.

    Returns:
        The same AtomArray with ``coord`` updated in place (and returned for chaining).
    """
    if not SA_CWI.has_annotation(atom_array):
        return atom_array

    coord_groups = C_CRDGRP.annotation(atom_array, default="generate")
    cwi = SA_CWI.annotation(atom_array)
    if cwi.ndim == 3:
        cwi = cwi[0]
    cwi_valid = ~np.isnan(cwi).any(axis=-1)

    if crdgrp_id is None:
        absolute_mask = (coord_groups == C_CRDGRP.KNOWN_ABSOLUTE_COORDS) & cwi_valid
        if absolute_mask.any():
            anchor_mask = absolute_mask
        else:
            intcrd_groups = np.unique(
                coord_groups[
                    cwi_valid
                    & (coord_groups != C_CRDGRP.KNOWN_ABSOLUTE_COORDS)
                    & (coord_groups != C_CRDGRP.UNKNOWN_COORDS)
                ]
            )
            if intcrd_groups.size == 0:
                return atom_array
            sizes = {int(g): int(((coord_groups == g) & cwi_valid).sum()) for g in intcrd_groups}
            anchor_group = max(sizes, key=sizes.get)
            anchor_mask = (coord_groups == anchor_group) & cwi_valid
    else:
        anchor_mask = (coord_groups == crdgrp_id) & cwi_valid
        if not anchor_mask.any():
            return atom_array

    if anchor_mask.sum() < 3:
        return atom_array

    mobile = atom_array.coord[anchor_mask]
    _, transform = superimpose(fixed=cwi[anchor_mask], mobile=mobile)
    atom_array.coord = transform.apply(atom_array.coord)
    return atom_array
