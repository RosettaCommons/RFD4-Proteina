import copy
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import biotite.structure as struc
import numpy as np
from atomworks.io.utils.atom_array_plus import concatenate_any
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.ml.conditions import C_CRD, C_CTR, C_DIS, C_IDX, C_NTR, C_SEQ
from atomworks.ml.conditions.annotator import (
    ensure_annotations,
)
from atomworks.ml.conditions.base import CONDITIONS
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
from atomworks.ml.transforms._checks import (
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atom_array import (
    add_global_token_id_annotation,
    add_global_atom_id_annotation,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import (
    apply_and_spread_token_wise,
    get_token_count,
)
from biotite.structure import AtomArray
from biotite.structure.residues import apply_residue_wise, get_residue_starts, spread_residue_wise
from foundry.utils.ddp import RankedLogger

from proteinfoundation.conditions import C_IDXGRP, C_INTCRD
from proteinfoundation.transforms.utils.conditioning_utils import (
    set_specified_conditions_to_default,
)

global_logger = RankedLogger(__name__, rank_zero_only=False)


class UnindexFlaggedTokens(Transform):
    """Create in-context conditioning for tokens with unknown indices."""

    requires_previous_transforms = [
        "EnsureRepresentativeAndCenterAtomAnnotations",
    ]

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def expand_unindexed_motifs(self, atom_array: AtomArray, is_inference: bool) -> AtomArray:
        """
        Takes atom array with C_IDX annotation and padds it to include unindexed motif atoms.

        During training, the original coordinates are left behind for the model to learn to diffuse, with all Conditions
        set to their defaults. During inference, the original tokens are removed.
        """

        # Assemble necessary annotations
        is_motif_atom_unindexed = ~C_IDX.mask(atom_array, default="generate")
        ensure_annotations(atom_array, "is_protein", "within_chain_res_idx")

        # This will be modified if any unindexed tokens are present to give the correct ground-truth index gaps (within
        # unindexed groups). This enables specification of defined index gaps greater than 1 between unindexed tokens.
        atom_array.set_annotation("within_index_group_res_idx", atom_array.within_chain_res_idx.copy())

        if not np.any(is_motif_atom_unindexed):
            return atom_array

        unique_chain_iids = np.unique(atom_array.chain_iid)
        max_resid_by_chain_iid = {
            chain_iid: np.max(atom_array.res_id[atom_array.chain_iid == chain_iid]) for chain_iid in unique_chain_iids
        }
        max_atom_idx = (
            atom_array.array_length() - 1  # start one less because of the way we increment it
        )
        starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)

        # Identify tokens which have at least one unindexed atom
        token_to_unindex = struc.spread_residue_wise(
            atom_array,
            struc.apply_residue_wise(
                atom_array,
                is_motif_atom_unindexed,
                function=lambda x: np.any(x),
            ),
        )
        assert token_to_unindex.sum() > 0, "No tokens to unindex!"

        # record original annotations and masks, to be added back later where appropriate
        original_1d_conditions = {}
        original_2d_conditions = {}
        annotation_categories = get_annotation_categories(atom_array, n_body="all")
        for cond in CONDITIONS:
            # Add to the appropriate dict for the current n_body
            if cond.n_body == 0:
                continue
            elif cond.n_body == 1:
                dict_to_use = original_1d_conditions
            elif cond.n_body == 2:
                dict_to_use = original_2d_conditions
            else:
                raise NotImplementedError(f"Unindexing does not yet support conditions with n_body={cond.n_body}")

            has_mask = cond.get_mask_name() in annotation_categories
            has_annot = cond.get_full_name() in annotation_categories

            # Store the annotation and/or mask if present
            if has_mask:
                if cond.name not in dict_to_use:
                    dict_to_use[cond.name] = {}
                dict_to_use[cond.name]["mask"] = cond.mask(atom_array, default="raise")
            if has_annot:
                if cond.name not in dict_to_use:
                    dict_to_use[cond.name] = {}
                dict_to_use[cond.name]["annotation"] = cond.annotation(atom_array, default="raise")

        idxs = np.arange(atom_array.array_length())
        unindexed_tokens = []
        old_to_new_atom_idx_map = {}
        for i, (start, end) in enumerate(zip(starts[:-1], starts[1:], strict=False)):
            if not token_to_unindex[start]:
                continue

            # Subset to only the unindexed atoms
            subset_mask = np.isin(idxs, idxs[start:end])
            token = copy.deepcopy(atom_array[subset_mask])
            within_residue_is_motif_atom_unindexed = is_motif_atom_unindexed[subset_mask]
            used_idxs = idxs[start:end][within_residue_is_motif_atom_unindexed]
            token = token[within_residue_is_motif_atom_unindexed]
            chain_iid = token.chain_iid[0]

            # Break in contiguity between indexed and unindexed regions is fine since they never get relative indices
            token.res_id = token.res_id + max_resid_by_chain_iid[chain_iid]

            # Do not leak index through termini features
            C_NTR.set_mask(token, np.full(token.array_length(), False, dtype=bool))
            C_CTR.set_mask(token, np.full(token.array_length(), False, dtype=bool))

            # TODO: When we introduce the is_canonical annotation, we should enforce that here instead
            assert token.is_protein.all(), f"Cannot unindex non-protein token: {token}"

            unindexed_tokens.append(token)

            # Label the first atom of a token fragment as the representative atom and the central atom
            is_token_rep = token.is_token_representative_atom
            if is_token_rep.sum() > 1:
                raise ValueError(f"Token {token} has multiple representative atoms.")
            elif is_token_rep.sum() == 0:
                is_token_rep[0] = True
            is_token_center = token.is_token_center_atom
            if is_token_center.sum() > 1:
                raise ValueError(f"Token {token} has multiple center atoms.")
            elif is_token_center.sum() == 0:
                is_token_center[0] = True

            # update index map for future 2d conditioning updates
            old_to_new_atom_idx_map.update({old_idx: (max_atom_idx := max_atom_idx + 1) for old_idx in used_idxs})

        assert len(unindexed_tokens) > 0, "No unindexed tokens found!"

        # ... Remove original tokens e.g. during inference
        if is_inference:
            # some indexed atoms might change location when we delete the unindexed tokens
            original_indexed_positions = np.arange(atom_array.array_length())[~token_to_unindex]
            new_indexed_positions = np.arange((~token_to_unindex).sum())

            # update old_to_new_atom_idx_map to reflect shortened atom array
            old_to_new_atom_idx_map = {
                old_idx: new_idx - token_to_unindex.sum() for old_idx, new_idx in old_to_new_atom_idx_map.items()
            }

            # update old_to_new_atom_idx_map to reflect moving indexed atoms
            old_to_new_atom_idx_map.update(
                {
                    old_idx: new_idx
                    for old_idx, new_idx in zip(original_indexed_positions, new_indexed_positions, strict=False)
                }
            )

            # Actually delete the unindexed tokens
            atom_array = atom_array[~token_to_unindex]

        else:
            # Temporarily set all previously-existing conditions to their default values
            conditions_to_remove_dict = {}
            for cond_name, val in original_1d_conditions.items():
                cond = CONDITIONS[cond_name]
                conditions_to_remove_dict[cond] = {}
                for k in ["mask", "annotation"]:
                    if k in val:
                        conditions_to_remove_dict[cond][k] = True
                    else:
                        conditions_to_remove_dict[cond][k] = False
            set_specified_conditions_to_default(atom_array, conditions=conditions_to_remove_dict)

            # Overwrite the null conditions everywhere except on tokens with atoms which were copied for unindexing
            # TODO: Doesn't this procedure of setting annotations to their defaults partially leak the identity of the
            # ground-truth tokens that were copied for unindexing? Keeping them also partially leaks it, so maybe we
            # should do this reset to defaults stochastically? To avoid the model learning something undesirable. And
            # unindexed tokens should also get some of their non-positional Conditions set to default stochastically.
            for cond_name, condition_dict in original_1d_conditions.items():
                cond = CONDITIONS[cond_name]
                if "mask" in condition_dict:
                    current_mask = condition_dict["mask"]
                    default_mask = cond.mask(atom_array, default="raise")
                    current_mask[token_to_unindex] = default_mask[token_to_unindex]
                    cond.set_mask(atom_array, current_mask)
                if "annotation" in condition_dict:
                    current_annotation = condition_dict["annotation"]
                    default_annotation = cond.annotation(atom_array, default="raise")
                    current_annotation[token_to_unindex] = default_annotation[token_to_unindex]
                    cond.set_annotation(atom_array, current_annotation)
            
            # Index groups present an edge case -- resetting to defaults only in the regions to be 
            # unindexed may cause the (now-indexed) source residues to have different index groups from
            # their same-chain indexed neighbors. To avoid this, we reset the full indexed region
            C_IDXGRP.set_annotation(atom_array, C_IDXGRP.default_annotation(atom_array))

        # Concatenate unindexed parts to the end
        atom_array_full = concatenate_any([atom_array] + unindexed_tokens)

        # Ensure that the newly-created index groups in the indexed region do not overlap with
        # those in the tokens that were copied for unindexing
        index_groups = C_IDXGRP.annotation(atom_array_full, default="raise")
        is_indexed = C_IDX.mask(atom_array_full, default="raise")
        max_index_group_in_indexed_region = index_groups[is_indexed].max()
        index_groups[~is_indexed] += max_index_group_in_indexed_region + 1
        C_IDXGRP.set_annotation(atom_array_full, index_groups)

        # Reset indices to zero to avoid any bias.
        for index_group in np.unique(index_groups):
            group_mask = index_groups == index_group

            # We leave the residue indices unchanged for globally indexed groups
            is_indexed = C_IDX.mask(atom_array_full[group_mask], default="raise")
            if (is_indexed != is_indexed[0]).any():
                raise ValueError(f"Index group {index_group} contains a mix of indexed and unindexed atoms.")
            if is_indexed[0]:
                continue

            # HACK: In legacy inference code, user-requested index gaps are only preserved as res_id gaps within the unindexed region
            # This should be cleaned up after the inference refactor
            if is_inference:
                group_res_ids = atom_array_full.res_id[group_mask]
                atom_array_full.within_index_group_res_idx[group_mask] = group_res_ids - group_res_ids.min()
            else:
                atom_array_full.within_index_group_res_idx[group_mask] -= atom_array_full.within_index_group_res_idx[
                    group_mask
                ].min()

        # update pairs in place using the old_to_new_atom_idx_map
        # vectorize for efficiency and one-shot update
        vectorized_map = np.vectorize(lambda x: old_to_new_atom_idx_map.get(x, x))

        # Update all 2D conditions to reflect movement of unindexed atoms
        for cond_name, condition_dict in original_2d_conditions.items():
            cond = CONDITIONS[cond_name]
            if "mask" in condition_dict:
                mask = condition_dict["mask"]
                pairs, values = mask.pairs, mask.values
                if len(pairs) > 0:
                    pairs = vectorized_map(pairs)
                    cond.set_mask(atom_array_full, pairs=pairs, values=values)
            if "annotation" in condition_dict:
                annotation = condition_dict["annotation"]
                pairs, values = annotation.pairs, annotation.values
                if len(pairs) > 0:
                    pairs = vectorized_map(pairs)
                    cond.set_annotation(atom_array_full, pairs=pairs, values=values)

        # Ensure tokens are recognised as separate
        n_unindexed_tokens = len(unindexed_tokens)
        assert get_token_count(atom_array_full) == get_token_count(atom_array) + n_unindexed_tokens, (
            f"Failed to create uniquely recognised tokens after concatenation.\n"
            f"Concatenated tokens: {get_token_count(atom_array_full)}, unindexed: {n_unindexed_tokens}"
        )

        return atom_array_full

    def forward(self, data: dict):
        atom_array = data["atom_array"]
        if "feats" not in data:
            data["feats"] = {}

        is_unindexed_atom = ~C_IDX.mask(atom_array, default="generate")

        is_token_with_unindexed_atoms = apply_and_spread_token_wise(
            atom_array,
            is_unindexed_atom.copy(),
            function=lambda x: np.any(x),
        )

        # Expand unindexed motifs if necessary
        atom_array_expanded = self.expand_unindexed_motifs(
            atom_array,
            is_inference=data["is_inference"],
        )

        # Record the identity of the ground-truth tokens copied for unindexing
        n_expanded_atoms = atom_array_expanded.array_length() - atom_array.array_length()
        mask = np.concatenate([is_token_with_unindexed_atoms, np.zeros(n_expanded_atoms)])
        if "ground_truth" not in data:
            data["ground_truth"] = {}
        data["ground_truth"]["is_ground_truth_token_corresponding_to_unindexed"] = mask.astype(bool)

        # Reset global token IDs and atom IDs after possible padding
        atom_array_expanded = add_global_token_id_annotation(atom_array_expanded)
        atom_array_expanded = add_global_atom_id_annotation(atom_array_expanded)
        data["atom_array"] = atom_array_expanded
        return data


def _get_target_indices_corresponding_to_inserted(
    atom_array: AtomArray,
    chain_mask,
    insertion_res_id,
    target_res_id,
    has_index,
    res_id_annotation="res_id",
    allow_element_fallback=False,
):
    """Finds the indices to the target residue that will give the same atom names/ordering as the inserted residue.
    If allow_element_fallback is True, then any atom names not found in the target residue will be matched by element,
    with ties broken by minimizing the total distance.

    """
    # Get insertion indices and atom names
    res_ids = atom_array.get_annotation(res_id_annotation)
    insertion_mask = chain_mask & (res_ids == insertion_res_id) & ~has_index
    insertion_atom_names = atom_array.atom_name[insertion_mask]

    # Get target residue data
    target_mask = chain_mask & (res_ids == target_res_id) & has_index
    target_atoms = atom_array[target_mask]
    target_sorted_indices = np.argsort(target_atoms.atom_name)
    target_sorted_atom_names = target_atoms.atom_name[target_sorted_indices]

    # Attempt to match insertion atoms to target atoms by name
    search_positions = np.searchsorted(target_sorted_atom_names, insertion_atom_names)
    unmatched_insertion_atoms_mask = ~np.isin(insertion_atom_names, target_sorted_atom_names)

    if np.any(unmatched_insertion_atoms_mask):
        if not allow_element_fallback:
            raise ValueError("Inserted atom names not found in target residue atom names.")
        else:
            # Identify which target atoms don't match any insertion atom names
            unmatched_target_atoms_mask = ~np.isin(target_sorted_atom_names, insertion_atom_names)
            unmatched_target_indices = np.where(unmatched_target_atoms_mask)[0]
            unmatched_insertion_indices = np.where(unmatched_insertion_atoms_mask)[0]

            # Initialize output array and fill in name-based matches
            target_indices_corresponding_to_inserted = np.full(len(insertion_atom_names), np.nan)
            target_indices_corresponding_to_inserted[~unmatched_insertion_atoms_mask] = target_sorted_indices[
                search_positions[~unmatched_insertion_atoms_mask]
            ]

            # Get coordinates and elements for unmatched atoms
            unmatched_target_coords = target_atoms.coord[target_sorted_indices][unmatched_target_atoms_mask]
            unmatched_insertion_coords = atom_array.coord[insertion_mask, :][unmatched_insertion_atoms_mask]

            unmatched_target_elements = target_atoms.element[target_sorted_indices][unmatched_target_atoms_mask]
            unmatched_insertion_elements = atom_array.element[insertion_mask][unmatched_insertion_atoms_mask]
            same_element_mask = unmatched_insertion_elements[:, np.newaxis] == unmatched_target_elements[np.newaxis, :]

            # Compute distance matrix, setting inf for different elements
            dists = np.linalg.norm(
                unmatched_insertion_coords[:, np.newaxis, :] - unmatched_target_coords[np.newaxis, :, :],
                axis=-1,
            )
            dists[~same_element_mask] = np.inf

            # Use linear sum assignment to match by element, minimizing total distance
            row_indices, col_indices = linear_sum_assignment(dists)

            # Fill in element-based matches
            for r, c in zip(row_indices, col_indices):
                if not np.isfinite(dists[r, c]):
                    raise ValueError("Could not find matching target atom for inserted atom based on element type.")
                assert np.isnan(
                    target_indices_corresponding_to_inserted[unmatched_insertion_indices[r]]
                ), "Duplicate target element assignemnt"
                target_indices_corresponding_to_inserted[unmatched_insertion_indices[r]] = target_sorted_indices[
                    unmatched_target_indices[c]
                ]

            if np.any(np.isnan(target_indices_corresponding_to_inserted)):
                raise ValueError(
                    "Some inserted atoms could not be matched to target atoms even after element fallback."
                )

    else:
        # All insertion atoms matched by name
        target_indices_corresponding_to_inserted = target_sorted_indices[search_positions]

    # Convert to int (may be float64 if initialized with NaN)
    return target_indices_corresponding_to_inserted.astype(int)


def find_optimal_insertion_points_for_unindexed_regions(
    atom_array: AtomArray,
    inputs_in_same_frame=True,
):
    """This function maps the given unindexed regions (in-context conditioning) to the corresponding regions diffused by the model.

    Currently, it only supports matching by atom name (if sequence-conditioned) and absolute coordinate conditioning (but not both). It is
    possible to provide many other forms of unindexed conditioning, but to find their optimal insertion points, some method
    of determining relative importance is necessary. For simplicity, this is left out for now.

    Args:
        atom_array (AtomArray): AtomArray including the unindexed regions (in-context conditioning).
        inputs_in_same_frame (bool): Whether the inputs can be relied on to be in the same global frame. If False, will
    """

    # Enforce uniqueness of res_ids within each chain
    for chain_id in np.unique(atom_array.chain_id):
        chain_array = atom_array[atom_array.chain_id == chain_id]
        chain_residue_starts = get_residue_starts(chain_array)
        res_ids_in_chain = chain_array.res_id[chain_residue_starts]
        if len(np.unique(res_ids_in_chain)) < len(res_ids_in_chain):
            raise ValueError(f"Bad input: Duplicate res_ids found in chain {chain_id}.")

    global_res_id = np.arange(len(get_residue_starts(atom_array)))
    atom_array.set_annotation("global_res_id", spread_residue_wise(atom_array, global_res_id))
    ensure_annotations(atom_array, "is_ligand")

    has_index = C_IDX.mask(atom_array, default="generate")
    model_generated_atom_array = atom_array[has_index].copy()
    in_context_conditioning = atom_array[~has_index].copy()

    # NOTE: If it is desirable to support the case where unindexed tokens are not all at the end, we could make a
    # rearranged copy of the AtomArray before computing global_res_id. For now, this is kept as a sanity check
    for chain_id in np.unique(model_generated_atom_array.chain_id):
        chain_array = model_generated_atom_array[model_generated_atom_array.chain_id == chain_id]
        chain_residue_starts = get_residue_starts(chain_array)
        global_res_id_residue_level = chain_array.global_res_id[chain_residue_starts]
        if not (np.diff(global_res_id_residue_level) == 1).all():
            raise ValueError(f"Bad input: Unindexed regions are not all at the end of {chain_id}.")

    # Validate inputs and determine matching strategy
    has_absolute_coordinate_conditioning = C_CRD.mask(in_context_conditioning, default="generate")

    has_internal_coordinate_conditioning = C_INTCRD.mask(in_context_conditioning, default="generate")
    if has_internal_coordinate_conditioning.any():
        raise NotImplementedError(
            "We currently do not support finding optimal insertion points for unindexed regions with internal coordinate conditioning. "
        )

    has_distance_conditioning = C_DIS.mask(in_context_conditioning, default="generate").as_dense_array().any(axis=0)
    if has_distance_conditioning.any():
        raise NotImplementedError(
            "We currently do not support finding optimal insertion points for unindexed regions with 2D distance conditioning. "
        )
    elif has_absolute_coordinate_conditioning.any():
        unindexed_residue_has_3d_conditioning = apply_residue_wise(
            in_context_conditioning, has_absolute_coordinate_conditioning, np.any
        )
        if not unindexed_residue_has_3d_conditioning.all():
            raise ValueError(
                "If using 3d conditioning for unindexed regions, all unindexed residues must have at least some 3d conditioning. "
                "Otherwise, there is ambiguity as to where they should be inserted."
            )
    else:
        raise ValueError("No 3D conditioning found for unindexed regions. Cannot determine optimal insertion points.")

    unindexed_group_ids = C_IDXGRP.annotation(in_context_conditioning, default="generate")

    # Get valid starts for each unindexed group.
    # Note that we increment by residue since all unindexed regions should consist exclusively of residues.
    generated_residue_starts = get_residue_starts(model_generated_atom_array)
    generated_residue_starts_atoms = model_generated_atom_array[generated_residue_starts]
    valid_first_residue_starts_by_group = defaultdict(list)
    chain_ids_by_group = {}
    unique_group_ids = np.unique(unindexed_group_ids)
    for group_id in unique_group_ids:
        unindexed_group = in_context_conditioning[unindexed_group_ids == group_id]

        # Get the unique chain id for the group.
        if len(np.unique(unindexed_group.chain_id)) > 1:
            raise ValueError("A single unindexed group must not span multiple chains.")
        chain_ids_by_group[group_id] = unindexed_group.chain_id[0]

        # This may be supported eventually as a method of chain-agnostic unindexing, but adds complexity so is left out for now
        if chain_ids_by_group[group_id] not in generated_residue_starts_atoms.chain_id:
            raise NotImplementedError(
                "Unindexed group chain ID not found in generated atom array. Chain-agnostic unindexing is not currently supported."
            )

        chain_id_mask = generated_residue_starts_atoms.chain_id == chain_ids_by_group[group_id]

        # Find valid starts for the first residue of this group.
        residue_extent_of_group = (
            unindexed_group.within_index_group_res_idx.max() - unindexed_group.within_index_group_res_idx.min()
        )
        max_indexed_global_res_id_current_chain = generated_residue_starts_atoms[chain_id_mask].global_res_id.max()
        valid_first_residue_starts = generated_residue_starts[
            (chain_id_mask)
            & ~(generated_residue_starts_atoms.is_ligand)
            & (
                generated_residue_starts_atoms.global_res_id + residue_extent_of_group
                <= max_indexed_global_res_id_current_chain
            )
        ]
        valid_first_residue_starts_by_group[group_id] = valid_first_residue_starts

    has_seq = C_SEQ.mask(atom_array, default="generate")

    # Filter valid residue mappings with the criterion that atom names must match for sequence-conditioned residues
    for group_id, valid_starts in valid_first_residue_starts_by_group.items():
        unindexed_group = in_context_conditioning[unindexed_group_ids == group_id]
        unindexed_group_residue_starts = get_residue_starts(unindexed_group)
        global_res_ids = unindexed_group.global_res_id[unindexed_group_residue_starts]
        within_index_group_res_idxs = unindexed_group.within_index_group_res_idx[unindexed_group_residue_starts]
        filtered_valid_starts = []
        current_chain_indices = np.where(atom_array.chain_id == chain_ids_by_group[group_id])[0]
        current_chain = atom_array[current_chain_indices]
        current_chain_has_seq = has_seq[current_chain_indices]
        for group_start in valid_starts:
            group_start_global_res_id = model_generated_atom_array[group_start].global_res_id

            unindexed_to_generated_global_res_id_map = {
                unindexed_global_res_id: group_start_global_res_id
                + (within_index_group_res_idx - within_index_group_res_idxs.min())
                for (unindexed_global_res_id, within_index_group_res_idx) in zip(
                    global_res_ids, within_index_group_res_idxs, strict=False
                )
            }
            current_start_is_valid = True
            for k, v in unindexed_to_generated_global_res_id_map.items():
                k_is_seq_conditioned = (current_chain_has_seq[current_chain.global_res_id == k]).any()
                if k_is_seq_conditioned:
                    assert (
                        current_chain_has_seq[current_chain.global_res_id == k]
                    ).all(), "Residues should not be partially sequence-conditioned."

                if k_is_seq_conditioned:
                    # Check if unindexed atom names are a subset of indexed ones
                    if not np.isin(
                        current_chain[current_chain.global_res_id == k].atom_name,
                        current_chain[current_chain.global_res_id == v].atom_name,
                    ).all():
                        current_start_is_valid = False
                        break
                else:
                    # Check if the unindexed elements are a subset of the indexed ones
                    unique_elements_k, counts_k = np.unique(
                        current_chain[current_chain.global_res_id == k].element, return_counts=True
                    )
                    unique_elements_v, counts_v = np.unique(
                        current_chain[current_chain.global_res_id == v].element, return_counts=True
                    )
                    elements_to_counts_k = dict(zip(unique_elements_k, counts_k))
                    elements_to_counts_v = dict(zip(unique_elements_v, counts_v))

                    if not all(
                        elements_to_counts_k[element] <= elements_to_counts_v.get(element, 0)
                        for element in unique_elements_k
                    ):
                        current_start_is_valid = False
                        break

            if current_start_is_valid:
                filtered_valid_starts.append(group_start)

        if len(filtered_valid_starts) == 0:
            raise ValueError(
                f"No valid insertion points found for unindexed group {group_id} after applying sequence conditioning atom name matching."
            )
        valid_first_residue_starts_by_group[group_id] = np.array(filtered_valid_starts)

    if not inputs_in_same_frame:
        for group_id, valid_starts in valid_first_residue_starts_by_group.items():
            if len(valid_starts) > 1:
                raise ValueError(
                    f"Multiple valid insertion points found for unindexed group {group_id} "
                    "when inputs are not guaranteed to be in the same frame."
                )

    optimal_first_residue_starts_by_group = {}
    occupied_res_ids_by_chain = defaultdict(set)  # Track which residues are already assigned
    for group_id, chain_id in chain_ids_by_group.items():
        unindexed_group = in_context_conditioning[unindexed_group_ids == group_id]
        unindexed_group_res_starts = get_residue_starts(unindexed_group)
        chain_mask = atom_array.chain_id == chain_id
        minimum_cost_across_groups = None
        minimum_cost_group_start = None
        minimum_cost_target_res_ids = None

        for group_start in valid_first_residue_starts_by_group[group_id]:
            group_start_global_res_id = model_generated_atom_array[group_start].global_res_id
            within_index_group_res_idx_residue_level = unindexed_group.within_index_group_res_idx[
                unindexed_group_res_starts
            ]
            unindexed_global_res_id_residue_level = unindexed_group.global_res_id[unindexed_group_res_starts]

            unindexed_to_generated_global_res_id_map = {
                unindexed_global_res_id: group_start_global_res_id
                + (within_index_group_res_idx - within_index_group_res_idx_residue_level.min())
                for (unindexed_global_res_id, within_index_group_res_idx) in zip(
                    unindexed_global_res_id_residue_level, within_index_group_res_idx_residue_level, strict=False
                )
            }

            # Check if this start would overlap with already-assigned groups
            target_global_res_ids = set(unindexed_to_generated_global_res_id_map.values())
            if target_global_res_ids & occupied_res_ids_by_chain[chain_id]:
                continue

            cost_for_this_group_start = 0
            for (
                inserted_res_id,
                target_res_id,
            ) in unindexed_to_generated_global_res_id_map.items():
                inserted_array = atom_array[chain_mask & (atom_array.global_res_id == inserted_res_id)]
                target_array = atom_array[chain_mask & (atom_array.global_res_id == target_res_id)]

                target_indices_corresponding_to_inserted = _get_target_indices_corresponding_to_inserted(
                    atom_array,
                    chain_mask,
                    inserted_res_id,
                    target_res_id,
                    has_index,
                    res_id_annotation="global_res_id",
                    allow_element_fallback=True,
                )

                cost_for_this_group_start += np.linalg.norm(
                    inserted_array.coord - target_array.coord[target_indices_corresponding_to_inserted],
                    axis=-1,
                ).sum()

            if minimum_cost_across_groups is None or cost_for_this_group_start < minimum_cost_across_groups:
                minimum_cost_group_start = group_start
                minimum_cost_across_groups = cost_for_this_group_start
                minimum_cost_target_res_ids = target_global_res_ids

        if minimum_cost_group_start is None:
            raise ValueError(f"Could not find a non-overlapping start for unindexed group {group_id}")

        optimal_first_residue_starts_by_group[group_id] = minimum_cost_group_start
        occupied_res_ids_by_chain[chain_id].update(minimum_cost_target_res_ids)

    # Convert the optimal first residue starts to the full residue mapping
    # Here we will use the actual res_id annotation for a more interpretable output mapping
    final_residue_mapping_by_chain = defaultdict(dict)
    for group_id, group_start in optimal_first_residue_starts_by_group.items():
        chain_id = chain_ids_by_group[group_id]
        unindexed_group = in_context_conditioning[unindexed_group_ids == group_id]
        group_start_global_res_id = model_generated_atom_array[group_start].global_res_id
        for res_id in np.unique(unindexed_group.res_id):
            within_group_res_id = unindexed_group[unindexed_group.res_id == res_id].within_index_group_res_idx[0]
            target_global_res_id = group_start_global_res_id + (
                within_group_res_id - unindexed_group.within_index_group_res_idx.min()
            )
            target_res_id = atom_array[atom_array.global_res_id == target_global_res_id].res_id[0]
            final_residue_mapping_by_chain[chain_id][res_id] = target_res_id

    return final_residue_mapping_by_chain


def process_unindexed_outputs(
    atom_array,
    insert_unindexed_regions=False,
    verbose=False,
    inputs_in_same_frame=True,
):
    """
    Process design outputs containing unindexed tokens. Returns metadata such as the assigned positional indices from
    the input indices and the RMSD of the unindexed tokens.

    Args:
        atom_array (AtomArray): AtomArray including the unindexed regions (in-context conditioning).
        insert_unindexed_regions (bool): Whether to insert the coordinates of the unindexed regions, overwriting the regions they mapped to.
        verbose (bool): Whether to save verbose insertion quality statistics, including breakdowns by residue type and individual residue.
        inputs_in_same_frame (bool): Whether the inputs can be relied on to be in the same global frame. If False, will
            attempt to map each unindexed group to a single generated region, and fails to find a mapping if this is ambiguous.

    Returns:
        - Diffused atom array (without additional unindexed tokens)
        - Metadata:
            - diffused_indices: keys = original (contig) indices, values = diffused indices
            - insertion_rmsd: overall RMSD of insertion
            - insertion_rmsd_by_residue: RMSD of insertion for each token

    TODO: In the longer term, having the model predict the insertion points will be much nicer, and then we can simply
    do the insertions in this function -- Metrics should be done elsewhere. Kept for now as we iterate towards that.
    """

    has_index = C_IDX.mask(atom_array, default="generate")

    model_generated_atom_array = atom_array[has_index].copy()
    in_context_conditioning = atom_array[~has_index].copy()

    # Validate inputs and determine matching strategy
    has_3d_conditioning = C_CRD.mask(in_context_conditioning, default="generate")

    has_distance_conditioning = C_DIS.mask(in_context_conditioning, default="generate").as_dense_array().any(axis=0)

    if has_distance_conditioning.any():
        raise NotImplementedError(
            "We currently do not support finding optimal insertion points for 2d distance-conditioned unindexed regions."
        )

    metadata = {
        "diffused_index_map": {},
    }

    metadata |= {
        "insertion_rmsd_by_token": {},
        "join_point_rmsd_by_token": {},
        "insertion_rmsd_by_restype": {},
    }
    token_maes = []
    token_rmcds = []

    # Track which indices have been mapped to unindexed regions and their remapped coordinates
    mapped_to_coordinate_conditioned_region = np.zeros(model_generated_atom_array.array_length(), dtype=bool)
    remapped_coord = np.full_like(model_generated_atom_array.coord, np.nan)

    # If we cannot find a valid insertion mapping, fall back to simply removing the unindexed region
    try:
        optimal_residue_mapping_by_chain = find_optimal_insertion_points_for_unindexed_regions(
            atom_array,
            inputs_in_same_frame=inputs_in_same_frame,
        )
    except ValueError as e:
        global_logger.warning(
            f"Could not find a valid insertion mapping for unindexed regions: {e} "
            f"No mapping will be reported and the unindexed region will not be reinserted."
        )
        model_generated_atom_array.set_annotation("mapped_to_coordinate_conditioned_region", mapped_to_coordinate_conditioned_region)
        return model_generated_atom_array, {}

    old_to_new_atom_index_map = {}
    global_indices = np.arange(atom_array.array_length())
    generated_indices = np.arange(model_generated_atom_array.array_length())

    for chain_id in optimal_residue_mapping_by_chain:
        for unindexed_res_id, target_res_id in optimal_residue_mapping_by_chain[chain_id].items():
            # Create some useful masks
            chain_mask = atom_array.chain_id == chain_id
            unindexed_token_mask = (in_context_conditioning.chain_id == chain_id) & (
                in_context_conditioning.res_id == unindexed_res_id
            )
            unindexed_token = in_context_conditioning[unindexed_token_mask]

            # Annotate the identified mapping
            if "src_component" in unindexed_token.get_annotation_categories():
                token_src_id = unindexed_token.src_component[0]
            else:
                token_src_id = f"{chain_id}{unindexed_res_id}"
            metadata["diffused_index_map"][token_src_id] = f"{chain_id}{target_res_id}"

            # Get local permutation/subset of indices within the target residue that corresponds to the atom ordering/subset in the unindexed token
            target_indices_corresponding_to_inserted = _get_target_indices_corresponding_to_inserted(
                atom_array,
                chain_mask,
                unindexed_res_id,
                target_res_id,
                has_index,
                allow_element_fallback=True,
            )

            # Get global indices that can be applied to the full atom array, combining model-generated components and in-context conditioning
            target_mask_global = chain_mask & (atom_array.res_id == target_res_id) & has_index
            permuted_target_indices_global = global_indices[target_mask_global][
                target_indices_corresponding_to_inserted
            ]

            # Subset to an index array that can be used on the model-generated atom array
            target_mask_generated = (model_generated_atom_array.chain_id == chain_id) & (
                model_generated_atom_array.res_id == target_res_id
            )
            permuted_target_indices_generated = generated_indices[target_mask_generated][
                target_indices_corresponding_to_inserted
            ]

            # Record that these indices have been mapped to unindexed regions
            mapped_to_coordinate_conditioned_region[permuted_target_indices_generated] = True

            # We'll need this later if doing 2d conditioning
            if has_distance_conditioning.any():
                insertion_residue_indices = np.where(chain_mask & (atom_array.res_id == unindexed_res_id))[0]
                old_to_new_atom_index_map.update(
                    dict(zip(insertion_residue_indices, permuted_target_indices_global, strict=False))
                )

            diff = unindexed_token.coord - model_generated_atom_array.coord[permuted_target_indices_generated]
            token_rmsd = float(np.sqrt((diff**2).sum(-1).mean()))
            token_rmcd = float(np.cbrt((np.abs(diff) ** 3).sum(-1).mean()))
            token_mae = float((np.abs(diff)).sum(-1).mean())

            metadata["insertion_rmsd_by_token"][token_src_id] = token_rmsd
            token_maes.append(token_mae)
            token_rmcds.append(token_rmcd)
            res_name = unindexed_token.res_name[0]

            if res_name not in metadata["insertion_rmsd_by_restype"]:
                metadata["insertion_rmsd_by_restype"][res_name] = []
            metadata["insertion_rmsd_by_restype"][res_name].append(token_rmsd)

            if not np.any(np.isin(unindexed_token.atom_name, ["N", "CA", "C", "O"])):
                # Infer join point as the unindexed atom closest to the backbone with ties broken (arbitrarily) by the encoding order
                atom_name_encoding_indices = np.array(
                    [
                        UNIFIED_ATOM37_ENCODING.atom_to_idx[(res_name, atom_name)]
                        for atom_name in unindexed_token.atom_name
                    ]
                )
                join_atom = np.argmin(atom_name_encoding_indices)

                dist = np.linalg.norm(
                    unindexed_token.coord[join_atom]
                    - model_generated_atom_array.coord[permuted_target_indices_generated[join_atom]]
                )
                metadata["join_point_rmsd_by_token"][token_src_id] = float(dist)

            if has_3d_conditioning.any():
                # NOTE: If we eventually support other kinds of unindexed conditioning (rasa etc) we may want to copy those over as well
                # Sequence conditioning is not copied over to avoid potential sequence/structure mismatches for tip-atom cases
                if insert_unindexed_regions:
                    model_generated_atom_array.coord[permuted_target_indices_generated] = unindexed_token.coord
                    coord_mask = C_CRD.mask(model_generated_atom_array, default="generate")
                    coord_mask[permuted_target_indices_generated & C_CRD.mask(unindexed_token)] = True
                    C_CRD.set_mask(model_generated_atom_array, coord_mask)

                # We always annotate the input conditioning for use in downstream Metrics
                remapped_coord[permuted_target_indices_generated] = unindexed_token.coord

    # Add an annotation to indicate which residues mapped to unindexed regions, whether inserting or not
    # NOTE: This annotation should be used by other mapping methods as well, e.g. model-predicted mappings
    model_generated_atom_array.set_annotation("mapped_to_coordinate_conditioned_region", mapped_to_coordinate_conditioned_region)

    # Validate remapped coordinates, then add as an annotation
    if has_3d_conditioning.any() and mapped_to_coordinate_conditioned_region.any():
        if np.isnan(remapped_coord[mapped_to_coordinate_conditioned_region]).any():
            raise ValueError("NaN values found in remapped coordinates")
        model_generated_atom_array.coords_with_insertion[mapped_to_coordinate_conditioned_region] = remapped_coord[
            mapped_to_coordinate_conditioned_region
        ]

    if not model_generated_atom_array.mapped_to_coordinate_conditioned_region.any():
        global_logger.warning(
            "Did not map any indexed regions to unindexed regions. "
            "No mapping will be reported and the unindexed region will not be reinserted."
        )

    # ... Calculate global metrics
    metadata["insertion_mae"] = float(np.mean(token_maes))
    metadata["insertion_rmcd"] = float(np.mean(token_rmcds))
    metadata["insertion_rmsd"] = float(np.mean(list(metadata["insertion_rmsd_by_token"].values())))
    metadata["join_point_rmsd"] = float(np.mean(list(metadata["join_point_rmsd_by_token"].values())))
    metadata["insertion_rmsd_by_restype"] = {
        a: float(np.mean(v)) for a, v in metadata["insertion_rmsd_by_restype"].items()
    }

    if not verbose:
        metadata = {k: v for k, v in metadata.items() if not k.startswith("insertion_rmsd_by_")}

    return model_generated_atom_array, metadata
