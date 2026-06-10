import copy
import itertools
from collections import defaultdict

import biotite.structure as struc
import numpy as np
from atomworks.io.utils.atom_array_plus import concatenate_any
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.ml.conditions import C_CRD, C_CTR, C_DIS, C_IDX, C_NTR, C_SEQ
from atomworks.ml.conditions.base import CONDITIONS
from atomworks.ml.transforms._checks import (
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atom_array import (
    add_global_atom_id_annotation,
    add_global_token_id_annotation,
    get_within_group_source_res_idx,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.annotator import (
    ensure_annotations,
)
from atomworks.ml.utils.token import (
    apply_and_spread_token_wise,
    apply_token_wise,
    get_token_count,
    get_token_starts,
)
from biotite.structure import AtomArray
from biotite.structure.residues import apply_residue_wise, get_residue_starts, spread_residue_wise
from scipy.optimize import linear_sum_assignment

from foundry.utils.ddp import RankedLogger
from foundry.utils.standard_annotations import SA_CRDGRPSRC, SA_CRDSRC, SA_CWI, SA_SRC, SA_UII
from proteinfoundation.conditions import C_CRDGRP, C_IDXGRP, C_INTCRD
from proteinfoundation.transforms.utils.conditioning_utils import (
    set_specified_conditions_to_default,
)

global_logger = RankedLogger(__name__, rank_zero_only=False)


class UnindexFlaggedTokens(Transform):
    """Create in-context conditioning for tokens with unknown indices."""

    def check_input(self, data: dict) -> None:
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
        ensure_annotations(atom_array, "is_polymer", "within_chain_source_res_idx")

        # This will be modified if any unindexed tokens are present to give the correct ground-truth index gaps (within
        # unindexed groups). This enables specification of defined index gaps greater than 1 between unindexed tokens.
        atom_array.set_annotation("within_index_group_res_idx", atom_array.within_chain_source_res_idx.copy())

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
        for _i, (start, end) in enumerate(itertools.pairwise(starts)):
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
            assert token.is_polymer.all(), f"Cannot unindex non-polymer token: {token}"

            unindexed_tokens.append(token)

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
            old_to_new_atom_idx_map.update(dict(zip(original_indexed_positions, new_indexed_positions, strict=False)))

            # Shift res_ids of surviving atoms down by the number of unique unindexed
            # res_ids preceding them in the same chain. This collapses the holes left
            # by deletion while preserving user-specified gaps in the indexed region,
            # so the recomputed within_chain_source_res_idx reflects only the latter.
            shifted_res_ids = atom_array.res_id.copy()
            for chain_iid in np.unique(atom_array.chain_iid):
                chain_mask = atom_array.chain_iid == chain_iid
                removed_res_ids = np.unique(atom_array.res_id[chain_mask & token_to_unindex])
                if len(removed_res_ids) == 0:
                    continue
                chain_res_ids = shifted_res_ids[chain_mask]
                shift = np.searchsorted(removed_res_ids, chain_res_ids, side="right")
                shifted_res_ids[chain_mask] = chain_res_ids - shift
            atom_array.res_id = shifted_res_ids

            atom_array = atom_array[~token_to_unindex]
            new_within_chain_source_res_idx = get_within_group_source_res_idx(atom_array, group_by="chain_iid")
            atom_array.set_annotation("within_chain_source_res_idx", new_within_chain_source_res_idx)
            atom_array.set_annotation(
                "within_index_group_res_idx", new_within_chain_source_res_idx.copy()
            )  # will be redefined properly later in this method

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
        atom_array_full = concatenate_any([atom_array, *unindexed_tokens])

        # Ensure that the newly-created index groups in the indexed region do not overlap with
        # those in the tokens that were copied for unindexing
        index_groups = C_IDXGRP.annotation(atom_array_full, default="raise")
        is_indexed = C_IDX.mask(atom_array_full, default="raise")
        max_index_group_in_indexed_region = index_groups[is_indexed].max()
        min_index_group_in_unindexed_region = index_groups[~is_indexed].min()
        index_groups[~is_indexed] += max_index_group_in_indexed_region - min_index_group_in_unindexed_region + 1
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

        return atom_array_full

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        if "feats" not in data:
            data["feats"] = {}

        is_unindexed_atom = ~C_IDX.mask(atom_array, default="generate")

        is_token_with_unindexed_atoms = apply_and_spread_token_wise(
            atom_array,
            is_unindexed_atom.copy(),
            function=lambda x: np.any(x),
        )
        has_any_unindexed = bool(np.any(is_token_with_unindexed_atoms))

        # Expand unindexed motifs if necessary
        atom_array_expanded = self.expand_unindexed_motifs(
            atom_array,
            is_inference=data["is_inference"],
        )

        # Record the identity of the ground-truth tokens copied for unindexing
        n_original_atoms = atom_array.array_length()
        n_expanded_atoms = atom_array_expanded.array_length() - n_original_atoms
        mask = np.concatenate([is_token_with_unindexed_atoms, np.zeros(n_expanded_atoms)])
        if "ground_truth" not in data:
            data["ground_truth"] = {}
        data["ground_truth"]["is_ground_truth_token_corresponding_to_unindexed"] = mask.astype(bool)

        # Compute per-atom source index for training before resetting global_token_id
        if has_any_unindexed and not data["is_inference"]:
            insertion_source_idx = _compute_insertion_source_indices(
                atom_array_expanded=atom_array_expanded,
                n_original_atoms=n_original_atoms,
            )
        else:
            insertion_source_idx = np.full(atom_array_expanded.array_length(), -1, dtype=np.int32)
        atom_array_expanded.set_annotation("insertion_source_idx", insertion_source_idx)

        # Expose the unindexed-source mask as an atom_array annotation for
        # downstream consumers (e.g. loss-weight predicates).
        if data.get("is_inference", False):
            is_unindexed_source = np.zeros(atom_array_expanded.array_length(), dtype=bool)
        else:
            is_unindexed_source = mask.astype(bool)
        atom_array_expanded.set_annotation("is_unindexed_source", is_unindexed_source)

        # Reset global token IDs and atom IDs after possible expansion

        atom_array_expanded = add_global_token_id_annotation(atom_array_expanded)
        atom_array_expanded = add_global_atom_id_annotation(atom_array_expanded)

        # Ensure newly-added tokens are recognised as separate
        if has_any_unindexed and not data["is_inference"]:
            n_unindexed_tokens = apply_token_wise(atom_array, is_unindexed_atom, np.any).sum()
            assert get_token_count(atom_array_expanded) == get_token_count(atom_array) + n_unindexed_tokens, (
                f"Failed to create uniquely recognised tokens after concatenation.\n"
                f"Concatenated tokens: {get_token_count(atom_array_expanded)}, unindexed: {n_unindexed_tokens}"
            )

        data["atom_array"] = atom_array_expanded
        return data


def _compute_insertion_source_indices(
    atom_array_expanded: AtomArray,
    n_original_atoms: int,
) -> np.ndarray:
    """Return per-atom insertion source indices for the expanded atom array.

    Each unindexed (appended) token inherits its source token's ``token_id``
    via deepcopy in :py:meth:`UnindexFlaggedTokens.expand_unindexed_motifs`.
    That inherited ID is the 0-indexed position of the source token within
    the original (pre-expansion) sequence, which is exactly what the CE loss
    needs as its target.

    Args:
        atom_array_expanded: Full atom array (indexed tokens then unindexed copies appended).
            Must have ``token_id`` annotation set (by :py:class:`AddGlobalTokenIdAnnotation`
            earlier in the pipeline) — the copies inherit it from the source tokens.
        n_original_atoms: Number of atoms in the original (pre-expansion) array.

    Returns:
        Int32 array of length ``atom_array_expanded.array_length()``.  Indexed-token
        positions are ``-1``; each atom of an unindexed token holds the 0-indexed
        token position of its source in the original sequence.

    Raises:
        ValueError: If the inherited token IDs on unindexed copies are out of range
            or the number of unindexed tokens does not match the number of distinct
            source tokens recorded in the inherited IDs.
    """
    result = np.full(atom_array_expanded.array_length(), -1, dtype=np.int32)

    unindexed_atoms = atom_array_expanded[n_original_atoms:]
    if unindexed_atoms.array_length() == 0:
        return result

    original_atoms = atom_array_expanded[:n_original_atoms]
    n_original_tok = len(get_token_starts(original_atoms))

    unindexed_tok_starts = get_token_starts(unindexed_atoms)
    unindexed_tok_ends = np.append(unindexed_tok_starts[1:], unindexed_atoms.array_length())
    n_unindexed_tok = len(unindexed_tok_starts)

    # Each unindexed token's first atom carries the inherited token_id of its source.
    inherited_ids = atom_array_expanded.token_id[n_original_atoms + unindexed_tok_starts]  # [n_unindexed_tok]

    if inherited_ids.max() >= n_original_tok:
        raise ValueError(
            f"Inherited token_id {inherited_ids.max()} >= n_original_tok {n_original_tok}. "
            "token_id must be set by AddGlobalTokenIdAnnotation before UnindexFlaggedTokens."
        )

    n_unique_sources = len(np.unique(inherited_ids))
    if n_unique_sources != n_unindexed_tok:
        raise ValueError(
            f"Number of unindexed tokens ({n_unindexed_tok}) does not match number of "
            f"unique source token IDs ({n_unique_sources}). Each source should produce exactly one copy."
        )

    for i in range(n_unindexed_tok):
        start = n_original_atoms + unindexed_tok_starts[i]
        end = n_original_atoms + unindexed_tok_ends[i]
        result[start:end] = int(inherited_ids[i])

    return result


def _get_target_indices_corresponding_to_inserted(
    atom_array: AtomArray,
    chain_mask: np.ndarray,
    insertion_res_id: int,
    target_res_id: int,
    has_index: np.ndarray,
    res_id_annotation: str = "res_id",
    allow_element_fallback: bool = False,
) -> np.ndarray:
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
            for r, c in zip(row_indices, col_indices, strict=False):
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
    inputs_in_same_frame: bool = True,
) -> dict:
    """Map unindexed regions (in-context conditioning) to the corresponding regions diffused by the model.

    Currently, it only supports matching by atom name (if sequence-conditioned) and absolute coordinate
    conditioning (but not both). It is possible to provide many other forms of unindexed conditioning,
    but to find their optimal insertion points, some method of determining relative importance is
    necessary. For simplicity, this is left out for now.

    Args:
        atom_array (AtomArray): AtomArray including the unindexed regions (in-context conditioning).
        inputs_in_same_frame (bool): Whether the inputs can be relied on to be in the same global
            frame. If False, will attempt to map each group to a single generated region.
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
            "We currently do not support finding optimal insertion points for unindexed regions "
            "with internal coordinate conditioning. "
        )

    has_distance_conditioning = C_DIS.mask(in_context_conditioning, default="generate").as_dense_array().any(axis=0)
    if has_distance_conditioning.any():
        raise NotImplementedError(
            "We currently do not support finding optimal insertion points for unindexed regions "
            "with 2D distance conditioning. "
        )
    elif has_absolute_coordinate_conditioning.any():
        unindexed_residue_has_3d_conditioning = apply_residue_wise(
            in_context_conditioning, has_absolute_coordinate_conditioning, np.any
        )
        if not unindexed_residue_has_3d_conditioning.all():
            raise ValueError(
                "If using 3d conditioning for unindexed regions, all unindexed residues must have "
                "at least some 3d conditioning. "
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

        # This may be supported eventually as a method of chain-agnostic unindexing, but adds
        # complexity so is left out for now
        if chain_ids_by_group[group_id] not in generated_residue_starts_atoms.chain_id:
            raise NotImplementedError(
                "Unindexed group chain ID not found in generated atom array. "
                "Chain-agnostic unindexing is not currently supported."
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
                    elements_to_counts_k = dict(zip(unique_elements_k, counts_k, strict=False))
                    elements_to_counts_v = dict(zip(unique_elements_v, counts_v, strict=False))

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
                f"No valid insertion points found for unindexed group {group_id} after applying "
                "sequence conditioning atom name matching."
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


def annotate_unindexed_insertion(
    atom_array: AtomArray,
    insert_unindexed_regions: bool = False,
    inputs_in_same_frame: bool = True,
) -> AtomArray:
    """Map unindexed in-context tokens onto generated tokens and annotate the result.

    Adds the following annotations
    - ``SA_CRDSRC`` - which generated atoms each unindexed token mapped to
    - ``SA_CWI`` - the target-frame unindexed coordinates at those atom positions

    Args:
        atom_array: AtomArray including the unindexed regions in-context
            conditioning appended to the generated tokens.
        insert_unindexed_regions: Deprecated. Must be False; passing True
            raises ``ValueError``. Kept temporarily for signature stability.
        inputs_in_same_frame: Whether the inputs can be relied on to be in the
            same global frame. If False attempts to map each unindexed group to
            a single generated region and fails when ambiguous.

    Returns:
        Annotated generated AtomArray with the unindexed conditioning slice
        removed. On a mapping failure ``SA_CRDSRC`` is empty and the array is
        returned otherwise unmodified.
    """
    if insert_unindexed_regions:
        raise ValueError(
            "insert_unindexed_regions=True is deprecated. Unindexed coordinate "
            "substitution is no longer supported; downstream metrics consume the "
            "SA_CWI annotation instead."
        )
    has_index = C_IDX.mask(atom_array, default="generate")

    model_generated_atom_array = atom_array[has_index].copy()
    in_context_conditioning = atom_array[~has_index].copy()

    # Short-circuit: no unindexed tokens at all — set default annotations and populate indexed ones.
    if in_context_conditioning.array_length() == 0:
        n = model_generated_atom_array.array_length()
        SA_CRDSRC.set_mask(model_generated_atom_array, np.zeros(n, dtype=bool))
        SA_CRDSRC.set_annotation(model_generated_atom_array, SA_CRDSRC.default_annotation(model_generated_atom_array))
        SA_CRDGRPSRC.set_mask(model_generated_atom_array, np.zeros(n, dtype=bool))
        SA_CRDGRPSRC.set_annotation(
            model_generated_atom_array, SA_CRDGRPSRC.default_annotation(model_generated_atom_array)
        )
        SA_UII.set_mask(model_generated_atom_array, np.zeros(n, dtype=bool))
        return _merge_indexed_coord_conditioning_into_src(model_generated_atom_array)

    # Validate inputs and determine matching strategy
    has_3d_conditioning = C_CRD.mask(in_context_conditioning, default="generate")
    has_distance_conditioning = C_DIS.mask(in_context_conditioning, default="generate").as_dense_array().any(axis=0)
    if has_distance_conditioning.any():
        raise NotImplementedError(
            "We currently do not support finding optimal insertion points for "
            "2d distance-conditioned unindexed regions."
        )

    # Track which generated atoms have been mapped to unindexed regions and stage their remapped coordinates
    n_atoms = model_generated_atom_array.array_length()
    mapped_to_coordinate_conditioned_region = np.zeros(n_atoms, dtype=bool)
    remapped_coord = np.full_like(model_generated_atom_array.coord, np.nan)
    src_coordinate_conditioned_region = SA_CRDSRC.default_annotation(model_generated_atom_array)
    src_coord_group = SA_CRDGRPSRC.default_annotation(model_generated_atom_array)

    coord_groups_all = C_CRDGRP.annotation(atom_array, default="generate")
    unindexed_coord_groups = coord_groups_all[~has_index]

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
        SA_CRDSRC.set_mask(model_generated_atom_array, mapped_to_coordinate_conditioned_region)
        SA_CRDSRC.set_annotation(model_generated_atom_array, src_coordinate_conditioned_region)
        SA_CRDGRPSRC.set_mask(model_generated_atom_array, mapped_to_coordinate_conditioned_region)
        SA_CRDGRPSRC.set_annotation(model_generated_atom_array, src_coord_group)
        SA_UII.set_mask(model_generated_atom_array, mapped_to_coordinate_conditioned_region)
        return _merge_indexed_coord_conditioning_into_src(model_generated_atom_array)

    generated_indices = np.arange(n_atoms)

    for chain_id in optimal_residue_mapping_by_chain:
        for unindexed_res_id, target_res_id in optimal_residue_mapping_by_chain[chain_id].items():
            # Create some useful masks
            chain_mask = atom_array.chain_id == chain_id
            unindexed_token_mask = (in_context_conditioning.chain_id == chain_id) & (
                in_context_conditioning.res_id == unindexed_res_id
            )
            unindexed_token = in_context_conditioning[unindexed_token_mask]

            # Annotate the identified mapping
            if SA_SRC.has_annotation(unindexed_token):
                token_src_id = SA_SRC.annotation(unindexed_token)[0]
            else:
                token_src_id = f"{chain_id}{unindexed_res_id}"

            # Get local permutation/subset of indices within the target residue that corresponds to
            # the atom ordering/subset in the unindexed token
            target_indices_corresponding_to_inserted = _get_target_indices_corresponding_to_inserted(
                atom_array,
                chain_mask,
                unindexed_res_id,
                target_res_id,
                has_index,
                allow_element_fallback=True,
            )

            # Subset to an index array that can be used on the model-generated atom array
            target_mask_generated = (model_generated_atom_array.chain_id == chain_id) & (
                model_generated_atom_array.res_id == target_res_id
            )
            permuted_target_indices_generated = generated_indices[target_mask_generated][
                target_indices_corresponding_to_inserted
            ]

            # Record that these indices have been mapped to unindexed regions
            mapped_to_coordinate_conditioned_region[permuted_target_indices_generated] = True
            src_coordinate_conditioned_region[permuted_target_indices_generated] = token_src_id

            # Record the source CoordinateGroup values
            src_coord_group[permuted_target_indices_generated] = unindexed_coord_groups[unindexed_token_mask]

            if has_3d_conditioning.any():
                # NOTE: If we eventually support other kinds of unindexed conditioning (rasa etc)
                # we may want to copy those over as well. Sequence conditioning is not copied over
                # to avoid potential sequence/structure mismatches for tip-atom cases.
                remapped_coord[permuted_target_indices_generated] = unindexed_token.coord

    # Add an annotation to indicate which residues mapped to unindexed regions, whether inserting or not
    SA_CRDSRC.set_mask(model_generated_atom_array, mapped_to_coordinate_conditioned_region)
    SA_CRDSRC.set_annotation(model_generated_atom_array, src_coordinate_conditioned_region)
    SA_CRDGRPSRC.set_mask(model_generated_atom_array, mapped_to_coordinate_conditioned_region)
    SA_CRDGRPSRC.set_annotation(model_generated_atom_array, src_coord_group)
    SA_UII.set_mask(model_generated_atom_array, mapped_to_coordinate_conditioned_region)

    # Validate remapped coordinates, then add as an SA_CWI annotation
    if has_3d_conditioning.any() and mapped_to_coordinate_conditioned_region.any():
        if np.isnan(remapped_coord[mapped_to_coordinate_conditioned_region]).any():
            raise ValueError("NaN values found in remapped coordinates")
        cwi = SA_CWI.annotation(model_generated_atom_array)
        cwi[mapped_to_coordinate_conditioned_region] = remapped_coord[mapped_to_coordinate_conditioned_region]
        SA_CWI.set_annotation(model_generated_atom_array, cwi)

    # Sanity check: number of remapped atoms should equal the number of unindexed atoms that had 3D conditioning
    if has_3d_conditioning.any():
        expected_unindexed_count = int(has_3d_conditioning.sum())
        actual_unindexed_count = int(mapped_to_coordinate_conditioned_region.sum())
        if actual_unindexed_count != expected_unindexed_count:
            global_logger.warning(
                f"Unindexed coord conditioning count ({expected_unindexed_count}) does not match "
                f"remapped count ({actual_unindexed_count}). "
                "The model has failed to reproduce the input unindexed motif."
            )
            # Reset SA_CWI to NaN so downstream insertion metrics propagate NaN rather than reporting an
            # optimistic value computed on the partial mapping.
            SA_CWI.set_annotation(model_generated_atom_array, SA_CWI.default_annotation(model_generated_atom_array))

    if not SA_CRDSRC.mask(model_generated_atom_array).any():
        global_logger.warning(
            "Did not map any indexed regions to unindexed regions. "
            "No mapping will be reported and the unindexed region will not be reinserted."
        )

    return _merge_indexed_coord_conditioning_into_src(model_generated_atom_array)


def _merge_indexed_coord_conditioning_into_src(generated_atom_array: AtomArray) -> AtomArray:
    """Fold directly-indexed ``C_CRD``/``C_INTCRD`` atoms into ``SA_CRDSRC``.

    Two kinds of atoms qualify as "their coords came from input conditioning":
      (a) atoms remapped from unindexed in-context tokens — already flagged in
          ``SA_CRDSRC`` by :py:func:`annotate_unindexed_insertion`.
      (b) atoms whose ``C_CRD`` / ``C_INTCRD`` was set directly in the input
          (the indexed motif scaffold case) — NOT yet in ``SA_CRDSRC``.
    """
    # (b) — atoms with directly-indexed coordinate conditioning in the input
    indexed_coord_conditioned_mask = C_CRD.mask(generated_atom_array) | C_INTCRD.mask(generated_atom_array)

    # Union (a) + (b) into the SA_CRDSRC mask
    mapped_mask = SA_CRDSRC.mask(generated_atom_array) | indexed_coord_conditioned_mask
    SA_CRDSRC.set_mask(generated_atom_array, mapped_mask)

    # For the (b) atoms only, copy SA_SRC ids over so the source-region label
    # exists for every CRDSRC-flagged atom (the (a) atoms got their ids set
    # inside annotate_unindexed_insertion).
    src_crd = SA_CRDSRC.annotation(generated_atom_array)
    src_crd[indexed_coord_conditioned_mask] = SA_SRC.annotation(generated_atom_array)[indexed_coord_conditioned_mask]
    SA_CRDSRC.set_annotation(generated_atom_array, src_crd)

    # Propagate nontrivial CoordGroup values so downstream metrics can group by coordinate group.
    coord_groups = C_CRDGRP.annotation(generated_atom_array, default="generate")
    src_crd_grp = SA_CRDGRPSRC.annotation(generated_atom_array)
    nontrivial = coord_groups != C_CRDGRP.UNKNOWN_COORDS
    src_crd_grp[nontrivial] = coord_groups[nontrivial]
    SA_CRDGRPSRC.set_annotation(generated_atom_array, src_crd_grp)

    # Token-level mask: True for every token whose atoms include at least one
    # SA_CRDSRC-flagged atom. Consumed by motif-RMSD / internal-coord metrics.
    generated_atom_array.set_annotation(
        "mapped_to_coordinate_conditioned_region_token_lvl",
        apply_and_spread_token_wise(generated_atom_array, SA_CRDSRC.mask(generated_atom_array), np.any),
    )

    return generated_atom_array
