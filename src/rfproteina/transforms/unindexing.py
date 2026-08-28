import copy
import itertools

import biotite.structure as struc
import numpy as np
from atomworks.io.utils.atom_array_plus import concatenate_any
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.ml.conditions import C_CRD, C_CTR, C_DIS, C_IDX, C_NTR
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
)
from biotite.structure import AtomArray
from scipy.optimize import linear_sum_assignment

from rfproteina.conditions import C_CRDGRP, C_IDXGRP, C_INTCRD
from rfproteina.transforms.utils.conditioning_utils import (
    set_specified_conditions_to_default,
)
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.standard_annotations import SA_CRDGRPSRC, SA_CRDSRC, SA_CWI, SA_SRC, SA_UII

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
            atom_array.set_annotation("insertion_source_idx", np.full(atom_array.array_length(), -1, dtype=np.int32))
            return atom_array

        unique_chain_iids = np.unique(atom_array.chain_iid)
        max_resid_by_chain_iid = {
            chain_iid: np.max(atom_array.res_id[atom_array.chain_iid == chain_iid]) for chain_iid in unique_chain_iids
        }

        # Each unindexed copy gets a fresh token_id (the source id is recorded in insertion_source_idx)
        next_copy_token_id = int(np.max(atom_array.token_id)) + 1
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

            # At inference there is no indexed source to point back to, so leave insertion_source_idx at -1.
            source_token_id = -1 if is_inference else int(token.token_id[0])
            token.set_annotation("insertion_source_idx", np.full(token.array_length(), source_token_id, dtype=np.int32))
            token.token_id = np.full(token.array_length(), next_copy_token_id, dtype=token.token_id.dtype)
            next_copy_token_id += 1

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

        # Indexed (source) tokens are never insertion queries; set insertion_source_idx=-1 there.
        if is_inference:
            atom_array.set_annotation("insertion_source_idx", np.full(atom_array.array_length(), -1, dtype=np.int32))
        else:
            # Cast token_id to signed int for NumPy 2 (NEP 50) compatibility: otherwise np.where
            # resolves to uint32 and rejects the -1 sentinel with an OverflowError.
            atom_array.set_annotation(
                "insertion_source_idx",
                np.where(C_IDX.mask(atom_array, default="raise"), -1, atom_array.token_id.astype(np.int32)),
            )

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

        # Per-atom mask over the expanded array: True for atoms of the original
        # ground-truth tokens whose atoms were copied into the unindexed conditioning slice.
        n_original_atoms = atom_array.array_length()
        n_expanded_atoms = atom_array_expanded.array_length() - n_original_atoms
        mask = np.concatenate([is_token_with_unindexed_atoms, np.zeros(n_expanded_atoms)])

        # Expose the same mask as an atom_array annotation for downstream bookkeeping
        # (These are the ORIGINAl ground-truth tokens, not e.g., the appended in-context copies)
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
            n_unindexed_tokens = int(apply_token_wise(atom_array, is_unindexed_atom, np.any).sum())
            tc_orig = get_token_count(atom_array)
            tc_exp = get_token_count(atom_array_expanded)
            assert tc_exp == tc_orig + n_unindexed_tokens, (
                "UnindexFlaggedTokens failed to create uniquely recognised tokens after "
                f"concatenation: expanded tokens={tc_exp}, original={tc_orig}, "
                f"unindexed={n_unindexed_tokens} (expected expanded == original + unindexed)."
            )

        data["atom_array"] = atom_array_expanded
        return data


def _get_target_indices_corresponding_to_inserted(
    atom_array: AtomArray,
    insertion_mask: np.ndarray,
    target_mask: np.ndarray,
    allow_element_fallback: bool = False,
) -> np.ndarray:
    """Find indices into the target token giving the same atom names/ordering as the inserted token.

    ``insertion_mask`` and ``target_mask`` select the inserted and target atoms from
    ``atom_array``. If ``allow_element_fallback`` is True, atom names not found in the
    target token are matched by element, with ties broken by minimizing total distance.
    """
    # Get insertion atom names
    insertion_atom_names = atom_array.atom_name[insertion_mask]

    # Get target token data
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


def annotate_unindexed_insertion(
    atom_array: AtomArray,
    token_mapping: dict[int, int],
) -> AtomArray:
    """Annotate generated tokens provided a token mapping.

    For each ``{unindexed_token_id: target_token_id}`` entry the unindexed in-context
    token is transferred onto its mapped generated token, recording:

    - ``SA_CRDSRC`` -- generated atoms that carry coordinate conditioning (atom-level; under
      partial conditioning such as tip-only internal coords, only the conditioned atoms);
    - ``SA_CWI`` -- the unindexed coordinates at those coord-conditioned atom positions;
    - ``SA_CRDGRPSRC`` -- the source CoordinateGroup for each coord-conditioned atom;
    - ``SA_UII`` -- per-atom bool flagging *all* atoms that map to unindexed insertion targets
      (whole token, regardless of whether they are coordinate-conditioned).

    Failed mappings (e.g., that conflict in sequence) leave the unindexed region unmapped.

    Args:
        atom_array: AtomArray with the unindexed in-context conditioning appended to the
            generated tokens.
        token_mapping: ``{unindexed_token_id: target_token_id}`` (``token_id`` is globally
            unique)

    Returns:
        Annotated generated AtomArray with the unindexed conditioning slice removed.
    """
    has_index = C_IDX.mask(atom_array, default="generate")
    model_generated_atom_array = atom_array[has_index].copy()
    in_context_conditioning = atom_array[~has_index].copy()

    n_atoms = model_generated_atom_array.array_length()

    def _set_empty_unindexed_annotations() -> AtomArray:
        """Set all unindexed SAs to empty defaults and merge indexed conditioning into SA_CRDSRC.

        Used both when there are no unindexed tokens and when the token mapping fails.
        """
        empty_bool = np.zeros(n_atoms, dtype=bool)
        SA_CRDSRC.set_mask(model_generated_atom_array, empty_bool)
        SA_CRDSRC.set_annotation(model_generated_atom_array, SA_CRDSRC.default_annotation(model_generated_atom_array))
        SA_CRDGRPSRC.set_mask(model_generated_atom_array, empty_bool)
        SA_CRDGRPSRC.set_annotation(
            model_generated_atom_array, SA_CRDGRPSRC.default_annotation(model_generated_atom_array)
        )
        SA_UII.set_mask(model_generated_atom_array, empty_bool)
        return _merge_indexed_coord_conditioning_into_src(model_generated_atom_array)

    # Short-circuit: no unindexed tokens at all — populate indexed conditioning and return.
    if in_context_conditioning.array_length() == 0:
        return _set_empty_unindexed_annotations()

    # Union of absolute and internal coord conditioning
    has_3d_conditioning = C_CRD.mask(in_context_conditioning, default="generate") | C_INTCRD.mask(
        in_context_conditioning, default="generate"
    )
    has_distance_conditioning = C_DIS.mask(in_context_conditioning, default="generate").as_dense_array().any(axis=0)
    if has_distance_conditioning.any():
        raise NotImplementedError(
            "Insertion annotation for 2d distance-conditioned unindexed regions is not supported."
        )

    # Track (a) every generated atom mapped from an unindexed token (whole token, drives SA_UII) and
    # (b) the subset of those atoms that actually carry coord conditioning
    mapped_to_unindexed_insertion = np.zeros(n_atoms, dtype=bool)
    mapped_to_coordinate_conditioned_region = np.zeros(n_atoms, dtype=bool)
    remapped_coord = np.full_like(model_generated_atom_array.coord, np.nan)
    src_coordinate_conditioned_region = SA_CRDSRC.default_annotation(model_generated_atom_array)
    src_coord_group = SA_CRDGRPSRC.default_annotation(model_generated_atom_array)
    generated_indices = np.arange(n_atoms)

    reference_coord = C_INTCRD.annotation(model_generated_atom_array, default="generate").copy()
    unindexed_internal_coords = C_INTCRD.annotation(in_context_conditioning, default="generate")

    coord_groups_all = C_CRDGRP.annotation(atom_array, default="generate")
    unindexed_coord_groups = coord_groups_all[~has_index]

    # Only coordinate-conditioned atoms are transferred onto the target token, so the rest (e.g. the
    # sidechain of a sequence-free residue) must not be matched -- they have no partner to find.
    is_coord_conditioned_in_context = np.zeros(atom_array.array_length(), dtype=bool)
    is_coord_conditioned_in_context[~has_index] = has_3d_conditioning

    # Apply the token mapping
    try:
        for unindexed_token_id, target_token_id in token_mapping.items():
            unindexed_token_mask = in_context_conditioning.token_id == unindexed_token_id
            unindexed_token = in_context_conditioning[unindexed_token_mask]

            if SA_SRC.has_annotation(unindexed_token):
                token_src_id = SA_SRC.annotation(unindexed_token)[0]
            else:
                token_src_id = str(unindexed_token_id)

            # Local permutation of atoms within the target token matching the unindexed token's atom ordering/subset
            insertion_mask = (atom_array.token_id == unindexed_token_id) & is_coord_conditioned_in_context
            target_mask = (atom_array.token_id == target_token_id) & has_index
            target_indices_corresponding_to_inserted = _get_target_indices_corresponding_to_inserted(
                atom_array,
                insertion_mask,
                target_mask,
                allow_element_fallback=True,
            )
            target_mask_generated = model_generated_atom_array.token_id == target_token_id
            permuted_target_indices_generated = generated_indices[target_mask_generated][
                target_indices_corresponding_to_inserted
            ]

            mapped_to_unindexed_insertion[target_mask_generated] = True

            # Selects this token's coord-conditioned atoms out of its in-context slice, in the same
            # order as the matched atoms in permuted_target_indices_generated.
            token_coord_mask = has_3d_conditioning[unindexed_token_mask]
            mapped_to_coordinate_conditioned_region[permuted_target_indices_generated] = True
            src_coordinate_conditioned_region[permuted_target_indices_generated] = token_src_id
            src_coord_group[permuted_target_indices_generated] = unindexed_coord_groups[unindexed_token_mask][
                token_coord_mask
            ]
            remapped_coord[permuted_target_indices_generated] = unindexed_token.coord[token_coord_mask]
            reference_coord[permuted_target_indices_generated] = unindexed_internal_coords[unindexed_token_mask][
                token_coord_mask
            ]

        SA_CRDSRC.set_mask(model_generated_atom_array, mapped_to_coordinate_conditioned_region)
        SA_CRDSRC.set_annotation(model_generated_atom_array, src_coordinate_conditioned_region)
        SA_CRDGRPSRC.set_mask(model_generated_atom_array, mapped_to_coordinate_conditioned_region)
        SA_CRDGRPSRC.set_annotation(model_generated_atom_array, src_coord_group)
        SA_UII.set_mask(model_generated_atom_array, mapped_to_unindexed_insertion)

        # Stage the unindexed coordinates as SA_CWI (consumed by downstream metrics)
        if mapped_to_coordinate_conditioned_region.any():
            if np.isnan(remapped_coord[mapped_to_coordinate_conditioned_region]).any():
                raise ValueError("NaN values found in remapped coordinates")
            cwi = SA_CWI.annotation(model_generated_atom_array)
            cwi[mapped_to_coordinate_conditioned_region] = remapped_coord[mapped_to_coordinate_conditioned_region]
            SA_CWI.set_annotation(model_generated_atom_array, cwi)

    except ValueError as e:
        global_logger.warning(
            f"Could not apply the insertion mapping ({e}); the unindexed region will not be annotated."
        )
        return _set_empty_unindexed_annotations()

    # Fold directly-indexed coord conditioning into SA_CRDSRC so the per-group re-staging below
    # sees indexed and unindexed atoms of the same coordinate group together.
    model_generated_atom_array = _merge_indexed_coord_conditioning_into_src(model_generated_atom_array)

    # Re-stage SA_CWI for internal-coord groups with a single per-group superposition
    model_generated_atom_array = _restage_internal_coord_cwi_per_group(model_generated_atom_array, reference_coord)

    if has_3d_conditioning.any():
        # Both counts are atom-level over coord-conditioned atoms, so partial conditioning
        # (e.g. tip-only) no longer trips this; a genuine mismatch means some conditioned atoms
        # were not mapped (the model failed to reproduce part of the motif).
        expected_unindexed_count = int(has_3d_conditioning.sum())
        actual_unindexed_count = int(mapped_to_coordinate_conditioned_region.sum())
        if actual_unindexed_count != expected_unindexed_count:
            global_logger.warning(
                f"Unindexed coord conditioning count ({expected_unindexed_count}) does not match "
                f"remapped count ({actual_unindexed_count}). "
                "The model has failed to reproduce the input unindexed motif."
            )
            # Invalidate only the unindexed motif coords; leave any directly-indexed groups intact.
            cwi = SA_CWI.annotation(model_generated_atom_array)
            cwi[mapped_to_coordinate_conditioned_region] = np.nan
            SA_CWI.set_annotation(model_generated_atom_array, cwi)

    if not mapped_to_unindexed_insertion.any():
        global_logger.warning("Did not map any unindexed regions to generated tokens.")

    return model_generated_atom_array


def _restage_internal_coord_cwi_per_group(generated_atom_array: AtomArray, reference_coord: np.ndarray) -> AtomArray:
    """Re-stage ``SA_CWI`` for each internal-coord group with a single rigid superposition.

    Indexed and unindexed atoms of the same internal coordinate group are otherwise staged into
    ``SA_CWI`` in different frames — indexed atoms are fit to the generated structure by
    :py:func:`~rfproteina.transforms.utils.conditioning_utils.insert_internal_coord_conditioned_groups`,
    while unindexed atoms are staged in the raw input frame by
    :py:func:`annotate_unindexed_insertion`. Fitting the whole group's ``reference_coord``
    (the ``C_INTCRD`` internal coords, which share one frame per group and so carry correct
    relative geometry across indexed and unindexed atoms) onto the model-generated coords in one
    superposition puts the group in a common frame, which the per-coordinate-group RMSD metrics
    (e.g. ``design_motif_allatom_rmsd``) assume. Absolute (``KNOWN_ABSOLUTE_COORDS``) and
    unconditioned groups are left untouched.

    Args:
      generated_atom_array: Generated AtomArray with ``SA_CRDSRC`` / ``SA_CRDGRPSRC`` populated.
      reference_coord: Per-atom internal-coord (``C_INTCRD``) reference of shape ``(n_atoms, 3)``;
        NaN for atoms without internal-coord conditioning.
    """
    crd_src = SA_CRDSRC.mask(generated_atom_array)
    groups = SA_CRDGRPSRC.annotation(generated_atom_array)
    is_internal = crd_src & (groups != C_CRDGRP.KNOWN_ABSOLUTE_COORDS) & (groups != C_CRDGRP.UNKNOWN_COORDS)
    if not is_internal.any():
        return generated_atom_array

    cwi = SA_CWI.annotation(generated_atom_array)
    generated_coord = generated_atom_array.coord
    for group_id in np.unique(groups[is_internal]):
        group_mask = is_internal & (groups == group_id)
        reference = reference_coord[group_mask]
        generated = generated_coord[group_mask]
        if np.isnan(reference).any() or np.isnan(generated).any():
            global_logger.warning(
                f"Skipping SA_CWI re-staging for internal coord group {group_id}: NaN coords present."
            )
            continue
        aligned, _ = struc.superimpose(generated, reference)
        cwi[group_mask] = aligned
    SA_CWI.set_annotation(generated_atom_array, cwi)
    return generated_atom_array


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
