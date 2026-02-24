"""Utilities for copying coordinates from generated structures back to input structures.

Handles coordinate transfer for workflows where residue types may change but we want to preserve annotations.
"""

import biotite.structure as struc
import numpy as np
from atomworks.common import exists
from atomworks.io.template import get_empty_ccd_template
from atomworks.io.utils.atom_array_plus import insert_atoms
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.ml.conditions.annotator import ensure_annotations
from atomworks.ml.utils.condition import default_annotations_and_conditions_from_registry
from atomworks.ml.conditions.base import CONDITIONS, Level
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray
from foundry.utils.logging import RankedLogger

logger = RankedLogger(rank_zero_only=False)

# Default atom-level annotations that vary within residues
DEFAULT_ATOM_LEVEL_ANNOTATIONS = (
    {
        "atom_name",
        "element",
        "coord",
        "hetero",
        "atom_id",
        "b_factor",
        "occupancy",
        "charge",
    }
    | {cond.mask_name for cond in CONDITIONS if (cond.n_body == 1 and cond.level == Level.ATOM)}
    | {cond.full_name for cond in CONDITIONS if (cond.n_body == 1 and cond.level == Level.ATOM)}
)


def _copy_coords_by_atom_name(
    source_array: AtomArray,
    source_mask: np.ndarray,
    target_array: AtomArray,
    target_mask: np.ndarray,
) -> None:
    """Copy coordinates from source to target for all matching atom names (in-place)."""
    source_atom_names = set(source_array.atom_name[source_mask])
    target_atom_names = target_array.atom_name[target_mask]

    for atom_name in target_atom_names:
        # Only copy if atom exists in source
        if atom_name not in source_atom_names:
            continue

        source_atom_mask = source_mask & (source_array.atom_name == atom_name)
        target_atom_mask = target_mask & (target_array.atom_name == atom_name)
        target_array.coord[target_atom_mask] = source_array.coord[source_atom_mask]


def get_residue_level_annotations(
    atom_array: AtomArray,
    token_mask: np.ndarray,
    annotations_to_exclude: set[str] | None = None,
) -> dict[str, int | float | str]:
    """Extract residue-level one-body annotations from a token.

    Returns constant annotation values that are uniform across all atoms
    in a residue/token.

    Args:
      atom_array: The atom array containing the residue.
      token_mask: Boolean mask selecting atoms from a single token/residue.
      annotations_to_exclude: Annotation names to exclude. Defaults to
        standard atom-level annotations.

    Returns:
      Mapping of annotation names to their constant values for this residue.
    """
    if annotations_to_exclude is None:
        annotations_to_exclude = DEFAULT_ATOM_LEVEL_ANNOTATIONS

    all_annots = set(get_annotation_categories(atom_array, n_body=1))
    residue_level_annots = all_annots - annotations_to_exclude

    residue_annots = {}
    for annot in residue_level_annots:
        values = atom_array.get_annotation(annot)[token_mask]

        # Skip array annotations (they indexing errors downstream)
        if isinstance(values, np.ndarray) and values.ndim > 1:
            continue

        unique_values = np.unique(values)

        # Verify annotation is actually constant across the residue
        if len(unique_values) == 1:
            value = unique_values[0]
            residue_annots[annot] = value

    return residue_annots


def copy_generated_coordinates_to_input(
    generated_atom_array: AtomArray,
    input_atom_array: AtomArray,
) -> AtomArray:
    """Copy generated coordinates back to input structure.

    Handles coordinate transfer for three cases:

    1. Single-atom tokens: Direct copy
    2. Matching residues (seq-conditioned or same type): Atom-wise copy
    3. Mismatched residues (design): Delete extra atoms, add missing atoms

    For case 3, preserves all residue-level annotations from input and creates
    new atoms, defaulting atom-level annotations where possible.

    Args:
      generated_atom_array: Structure with generated coordinates. Must have
        ``token_id`` annotation.
      input_atom_array: Original structure to update. A deep copy is made
        internally, so the original is not modified.

    Returns:
      Updated atom array with coordinates from generated structure.
      May have different atoms if residue types changed for non-seq-conditioned
      tokens.
    """
    ensure_annotations(generated_atom_array, "token_id")
    ensure_annotations(input_atom_array, "token_id")

    # Deep copy to avoid modifying the original input
    input_atom_array = input_atom_array.copy()

    assert set(input_atom_array.token_id) == set(
        generated_atom_array.token_id
    ), "Input and generated atom arrays must have the same tokens"

    input_atom_array.set_annotation("original_global_index", np.arange(input_atom_array.array_length()))
    original_input_coords = input_atom_array.coord.copy()

    extra_atoms_mask = np.zeros(len(input_atom_array), dtype=bool)
    missing_atoms_and_indices = []

    for token_id in np.unique(input_atom_array.token_id):
        # Get annotations for input token
        input_token_mask = input_atom_array.token_id == token_id
        input_res_name = input_atom_array.res_name[input_token_mask][0]
        input_atom_names = input_atom_array.atom_name[input_token_mask]

        # Get corresponding annotations for the generated token
        generated_token_mask = generated_atom_array.token_id == token_id
        generated_res_name = generated_atom_array.res_name[generated_token_mask][0]
        generated_atom_names = generated_atom_array.atom_name[generated_token_mask]

        # + ---------------------------------------------------------------------------------- +
        # CASE 1: Single-atom token (e.g., small molecule, atomized) -> copy directly
        # + ---------------------------------------------------------------------------------- +
        if np.sum(input_token_mask) == 1:
            assert np.sum(generated_token_mask) == 1, "Generated token should also be single-atom"
            input_atom_array.coord[input_token_mask] = generated_atom_array.coord[generated_token_mask]
            continue

        # + ---------------------------------------------------------------------------------- +
        # CASE 2: Matching residue names -> copy by atom name (to ensure order invariance)
        # + ---------------------------------------------------------------------------------- +
        if generated_res_name == input_res_name:
            input_atom_name_set = set(input_atom_names)
            generated_atom_name_set = set(generated_atom_names)
            assert input_atom_name_set.issubset(generated_atom_name_set), (
                f"Input atom names must be a subset of generated atom names for matching residues. "
                f"Input token {token_id} has atoms {input_atom_names}, generated has atoms {generated_atom_names}."
            )
            if input_atom_name_set != generated_atom_name_set:
                atom_names_diff = generated_atom_name_set - input_atom_name_set
                logger.warning(
                    f"Some generated atom names are not present in the input AtomArray: {atom_names_diff} "
                    f"for residue type {input_res_name}. This may result from an RCSB PDB structure with missing atoms."
                )
            else:
                _copy_coords_by_atom_name(
                    generated_atom_array, generated_token_mask, input_atom_array, input_token_mask
                )
                continue

        # + ---------------------------------------------------------------------------------- +
        # CASE 3: Residue name changed -> keep matching atoms; add/remove atoms as needed
        # + ---------------------------------------------------------------------------------- +
        missing_atoms = np.setdiff1d(generated_atom_names, input_atom_names)
        extra_atoms = np.setdiff1d(input_atom_names, generated_atom_names)

        # Keep track of atoms that we need to remove later
        extra_atoms_mask |= input_token_mask & np.isin(input_atom_array.atom_name, extra_atoms)

        # Copy matching atoms
        _copy_coords_by_atom_name(generated_atom_array, generated_token_mask, input_atom_array, input_token_mask)

        # Update residue name
        input_atom_array.res_name[input_token_mask] = generated_res_name

        # Prepare to add missing atoms later
        if len(missing_atoms) > 0:
            residue_annots = get_residue_level_annotations(input_atom_array, input_token_mask)

            # Use the values in the template, matching what the model generated
            residue_annots.pop("res_name", None)
            residue_annots.pop("atom_name", None)
            residue_annots.pop("element", None)

            residue_annots["original_global_index"] = np.full_like(input_token_mask.sum(), -1, dtype=int)

            # Create template for new residue type with all residue-level annotations
            template = get_empty_ccd_template(
                ccd_code=generated_res_name,
                remove_hydrogens=True,
                **residue_annots,
            )

            # Subset to missing atoms only
            template = template[np.isin(template.atom_name, missing_atoms)]

            # Spoof all annotations with defaults so they survive insert_atoms()
            annotations_to_default = get_annotation_categories(input_atom_array, n_body="all")
            existing_annotations = get_annotation_categories(template, n_body="all")
            annotations_to_default = set(annotations_to_default) - set(existing_annotations)
            template = default_annotations_and_conditions_from_registry(template, annotations=annotations_to_default)

            # Set coordinates from generated atom array
            _copy_coords_by_atom_name(
                generated_atom_array, generated_token_mask, template, np.ones(len(template), dtype=bool)
            )

            missing_atoms_and_indices.append((int(token_id), template))

    # Delete the extra atoms...
    input_atom_array = input_atom_array[~extra_atoms_mask]

    # ... and add the missing atoms
    if len(missing_atoms_and_indices) > 0:
        # Shift by one so we insert at the END of each token
        token_starts_shifted = get_token_starts(input_atom_array, add_exclusive_stop=True)[1:]
        insertion_indices = [int(token_starts_shifted[token_id]) for token_id, _ in missing_atoms_and_indices]
        atoms_to_add = [atom_template for _, atom_template in missing_atoms_and_indices]

        # NOTE: We insert BEFORE the specified index
        input_atom_array = insert_atoms(input_atom_array, atoms_to_add, insertion_indices)

        # Standardize atom order
        input_atom_array = input_atom_array[struc.info.standardize_order(input_atom_array)]

        # Add any missing canonical bonds after atom insertion
        canonical_intra_residue_bonds = struc.bonds.connect_via_residue_names(
            input_atom_array[~input_atom_array.atomize], inter_residue=True
        )
        input_atom_array.bonds = input_atom_array.bonds.merge(canonical_intra_residue_bonds)

    # Add an annotation for the original coords, with NaNs for added atoms
    derived_from_original_input = np.where(input_atom_array.original_global_index != -1)[0]
    reshaped_original_global_index = input_atom_array.original_global_index[derived_from_original_input]
    reshaped_original_input_coords = np.full_like(input_atom_array.coord, np.nan)
    reshaped_original_input_coords[derived_from_original_input] = original_input_coords[reshaped_original_global_index]
    input_atom_array.set_annotation("original_input_coord", reshaped_original_input_coords)

    return input_atom_array
