import sys

import numpy as np
import pytest
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from biotite.structure import AtomArray

from rfproteina.conditions import C_IDXGRP
from rfproteina.transforms.unindexing import (
    find_optimal_insertion_points_for_unindexed_regions,
)

# ============================================================================
# Helper Functions
# ============================================================================


def _create_indexed_atom_array(n_residues_per_chain, residue_types=None):
    """
    Create an atom array with indexed residues across multiple chains.

    Args:
        n_residues_per_chain: List of number of residues per chain
        residue_types: List of residue type codes (cycles through if needed)

    Returns:
        atom_array: AtomArray with only indexed residues
        residue_atoms: Dict mapping residue types to their atom lists
    """
    if residue_types is None:
        residue_types = ["ALA", "GLY", "VAL", "LEU", "SER"]

    # Define atoms for each residue type
    residue_atoms = {
        "ALA": ["N", "CA", "C", "O", "CB"],
        "GLY": ["N", "CA", "C", "O"],
        "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
        "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
        "SER": ["N", "CA", "C", "O", "CB", "OG"],
    }

    all_coords = []
    all_chain_ids = []
    all_res_ids = []
    all_res_names = []
    all_atom_names = []

    np.random.seed(42)
    chain_letters = ["A", "B", "C", "D", "E", "F"]

    for chain_idx, n_residues in enumerate(n_residues_per_chain):
        chain_id = chain_letters[chain_idx]

        for res_idx in range(n_residues):
            # Cycle through residue types
            res_type = residue_types[res_idx % len(residue_types)]
            atoms = residue_atoms[res_type]
            res_id = res_idx + 1

            for atom_name in atoms:
                all_chain_ids.append(chain_id)
                all_res_ids.append(res_id)
                all_res_names.append(res_type)
                all_atom_names.append(atom_name)
                all_coords.append(np.random.rand(3) * 20.0)

    n_atoms = len(all_coords)
    atom_array = AtomArray(n_atoms)
    atom_array.coord = np.array(all_coords)
    atom_array.set_annotation("chain_id", np.array(all_chain_ids))
    atom_array.set_annotation("res_id", np.array(all_res_ids))
    atom_array.set_annotation("res_name", np.array(all_res_names))
    atom_array.set_annotation("atom_name", np.array(all_atom_names))
    atom_array.set_annotation("element", np.array([name[0] for name in all_atom_names]))
    atom_array.set_annotation("is_ligand", np.full(n_atoms, False))
    atom_array.set_annotation("within_index_group_res_idx", np.arange(n_atoms))

    ensure_annotations(atom_array, "token_id")

    atom_array = as_atom_array_plus(atom_array)
    C_IDX.set_mask(atom_array, np.ones(n_atoms, dtype=bool))
    C_IDXGRP.set_annotation(atom_array, np.zeros(n_atoms, dtype=int))

    return atom_array, residue_atoms


def _add_unindexed_residues(atom_array, residue_atoms, unindexed_specs):
    """
    Add unindexed residues to an existing atom array.

    Args:
        atom_array: Existing AtomArray with indexed residues
        residue_atoms: Dict mapping residue types to their atom lists
        unindexed_specs: List of tuples (chain_id, res_id, res_type, group_id, atom_subset)
                        atom_subset can be 'all' or a list of atom names

    Returns:
        Extended atom array with unindexed residues
    """
    # Extract existing data
    existing_coords = list(atom_array.coord)
    existing_chain_ids = list(atom_array.chain_id)
    existing_res_ids = list(atom_array.res_id)
    existing_res_names = list(atom_array.res_name)
    existing_atom_names = list(atom_array.atom_name)
    existing_elements = list(atom_array.element)
    existing_has_index = list(C_IDX.mask(atom_array, default="raise"))
    existing_index_groups = list(C_IDXGRP.annotation(atom_array))
    existing_within_index_group_res_idx = list(atom_array.within_index_group_res_idx)

    for chain_id, res_id, res_type, group_id, atom_subset, within_index_group_res_idx in unindexed_specs:
        all_atoms = residue_atoms[res_type]
        atoms_to_include = all_atoms if atom_subset == "all" else atom_subset

        for atom_name in atoms_to_include:
            if atom_name not in all_atoms:
                raise ValueError(f"Atom {atom_name} not valid for residue type {res_type}")

            existing_chain_ids.append(chain_id)
            existing_res_ids.append(res_id)
            existing_res_names.append(res_type)
            existing_atom_names.append(atom_name)
            existing_elements.append(atom_name[0])
            existing_has_index.append(False)
            existing_index_groups.append(group_id)
            existing_coords.append(np.zeros(3))
            existing_within_index_group_res_idx.append(within_index_group_res_idx)

    n_atoms = len(existing_coords)
    new_atom_array = AtomArray(n_atoms)
    new_atom_array.coord = np.array(existing_coords)
    new_atom_array.set_annotation("chain_id", np.array(existing_chain_ids))
    new_atom_array.set_annotation("res_id", np.array(existing_res_ids))
    new_atom_array.set_annotation("res_name", np.array(existing_res_names))
    new_atom_array.set_annotation("atom_name", np.array(existing_atom_names))
    new_atom_array.set_annotation("element", np.array(existing_elements))
    new_atom_array.set_annotation("is_ligand", np.full(n_atoms, False))
    new_atom_array.set_annotation("within_index_group_res_idx", np.array(existing_within_index_group_res_idx))

    ensure_annotations(new_atom_array, "token_id")

    new_atom_array = as_atom_array_plus(new_atom_array)
    C_IDX.set_mask(new_atom_array, np.array(existing_has_index, dtype=bool))
    C_IDXGRP.set_annotation(new_atom_array, np.array(existing_index_groups, dtype=int))

    return new_atom_array


def _setup_coordinates_for_mapping(atom_array, ground_truth_mapping, jitter_scale=0.2, seed=123):
    """
    Set up coordinates so that ground truth mapping is optimal.

    Args:
        atom_array: AtomArray with indexed and unindexed residues
        ground_truth_mapping: Dict {chain_id: {unindexed_res_id: indexed_res_id}}
        jitter_scale: Standard deviation for jitter
        seed: Random seed

    Returns:
        Modified atom_array with coordinates set
    """
    np.random.seed(seed)
    has_index = C_IDX.mask(atom_array, default="raise")

    # Set indexed residue coordinates (well-separated)
    for chain_id in np.unique(atom_array.chain_id):
        chain_mask = atom_array.chain_id == chain_id
        chain_offset = ord(chain_id) * 200  # Separate chains in space

        for res_id in np.unique(atom_array.res_id[chain_mask & has_index]):
            res_mask = (atom_array.res_id == res_id) & (atom_array.chain_id == chain_id) & has_index
            base_coords = np.array([chain_offset + res_id * 15.0, 0.0, 0.0])

            for i, atom_idx in enumerate(np.where(res_mask)[0]):
                atom_array.coord[atom_idx] = base_coords + np.array([i * 0.3, 0, 0])

    # Apply jittered coordinates to unindexed regions
    for chain_id, res_mapping in ground_truth_mapping.items():
        for unindexed_res_id, indexed_res_id in res_mapping.items():
            unindexed_mask = (atom_array.res_id == unindexed_res_id) & (atom_array.chain_id == chain_id) & ~has_index
            indexed_mask = (atom_array.res_id == indexed_res_id) & (atom_array.chain_id == chain_id) & has_index

            unindexed_atoms = atom_array.atom_name[unindexed_mask]
            indexed_atoms = atom_array.atom_name[indexed_mask]

            for atom_name in unindexed_atoms:
                if atom_name in indexed_atoms:
                    unindexed_idx = np.where(unindexed_mask & (atom_array.atom_name == atom_name))[0][0]
                    indexed_idx = np.where(indexed_mask & (atom_array.atom_name == atom_name))[0][0]

                    jitter = np.random.randn(3) * jitter_scale
                    atom_array.coord[unindexed_idx] = atom_array.coord[indexed_idx] + jitter

    return atom_array


def _verify_mapping(result_mapping, ground_truth_mapping):
    """
    Verify that result mapping matches ground truth.

    Args:
        result_mapping: Dict {unindexed_token_id: target_token_id}
        ground_truth_mapping: Dict {unindexed_token_id: target_token_id}
    """
    assert result_mapping is not None, "Function should return a mapping"
    assert (
        result_mapping == ground_truth_mapping
    ), f"Mapping mismatch: got {result_mapping}, expected {ground_truth_mapping}"


# ============================================================================
# Tests
# ============================================================================

# NOTE: We only test the optimal insertion functionality for 3d, since the 3d reinsertion & metrics computation are
# covered elsewhere (test_unindexing.py). That test does not cover the 2d case, so the 2d test below is all-in-one.


@pytest.mark.fast
def test_find_optimal_insertion_points_3d_conditioning():
    """
    Test with 3D coordinate conditioning.

    Features tested:
    - Multiple chains (A and B)
    - Varying residue types with different atom counts
    - Tip-atom residues (subset of atoms)
    - Overlapping res_ids across chains (res_id 100 in both chains)
    - Non-contiguous res_ids within a group (100, 101, 103 with gap at 102)

    """
    # Create indexed residues: ALA, GLY, VAL, LEU, SER pattern
    residue_pattern = ["ALA", "GLY", "VAL", "LEU", "SER"]
    atom_array, residue_atoms = _create_indexed_atom_array(
        n_residues_per_chain=[8, 6],
        residue_types=residue_pattern,
    )

    # Add unindexed residues:
    # Chain A: Group 1 with NON-CONTIGUOUS res_ids (100, 101, 103) mapping to (1, 2, 4)
    #          Note: gap at 102 in unindexed matches gap at 3 in indexed
    # Chain B: Group 2 with res_ids (100, 101) - OVERLAPS with chain A res_ids!
    unindexed_specs = [
        # Group 1 in chain A - non-contiguous res_ids
        ("A", 100, "ALA", 1, "all", 0),
        ("A", 101, "GLY", 1, "all", 1),
        ("A", 103, "LEU", 1, ["CG", "CD1", "CD2"], 3),  # Tip-atom example
        # Group 2 in chain B - overlapping res_ids with chain A
        ("B", 100, "ALA", 2, "all", 0),
        ("B", 101, "GLY", 2, "all", 1),
    ]

    atom_array = _add_unindexed_residues(atom_array, residue_atoms, unindexed_specs)

    # Coordinate setup uses the chain-aware view: {chain_id: {unindexed_res_id: indexed_res_id}}
    # Chain A: (100, 101, 103) -> (1, 2, 4) which are ALA, GLY, LEU
    # Chain B: (100, 101) -> (1, 2) which are ALA, GLY
    coordinate_mapping = {
        "A": {100: 1, 101: 2, 103: 4},
        "B": {100: 1, 101: 2},
    }

    # Ground truth mapping returned by the function: {unindexed_token_id: target_token_id}.
    # Indexed residues are tokens 0-13 (chain A res 1-8 -> tokens 0-7, chain B res 1-6 -> tokens 8-13).
    # Unindexed residues are appended as tokens 14-18 in spec order.
    #   token 14 (A/100) -> A res 1  -> token 0
    #   token 15 (A/101) -> A res 2  -> token 1
    #   token 16 (A/103) -> A res 4  -> token 3
    #   token 17 (B/100) -> B res 1  -> token 8
    #   token 18 (B/101) -> B res 2  -> token 9
    ground_truth_mapping = {14: 0, 15: 1, 16: 3, 17: 8, 18: 9}

    # Setup coordinates
    atom_array = _setup_coordinates_for_mapping(atom_array, coordinate_mapping, jitter_scale=0.2)

    # Add 3D coordinate conditioning
    has_index = C_IDX.mask(atom_array, default="raise")
    C_CRD.set_mask(atom_array, ~has_index)
    C_SEQ.set_mask(atom_array, ~has_index)

    # Run function
    result_mapping = find_optimal_insertion_points_for_unindexed_regions(atom_array)

    # Verify
    _verify_mapping(result_mapping, ground_truth_mapping)


if __name__ == "__main__":
    pytest.main(sys.argv)
