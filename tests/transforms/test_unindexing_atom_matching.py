"""Test _get_target_indices_corresponding_to_inserted function.

Verifies atom matching with name priority and element-based distance fallback.
"""

import numpy as np
import pytest
from biotite.structure import AtomArray

from proteinfoundation.transforms.unindexing import _get_target_indices_corresponding_to_inserted


def test_atom_matching_with_name_priority_and_element_fallback():
    """Test comprehensive atom matching: name priority, element fallback, distance minimization."""

    # Helper to create atom array with specified properties
    def make_residue(res_id, names, elements, coords):
        atoms = AtomArray(len(names))
        atoms.atom_name = np.array(names)
        atoms.element = np.array(elements)
        atoms.coord = np.array(coords)
        atoms.res_id = np.full(len(names), res_id)
        atoms.chain_id = np.array(["A"] * len(names))
        return atoms

    # Test 1: Pure name matching (all atoms match by name)
    insertion = make_residue(1, ["CA", "CB", "CG"], ["C", "C", "C"], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    target = make_residue(2, ["CA", "CB", "CG"], ["C", "C", "C"], [[10, 0, 0], [11, 0, 0], [12, 0, 0]])
    atom_array = insertion + target
    chain_mask = atom_array.chain_id == "A"
    has_index = atom_array.res_id == 2

    indices = _get_target_indices_corresponding_to_inserted(
        atom_array, chain_mask, insertion_res_id=1, target_res_id=2, has_index=has_index
    )

    target_atoms = atom_array[has_index]
    assert np.array_equal(target_atoms.atom_name[indices], ["CA", "CB", "CG"]), "Name matching failed"

    # Test 2: Mixed name and element matching (name takes priority)
    insertion = make_residue(1, ["CA", "CD", "CG"], ["C", "C", "C"], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    target = make_residue(2, ["CA", "CB", "CG"], ["C", "C", "C"], [[10, 0, 0], [11, 0, 0], [12, 0, 0]])
    atom_array = insertion + target
    chain_mask = atom_array.chain_id == "A"
    has_index = atom_array.res_id == 2

    indices = _get_target_indices_corresponding_to_inserted(
        atom_array, chain_mask, insertion_res_id=1, target_res_id=2, has_index=has_index, allow_element_fallback=True
    )

    target_atoms = atom_array[has_index]
    # CA and CG match by name, OB matches CB by element
    assert target_atoms.atom_name[indices[0]] == "CA", "Name match CA failed"
    assert target_atoms.atom_name[indices[1]] == "CB", "Element fallback CD→CB failed"
    assert target_atoms.atom_name[indices[2]] == "CG", "Name match CG failed"

    # Test 3: Distance minimization in element matching
    # Two insertion C atoms should match to closest two target C atoms, not cross-match
    insertion = make_residue(1, ["CG", "CD"], ["C", "C"], [[0, 0, 0], [5, 0, 0]])
    target = make_residue(2, ["CA", "CB"], ["C", "C"], [[1, 0, 0], [6, 0, 0]])  # CA close to CG, CB close to CD
    atom_array = insertion + target
    chain_mask = atom_array.chain_id == "A"
    has_index = atom_array.res_id == 2

    indices = _get_target_indices_corresponding_to_inserted(
        atom_array, chain_mask, insertion_res_id=1, target_res_id=2, has_index=has_index, allow_element_fallback=True
    )

    target_atoms = atom_array[has_index]
    insertion_coords = atom_array.coord[:2]
    for i, target_idx in enumerate(indices):
        dist = np.linalg.norm(insertion_coords[i] - target_atoms.coord[target_idx])
        assert dist < 2.0, f"Atom {i} matched to distant target (dist={dist:.1f}), distance minimization failed"

    # Test 4: Error when names don't match and fallback disabled
    insertion = make_residue(1, ["CA", "CD", "CG"], ["C", "C", "C"], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    target = make_residue(2, ["CA", "CB", "CG"], ["C", "C", "C"], [[10, 0, 0], [11, 0, 0], [12, 0, 0]])
    atom_array = insertion + target
    chain_mask = atom_array.chain_id == "A"
    has_index = atom_array.res_id == 2

    with pytest.raises(ValueError, match="Inserted atom names not found"):
        _get_target_indices_corresponding_to_inserted(
            atom_array,
            chain_mask,
            insertion_res_id=1,
            target_res_id=2,
            has_index=has_index,
            allow_element_fallback=False,
        )
