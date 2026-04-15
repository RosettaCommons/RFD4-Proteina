"""Atom-level bond utilities (bond order, aromaticity, distances).

Extracted from :py:mod:`~proteinfoundation.datapipes.pipeline` to keep that
module focused on transform composition.
"""

import numpy as np
from atomworks.constants import BIOTITE_BOND_TYPE_TO_BOND_ORDER
from biotite.structure import AtomArray
from biotite.structure.bonds import BondType


def _build_symmetric_bond_matrix(
    atom_array: AtomArray, per_bond_values: np.ndarray, dtype: np.dtype, fill: float | bool | int = 0
) -> np.ndarray:
    """Scatter ``per_bond_values`` into a symmetric ``(n_atoms, n_atoms)`` matrix."""
    atom1_idxs, atom2_idxs, _ = atom_array.bonds.as_array().T
    n_atoms = atom_array.array_length()
    matrix = np.full((n_atoms, n_atoms), fill, dtype=dtype)
    matrix[atom1_idxs, atom2_idxs] = per_bond_values
    matrix[atom2_idxs, atom1_idxs] = per_bond_values
    return matrix


def get_bond_order_matrix(atom_array: AtomArray) -> np.ndarray:
    """Get the bond order matrix using Kekulé assignments for aromatic bonds.

    Values: 0=no bond, 1=single, 2=double, 3=triple, 4=coordination.
    Aromatic bonds get their Kekulé order (AROMATIC_SINGLE->1, AROMATIC_DOUBLE->2, AROMATIC_TRIPLE->3).
    QUADRUPLE and generic AROMATIC (no Kekulé info) are mapped to single (1).
    """
    _, _, bond_types = atom_array.bonds.as_array().T

    # Build lookup from BIOTITE_BOND_TYPE_TO_BOND_ORDER constant (sized to the biotite enum)
    lookup = np.ones(max(int(bt) for bt in BondType) + 1, dtype=np.int16)  # default: unknown -> single
    for bt, order in BIOTITE_BOND_TYPE_TO_BOND_ORDER.items():
        lookup[int(bt)] = order
    lookup[BondType.QUADRUPLE] = 1  # QUADRUPLE -> single
    lookup[BondType.COORDINATION] = 4  # COORDINATION -> own class
    lookup[BondType.AROMATIC] = 1  # generic AROMATIC (no Kekulé info) -> single

    bond_orders = lookup[bond_types]
    assert np.all(bond_orders >= 0) and np.all(bond_orders <= 4), "Invalid bond order detected"

    return _build_symmetric_bond_matrix(atom_array, bond_orders, dtype=np.int16, fill=0)


_AROMATIC_BOND_TYPES = (
    BondType.AROMATIC_SINGLE,
    BondType.AROMATIC_DOUBLE,
    BondType.AROMATIC_TRIPLE,
    BondType.AROMATIC,
)


def get_is_aromatic_matrix(atom_array: AtomArray) -> np.ndarray:
    """Get boolean aromaticity matrix for an atom array.

    A bond is aromatic if its biotite BondType is any of AROMATIC_{SINGLE,DOUBLE,TRIPLE}
    or the generic AROMATIC.
    """
    _, _, bond_types = atom_array.bonds.as_array().T
    is_aromatic = np.isin(bond_types, [int(bt) for bt in _AROMATIC_BOND_TYPES])
    return _build_symmetric_bond_matrix(atom_array, is_aromatic, dtype=bool, fill=False)


def get_bond_distances(atom_array: AtomArray) -> np.ndarray:
    """Returns the bond distance (adjacency) list as a 1D array."""
    coords = atom_array.coord
    atom1_idxs, atom2_idxs, _ = atom_array.bonds.as_array().T
    return np.linalg.norm(coords[atom1_idxs] - coords[atom2_idxs], axis=1)


def get_bond_distance_matrix(atom_array: AtomArray) -> np.ndarray:
    """Returns the bond adjacency matrix with bond distances as values."""
    return _build_symmetric_bond_matrix(atom_array, get_bond_distances(atom_array), dtype=float, fill=np.inf)
