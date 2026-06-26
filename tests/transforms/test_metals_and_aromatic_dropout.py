"""Tests for metal-isolation and bond-dropout transforms."""

import biotite.structure as struc
import numpy as np
import pytest

from proteinfoundation.datapipes.pipeline import DropoutAromaticFlag
from proteinfoundation.transforms.metals import (
    DropoutCoordinationBonds,
    IsolateMetalsToOwnChain,
)

# ------------------------------------------------------------------------- #
# IsolateMetalsToOwnChain                                                   #
# ------------------------------------------------------------------------- #


def _make_zn_coordinated_array() -> struc.AtomArray:
    """Tiny synthetic structure: chain A with 4 N atoms + a Zn metal sharing the chain."""
    n = 5
    atoms = struc.AtomArray(n)
    atoms.coord = np.array(
        [
            [0.0, 0.0, 0.0],  # 0: Zn (shares chain A with neighbors -- pre-isolation)
            [2.0, 0.0, 0.0],  # 1: N
            [-2.0, 0.0, 0.0],  # 2: N
            [0.0, 2.0, 0.0],  # 3: N
            [0.0, -2.0, 0.0],  # 4: N
        ],
        dtype=np.float32,
    )
    atoms.element = np.array(["ZN", "N", "N", "N", "N"])
    atoms.atom_name = np.array(["ZN", "N1", "N2", "N3", "N4"])
    atoms.res_name = np.array(["LIG", "LIG", "LIG", "LIG", "LIG"])
    atoms.chain_id = np.array(["A", "A", "A", "A", "A"])
    atoms.res_id = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    atoms.hetero = np.array([True, True, True, True, True])
    atoms.set_annotation("is_polymer", np.array([False, False, False, False, False]))

    # Pre-isolation: every Zn–N bond is SINGLE (CCD-style)
    bonds = struc.BondList(
        n,
        np.array(
            [
                [0, 1, struc.BondType.SINGLE],
                [0, 2, struc.BondType.SINGLE],
                [0, 3, struc.BondType.SINGLE],
                [0, 4, struc.BondType.SINGLE],
            ],
            dtype=np.uint32,
        ),
    )
    atoms.bonds = bonds
    return atoms


def test_isolate_metals_moves_zn_to_own_chain_and_rebonds_as_coordination():
    atom_array = _make_zn_coordinated_array()
    data = IsolateMetalsToOwnChain().forward({"atom_array": atom_array})
    out = data["atom_array"]

    # Metals are moved to the end; find Zn by element rather than by position
    zn_indices = np.where(np.char.upper(out.element.astype(str)) == "ZN")[0]
    assert len(zn_indices) == 1
    zn_idx = zn_indices[0]
    non_zn_indices = np.where(np.char.upper(out.element.astype(str)) != "ZN")[0]

    # Zn is now on a chain distinct from the ligand atoms
    assert out.chain_id[zn_idx] != "A"
    assert all(out.chain_id[i] == "A" for i in non_zn_indices)

    # res_name is the element symbol
    assert out.res_name[zn_idx] == "ZN"

    # Every bond touching Zn is now COORDINATION
    bond_arr = out.bonds.as_array()
    for i, j, btype in bond_arr:
        if zn_idx in (i, j):
            assert btype == int(struc.BondType.COORDINATION), f"bond {(i, j)} should be COORDINATION, got {btype}"


def test_isolate_metals_is_noop_without_metals():
    atom_array = _make_zn_coordinated_array()
    # Strip the Zn so no metal remains
    atom_array = atom_array[1:]
    pre_chain = atom_array.chain_id.copy()
    pre_bonds = atom_array.bonds.as_array().copy()

    out = IsolateMetalsToOwnChain().forward({"atom_array": atom_array})["atom_array"]

    assert np.array_equal(out.chain_id, pre_chain)
    assert np.array_equal(out.bonds.as_array(), pre_bonds)


# ------------------------------------------------------------------------- #
# DropoutCoordinationBonds                                                  #
# ------------------------------------------------------------------------- #


def _make_bond_feats(n: int = 6, n_coord_bonds: int = 4) -> dict:
    """Build a symmetric (bond_order, bond_mask, is_aromatic) trio with coord bonds."""
    bond_order = np.zeros((n, n), dtype=np.int16)
    # Add a few SINGLE bonds (order 1) and a few COORDINATION (order 4)
    bond_order[0, 1] = bond_order[1, 0] = 1
    bond_order[2, 3] = bond_order[3, 2] = 1
    coord_pairs = [(4, 0), (4, 1), (4, 2), (4, 3)][:n_coord_bonds]
    for i, j in coord_pairs:
        bond_order[i, j] = bond_order[j, i] = 4

    bond_mask = bond_order != 0
    is_aromatic = np.zeros((n, n), dtype=bool)
    return {"feats": {"bond_order": bond_order, "bond_mask": bond_mask, "is_aromatic": is_aromatic}}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dropout_coordination_p_all_drops_every_coordination_bond(seed: int):
    """With p_all=1.0, ALL coordination bonds must vanish across all matrices, deterministically."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        data = _make_bond_feats()
        out = DropoutCoordinationBonds(p_all=1.0, p_individual=0.0).forward(data)
        bo = out["feats"]["bond_order"]
        bm = out["feats"]["bond_mask"]
        ia = out["feats"]["is_aromatic"]

        assert (bo == 4).sum() == 0, "no COORDINATION bonds should remain"
        # Non-coordination bonds untouched
        assert bo[0, 1] == 1 and bm[0, 1] and not ia[0, 1]
        # All three matrices stay symmetric
        assert np.array_equal(bo, bo.T)
        assert np.array_equal(bm, bm.T)
        assert np.array_equal(ia, ia.T)
    finally:
        np.random.set_state(rng_state)


def test_dropout_coordination_p_all_zero_p_individual_zero_is_noop():
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        data = _make_bond_feats()
        before_bo = data["feats"]["bond_order"].copy()
        before_bm = data["feats"]["bond_mask"].copy()
        out = DropoutCoordinationBonds(p_all=0.0, p_individual=0.0).forward(data)
        assert np.array_equal(out["feats"]["bond_order"], before_bo)
        assert np.array_equal(out["feats"]["bond_mask"], before_bm)
    finally:
        np.random.set_state(rng_state)


def test_dropout_coordination_individual_drop_is_symmetric():
    """When dropping per-edge, dropped (i,j) pairs must be zeroed in both directions."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(7)
        data = _make_bond_feats(n_coord_bonds=4)
        out = DropoutCoordinationBonds(p_all=0.0, p_individual=0.5).forward(data)
        bo = out["feats"]["bond_order"]
        # Symmetry preserved across the whole matrix
        assert np.array_equal(bo, bo.T)
        # The bond_mask / is_aromatic edits happen at the same (i,j) positions
        assert np.array_equal(bo == 0, ~out["feats"]["bond_mask"] | (data["feats"]["bond_order"] == 0))
    finally:
        np.random.set_state(rng_state)


# ------------------------------------------------------------------------- #
# DropoutAromaticFlag                                                       #
# ------------------------------------------------------------------------- #


def test_dropout_aromatic_p_one_zeros_all_aromatic_flags():
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        feats = {
            "is_aromatic": np.array(
                [[False, True, False], [True, False, True], [False, True, False]],
                dtype=bool,
            )
        }
        out = DropoutAromaticFlag(p=1.0).forward({"feats": feats})
        assert not out["feats"]["is_aromatic"].any()
    finally:
        np.random.set_state(rng_state)


def test_dropout_aromatic_p_zero_preserves_flags():
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        original = np.array([[False, True], [True, False]], dtype=bool)
        feats = {"is_aromatic": original.copy()}
        out = DropoutAromaticFlag(p=0.0).forward({"feats": feats})
        assert np.array_equal(out["feats"]["is_aromatic"], original)
    finally:
        np.random.set_state(rng_state)
