"""Tests for metal-isolation and bond-dropout transforms."""

import numpy as np
import pytest

from rfproteina.datapipes.pipeline import DropoutAromaticFlag
from rfproteina.transforms.metals import DropoutCoordinationBonds

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
