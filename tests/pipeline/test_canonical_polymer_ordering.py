"""Topology of the pair-track bond features across the CP-SEA cyclisation types.

Direction in ``canonical_polymer_ordering`` comes from the bonded
``(leaving, entering)`` atom pair, so a closure is read as chemistry rather than
as a gap between array positions. Two behaviours follow, and the fixtures pin both:

* a **backbone linkage** carries ``+-1`` ordering and is suppressed in ``bond_mask``;
* a **crosslink** (disulfide, isopeptide) carries no ordering and stays visible.
"""

import numpy as np
import pytest
from atomworks.constants import STANDARD_AA
from atomworks.io.utils.standard_annotations import S_SEGMIN
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray, BondList, BondType

from rfproteina.datapipes.bonds import compute_token_bond_features
from tests.conftest import TEST_DATA_DIR
from tests.e2e.helpers.regression_baselines import load_design_cif

CYCLIC_DIR = TEST_DATA_DIR / "cyclic_peptides"

# file, closure endpoints (upstream, downstream), closure is a backbone linkage
CYCLIC_CASES = [
    pytest.param("cpsea_head_tail_4djs.cif.gz", ("THR183", "ASN179"), True, id="head_tail"),
    pytest.param("cpsea_disulfide_7bob.cif.gz", ("CYS118", "CYS123"), False, id="disulfide"),
    pytest.param("cpsea_isopeptide_6rlv.cif.gz", ("LYS899", "ASP903"), False, id="isopeptide"),
]


def _load(name: str) -> AtomArray:
    """Load a fixture and apply the annotations the pipeline would set."""
    aa = load_design_cif(CYCLIC_DIR / name)
    # ACE/NME caps are non-standard, so AtomizeByCCDName would atomize them.
    aa.set_annotation("atomize", ~np.isin(aa.res_name, list(STANDARD_AA)))
    aa.set_annotation("insertion_source_idx", np.full(aa.array_length(), -1, dtype=np.int32))
    return aa


def _token_of(aa: AtomArray, selection: str) -> int:
    """Resolve ``"CYS118"`` to its token index, so expectations read chemically."""
    res_name, res_id = selection[:3], int(selection[3:])
    starts = get_token_starts(aa)
    hits = np.flatnonzero((aa.res_name[starts] == res_name) & (aa.res_id[starts] == res_id))
    assert len(hits) == 1, f"{selection} matched {len(hits)} tokens, expected exactly 1"
    return int(hits[0])


@pytest.mark.parametrize("name,closure,is_backbone_linkage", CYCLIC_CASES)
def test_closure_annotations(name: str, closure: tuple[str, str], is_backbone_linkage: bool):
    """A closure is ordered and suppressed iff it is a backbone linkage.

    For ``head_tail`` the closure runs ``THR183.C -> ASN179.N``, making ASN179 the
    downstream partner even though it sits at the lower token index.
    """
    aa = _load(name)
    feats = compute_token_bond_features(aa)
    cpo, bond_mask = feats["canonical_polymer_ordering"], feats["bond_mask"]
    upstream, downstream = (_token_of(aa, s) for s in closure)

    # Holds for every topology, including the crosslinks that carry no ordering.
    assert feats["token_bond_path_length"][upstream, downstream] == 1

    expected = 1 if is_backbone_linkage else 0
    assert cpo[downstream, upstream] == expected
    assert cpo[upstream, downstream] == -expected

    # Suppression is scoped to the backbone, so a crosslink stays visible to the model.
    assert bond_mask[upstream, downstream] == (not is_backbone_linkage)
    assert bond_mask[downstream, upstream] == bond_mask[upstream, downstream]


@pytest.mark.parametrize("name,closure,is_backbone_linkage", CYCLIC_CASES)
def test_annotation_consistency(name: str, closure: tuple[str, str], is_backbone_linkage: bool):
    """Cross-feature invariants that hold whatever the topology."""
    aa = _load(name)
    feats = compute_token_bond_features(aa)
    cpo, bond_mask = feats["canonical_polymer_ordering"], feats["bond_mask"]
    path_len = feats["token_bond_path_length"]

    assert np.array_equal(cpo, -cpo.T), "ordering must be antisymmetric"
    assert np.array_equal(bond_mask, bond_mask.T), "bond_mask must be symmetric"
    assert np.array_equal(path_len, path_len.T), "path lengths must be symmetric"
    assert (np.diagonal(cpo) == 0).all(), "no token is upstream of itself"
    assert set(np.unique(cpo)) <= {-1, 0, 1}, "ordering is ternary"

    ordered = cpo != 0
    assert (path_len[ordered] == 1).all(), "ordered pairs are directly bonded"
    assert not bond_mask[ordered].any(), "backbone linkages are suppressed in bond_mask"


def _masked_ring(n: int = 4) -> AtomArray:
    """A ring of ``n`` expandable-segment sentinels, so every residue is a stand-in.

    This is the shape an unconstrained cyclic design takes: chemistry cannot orient any
    bond, so the tie-break governs all of them.
    """
    aa = AtomArray(n)
    aa.coord = np.zeros((n, 3), dtype=np.float32)
    aa.chain_id = np.array(["A"] * n)
    aa.res_id = np.arange(1, n + 1)
    aa.res_name = np.array(["UNK"] * n)
    aa.atom_name = np.array(["CA"] * n)
    aa.element = np.array(["C"] * n)
    aa.set_annotation("token_type", np.zeros(n, dtype=np.int16))  # PROTEIN
    aa.set_annotation("atomize", np.zeros(n, dtype=bool))
    aa.set_annotation("insertion_source_idx", np.full(n, -1, dtype=np.int32))
    aa.set_annotation("molecule_iid", np.zeros(n, dtype=int))

    # Mark every residue an expandable-segment sentinel, so each is a stand-in (S_SEGMIN).
    S_SEGMIN.set_annotation(aa, np.ones(n, dtype=int))
    aa.bonds = BondList(n, np.array([(i, (i + 1) % n, BondType.SINGLE) for i in range(n)]))
    return aa


def test_masked_ring_closure_runs_high_to_low():
    """With no chemistry to read, ensure the ring-closing bond still points the physical way."""
    cpo = compute_token_bond_features(_masked_ring(4))["canonical_polymer_ordering"]

    assert np.array_equal(cpo, -cpo.T), "ordering must be antisymmetric"

    # Linear bonds i -> i+1: lower token upstream, so cpo[downstream, upstream] == +1.
    assert cpo[1, 0] == 1 and cpo[2, 1] == 1 and cpo[3, 2] == 1

    # Ring closure joins tokens 3 (high, C-side) and 0 (low, N-side); 3 is upstream, so cpo[0, 3] == +1.
    # Without the wrap-around flip the tie-break would call 0 upstream and this would be -1.
    assert cpo[0, 3] == 1
