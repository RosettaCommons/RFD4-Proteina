"""Regression tests for unindexed motif conditioning across sequence redesign."""

import numpy as np
import pytest
from atomworks.io.utils.atom_array_plus import as_atom_array_plus, concatenate_any
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from atomworks.ml.conditions.base import ConditionConflictError
from biotite.structure import AtomArray

from rfproteina.conditions import (
    C_CRDGRP,
    C_IDXGRP,
    C_INTCRD,
    C_RAS,
    C_SEC,
    C_SEQ_PROB_CYS,
    RasaEnum,
    SSEnum,
)
from rfproteina.transforms.unindexing import annotate_unindexed_insertion
from rfproteina.utils.standard_annotations import SA_CRDSRC, SA_CWI, SA_IDXGRPSRC, SA_SRC, SA_UII

_BACKBONE = ("N", "CA", "C", "O")
_RESIDUE_ATOMS = {
    "GLY": _BACKBONE,
    "ALA": (*_BACKBONE, "CB"),
    "LEU": (*_BACKBONE, "CB", "CG", "CD1", "CD2"),
    "TRP": (*_BACKBONE, "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}


def _make_redesigned_motif_pair(target_residue: str, coordinate_mode: str = "absolute") -> AtomArray:
    """Build a generated residue followed by a backbone-conditioned, sequence-free LEU motif."""
    target_names = _RESIDUE_ATOMS[target_residue]
    source_names = _RESIDUE_ATOMS["LEU"]
    names = (*target_names, *source_names)
    aa = as_atom_array_plus(AtomArray(len(names)))
    aa.atom_name = np.array(names)
    aa.element = np.array([name[0] for name in names])
    aa.res_name = np.array([target_residue] * len(target_names) + ["LEU"] * len(source_names))
    aa.res_id = np.array([1] * len(target_names) + [2] * len(source_names))
    aa.chain_id[:] = "A"
    aa.set_annotation("token_id", aa.res_id - 1)
    aa.set_annotation("molecule_iid", np.zeros(len(aa), dtype=int))
    aa.set_annotation("is_polymer", np.ones(len(aa), dtype=bool))
    aa.set_annotation("atomize", np.zeros(len(aa), dtype=bool))

    source_coords = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 1], [3, 0, 1], [4, 0, 1], [3, 1, 1]],
        dtype=np.float32,
    )
    # Shared backbone geometry is translated in the generated residue. Extra sidechain geometry
    # is arbitrary: it must never become a coordinate target merely because the token is mapped.
    target_coords = np.array(
        [source_coords[source_names.index(name)] if name in source_names else [5, 1, 2] for name in target_names],
        dtype=np.float32,
    )
    aa.coord = np.concatenate([target_coords + 10, source_coords])
    source = aa.token_id == 1
    backbone = source & np.isin(aa.atom_name, _BACKBONE)
    C_IDX.set_annotation(aa, ~source)
    C_IDXGRP.set_annotation(aa, np.where(source, 7, 0))
    C_SEQ.set_annotation(aa, C_SEQ.default_annotation(aa))

    absolute = C_CRD.default_annotation(aa)
    if coordinate_mode == "absolute":
        absolute[backbone] = aa.coord[backbone]
    C_CRD.set_annotation(aa, absolute)
    groups = C_CRDGRP.default_annotation(aa)
    if coordinate_mode == "internal":
        groups[backbone] = 3
    C_CRDGRP.set_annotation(aa, groups)
    internal = C_INTCRD.default_annotation(aa)
    if coordinate_mode == "internal":
        internal[backbone] = aa.coord[backbone]
    C_INTCRD.set_annotation(aa, internal)
    SA_SRC.set_annotation(aa, np.where(source, "motif", "generated"))
    SA_CWI.set_annotation(aa, SA_CWI.default_annotation(aa))
    return aa


@pytest.mark.parametrize("target_residue", ["ALA", "GLY", "TRP"])
@pytest.mark.parametrize("coordinate_mode", ["absolute", "internal"])
def test_backbone_only_unindexed_motif_allows_sequence_redesign(target_residue, coordinate_mode):
    """Different sidechain inventories must not discard frozen backbone targets or fix sequence."""
    aa = _make_redesigned_motif_pair(target_residue, coordinate_mode)
    source = aa.token_id == 1
    source_backbone = source & np.isin(aa.atom_name, _BACKBONE)
    coordinate_condition = C_CRD if coordinate_mode == "absolute" else C_INTCRD
    frozen = coordinate_condition.annotation(aa)[source_backbone].copy()
    generated_coords = aa.coord[~source].copy()
    # Only stored condition targets are authoritative; live motif geometry may have changed.
    aa.coord[source] += 50

    out = annotate_unindexed_insertion(aa, {1: 0})

    backbone = np.isin(out.atom_name, _BACKBONE)
    assert len(out) == len(_RESIDUE_ATOMS[target_residue])
    assert SA_UII.mask(out).all()
    assert C_IDX.mask(out).all()
    assert (out.res_name == target_residue).all()
    assert not C_SEQ.mask(out).any()
    assert (SA_IDXGRPSRC.annotation(out) == 7).all()
    np.testing.assert_array_equal(SA_CRDSRC.mask(out), backbone)
    assert (SA_CRDSRC.annotation(out)[backbone] == "motif").all()
    np.testing.assert_array_equal(coordinate_condition.mask(out), backbone)
    np.testing.assert_array_equal(coordinate_condition.annotation(out)[backbone], frozen)
    expected_group = C_CRDGRP.KNOWN_ABSOLUTE_COORDS if coordinate_mode == "absolute" else 3
    assert (C_CRDGRP.annotation(out)[backbone] == expected_group).all()
    assert (C_CRDGRP.annotation(out)[~backbone] == C_CRDGRP.UNKNOWN_COORDS).all()
    assert not (C_INTCRD if coordinate_mode == "absolute" else C_CRD).mask(out).any()
    expected_reference = frozen if coordinate_mode == "absolute" else generated_coords[backbone]
    np.testing.assert_allclose(SA_CWI.annotation(out)[backbone], expected_reference, atol=1e-6)
    assert np.isnan(SA_CWI.annotation(out)[~backbone]).all()
    np.testing.assert_array_equal(out.coord, generated_coords)


@pytest.mark.parametrize("condition, value", [(C_SEC, SSEnum.HELIX), (C_SEQ_PROB_CYS, 0.0)])
def test_unindexed_residue_conditions_cover_redesigned_sidechains(condition, value):
    """Residue conditions cover added sidechain atoms, including an explicit zero probability."""
    aa = _make_redesigned_motif_pair("TRP")
    values = condition.default_annotation(aa)
    values[aa.token_id == 1] = value
    condition.set_annotation(aa, values)

    out = annotate_unindexed_insertion(aa, {1: 0})

    assert SA_UII.mask(out).all()
    assert condition.mask(out).all()
    assert (condition.annotation(out) == value).all()
    assert not C_SEQ.mask(out).any()


def test_unindexed_atom_condition_does_not_fix_sidechain_coordinates():
    """A sidechain exposure request transfers without adding coordinate conditioning."""
    aa = _make_redesigned_motif_pair("ALA")
    rasa = C_RAS.default_annotation(aa)
    rasa[(aa.token_id == 1) & (aa.atom_name == "CB")] = RasaEnum.EXPOSED
    C_RAS.set_annotation(aa, rasa)

    out = annotate_unindexed_insertion(aa, {1: 0})

    assert SA_UII.mask(out).all()
    np.testing.assert_array_equal(C_RAS.mask(out), out.atom_name == "CB")
    assert (C_RAS.annotation(out)[out.atom_name == "CB"] == RasaEnum.EXPOSED).all()
    assert not C_CRD.mask(out)[out.atom_name == "CB"].any()
    assert not C_SEQ.mask(out).any()


def test_unmatched_conditioned_sidechain_discards_all_motif_transfers():
    """An unmatched required atom rolls back earlier transfers as well as the failing motif."""
    first = _make_redesigned_motif_pair("ALA")
    second = _make_redesigned_motif_pair("GLY")
    C_RAS.set_annotation(first, C_RAS.default_annotation(first))
    rasa = C_RAS.default_annotation(second)
    rasa[(second.token_id == 1) & (second.atom_name == "CD1")] = RasaEnum.EXPOSED
    C_RAS.set_annotation(second, rasa)
    second.token_id += 2
    second.res_id += 2
    aa = concatenate_any([first, second])

    out = annotate_unindexed_insertion(aa, {1: 0, 3: 2})

    assert not SA_UII.mask(out).any()
    assert not SA_CRDSRC.mask(out).any()
    assert not C_CRD.mask(out).any()
    assert not C_RAS.mask(out).any()
    assert not C_SEQ.mask(out).any()
    np.testing.assert_array_equal(out.coord, aa.coord[C_IDX.mask(aa)])
    assert not C_CRD.mask(aa)[aa.token_id == 0].any()


def test_unindexed_coordinate_conflict_still_raises():
    """Sequence redesign must not suppress contradictory coordinate targets on a mapped atom."""
    aa = _make_redesigned_motif_pair("ALA")
    absolute = C_CRD.annotation(aa).copy()
    absolute[(aa.token_id == 0) & (aa.atom_name == "N")] = [100, 100, 100]
    C_CRD.set_annotation(aa, absolute)
    C_CRDGRP.set_annotation(aa, C_CRDGRP.default_annotation(aa))

    with pytest.raises(ConditionConflictError):
        annotate_unindexed_insertion(aa, {1: 0})


def test_sequence_only_unindexed_motif_does_not_require_atom_partners():
    """A sequence target transfers across unequal atom inventories without changing live identity."""
    aa = _make_redesigned_motif_pair("GLY", coordinate_mode="none")
    sequence = C_SEQ.annotation(aa).copy()
    sequence[aa.token_id == 1] = "LEU"
    C_SEQ.set_annotation(aa, sequence)

    out = annotate_unindexed_insertion(aa, {1: 0})

    assert SA_UII.mask(out).all()
    assert (C_SEQ.annotation(out) == "LEU").all()
    assert (out.res_name == "GLY").all()
    assert not C_CRD.mask(out).any()
    assert not C_INTCRD.mask(out).any()


@pytest.mark.parametrize("coordinate_mode", ["absolute", "internal"])
def test_indexed_reuse_preserves_redesigned_motif_conditions(coordinate_mode):
    """A subsequent indexed insertion pass retains the transferred frozen condition targets."""
    aa = _make_redesigned_motif_pair("ALA", coordinate_mode)
    mapped = annotate_unindexed_insertion(aa, {1: 0})
    assert SA_UII.mask(mapped).all()

    reused = annotate_unindexed_insertion(mapped, {})

    for condition in (C_CRD, C_INTCRD, C_CRDGRP, C_SEQ):
        np.testing.assert_array_equal(condition.annotation(reused), condition.annotation(mapped))
