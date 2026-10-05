"""Scoring-copy compatibility with AtomWorks component labels and aliases."""

import numpy as np
import pytest
from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from atomworks.io.utils.ccd import get_chem_comp_type
from biotite.structure import BondList

pytest.importorskip("tmol")

from tmol.io import canonical_ordering_for_biotite  # noqa: E402
from tmol.ligand import detect_nonstandard_residues  # noqa: E402

from rfproteina.conditions import C_HOT  # noqa: E402
from rfproteina.metrics.physics.tmol_ddg_core import prepare_tmol_atom_array  # noqa: E402


def metadata_fixture():
    array = AtomArrayPlus(9)
    array.res_name = np.array(["ARG", "ARG", "PRO", "PRO", "MSE", "SAM", "ARG", "GLY", "ZZZ"])
    array.res_id = np.arange(1, 10)
    array.chain_id[:] = "A"
    array.coord = np.arange(27, dtype=np.float32).reshape(9, 3)
    array.bonds = BondList(9, np.array([[0, 1, 1]], dtype=np.uint32))
    array.set_annotation(
        "chem_comp_type",
        np.array(
            [
                "PEPTIDE LINKING",
                "L-PEPTIDE LINKING",
                "L-PEPTIDE LINKING",
                "PEPTIDE LINKING",
                "PEPTIDE LINKING",
                "NON-POLYMER",
                "D-PEPTIDE LINKING",
                "PEPTIDE LINKING",
                "PEPTIDE LINKING",
            ]
        ),
    )
    return array


def assert_unchanged(actual, expected):
    assert actual.get_annotation_categories() == expected.get_annotation_categories()
    np.testing.assert_array_equal(actual.coord, expected.coord)
    np.testing.assert_array_equal(actual.bonds.as_array(), expected.bonds.as_array())
    for name in actual.get_annotation_categories():
        np.testing.assert_array_equal(actual.get_annotation(name), expected.get_annotation(name))


@pytest.mark.parametrize("also_canonical", [False, True])
def test_aliases_are_canonicalized_only_on_scoring_copy(also_canonical):
    atoms = metadata_fixture()
    values = np.arange(len(atoms)) % 2 == 0
    atoms.set_annotation("mask_hotspot_1_atom", values)
    if also_canonical:
        C_HOT.set_annotation(atoms, values)
    before = atoms.copy()
    prepared = prepare_tmol_atom_array(atoms)
    assert prepared is not atoms
    assert "mask_hotspot_1_atom" not in prepared.get_annotation_categories()
    np.testing.assert_array_equal(C_HOT.annotation(prepared, default="raise"), values)
    assert_unchanged(atoms, before)
    np.testing.assert_array_equal(prepared.coord, atoms.coord)
    np.testing.assert_array_equal(prepared.bonds.as_array(), atoms.bonds.as_array())


def test_only_generic_standard_amino_acid_types_are_refined():
    atoms = metadata_fixture()
    before = atoms.copy()
    prepared = prepare_tmol_atom_array(atoms)
    assert list(prepared.chem_comp_type[:4]) == [
        get_chem_comp_type("ARG", mode="raise"),
        "L-PEPTIDE LINKING",
        "L-PEPTIDE LINKING",
        get_chem_comp_type("PRO", mode="raise"),
    ]
    assert prepared.chem_comp_type[4] == "PEPTIDE LINKING"  # Modified residue retains its declaration.
    assert prepared.chem_comp_type[5] == "NON-POLYMER"
    assert prepared.chem_comp_type[6] == "D-PEPTIDE LINKING"  # Explicit stereochemistry is not overwritten.
    assert prepared.chem_comp_type[7] == get_chem_comp_type("GLY", mode="raise")
    assert prepared.chem_comp_type[8] == "PEPTIDE LINKING"
    assert_unchanged(atoms, before)


def test_conflicting_alias_values_are_rejected():
    atoms = metadata_fixture()
    atoms.set_annotation("mask_hotspot_1_atom", np.zeros(len(atoms), dtype=bool))
    atoms.set_annotation(C_HOT.full_name, np.ones(len(atoms), dtype=bool))
    before = atoms.copy()
    with pytest.raises(ValueError, match="Conflicting fields"):
        prepare_tmol_atom_array(atoms)
    assert_unchanged(atoms, before)


def test_stereochemical_conflicts_still_reach_native_validation():
    atoms = metadata_fixture()
    prepared = prepare_tmol_atom_array(atoms)
    with pytest.raises(ValueError, match="Conflicting chem_comp_type annotations for ARG"):
        detect_nonstandard_residues(prepared, canonical_ordering_for_biotite())
