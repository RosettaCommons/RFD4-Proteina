"""Nucleic-acid base-pair detection from AtomWorks protonation and interaction typing."""

import biotite.structure as struc
import numpy as np
import pytest
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.utils.annotator import ensure_annotations

from rfproteina.conditions import C_BP, BasePairEnum
from rfproteina.transforms.utils.na_geom_utils import annotate_na_ss, calculate_hb_counts
from tests.conftest import TEST_DATA_DIR


@pytest.mark.parametrize(
    "pdb_id,expected",
    [("1bna", {(0, 3): 3, (1, 2): 2}), ("1ehz", {(0, 1): 3})],
)
def test_base_pairs_from_real_dna_and_rna(pdb_id, expected):
    """Recover known base pairs from DNA (1BNA) and RNA (1EHZ) structure fragments.

    Check three base-to-base hydrogen bonds for G-C pairs and two for the DNA
    A-T pair, with symmetric counts even when chain instance IDs are long.
    Counting must preserve input coordinates, bonds, and annotation categories;
    secondary-structure annotation must identify exactly these pairs in both directions.
    """
    atoms = parse(
        TEST_DATA_DIR / "nucleic_hbonds" / f"{pdb_id}.cif.zst",
        config=ParseConfig.from_preset("minimal", add_id_and_entity_annotations=True),
    )["asym_unit"][0]
    atoms = as_atom_array_plus(atoms)
    atoms.set_annotation("chain_iid", np.char.add("long_assembly_chain_", atoms.chain_id))
    ensure_annotations(atoms, "token_id", "is_polymer")
    original = atoms.copy()
    starts = struc.get_residue_starts(atoms)
    counts = calculate_hb_counts(atoms, {"token_starts": starts})
    assert counts.dtype == np.int32
    np.testing.assert_array_equal(counts, counts.transpose(1, 0, 2))
    for (i, j), n_hbonds in expected.items():
        assert counts[i, j, 2] == n_hbonds
    np.testing.assert_array_equal(atoms.coord, original.coord)
    np.testing.assert_array_equal(atoms.bonds.as_array(), original.bonds.as_array())
    assert atoms.get_annotation_categories() == original.get_annotation_categories()

    annotated = annotate_na_ss(atoms, NA_only=True)
    pairs = C_BP.annotation(annotated, default="raise")
    token_pairs = np.searchsorted(starts, pairs.pairs)
    assert {tuple(p) for p in token_pairs} == {p for i, j in expected for p in ((i, j), (j, i))}
    assert np.all(pairs.values == BasePairEnum.PAIR.value)
