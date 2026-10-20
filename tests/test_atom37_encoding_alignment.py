"""Test UNIFIED_ATOM37_ENCODING structure and collate.py handling.

This test ensures that UNIFIED_ATOM37_ENCODING is correctly structured and that
collate.py derives its atom→slot mappings correctly from the single source of truth.


"""

import numpy as np
import pytest
from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.enums import ChainType
from atomworks.ml.transforms.encoding import atom_array_from_encoding, atom_array_to_encoding
from biotite.structure import AtomArray

from rfproteina.encoding import UNIFIED_ATOM37_ENCODING
from rfproteina.utils.token_encoding import build_atom37_slots


@pytest.mark.parametrize("res_name", ["TRP", "NAG", "SEP", "DA", "A"])
@pytest.mark.parametrize("chain_type", [ChainType.NON_POLYMER, ChainType.POLYPEPTIDE_L])
def test_vocabulary_slots_ignore_chain_classification(res_name, chain_type):
    """Verify vocabulary slots ignore chain classification while explicit atomization uses slot 1."""
    layout = UNIFIED_ATOM37_ENCODING.token_atoms[res_name]
    occupied = np.flatnonzero(layout != "")
    aa = AtomArray(len(occupied))
    aa.res_name[:] = res_name
    aa.atom_name[:] = layout[occupied]
    aa.set_annotation("chain_type", np.full(len(aa), chain_type, dtype=int))
    aa.set_annotation("atomize", np.zeros(len(aa), dtype=bool))
    np.testing.assert_array_equal(build_atom37_slots(aa), occupied)
    aa.atomize[:] = True
    np.testing.assert_array_equal(build_atom37_slots(aa), np.ones(len(aa), dtype=int))


def test_unified_atom37_slot_1_is_center_atom():
    """Verify that slot 1 contains the token center atom for all residue types.

    For proteins: CA
    For nucleic acids: C1'
    For atomized: first atom goes to slot 1 (tested in collate.py)
    """

    # Check proteins - CA should be at slot 1
    for aa in STANDARD_AA:
        if aa != "GLY":  # All except GLY have CA at slot 1
            atom37_layout = UNIFIED_ATOM37_ENCODING.token_atoms[aa]
            assert atom37_layout[1].strip() == "CA", f"{aa}: expected CA at slot 1, got '{atom37_layout[1]}'"

    # GLY has CA at slot 1 as well
    gly_layout = UNIFIED_ATOM37_ENCODING.token_atoms["GLY"]
    assert gly_layout[1].strip() == "CA", "GLY: expected CA at slot 1"

    # Check nucleic acids - C1' should be at slot 1
    for na in list(STANDARD_RNA) + list(STANDARD_DNA) + ["N", "DN"]:
        atom37_layout = UNIFIED_ATOM37_ENCODING.token_atoms[na]
        assert atom37_layout[1].strip() == "C1'", f"{na}: expected C1' at slot 1, got '{atom37_layout[1]}'"


def test_roundtrip_protein_structure():
    """Test encoding → decoding round-trip for a simple protein structure."""
    # Create simple protein: ALA-GLY-CYS
    atom_array = AtomArray(length=10)
    atom_array.coord = np.random.randn(10, 3).astype(np.float32)
    atom_array.set_annotation(
        "res_name", np.array(["ALA", "ALA", "ALA", "GLY", "GLY", "GLY", "CYS", "CYS", "CYS", "CYS"])
    )
    atom_array.set_annotation("atom_name", np.array(["N", "CA", "C", "N", "CA", "C", "N", "CA", "C", "SG"]))
    atom_array.set_annotation("tok_idx", np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]))
    atom_array.set_annotation("chain_id", np.array(["A"] * 10))
    atom_array.set_annotation("res_id", np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3]))
    atom_array.set_annotation("occupancy", np.ones(10, dtype=np.float32))
    atom_array.set_annotation("b_factor", np.ones(10, dtype=np.float32) * 30.0)
    atom_array.set_annotation("element", np.array(["N", "C", "C", "N", "C", "C", "N", "C", "C", "S"]))

    # Encode
    encoding = atom_array_to_encoding(atom_array, UNIFIED_ATOM37_ENCODING)

    # Verify encoding structure
    assert "xyz" in encoding
    assert "mask" in encoding
    assert "seq" in encoding
    assert encoding["xyz"].shape[0] == 3  # 3 tokens
    assert encoding["xyz"].shape[1] == 37  # 37 atom slots
    assert encoding["xyz"].shape[2] == 3  # xyz coordinates

    # Verify sequence encoding
    expected_seq_indices = [
        UNIFIED_ATOM37_ENCODING.token_to_idx["ALA"],
        UNIFIED_ATOM37_ENCODING.token_to_idx["GLY"],
        UNIFIED_ATOM37_ENCODING.token_to_idx["CYS"],
    ]
    np.testing.assert_array_equal(encoding["seq"], expected_seq_indices)

    # Decode back
    decoded_atom_array = atom_array_from_encoding(
        encoding["xyz"], encoding["seq"], UNIFIED_ATOM37_ENCODING, encoded_mask=encoding["mask"]
    )

    # Verify key properties are present and valid
    assert len(decoded_atom_array) > 0, "Decoded atom array should not be empty"
    assert hasattr(decoded_atom_array, "res_name"), "Should have res_name annotation"
    assert hasattr(decoded_atom_array, "atom_name"), "Should have atom_name annotation"
    assert hasattr(decoded_atom_array, "coord"), "Should have coordinates"

    # Verify residue types match
    decoded_res_names_unique = np.unique(decoded_atom_array.res_name)
    original_res_names_unique = np.array(["ALA", "CYS", "GLY"])
    np.testing.assert_array_equal(decoded_res_names_unique, original_res_names_unique)

    # Verify atom names exist for key backbone atoms (should be present in decoded structure)
    decoded_atom_names_set = set(decoded_atom_array.atom_name)
    assert "CA" in decoded_atom_names_set, "CA atoms should be present"
    assert "N" in decoded_atom_names_set, "N atoms should be present"
    assert "C" in decoded_atom_names_set, "C atoms should be present"


def test_roundtrip_nucleic_acid_structure():
    """Test encoding → decoding round-trip for RNA/DNA structure."""
    # Create simple RNA: A-C-G
    atom_array = AtomArray(length=9)
    atom_array.coord = np.random.randn(9, 3).astype(np.float32)
    atom_array.set_annotation("res_name", np.array(["A", "A", "A", "C", "C", "C", "G", "G", "G"]))
    atom_array.set_annotation("atom_name", np.array(["C1'", "C2'", "P", "C1'", "C2'", "P", "C1'", "C2'", "P"]))
    atom_array.set_annotation("tok_idx", np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]))
    atom_array.set_annotation("chain_id", np.array(["R"] * 9))
    atom_array.set_annotation("res_id", np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))
    atom_array.set_annotation("occupancy", np.ones(9, dtype=np.float32))
    atom_array.set_annotation("b_factor", np.ones(9, dtype=np.float32) * 30.0)
    atom_array.set_annotation("element", np.array(["C", "C", "P", "C", "C", "P", "C", "C", "P"]))

    # Encode
    encoding = atom_array_to_encoding(atom_array, UNIFIED_ATOM37_ENCODING)

    # Verify encoding structure
    assert encoding["xyz"].shape[0] == 3  # 3 tokens
    assert encoding["xyz"].shape[1] == 37  # 37 atom slots

    # Verify sequence encoding for RNA
    expected_seq_indices = [
        UNIFIED_ATOM37_ENCODING.token_to_idx["A"],  # RNA adenine
        UNIFIED_ATOM37_ENCODING.token_to_idx["C"],  # RNA cytosine
        UNIFIED_ATOM37_ENCODING.token_to_idx["G"],  # RNA guanine
    ]
    np.testing.assert_array_equal(encoding["seq"], expected_seq_indices)

    # Decode back
    decoded_atom_array = atom_array_from_encoding(
        encoding["xyz"], encoding["seq"], UNIFIED_ATOM37_ENCODING, encoded_mask=encoding["mask"]
    )

    # Verify key properties are present and valid
    assert len(decoded_atom_array) > 0, "Decoded atom array should not be empty"
    assert hasattr(decoded_atom_array, "res_name"), "Should have res_name annotation"
    assert hasattr(decoded_atom_array, "atom_name"), "Should have atom_name annotation"
    assert hasattr(decoded_atom_array, "coord"), "Should have coordinates"

    # Verify residue types match (RNA nucleotides)
    decoded_res_names_unique = np.unique(decoded_atom_array.res_name)
    original_res_names_unique = np.array(["A", "C", "G"])
    np.testing.assert_array_equal(decoded_res_names_unique, original_res_names_unique)

    # Verify atom names exist for key RNA atoms
    decoded_atom_names_set = set(decoded_atom_array.atom_name)
    assert "C1'" in decoded_atom_names_set, "C1' atoms should be present in RNA"
    assert "P" in decoded_atom_names_set, "P atoms should be present in RNA"


def test_roundtrip_mixed_protein_na_structure():
    """Test encoding → decoding round-trip for mixed protein + nucleic acid."""
    # Create mixed structure: ALA (protein) + DA (DNA) + GLY (protein)
    atom_array = AtomArray(length=9)
    atom_array.coord = np.random.randn(9, 3).astype(np.float32)
    atom_array.set_annotation("res_name", np.array(["ALA", "ALA", "ALA", "DA", "DA", "DA", "GLY", "GLY", "GLY"]))
    atom_array.set_annotation("atom_name", np.array(["N", "CA", "C", "C1'", "C2'", "P", "N", "CA", "C"]))
    atom_array.set_annotation("tok_idx", np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]))
    atom_array.set_annotation("chain_id", np.array(["A", "A", "A", "D", "D", "D", "A", "A", "A"]))
    atom_array.set_annotation("res_id", np.array([1, 1, 1, 1, 1, 1, 2, 2, 2]))
    atom_array.set_annotation("occupancy", np.ones(9, dtype=np.float32))
    atom_array.set_annotation("b_factor", np.ones(9, dtype=np.float32) * 30.0)
    atom_array.set_annotation("element", np.array(["N", "C", "C", "C", "C", "P", "N", "C", "C"]))

    # Encode
    encoding = atom_array_to_encoding(atom_array, UNIFIED_ATOM37_ENCODING)

    # Verify encoding structure
    assert encoding["xyz"].shape[0] == 3  # 3 tokens

    # Verify sequence encoding captures both protein and DNA
    expected_seq_indices = [
        UNIFIED_ATOM37_ENCODING.token_to_idx["ALA"],  # Protein
        UNIFIED_ATOM37_ENCODING.token_to_idx["DA"],  # DNA adenine
        UNIFIED_ATOM37_ENCODING.token_to_idx["GLY"],  # Protein
    ]
    np.testing.assert_array_equal(encoding["seq"], expected_seq_indices)

    # Decode back
    decoded_atom_array = atom_array_from_encoding(
        encoding["xyz"], encoding["seq"], UNIFIED_ATOM37_ENCODING, encoded_mask=encoding["mask"]
    )

    # Verify key properties are present and valid across molecule types
    assert len(decoded_atom_array) > 0, "Decoded atom array should not be empty"
    assert hasattr(decoded_atom_array, "res_name"), "Should have res_name annotation"
    assert hasattr(decoded_atom_array, "atom_name"), "Should have atom_name annotation"
    assert hasattr(decoded_atom_array, "coord"), "Should have coordinates"

    # Verify residue types match (protein + DNA)
    decoded_res_names_unique = np.unique(decoded_atom_array.res_name)
    original_res_names_unique = np.array(["ALA", "DA", "GLY"])
    np.testing.assert_array_equal(decoded_res_names_unique, original_res_names_unique)

    # Verify both protein and DNA atoms present
    decoded_atom_names_set = set(decoded_atom_array.atom_name)
    assert "CA" in decoded_atom_names_set, "Protein CA atoms should be present"
    assert "C1'" in decoded_atom_names_set, "DNA C1' atoms should be present"
