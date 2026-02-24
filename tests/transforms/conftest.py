"""Fixtures for design task tests."""

import copy

import numpy as np
import pytest
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.utils.testing import cached_parse
from synthetic_example_generator import add_chain_nearby_without_clash

cif_parser_args = STANDARD_PARSER_ARGS | {
    "hydrogen_policy": "remove",
}

# ============================== Meta-fixtures ==============================


@pytest.fixture(scope="module")
def atom_array_fixture(request):
    """Meta-fixture that returns the requested fixture by name.

    This allows parametrizing tests with different fixtures using indirect=True.

    Example:
        @pytest.mark.parametrize("atom_array_fixture", ["simple_protein", "complex_protein"], indirect=True)
        def test_something(atom_array_fixture):
            # atom_array_fixture will be the actual fixture value
            pass
    """
    return request.getfixturevalue(request.param)


# ============================== Fixtures ==============================


@pytest.fixture(scope="module")
def simple_protein():
    """Simple single-chain protein structure.

    Returns:
      Data dict with atom_array containing a single protein chain (6lyz - lysozyme).
    """
    data = cached_parse("6lyz", **cif_parser_args)
    return data


@pytest.fixture(scope="module")
def protein_with_non_canonical_aa():
    """Protein structure with non-canonical amino acids.

    Returns:
      Data dict with atom_array containing protein with non-canonical amino acids.
    """
    # 1a8o is a structure with a ncaa
    data = cached_parse("1a8o", **cif_parser_args)
    return data


@pytest.fixture(scope="module")
def protein_with_ligand():
    """Protein structure with bound ligand."""
    data = cached_parse("1mbo", **cif_parser_args)
    data["atom_array"] = data["atom_array"][np.isin(data["atom_array"].chain_id, ["A", "C"])]
    return data


@pytest.fixture(scope="module")
def protein_protein_interface():
    """Two-chain protein interface for PPI tasks.

    Returns:
      Data dict with atom_array containing a protein-protein interface.
    """
    # 1mso is a protein complex with small chains
    data = cached_parse("1mso", **cif_parser_args)
    return data


@pytest.fixture(scope="module")
def simple_dna():
    """Simple single-chain DNA structure."""
    data = cached_parse("1bna", **cif_parser_args)
    return data


@pytest.fixture(scope="module")
def simple_rna():
    """Simple single-chain RNA structure."""
    data = cached_parse("1ehz", **cif_parser_args)
    data["atom_array"] = data["atom_array"][data["atom_array"].chain_id == "A"]  # remove all the random ions
    return data


@pytest.fixture(scope="module")
def simple_ligand(protein_with_ligand):
    """Simple single-chain ligand structure (HEME)."""
    data = protein_with_ligand.copy()
    data["atom_array"] = data["atom_array"][data["atom_array"].chain_id == "C"]
    # data = data[data.occupancy > 0]  # remove unoccupied atoms for testing
    return data


@pytest.fixture(scope="module")
def complex_protein(protein_with_non_canonical_aa, simple_dna, simple_rna, simple_ligand):
    """Complex protein structure with non-canonical amino acids, DNA, RNA, and ligand."""

    protein_with_non_canonical_aa_copy = copy.deepcopy(protein_with_non_canonical_aa)
    simple_dna_copy = copy.deepcopy(simple_dna)
    simple_rna_copy = copy.deepcopy(simple_rna)
    simple_ligand_copy = copy.deepcopy(simple_ligand)

    atom_array = protein_with_non_canonical_aa_copy["atom_array"]
    atom_array = add_chain_nearby_without_clash(
        atom_array, simple_dna_copy["atom_array"], direction=np.array([0.0, 0.0, 1.0])
    )
    atom_array = add_chain_nearby_without_clash(
        atom_array, simple_rna_copy["atom_array"], direction=np.array([0.0, 1.0, 0.0])
    )
    atom_array = add_chain_nearby_without_clash(
        atom_array, simple_ligand_copy["atom_array"], direction=np.array([1.0, 0.0, 0.0])
    )
    return {
        "atom_array": atom_array,
        "example_id": "complex_protein",
    }


# ============================== Antibody Framework Fixtures ==============================


@pytest.fixture(scope="module")
def simple_protein_ab(protein_protein_interface):
    """Same as simple_protein but with proper extra_info"""
    data = protein_protein_interface.copy()
    data["extra_info"] = {
        "all_ig_chains": str(
            {
                "A": {
                    "seqstart_index": "1",
                    "seqend_index": "70",
                    "vdj_cdr1_start_pos": "5",
                    "vdj_cdr1": "QGP",
                    "vdj_cdr2_start_pos": "10",
                    "vdj_cdr2": "PFR",
                    "vdj_cdr3_start_pos": "15",
                    "vdj_cdr3": "VDR",
                    "chain_type": "H",
                }
            }
        )
    }
    return data


@pytest.fixture(scope="module")
def complex_protein_ab(complex_protein):
    """Same as complex_protein but with proper extra_info"""
    data = complex_protein.copy()
    data["extra_info"] = {
        "all_ig_chains": str(
            {
                "A": {
                    "seqstart_index": "1",
                    "seqend_index": "70",
                    "vdj_cdr1_start_pos": "5",
                    "vdj_cdr1": "QGP",
                    "vdj_cdr2_start_pos": "10",
                    "vdj_cdr2": "PFR",
                    "vdj_cdr3_start_pos": "15",
                    "vdj_cdr3": "VDR",
                    "chain_type": "H",
                }
            }
        )
    }
    return data


# ============================== PPI Fixtures ==============================


@pytest.fixture(scope="module")
def protein_protein_interface_crop_info(protein_protein_interface):
    """Crop info for protein-protein interface."""
    data = protein_protein_interface.copy()
    data["crop_info"] = {
        "atom_array": data["atom_array"].copy(),
    }
    data["query_pn_unit_iids"] = np.unique(data["atom_array"].pn_unit_iid)[:2]
    return data


@pytest.fixture(scope="module")
def complex_protein_crop_info(complex_protein):
    """Crop info for complex protein."""
    data = complex_protein.copy()
    data["crop_info"] = {
        "atom_array": data["atom_array"].copy(),
    }
    data["query_pn_unit_iids"] = np.unique(data["atom_array"].pn_unit_iid)[:2]
    return data
