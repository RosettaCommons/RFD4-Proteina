import copy

import numpy as np
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.utils.testing import cached_parse
from synthetic_example_generator import add_chain_nearby_without_clash

from proteinfoundation.tests.transforms.design_task_test_utils import apply_design_task
from proteinfoundation.transforms.design_task import (
    UnconditionalTask,
)

cif_parser_args = STANDARD_PARSER_ARGS | {
    "hydrogen_policy": "remove",
}


def simple_protein():
    """Simple single-chain protein structure.

    Returns:
      Data dict with atom_array containing a single protein chain (6lyz - lysozyme).
    """
    data = cached_parse("6lyz", **cif_parser_args)
    return data


def protein_with_non_canonical_aa():
    """Protein structure with non-canonical amino acids.

    Returns:
      Data dict with atom_array containing protein with non-canonical amino acids.
    """
    # 1a8o is a structure with a ncaa
    data = cached_parse("1a8o", **cif_parser_args)
    return data


def protein_with_ligand():
    """Protein structure with bound ligand."""
    data = cached_parse("1mbo", **cif_parser_args)
    data["atom_array"] = data["atom_array"][np.isin(data["atom_array"].chain_id, ["A", "C"])]
    return data


def protein_protein_interface():
    """Two-chain protein interface for PPI tasks.

    Returns:
      Data dict with atom_array containing a protein-protein interface.
    """
    # 1mso is a protein complex with small chains
    data = cached_parse("1mso", **cif_parser_args)
    return data


def simple_dna():
    """Simple single-chain DNA structure."""
    data = cached_parse("1bna", **cif_parser_args)
    return data


def simple_rna():
    """Simple single-chain RNA structure."""
    data = cached_parse("1ehz", **cif_parser_args)
    data["atom_array"] = data["atom_array"][data["atom_array"].chain_id == "A"]  # remove all the random ions
    return data


_PROTEIN_WITH_LIGAND_DATA = protein_with_ligand()


def simple_ligand() -> dict:
    """Simple single-chain ligand structure (HEME)."""
    data = _PROTEIN_WITH_LIGAND_DATA.copy()
    data["atom_array"] = data["atom_array"][data["atom_array"].chain_id == "C"]
    # data = data[data.occupancy > 0]  # remove unoccupied atoms for testing
    return data


def complex_protein() -> dict:
    """Complex protein structure with non-canonical amino acids, DNA, RNA, and ligand."""

    protein_with_non_canonical_aa_copy = copy.deepcopy(protein_with_non_canonical_aa())
    simple_dna_copy = copy.deepcopy(simple_dna())
    simple_rna_copy = copy.deepcopy(simple_rna())
    simple_ligand_copy = copy.deepcopy(simple_ligand())

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


_COMPLEX_PROTEIN_DATA = complex_protein()


def complex_protein_crop_info() -> dict:
    """Crop info for complex protein."""
    data = _COMPLEX_PROTEIN_DATA.copy()
    data["crop_info"] = {
        "atom_array": data["atom_array"].copy(),
    }
    data["query_pn_unit_iids"] = np.unique(data["atom_array"].pn_unit_iid)[:2]
    return data


def complex_protein_ab() -> dict:
    """Same as complex_protein but with proper extra_info"""
    data = _COMPLEX_PROTEIN_DATA.copy()
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


if __name__ == "__main__":
    # Parse the structure
    n_repeats = 1
    task = UnconditionalTask(
        rasa_kwargs={"p_add_rasa_condition": 0.0},
        hbond_kwargs={"p_add_hbond_condition": 0.0},
        secondary_structure_kwargs={"p_add_secondary_structure_condition": 1.0},
    )

    complex_protein_data = complex_protein()
    complex_protein_data_copies = [copy.deepcopy(complex_protein_data) for _ in range(n_repeats)]

    for complex_protein_data_copy in complex_protein_data_copies:
        data = apply_design_task(complex_protein_data_copy, task)
