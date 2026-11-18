"""``rfproteina.af3.af3_input``: the AF3 oracle's input construction, as functions of an AtomArray."""

import io
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from atomworks.io.utils.io_utils import CIFWriteConfig, to_cif_string
from biotite.structure.io import pdbx
from biotite.structure.io.pdbx import get_structure

from rfproteina.af3.af3_input import (
    AF3Input,
    build_af3_input,
    build_user_ccd_string,
    disambiguate_chain_ids,
    inject_empty_chem_comp_bond,
    renumber_res_ids_per_chain,
)
from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def design():
    """A real design with a covalent ligand bond and a ligand outside the CCD."""
    return get_structure(
        pdbx.CIFFile.read(TEST_DATA_DIR / "lencapavir_design_with_covale.cif"), model=1, include_bonds=True
    )


def _blocks(text: str) -> list[str]:
    return [b for b in re.split(r"(?m)^(?=data_)", text) if b.strip()]


def test_build_af3_input_returns_the_cif_and_a_user_ccd_for_non_standard_components(design):
    af3_input = build_af3_input(design, "lenacapavir")
    assert isinstance(af3_input, AF3Input)
    assert af3_input.cif.startswith("data_lenacapavir")
    assert af3_input.template is None
    assert af3_input.user_ccd is not None
    names = {b.split("\n", 1)[0].removeprefix("data_") for b in _blocks(af3_input.user_ccd)}
    standard = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
    assert names and not names & standard, "a block for every non-standard component, and no standard one"
    # every block has the bond table AF3's validator requires, even a one-atom component's
    assert all("_chem_comp_bond.atom_id_1" in b for b in _blocks(af3_input.user_ccd))


def test_atoms_without_coordinates_are_left_out_of_the_cif(design):
    broken = design.copy()
    broken.coord[0] = np.nan
    af3_input = build_af3_input(broken, "x")
    assert af3_input.atom_array.array_length() == broken.array_length() - 1
    assert af3_input.disambiguated_atom_array.array_length() == broken.array_length()


def test_residue_numbering_is_kept_or_renumbered_per_chain(design):
    gapped = design.copy()
    gapped.res_id = gapped.res_id + 100
    kept = build_af3_input(gapped, "x").atom_array
    assert kept.res_id.min() == gapped.res_id.min()
    renumbered = build_af3_input(gapped, "x", preserve_res_id=False).atom_array
    assert renumbered.res_id.min() == 1


def test_unl_is_refused(design):
    unl = design.copy()
    unl.res_name[unl.hetero] = "UNL"
    with pytest.raises(ValueError, match="UNL"):
        build_af3_input(unl, "x")


def test_the_oracle_builds_its_manifest_with_it(design):
    """The manifest entry is the same construction: nothing to drift."""
    with tempfile.TemporaryDirectory() as tmp:
        job = OracleJobInput(atom_array=design, example_id="d", out_dir=Path(tmp))
        entries, _ = AF3Oracle(preserve_res_id=True)._build_manifest([job], Path(tmp))
    mine = build_af3_input(design, entries[0]["name"])
    assert entries[0]["cif"].split("\n", 1)[1] == mine.cif.split("\n", 1)[1]
    assert entries[0].get("user_ccd") == mine.user_ccd


def test_the_previous_private_names_still_work(design):
    aa, _ = AF3Oracle._disambiguate_chain_ids(design)
    assert aa.array_length() == design.array_length()
    assert AF3Oracle._renumber_res_ids_per_chain(design).res_id.min() == 1
    assert AF3Oracle()._build_user_ccd_string(design) == build_user_ccd_string(design)


def test_helpers_are_importable_on_their_own():
    assert "_chem_comp_bond.atom_id_1" in inject_empty_chem_comp_bond("data_X\n_entry.id X\n#\n")
    assert disambiguate_chain_ids is not None and renumber_res_ids_per_chain is not None
    assert to_cif_string is not None and CIFWriteConfig is not None and io is not None and sys is not None
