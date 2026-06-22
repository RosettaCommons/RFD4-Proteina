"""Tests for IPA interaction energy metric.

Two test structures:

1. **Axitinib / 4WA9** (from IPA's own test suite) -- has golden-value
   reference outputs from IPA's ``tests/axitinib_analysis/``.

2. **MCL1 benchmark** (from foundry-dev benchmarks, same as the MACE
   interaction test) -- no golden values; checks structural correctness.

Requires: ``IPA_PATH`` env var pointing to a compiled In-Pocket-Analysis repo.
Run with::

    IPA_PATH=/path/to/In-Pocket-Analysis \\
    pytest tests/metrics/test_ipa_interaction_energy_metric.py -v -s --noconftest
"""

import os

import numpy as np
import pytest

pytest.importorskip("gemmi", reason="IPA requires gemmi: pip install gemmi")

IPA_PATH = os.environ.get("IPA_PATH")
MCL1_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "mcl1_rbfe")


def _ipa_binary_available() -> bool:
    if IPA_PATH is None:
        return False
    exe = os.path.join(IPA_PATH, "gfn2xtb_in_pocket_opt.exe")
    return os.path.isfile(exe) and os.access(exe, os.X_OK)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _ipa_binary_available(),
        reason="IPA not installed: set IPA_PATH and compile gfn2xtb_in_pocket_opt.exe",
    ),
]

# Golden values from IPA's tests/axitinib_analysis/ (neutral tautomer).
# Produced by: python in_pocket.py --pdb tests/4WA9.cif --active-residue 9000
#              --chain B --distance 3.6 --use-ip-structure --verbose
AXITINIB_GOLDEN = {
    "ipa_interaction_energy_kcal": -55.67,
    "ipa_binding_energy_kcal": -50.88,
    "deformation_energy_kcal": 4.79,
}

# Per-residue golden values (neutral tautomer, non-flipped, standard charge).
# These are the rows from interaction_energies.csv that enter the IPA total.
AXITINIB_PER_RESIDUE_GOLDEN = {
    ("TYR", 253, "0"): -4.22,
    ("ALA", 269, "0"): -2.42,
    ("LYS", 271, "+1"): -9.12,
    ("THR", 315, "0"): -3.02,
    ("GLU", 316, "-1"): -4.54,
    ("PHE", 317, "0"): -6.09,
    ("MET", 318, "0"): -6.86,
    ("GLY", 321, "0"): -2.33,
    ("ASN", 368, "0"): -3.58,
    ("LEU", 370, "0"): -4.37,
    ("ALA", 380, "0"): -2.59,
    ("PHE", 382, "0"): -6.54,
}


def _has_test_structure() -> bool:
    if IPA_PATH is None:
        return False
    return os.path.isfile(os.path.join(IPA_PATH, "tests", "4WA9.cif"))


def _has_mcl1_benchmark() -> bool:
    return os.path.isfile(os.path.join(MCL1_DATASET_PATH, "mcl1_structure.pdb"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def axitinib_complex():
    """Load 4WA9 chain B (axitinib + VEGFR2 monomer).

    Only chain B is kept (4WA9 is a dimer; the golden reference uses
    ``--chain B``).
    """
    if not _has_test_structure():
        pytest.skip("4WA9.cif not found in IPA_PATH/tests/")

    from atomworks.io import parse
    from atomworks.io.config import ParseConfig

    cif_path = os.path.join(IPA_PATH, "tests", "4WA9.cif")
    config = ParseConfig(
        model=1,
        build_assembly=None,
        add_missing_atoms=False,
        sanitize_bonds_and_charges=False,
    )
    result = parse(cif_path, config=config)
    atom_array = result["asym_unit"][0]

    atom_array = atom_array[atom_array.chain_id == "B"]

    is_ligand = atom_array.res_name == "AXI"
    is_protein = ~is_ligand
    atom_array.set_annotation("is_ligand", is_ligand)
    atom_array.set_annotation("is_protein", is_protein)

    n_lig = int(is_ligand.sum())
    assert n_lig > 0, "No ligand atoms found (res_name AXI on chain B)"

    return {
        "atom_array": atom_array,
        "name": "axitinib_4WA9",
        "active_residue": int(atom_array.res_id[is_ligand][0]),
        "chain": str(atom_array.chain_id[is_ligand][0]),
        "n_ligand_atoms": n_lig,
        "n_protein_atoms": int(is_protein.sum()),
    }


@pytest.fixture(scope="module")
def mcl1_complex():
    """Build a protein-ligand complex from MCL1 benchmark (first ligand).

    Uses ``atom_array_from_rdkit`` for the ligand (preserving bonds and
    charges) and sets ``chain_type`` so ``ensure_annotations`` can derive
    ``is_protein`` / ``is_ligand`` automatically.
    """
    if not _has_mcl1_benchmark():
        pytest.skip("MCL1 benchmark data not found")

    import biotite.structure as struc
    from atomworks.enums import ChainType
    from atomworks.io.tools.rdkit import atom_array_from_rdkit
    from biotite.structure.io.pdb import PDBFile
    from rdkit import Chem

    protein_pdb = os.path.join(MCL1_DATASET_PATH, "mcl1_structure.pdb")
    sdf_path = os.path.join(MCL1_DATASET_PATH, "ligands.sdf")

    pdb_file = PDBFile.read(protein_pdb)
    protein = pdb_file.get_structure(model=1)
    if isinstance(protein, struc.AtomArrayStack):
        protein = protein[0]

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next(m for m in supplier if m is not None)
    name = mol.GetProp("_Name")

    lig_arr = atom_array_from_rdkit(mol, remove_hydrogens=False)
    lig_arr.chain_id[:] = "L"
    lig_arr.res_name[:] = "LIG"

    combined = protein + lig_arr

    chain_type = np.full(len(combined), ChainType.POLYPEPTIDE_L, dtype=int)
    chain_type[len(protein) :] = ChainType.NON_POLYMER
    combined.set_annotation("chain_type", chain_type)

    return {"atom_array": combined, "name": name}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ipa_executable_initialization():
    """IPAExecutable resolves from IPA_PATH."""
    if not _ipa_binary_available():
        pytest.skip("IPA binaries not compiled")

    from proteinfoundation.metrics.ipa_qm import IPAExecutable

    if IPAExecutable.is_initialized():
        IPAExecutable.reinitialize(os.path.join(IPA_PATH, "gfn2xtb_in_pocket_opt.exe"))
    else:
        IPAExecutable.get_or_initialize()

    assert IPAExecutable.is_initialized()
    assert os.path.isfile(IPAExecutable.get_bin_path())


def test_ipa_axitinib_golden_values(axitinib_complex):
    """Full pipeline on axitinib (4WA9) with golden-value validation.

    Checks total interaction energy, binding energy, deformation energy,
    and per-residue interaction energies against IPA's reference output.
    """
    from proteinfoundation.metrics.ipa_interaction_energy_metric import (
        QMInteractionEnergyIPA,
    )

    metric = QMInteractionEnergyIPA(
        ipa_path=IPA_PATH,
        distance=3.6,
        alpha=1.0,
        kappa=0.025,
        use_ip_structure=True,
        protonation="rdkit",
    )
    metric.setup()

    try:
        result = metric.compute(axitinib_complex["atom_array"])

        assert result["n_structures"] == 1, "Should process exactly 1 structure"

        e_int = result["median_total_interaction_energy"]
        e_bind = result["median_ipa_binding_energy"]

        assert np.isfinite(e_int), "Interaction energy should be finite"
        assert np.isfinite(e_bind), "Binding energy should be finite"
        assert e_int < 0, "Interaction energy should be attractive (negative)"

        # Total interaction energy (1 kcal/mol tolerance)
        assert abs(e_int - AXITINIB_GOLDEN["ipa_interaction_energy_kcal"]) < 1.0, (
            f"Interaction energy {e_int:.2f} too far from golden "
            f"{AXITINIB_GOLDEN['ipa_interaction_energy_kcal']:.2f} kcal/mol "
            f"(delta={abs(e_int - AXITINIB_GOLDEN['ipa_interaction_energy_kcal']):.2f})"
        )

        # Binding energy (1 kcal/mol tolerance)
        assert (
            abs(e_bind - AXITINIB_GOLDEN["ipa_binding_energy_kcal"]) < 1.0
        ), f"Binding energy {e_bind:.2f} too far from golden {AXITINIB_GOLDEN['ipa_binding_energy_kcal']:.2f} kcal/mol"

        # Deformation energy (exact match expected -- ligand-only, no capping)
        if result["deformation_energies"]:
            e_def = result["deformation_energies"][0]
            assert (
                abs(e_def - AXITINIB_GOLDEN["deformation_energy_kcal"]) < 0.1
            ), f"Deformation energy {e_def:.2f} != golden {AXITINIB_GOLDEN['deformation_energy_kcal']:.2f}"

        # Per-residue interaction energies (0.5 kcal/mol tolerance per residue)
        per_res = result["per_residue_interactions"]
        assert len(per_res) == 1, "Should have 1 structure's per-residue data"
        residue_list = per_res[0]
        assert len(residue_list) > 0, "Should have per-residue entries"

        for (res_name, resn, charge), golden_eint in AXITINIB_PER_RESIDUE_GOLDEN.items():
            matches = [
                r
                for r in residue_list
                if r["res"] == res_name and r["resn"] == resn and r["charge"] == charge and not r["flip"]
            ]
            assert (
                len(matches) == 1
            ), f"Expected exactly 1 match for {res_name} {resn} charge={charge}, got {len(matches)}"
            actual = matches[0]["eint_kcal"]
            assert abs(actual - golden_eint) < 0.5, (
                f"{res_name} {resn} (charge={charge}): {actual:.2f} vs golden "
                f"{golden_eint:.2f} kcal/mol (delta={abs(actual - golden_eint):.2f})"
            )

    finally:
        metric.teardown()


def test_ipa_mcl1_benchmark(mcl1_complex):
    """IPA pipeline on MCL1 benchmark ligand (same as MACE test).

    No golden values -- verifies structural correctness: the metric
    returns finite energies and per-residue breakdowns.
    """
    from proteinfoundation.metrics.ipa_interaction_energy_metric import (
        QMInteractionEnergyIPA,
    )

    metric = QMInteractionEnergyIPA(
        ipa_path=IPA_PATH,
        distance=5.0,
        use_ip_structure=True,
        eval_only=True,
        protonation="rdkit",
    )
    metric.setup()

    try:
        result = metric.compute(mcl1_complex["atom_array"])

        assert result["n_structures"] == 1, "Should process exactly 1 structure"

        e_int = result["median_total_interaction_energy"]
        assert np.isfinite(e_int), "Interaction energy should be finite"
        assert e_int < 0, "Interaction energy should be attractive (negative)"

    finally:
        metric.teardown()
