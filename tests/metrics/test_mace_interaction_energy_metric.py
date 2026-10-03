"""Tests for MACE interaction energy metric.

Uses the MCL1 benchmark (first ligand) to verify the full pipeline:
AtomWorks protonation -> geometry optimization -> interaction energy decomposition.

The protonation test runs on CPU; the full pipeline requires a GPU with CUDA and the MACE-POLAR-1 model.
Run with: pytest tests/metrics/test_mace_interaction_energy_metric.py -v -s --noconftest
"""

import os
import time

import numpy as np
import pytest

from tests.helpers import BENCHMARKS_DIR

EV_TO_KCAL = 23.0609

MCL1_DATASET_PATH = BENCHMARKS_DIR / "small_molecule_binder" / "mcl1_rbfe"


@pytest.fixture(scope="module")
def mcl1_complex():
    """Build a protein-ligand complex from MCL1 benchmark (first ligand).

    Uses ``atom_array_from_rdkit`` for the ligand (preserving bonds and
    charges), reads the protein's bonds, and sets ``charge``, ``pn_unit_iid``
    and ``chain_type`` so protonation and ``ensure_annotations`` can run.
    """
    import biotite.structure as struc
    from atomworks.enums import ChainType
    from atomworks.io.tools.rdkit import atom_array_from_rdkit
    from biotite.structure.io.pdb import PDBFile
    from rdkit import Chem

    protein_pdb = os.path.join(MCL1_DATASET_PATH, "mcl1_structure.pdb")
    sdf_path = os.path.join(MCL1_DATASET_PATH, "ligands.sdf")

    pdb_file = PDBFile.read(protein_pdb)
    protein = pdb_file.get_structure(model=1, include_bonds=True)
    if isinstance(protein, struc.AtomArrayStack):
        protein = protein[0]

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next(m for m in supplier if m is not None)
    name = mol.GetProp("_Name")
    exp_dg = float(mol.GetProp("kcal/mol experimental dG"))

    lig_arr = atom_array_from_rdkit(mol, remove_hydrogens=False)
    lig_arr.chain_id[:] = "L"
    lig_arr.res_name[:] = "LIG"

    protein.set_annotation("charge", np.zeros(len(protein), dtype=lig_arr.charge.dtype))
    combined = protein + lig_arr
    combined.set_annotation("pn_unit_iid", np.char.add(combined.chain_id.astype(str), "_1"))

    chain_type = np.full(len(combined), ChainType.POLYPEPTIDE_L, dtype=int)
    chain_type[len(protein) :] = ChainType.NON_POLYMER
    combined.set_annotation("chain_type", chain_type)

    # Drop all hydrogens so protonation rebuilds them for the protein residues
    # and for the ligand (from its RDKit-derived heavy-atom bond graph).
    combined = combined[combined.element != "H"]

    return {"atom_array": combined, "name": name, "exp_dg": exp_dg}


def test_hydrogens_are_rebuilt_on_every_residue_and_keep_the_chain_types(mcl1_complex):
    """Each peptide nitrogen gains one hydrogen, and ``chain_type`` still marks the ligand."""
    from atomworks.enums import ChainType
    from atomworks.ml.conditions.annotator import ensure_annotations

    from rfproteina.metrics.physics.mace_interaction_energy_metric import _infer_hydrogens

    heavy = mcl1_complex["atom_array"]
    protonated = _infer_hydrogens(heavy, ph=7.0)

    is_h = protonated.element == "H"
    assert np.array_equal(protonated.coord[~is_h], heavy.coord)
    assert np.array_equal(protonated.atom_name[~is_h], heavy.atom_name)
    for n in np.flatnonzero((protonated.atom_name == "N") & (protonated.res_name != "PRO")):
        neighbours = protonated.bonds.get_bonds(int(n))[0]
        if any(protonated.atom_name[j] == "C" and protonated.res_id[j] != protonated.res_id[n] for j in neighbours):
            assert is_h[neighbours].sum() == 1, f"{protonated.res_name[n]}{protonated.res_id[n]}"
    ligand = protonated.chain_id == "L"
    assert is_h[ligand].any()
    assert np.all(protonated.chain_type[ligand] == ChainType.NON_POLYMER)
    ensure_annotations(protonated, "is_ligand", "is_protein")
    assert np.all(protonated.is_ligand == ligand)


@pytest.mark.gpu
@pytest.mark.xfail(
    strict=False,
    reason="MACE interaction energy is flaky on shared GPU (OOM / n_structures=0 after prior tests)",
)
def test_mace_interaction(mcl1_complex):
    """Full pipeline test with default settings on one MCL1 ligand.

    Uses the MACEInteractionEnergyMetric.compute() API with default
    parameters (LBFGS, AtomWorks protonation), with a single optimisation step.
    """
    from rfproteina.metrics.physics.mace_interaction_energy_metric import (
        MACEInteractionEnergyMetric,
    )

    model_path = os.environ.get("MACE_POLAR_MODEL_PATH") or os.environ.get("MACE_MODEL_PATH")
    if model_path is None:
        pytest.skip("MACE_POLAR_MODEL_PATH or MACE_MODEL_PATH not set")

    import torch

    def _cuda_free_gib() -> float:
        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / (1024**3)

    # GPU CI runs this after other tests; need headroom for MACE forward passes.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Fall back to CPU when the GPU lane is nearly full after earlier tests.
    device = "cuda" if _cuda_free_gib() >= 8.0 else "cpu"

    metric = MACEInteractionEnergyMetric(
        model_path=model_path,
        max_opt_steps=1,
        fmax=0.05,
        device=device,
        hydrogen_policy="infer",
        infer_charge="zeros",
        max_energy_magnitude=None,
        debug=False,
    )
    metric.setup()

    try:
        t0 = time.time()
        result = metric.compute(mcl1_complex["atom_array"])
        elapsed = time.time() - t0

        assert result["n_structures"] == 1, "Should process exactly 1 structure"
        assert np.isfinite(result["median_interaction_energy"]), "Interaction energy should be finite"
        assert np.isfinite(result["median_complex_energy"]), "Complex energy should be finite"
        assert np.isfinite(result["median_protein_energy"]), "Protein energy should be finite"
        assert np.isfinite(result["median_ligand_energy"]), "Ligand energy should be finite"

        e_int = result["median_interaction_energy"]

        print(f"\nLigand: {mcl1_complex['name']}")
        print(f"Experimental dG: {mcl1_complex['exp_dg']:.3f} kcal/mol")
        print(f"Interaction energy: {e_int:.4f} eV ({e_int * EV_TO_KCAL:.2f} kcal/mol)")
        print(f"Complex energy: {result['median_complex_energy']:.4f} eV")
        print(f"Protein energy: {result['median_protein_energy']:.4f} eV")
        print(f"Ligand energy: {result['median_ligand_energy']:.4f} eV")
        print(f"Optimization steps: {result['mean_n_opt_steps']:.0f}")
        print(f"Converged: {result['convergence_rate']:.0%}")
        print(f"Elapsed: {elapsed:.1f}s")

    finally:
        metric.teardown()
