"""Tests for MACE interaction energy metric.

Uses the MCL1 benchmark (first ligand) to verify the full pipeline:
hydride H addition -> geometry optimization -> interaction energy decomposition.

Requires: GPU with CUDA, MACE-POLAR-1 model, hydride installed.
Run with: pytest tests/metrics/test_mace_interaction_energy_metric.py -v -s --noconftest
"""

import os
import time
from unittest.mock import patch

import numpy as np
import pytest

EV_TO_KCAL = 23.0609

MCL1_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "mcl1_rbfe")


@pytest.fixture(scope="module")
def mcl1_complex():
    """Build a protein-ligand complex from MCL1 benchmark (first ligand).

    Uses ``atom_array_from_rdkit`` for the ligand (preserving bonds and
    charges) and sets ``chain_type`` so ``ensure_annotations`` can derive
    ``is_protein`` / ``is_ligand`` automatically.
    """
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
    exp_dg = float(mol.GetProp("kcal/mol experimental dG"))

    lig_arr = atom_array_from_rdkit(mol, remove_hydrogens=False)
    lig_arr.chain_id[:] = "L"
    lig_arr.res_name[:] = "LIG"

    combined = protein + lig_arr

    chain_type = np.full(len(combined), ChainType.POLYPEPTIDE_L, dtype=int)
    chain_type[len(protein) :] = ChainType.NON_POLYMER
    combined.set_annotation("chain_type", chain_type)

    # Drop all hydrogens to verify ensure_hydrogens can re-infer them for both
    # protein residues (via CCD lookup) and the ligand (via preserved
    # RDKit-derived heavy-atom bond graph).
    combined = combined[combined.element != "H"]

    return {"atom_array": combined, "name": name, "exp_dg": exp_dg}


@pytest.mark.skip(reason="Test input cannot be parsed by atomworks. Need to fix this")
def test_chain_type_restored_after_rdkit_protonation(mcl1_complex):
    """RDKit protonation fallback drops Biotite annotations; restore keeps ligand masks."""
    from atomworks.enums import ChainType
    from atomworks.ml.conditions.annotator import ensure_annotations

    import proteinfoundation.metrics.physics.mace_interaction_energy_metric as mace_metric

    prepared = mcl1_complex["atom_array"]
    chain_type_by_id: dict[str, int] = {}
    for cid in np.unique(prepared.chain_id):
        mask = prepared.chain_id == cid
        chain_type_by_id[str(cid)] = int(prepared.chain_type[mask][0])

    def _rdkit_protonation_only(atom_array, *, ph):
        from atomworks.io.tools.rdkit import atom_array_from_rdkit, atom_array_to_rdkit

        mol = atom_array_to_rdkit(atom_array, hydrogen_policy="infer", set_coord=True)
        return atom_array_from_rdkit(mol, remove_hydrogens=False, remove_inferred_atoms=False)

    with patch.object(
        mace_metric,
        "_infer_hydrogens",
        side_effect=_rdkit_protonation_only,
    ):
        protonated = mace_metric._infer_hydrogens(prepared, ph=7.0)

    assert "chain_type" not in protonated.get_annotation_categories()

    restored = np.empty(len(protonated), dtype=int)
    for cid in np.unique(protonated.chain_id):
        mask = protonated.chain_id == cid
        restored[mask] = chain_type_by_id.get(str(cid), ChainType.POLYPEPTIDE_L)
    protonated.set_annotation("chain_type", restored)

    ensure_annotations(protonated, "is_ligand", "is_protein")
    lig_mask = protonated.chain_id == "L"
    assert np.all(protonated.chain_type[lig_mask] == ChainType.NON_POLYMER)
    assert np.all(protonated.is_ligand[lig_mask])
    assert np.all(protonated.is_protein[~lig_mask])


@pytest.mark.gpu
@pytest.mark.xfail(
    strict=False,
    reason="MACE interaction energy is flaky on shared GPU (OOM / n_structures=0 after prior tests)",
)
def test_mace_interaction(mcl1_complex):
    """Full pipeline test with default settings on one MCL1 ligand.

    Uses the MACEInteractionEnergyMetric.compute() API with default
    parameters (50 opt steps, LBFGS, hydride H addition).
    """
    from proteinfoundation.metrics.physics.mace_interaction_energy_metric import (
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
