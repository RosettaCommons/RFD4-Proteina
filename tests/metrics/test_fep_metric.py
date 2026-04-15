"""Tests for the TMD-based RBFE metrics.

Uses the MCL1 benchmark (edge 23->44) from the JACS set.
Reference: ddG = 0.32 kcal/mol (standard mode), experimental = 0.16 kcal/mol.

Requires: GPU with CUDA, TMD installed with custom_ops.
Run with: pytest tests/metrics/test_fep_metric.py -v -s
"""

import os

import numpy as np
import pytest

tmd = pytest.importorskip("tmd", reason="TMD not installed -- skipping FEP tests")

pytestmark = pytest.mark.gpu

MCL1_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "mcl1_rbfe")


@pytest.fixture(scope="module")
def mcl1_dataset():
    """Load full MCL1 benchmark dataset."""
    from proteinfoundation.metrics.fep_utils import load_benchmark_dataset

    return load_benchmark_dataset(MCL1_DATASET_PATH)


@pytest.fixture(scope="module")
def mcl1_edge(mcl1_dataset):
    """Load MCL1 first edge (23->44) for testing."""
    ds = mcl1_dataset
    edge = ds["edges"][0]
    return {
        "mol_a": ds["mols"][edge["mol_a"]],
        "mol_b": ds["mols"][edge["mol_b"]],
        "core": np.array(edge["core"]),
        "protein_pdb": ds["protein_pdb"],
        "exp_ddg": 0.16,  # kcal/mol
        "ref_ddg": 0.32,  # kcal/mol (standard mode reference)
    }


def _build_protein_ligand_atom_array(protein_pdb: str, ligand_mol):
    """Build a combined protein+ligand AtomArray from a PDB path and RDKit Mol.

    Sets ``chain_type`` annotations so that ``extract_protein_atom_array`` and
    ``extract_ligand_as_rdkit`` work correctly.
    """
    import biotite.structure as struc
    from atomworks.enums import ChainType
    from atomworks.io.tools.rdkit import atom_array_from_rdkit
    from biotite.structure.io.pdb import PDBFile

    pdb_file = PDBFile.read(protein_pdb)
    protein_aa = pdb_file.get_structure(model=1)
    if isinstance(protein_aa, struc.AtomArrayStack):
        protein_aa = protein_aa[0]

    ligand_aa = atom_array_from_rdkit(ligand_mol, remove_hydrogens=False)
    ligand_aa.chain_id = np.full(len(ligand_aa), "L")
    ligand_aa.res_name = np.full(len(ligand_aa), "LIG")

    combined = protein_aa + ligand_aa

    chain_type = np.full(len(combined), ChainType.POLYPEPTIDE_L, dtype=int)
    chain_type[len(protein_aa) :] = ChainType.NON_POLYMER
    combined.set_annotation("chain_type", chain_type)

    return combined


# ---------------------------------------------------------------------------
# Raw API tests (existing)
# ---------------------------------------------------------------------------


def test_fep_smoke(mcl1_edge):
    """Smoke test (~4-8 min on H200). Confirms the full pipeline runs without error.

    Uses minimal frames/equilibration (50 frames, 8 windows) -- the ddG value
    will be noisy but the test verifies system building, minimization, HREX,
    BAR analysis, and result extraction all work end-to-end.
    Faster on repeat runs due to CUDA kernel caching.
    """
    from proteinfoundation.metrics.fep_utils import run_rbfe_edge

    result = run_rbfe_edge(
        mol_a=mcl1_edge["mol_a"],
        mol_b=mcl1_edge["mol_b"],
        core=mcl1_edge["core"],
        protein_pdb=mcl1_edge["protein_pdb"],
        mode="fast",
        seed=2025,
        n_frames_override=50,
        n_eq_steps_override=1000,
        n_windows_override=8,
    )

    assert not np.isnan(result["ddg_kcal_mol"]), "ddG should not be NaN"
    assert np.isfinite(result["ddg_err_kcal_mol"]), "Error should be finite"
    assert result["elapsed_seconds"] > 0


# ---------------------------------------------------------------------------
# FEPMetric (Foundry Metric API) tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fep_metric_fast_mode(mcl1_edge):
    """Full fast-mode FEPMetric via Foundry Metric API (~60 min on 1x H200).

    Same tolerance as test_fep_fast_mode (2.0 kcal/mol).
    """
    from proteinfoundation.metrics.fep_metric import FEPMetric

    gt_aa = _build_protein_ligand_atom_array(mcl1_edge["protein_pdb"], mcl1_edge["mol_a"])
    pred_aa = _build_protein_ligand_atom_array(mcl1_edge["protein_pdb"], mcl1_edge["mol_b"])

    metric = FEPMetric(mode="fast", seed=2025)
    result = metric.compute(
        predicted_atom_array=pred_aa,
        ground_truth_atom_array=gt_aa,
    )

    ddg = result["ddg_kcal_mol"]
    err = result["ddg_err_kcal_mol"]
    ref = mcl1_edge["ref_ddg"]

    print(f"\nFEPMetric ddG = {ddg:.3f} ± {err:.3f} kcal/mol")
    print(f"Reference: {ref:.3f} kcal/mol")

    assert not np.isnan(ddg), "ddG should not be NaN"
    assert "error" not in result
    assert abs(ddg - ref) < 2.0, (
        f"ddG={ddg:.3f} deviates from reference {ref:.3f} by " f"{abs(ddg - ref):.3f} kcal/mol (tolerance: 2.0)"
    )


# ---------------------------------------------------------------------------
# FEPBatchMetric tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fep_batch_metric_smoke(mcl1_dataset):
    """Batch RBFE via FEPBatchMetric with 2 MCL1 edges (~30 min on multi-GPU with MPS).

    Verifies the accumulate-then-flush pattern: setup -> compute x N -> teardown.
    """
    from proteinfoundation.metrics.fep_metric import FEPBatchMetric

    ds = mcl1_dataset
    edges = ds["edges"][:2]

    metric = FEPBatchMetric(mode="fast", seed=2025, n_mps_workers=16)
    metric.setup()

    for edge in edges:
        mol_a = ds["mols"][edge["mol_a"]]
        mol_b = ds["mols"][edge["mol_b"]]

        gt_aa = _build_protein_ligand_atom_array(ds["protein_pdb"], mol_a)
        pred_aa = _build_protein_ligand_atom_array(ds["protein_pdb"], mol_b)

        placeholder = metric.compute(
            predicted_atom_array=pred_aa,
            ground_truth_atom_array=gt_aa,
        )
        assert placeholder["pending"] is True

    metric.teardown()
    results = metric.get_results()

    assert len(results) == 2
    for r in results:
        assert not np.isnan(r["ddg_kcal_mol"]), f"ddG should not be NaN: {r}"
        assert np.isfinite(r["ddg_err_kcal_mol"]), f"Error should be finite: {r}"
