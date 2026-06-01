"""Tests for OracleMetric subclasses (RF3Oracle, AF3Oracle) and the full refolding pipeline.

Slow/GPU tests require:
  - RF3 checkpoint at $DATA_PATH/rf3_checkpoints/...
  - AF3 apptainer image at $AF3_APPTAINER_PATH
  - A CUDA device

Run with:
    pytest tests/test_oracles.py -v -m "not slow"   # unit tests only
    pytest tests/test_oracles.py -v                  # all tests (needs GPU)
"""

import io
import os
import time

import biotite.structure as struc
import numpy as np
import pytest
import rootutils
from atomworks.enums import ChainType
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.io.transforms.atom_array import (
    add_chain_iid_annotation,
    add_id_and_entity_annotations,
    add_molecule_iid_annotation,
    add_pn_unit_iid_annotation,
)
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import CIFWriteConfig, to_cif_string
from atomworks.ml.conditions import C_CRD
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from biotite.structure.io.pdbx import get_structure

from foundry.metrics import MetricManager
from proteinfoundation.metrics.af3_oracle import AF3Oracle
from proteinfoundation.metrics.design_metrics import ClashMetric
from proteinfoundation.metrics.gt_structure_comparison import StructureComparisonMetric
from proteinfoundation.metrics.metrics_utils import run_oracle_refolding_and_pass_criteria
from proteinfoundation.metrics.pass_criterion_metric import PassCriterionMetric
from proteinfoundation.metrics.rf3_oracle import RF3Oracle

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ["NCCL_DEBUG"] = "WARNING"

from tests.conftest import TEST_DATA_DIR, has_af3  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def modified_1a1e():
    """Modified 1a1e where PTR is split into TYR + custom phosphate ligand.

    To stress-test the custom CCD handling in AF3, the non-canonical amino acid
    PTR (phosphotyrosine) is split into TYR and a separate non-CCD ligand,
    which is featurized as a covalent modification.
    """
    path = TEST_DATA_DIR / "1a1e.cif"

    config = ParseConfig.from_preset("rcsb", hydrogen_policy="remove", add_missing_atoms=True)
    aa_1a1e = parse(path, config=config)["asym_unit"][0]
    aa_1a1e = as_atom_array_plus(aa_1a1e)

    # Design outputs don't have OXT or NaN coordinates
    aa_1a1e = aa_1a1e[aa_1a1e.atom_name != "OXT"]
    aa_1a1e = aa_1a1e[~np.isnan(aa_1a1e.coord).any(axis=-1)]

    mod_aa = aa_1a1e.copy()

    # Convert PTR to TYR + custom phosphate ligand
    ptr_tyr_mask = (mod_aa.res_name == "PTR") & ~(np.isin(mod_aa.atom_name, ["P", "O1P", "O2P", "O3P"]))
    ptr_phosphate_mask = (mod_aa.res_name == "PTR") & (np.isin(mod_aa.atom_name, ["P", "O1P", "O2P", "O3P"]))
    mod_aa.res_name[ptr_tyr_mask] = "TYR"
    mod_aa.res_name[ptr_phosphate_mask] = "L:0"
    mod_aa.res_id[ptr_phosphate_mask] = 1
    mod_aa.is_polymer[ptr_phosphate_mask] = False
    mod_aa.hetero[ptr_phosphate_mask] = True
    mod_aa.chain_type[ptr_phosphate_mask] = ChainType.NON_POLYMER
    mod_aa.chem_comp_type[ptr_phosphate_mask] = "NON-POLYMER"
    mod_aa.chain_id[ptr_phosphate_mask & (mod_aa.chain_id == "C")] = "E"
    mod_aa.chain_id[ptr_phosphate_mask & (mod_aa.chain_id == "D")] = "F"
    mod_aa = mod_aa[np.argsort(ptr_phosphate_mask, kind="stable")]

    # Add iid annotations which are typically present in design workflows
    mod_aa.set_annotation("transformation_id", np.full(mod_aa.array_length(), "1", dtype=str))
    mod_aa = add_id_and_entity_annotations(mod_aa)
    mod_aa = add_chain_iid_annotation(mod_aa)
    mod_aa = add_pn_unit_iid_annotation(mod_aa)
    mod_aa = add_molecule_iid_annotation(mod_aa)

    # Add an annotation for determining templating (no templating by default)
    mod_aa.set_annotation("is_templated", np.full(mod_aa.array_length(), False, dtype=bool))

    return mod_aa


@pytest.fixture(params=["rf3", "af3"], scope="module")
def oracle_info(request, modified_1a1e):
    """Instantiated RF3Oracle and AF3Oracle, as well as various other test inputs."""
    confidence_keys = [
        "chain_of_interest",
        "overall_plddt",
        "overall_pae",
        "plddt_coi",
        "ptm_coi",
        "pae_coi",
        "ipae_coi",
        "min_ipae_coi",
        "iptm_coi",
        "ranking_score",
    ]

    if request.param == "af3" and not has_af3:
        pytest.skip("AF3 apptainer image not available")

    if request.param == "rf3":
        oracle = RF3Oracle(
            num_steps=50,
            n_recycles=5,
            ckpt_path=os.path.join(
                os.environ["DATA_PATH"],
                "rf3_checkpoints/rf3-with-confidence-xaira-epoch-0711-fixed.ckpt",
            ),
            template_annotations="is_templated",
        )
    elif request.param == "af3":
        oracle = AF3Oracle(num_recycles=5, template_annotations="is_templated")
    else:
        raise ValueError(f"Unknown oracle type: {request.param}")

    oracle.setup()
    yield {
        "oracle": oracle,
        "atom_array": modified_1a1e,
        "example_id": "modified_1a1e",
        "confidence_keys": confidence_keys,
    }

    # Teardown runs at the end of the module
    oracle.teardown()


# ---------------------------------------------------------------------------
# Unit tests (no GPU required)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lencapavir_design_with_covale() -> AtomArray:
    """Real design that exhibits known failure modes"""
    path = TEST_DATA_DIR / "lencapavir_design_with_covale.cif"
    return get_structure(pdbx.CIFFile.read(path), model=1, include_bonds=True)


def test_renumber_emits_1_based_struct_conn_seq_ids(lencapavir_design_with_covale):
    """Renumber + ``to_cif_string`` produces ``_struct_conn`` partner seq_ids in ``[1..len(chain)]``."""
    aa = lencapavir_design_with_covale
    renumbered = AF3Oracle._renumber_res_ids_per_chain(aa)
    cif_text = to_cif_string(renumbered.copy(), config=CIFWriteConfig(id="test"))

    cif = pdbx.CIFFile.read(io.StringIO(cif_text))
    sc = cif.block["struct_conn"]
    chains = np.concatenate([sc["ptnr1_label_asym_id"].as_array(str), sc["ptnr2_label_asym_id"].as_array(str)])
    seqs = np.concatenate([sc["ptnr1_label_seq_id"].as_array(int), sc["ptnr2_label_seq_id"].as_array(int)])
    chain_len = {c: int(renumbered.res_id[renumbered.chain_id == c].max()) for c in np.unique(renumbered.chain_id)}
    for chain, seq in zip(chains, seqs, strict=False):
        assert (
            1 <= seq <= chain_len[chain]
        ), f"_struct_conn seq_id {seq} out of [1..{chain_len[chain]}] for chain {chain}"


def test_validate_inter_residue_bonds_match_passes(lencapavir_design_with_covale):
    """Identical input and AF3 arrays satisfy the validator."""
    aa = lencapavir_design_with_covale
    AF3Oracle._validate_inter_residue_bonds(aa, aa.copy())


def test_validate_inter_residue_bonds_missing_raises(lencapavir_design_with_covale):
    """Dropping the cross-chain covale bond raises ``ValueError`` (original silent-drop failure mode)."""
    aa = lencapavir_design_with_covale
    af3 = aa.copy()
    bonds = af3.bonds.as_array()
    same_chain = af3.chain_id[bonds[:, 0]] == af3.chain_id[bonds[:, 1]]
    af3.bonds = struc.BondList(af3.array_length(), bonds[same_chain])
    with pytest.raises(ValueError, match="inter-residue bond graph"):
        AF3Oracle._validate_inter_residue_bonds(aa, af3)


def test_validate_inter_residue_bonds_extra_raises(lencapavir_design_with_covale):
    """Adding an inter-residue bond on the AF3 side that wasn't in the input raises."""
    aa = lencapavir_design_with_covale
    af3 = aa.copy()
    n_atoms_idx = np.where(af3.atom_name == "N")[0]
    new_bonds = af3.bonds.as_array().tolist() + [
        [int(n_atoms_idx[0]), int(n_atoms_idx[-1]), int(struc.BondType.SINGLE)]
    ]
    af3.bonds = struc.BondList(af3.array_length(), np.asarray(new_bonds))
    with pytest.raises(ValueError, match="inter-residue bond graph"):
        AF3Oracle._validate_inter_residue_bonds(aa, af3)


def test_disambiguate_chain_ids():
    """Chains sharing chain_id but differing by chain_iid receive distinct chain IDs."""
    aa = AtomArray(6)
    aa.chain_id = np.array(["A", "A", "A", "A", "B", "B"])
    aa.add_annotation("chain_iid", dtype=str)
    aa.chain_iid = np.array(["A_1", "A_1", "A_2", "A_2", "B_1", "B_1"])

    result, _ = AF3Oracle._disambiguate_chain_ids(aa)

    a_iid0_id = result.chain_id[0]
    a_iid1_id = result.chain_id[2]
    assert a_iid0_id != a_iid1_id
    assert result.chain_id[1] == a_iid0_id
    assert result.chain_id[3] == a_iid1_id
    assert np.all(result.chain_id[4:] == "B")


# ---------------------------------------------------------------------------
# Slow GPU tests — oracle.compute directly
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Slow GPU integration test — full four-manager oracle pipeline
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
def test_oracle_pipeline_integration(oracle_info, modified_1a1e, tmp_path):
    """Full 4-stage pipeline with chain A templating: design → oracle → refolding → pass_criteria.

    Covers confidence keys, valid ranges, CIF persistence, resumability, backbone RMSD,
    and chain A RMSD (expected < 2 Å since chain A is templated).
    """
    oracle = oracle_info["oracle"]
    ensure_annotations(modified_1a1e, "is_protein_backbone")

    example_id = "chain_a_templated_pipeline_1a1e"

    # Template chain A, and ensure that chain B is the chain of interest
    aa_templated = modified_1a1e.copy()
    aa_templated.set_annotation("is_templated", aa_templated.chain_id == "A")
    C_CRD.set_mask(aa_templated, (aa_templated.chain_id != "B"))

    # --- design_mm ---
    design_mm = MetricManager({"general_metrics": ClashMetric()})
    design_mm.on_validation_epoch_start()
    design_result = design_mm(
        network_input=None,
        network_output=None,
        extra_info={"example_id": example_id, "metrics_tags": [], "output_base_dir": str(tmp_path)},
        predicted_atom_array=aa_templated,
    )
    design_mm.on_validation_epoch_end()
    assert "example_id" in design_result

    # --- oracle_mm + refold_mm + pass_mm ---
    oracle_mm = MetricManager({"oracle": oracle})
    refold_mm = MetricManager(
        {
            "backbone_rmsd": StructureComparisonMetric(
                align_mask="is_protein_backbone",
                subset_mask="is_protein_backbone",
                metric_type="rmsd",
            ),
            "chain_a_rmsd": StructureComparisonMetric(
                align_mask="is_protein_backbone & (chain_id == 'A')",
                subset_mask="is_protein_backbone & (chain_id == 'A')",
                metric_type="rmsd",
            ),
        }
    )
    pass_mm = MetricManager(
        {"unconditional": PassCriterionMetric(metric_thresholds={"backbone_rmsd.rmsd": {"max": 2.0}})}
    )
    for mm in [oracle_mm, refold_mm, pass_mm]:
        mm.on_validation_epoch_start()

    result = run_oracle_refolding_and_pass_criteria(
        atom_array=aa_templated,
        example_id=example_id,
        output_dir=tmp_path,
        oracle_mm=oracle_mm,
        refold_mm=refold_mm,
        pass_mm=pass_mm,
        metrics_tags=None,
        prior_metrics=design_result,
    )

    # Resumability: second oracle call should load from JSON without re-running
    t0 = time.time()
    result2 = oracle.compute(
        input_atom_array=aa_templated,
        example_id=example_id,
        out_dir=tmp_path,
        chain_of_interest=result.get("oracle.chain_of_interest"),
    )
    assert time.time() - t0 < 5, "Resumable oracle call took too long, likely re-running"
    assert result2["overall_plddt"] == pytest.approx(result["oracle.overall_plddt"])

    for mm in [oracle_mm, refold_mm, pass_mm]:
        mm.on_validation_epoch_end()

    assert result is not None

    # Confidence keys and value ranges
    for key in oracle_info["confidence_keys"]:
        prefixed = f"oracle.{key}"
        assert prefixed in result, f"Missing key: {prefixed}"
        if key not in ["chain_of_interest", "ranking_score"]:
            assert result[prefixed] > 0, f"Unrealistic value for {prefixed}: {result[prefixed]}"
            assert result[prefixed] < 100, f"Unrealistic value for {prefixed}: {result[prefixed]}"

    # CIF was saved
    cif_path = oracle.refolded_cif_path(tmp_path, example_id)
    assert cif_path.exists(), "Oracle did not save refolded CIF"

    # Structure quality metrics
    assert "backbone_rmsd.rmsd" in result
    assert "chain_a_rmsd.rmsd" in result
    assert "unconditional.pass" in result
    assert "unconditional.all_metrics_present" in result
    assert np.isfinite(result["oracle.overall_plddt"])
    assert np.isfinite(result["backbone_rmsd.rmsd"])
    assert result["unconditional.pass"] in (0.0, 1.0)
    assert result["unconditional.all_metrics_present"] in (0.0, 1.0)

    # Chain A RMSD should be low since chain A was templated
    assert (
        result["chain_a_rmsd.rmsd"] < 2.0
    ), f"Chain A RMSD {result['chain_a_rmsd.rmsd']:.2f} Å exceeds 2.0 Å despite being templated."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
