"""Tests for OracleMetric subclasses (AF3Oracle) and the full refolding pipeline.

Slow/GPU tests require:
  - AF3 runtime image at $AF3_IMAGE
  - A CUDA device

Run with:
    pytest tests/test_oracles.py -v -m "not slow"   # unit tests only
    pytest tests/test_oracles.py -v                  # all tests (needs GPU)
"""

import errno
import io
import os
import socket
import threading
import time
from pathlib import Path

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

from rfproteina.metrics import MetricManager
from rfproteina.metrics.design_metrics import ClashMetric
from rfproteina.metrics.gt_structure_comparison import StructureComparisonMetric
from rfproteina.metrics.oracles.af3_oracle import AF3Oracle, _send_to_daemon
from rfproteina.metrics.oracles.pass_criterion_metric import PassCriterionMetric
from rfproteina.metrics.oracles.pre_oracle_transform import OFF_TARGET_PARSE_CONFIG, OffTargetSubstitutionTransform
from rfproteina.metrics.utils.metrics_utils import run_oracle_refolding_and_pass_criteria
from rfproteina.utils.parse_presets import ANNOTATED_CIF, STANDARD_ANNOTATIONS_ONLY

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


@pytest.fixture(params=["af3"], scope="module")
def oracle_info(request, modified_1a1e):
    """Instantiated AF3Oracle."""
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

    if not has_af3 and not os.environ.get("AF3_LAUNCHER_SOCKET_TEMPLATE"):
        # See gpu_e2e.require_preval_oracle: when the cluster declares AF3, skipping
        # would make an absent runtime look like a pass.
        message = "AF3 apptainer image not available"
        if os.environ.get("REQUIRE_AF3") == "1":
            pytest.fail(message)
        pytest.skip(message)

    oracle = AF3Oracle(num_recycles=5, template_query="is_templated")

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


def test_send_to_daemon_retries_on_eagain(tmp_path, monkeypatch):
    """A connect() that raises EAGAIN (the errno Linux returns for an AF_UNIX
    socket whose listen backlog is momentarily full -- e.g. several rank
    processes connecting to the daemon in a burst) must be retried like
    ECONNREFUSED, not propagated.
    """
    sock_path = tmp_path / "daemon.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)

    def serve_once():
        conn, _ = server.accept()
        data = b""
        while not data.endswith(b"\n"):
            data += conn.recv(65536)
        conn.sendall(b'{"status": "ok"}\n')
        conn.close()

    thread = threading.Thread(target=serve_once, daemon=True)
    thread.start()

    real_connect = socket.socket.connect
    calls = {"n": 0}

    def flaky_connect(self, address):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", flaky_connect)

    try:
        response = _send_to_daemon(str(sock_path), {"cmd": "ping"}, connect_retry_interval_s=0.01)
    finally:
        thread.join(timeout=5)
        server.close()

    assert response == {"status": "ok"}
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Pre-oracle transforms
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def annotated_design() -> AtomArray:
    """Design with an unambiguous protein COI (``A_1``) plus a ligand partner (``B_1``)."""
    path = TEST_DATA_DIR / "lencapavir_design_with_covale.cif"
    return parse(path, config=ANNOTATED_CIF)["assemblies"]["1"][0]


@pytest.mark.slow
@pytest.mark.gpu
def test_off_target_substitution_refolds_each_off_target_with_af3(annotated_design, tmp_path):
    """AF3 folds one substituted complex per off-target, storing and scoring each on its own.

    Covers the transform end to end: the design chain, the renamed off-target
    chains, the aligned annotations and the merged bond graph all survive the
    round trip through AF3's CIF entrypoint and back, each save_id holds the
    off-target it names (checked by residue count, which differs between the
    two), and a second call reloads them all from the manifest.
    """
    if not has_af3:
        pytest.skip("AF3 apptainer image not available")

    # Two off-targets of very different size, so residue count identifies which one
    # a given refold contains. One carries cofactors, since substituting a real
    # target usually brings its ligands along.
    off_target_paths = {
        "5vht": TEST_DATA_DIR / "5vht_from_file.cif",
        "5ocm": TEST_DATA_DIR / "pdb" / "5ocm.cif",
    }
    oracle = AF3Oracle(
        oracle_name="af3_off_target",
        num_recycles=3,
        pre_oracle_transform=OffTargetSubstitutionTransform(off_target_paths),
    )
    oracle.setup()
    try:
        result = oracle.compute(
            input_atom_array=annotated_design,
            example_id="off_target_design",
            out_dir=tmp_path,
            chain_of_interest="A_1",
        )

        assert oracle.list_save_ids() == list(off_target_paths)
        coi_mask = annotated_design.chain_iid == "A_1"
        n_coi_residues = struc.get_residue_count(annotated_design[coi_mask])
        for save_id, off_target_path in off_target_paths.items():
            plddt = result[f"{save_id}.overall_plddt"]
            assert 0 < plddt < 100, f"{save_id}: implausible plddt {plddt}"
            assert result[f"{save_id}.min_ipae_coi"] > 0, f"{save_id}: no interface pae, off-target not folded with it"

            cif_path = Path(result[f"{save_id}.refolded_structure_path"])
            assert cif_path.exists() and save_id in cif_path.name
            refolded = parse(cif_path, config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]
            # The design chain came back whole...
            assert struc.get_residue_count(refolded[refolded.chain_id == "A"]) == n_coi_residues
            # ...next to this save_id's off-target and no other. The two off-targets
            # differ in size, so residue count identifies which one was folded.
            off_target = parse(off_target_path, config=OFF_TARGET_PARSE_CONFIG)["assemblies"]["1"][0]
            assert struc.get_residue_count(refolded) == n_coi_residues + struc.get_residue_count(off_target)

        # Resumability: the manifest + per-save_id JSONs reload without refolding.
        t0 = time.time()
        cached = oracle.compute(
            input_atom_array=annotated_design,
            example_id="off_target_design",
            out_dir=tmp_path,
            chain_of_interest="A_1",
        )
        assert time.time() - t0 < 1, "cached oracle call took too long, likely re-running"
        assert cached == result
    finally:
        oracle.teardown()


# ---------------------------------------------------------------------------
# Slow GPU integration test — full four-manager oracle pipeline
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.xfail(
    strict=True,
    reason=(
        "AF3 drops intra-chain covale bonds to non-standard residues, so the refold of chain C "
        "of 1a1e (ACE-TYR-GLU-DIY) loses ACE:C--TYR:N and GLU:C--DIY:N1 and "
        "_validate_inter_residue_bonds rejects it. Pre-existing AF3 limitation, not a regression."
    ),
)
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
            "refolded_backbone_rmsd": StructureComparisonMetric(
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
        {"unconditional": PassCriterionMetric(metric_thresholds={"refolded_backbone_rmsd.rmsd": {"max": 2.0}})}
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
    assert "refolded_backbone_rmsd.rmsd" in result
    assert "chain_a_rmsd.rmsd" in result
    assert "unconditional.pass" in result
    assert "unconditional.all_metrics_present" in result
    assert np.isfinite(result["oracle.overall_plddt"])
    assert np.isfinite(result["refolded_backbone_rmsd.rmsd"])
    assert result["unconditional.pass"] in (0.0, 1.0)
    assert result["unconditional.all_metrics_present"] in (0.0, 1.0)

    # Chain A RMSD should be low since chain A was templated
    assert (
        result["chain_a_rmsd.rmsd"] < 2.0
    ), f"Chain A RMSD {result['chain_a_rmsd.rmsd']:.2f} Å exceeds 2.0 Å despite being templated."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
