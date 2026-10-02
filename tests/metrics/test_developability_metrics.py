"""Tests for developability metrics: sequence (GRAVY, instability) and Rosetta scoring."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from atomworks.enums import ChainType
from atomworks.io.parser import parse
from atomworks.io.tools.inference import sequence_to_annotated_atom_array
from biotite.structure import AtomArray, concatenate

from rfproteina.metrics.physics.developability_metrics import (
    RosettaDevelopabilityMetric,
    SequenceDevelopabilityMetric,
    compute_gravy,
    compute_instability_index,
    protein_one_letter_sequence,
    subset_to_designed_protein,
)

# Ubiquitin (1UBQ) — accuracy anchor against published ProtParam values.
_UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

# GRAVY threshold from configs/metrics/pass_criteria: soluble designs pass, membrane fail.
_GRAVY_MAX = 0.1
_SOLUBLE = {
    "ubiquitin_1ubq": _UBQ,
    "lysozyme_6lyz": (
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCND"
        "GRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    ),
}
_MEMBRANE = {
    "rhodopsin_1f88": (
        "MNGTEGPNFYVPFSNKTGVVRSPFEAPQYYLAEPWQFSMLAAYMFLLIMLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTT"
        "TLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLVGWSRYIPEGMQCS"
        "CGIDYYTPHEETNNESFVIYMFVVHFIIPLIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWLPYAGVAFYIFTHQ"
        "GSDFGPIFMTIPAFFAKTSAVYNPVIYIMMNKQFRNCMVTTLCCGKNPLGDDEASTTVSKTETSQVAPA"
    ),
    "aquaporin_z_1rc2": (
        "MFRKLAAECFGTFWLVFGGCGSAVLAAGFPELGIGFAGVALAFGLTVLTMAFAVGHISGGHFNPAVTIGLWAGGRFPAKEVVGYVIAQVVGGI"
        "VAAALLYLIASGKTGFDAAASGFASNGYGEHSPGGYSMLSALVVELVLSAGFLLVIHGATDKFAPAGFAPIAIGLALTLIHLISIPVTNTSVN"
        "PARSTAVAIFQGGWALEQLWFFWVVPIVGGIIGGLIYRTLLEKRD"
    ),
}


def _atom_array_from_sequence(sequence: str, chain_id: str = "A") -> AtomArray:
    """Build an annotated protein AtomArray from a one-letter sequence (via atomworks)."""
    return sequence_to_annotated_atom_array(sequence, chain_id, chain_type=ChainType.POLYPEPTIDE_L)


def test_matches_protparam_ubiquitin():
    """GRAVY/instability match published ProtParam values, catching table typos."""
    assert compute_gravy(_UBQ) == pytest.approx(-0.489, abs=1e-2)
    assert compute_instability_index(_UBQ) == pytest.approx(36.06, abs=0.1)


def test_compute_gravy_normalizes_and_validates():
    assert compute_gravy(" acde ") == pytest.approx(compute_gravy("ACDE"))
    with pytest.raises(ValueError):
        compute_gravy("ACDEZ")


@pytest.mark.parametrize("sequence", _SOLUBLE.values(), ids=_SOLUBLE.keys())
def test_soluble_proteins_pass_gravy(sequence: str):
    assert compute_gravy(sequence) <= _GRAVY_MAX


@pytest.mark.parametrize("sequence", _MEMBRANE.values(), ids=_MEMBRANE.keys())
def test_membrane_proteins_fail_gravy(sequence: str):
    assert compute_gravy(sequence) > _GRAVY_MAX


def test_sequence_metric_reports_only_gravy_and_instability():
    result = SequenceDevelopabilityMetric().compute(_atom_array_from_sequence(_UBQ))
    assert set(result) == {"gravy", "instability_index"}
    assert result["gravy"] == pytest.approx(compute_gravy(_UBQ))
    assert result["instability_index"] == pytest.approx(compute_instability_index(_UBQ))


def test_sequence_metric_empty_is_nan():
    result = SequenceDevelopabilityMetric().compute(AtomArray(0))
    assert np.isnan(result["gravy"]) and np.isnan(result["instability_index"])


def test_subset_to_designed_chain_selects_chain_of_interest():
    complex_arr = concatenate(
        [_atom_array_from_sequence("ACDEFGHIKL", "A"), _atom_array_from_sequence("MNPQRSTVWY", "B")]
    )
    designed = subset_to_designed_protein(complex_arr, chain_of_interest="A")
    assert protein_one_letter_sequence(designed) == "ACDEFGHIKL"


def test_unselected_rosetta_metric_does_not_require_container(monkeypatch):
    monkeypatch.delenv("PYROSETTA_CONTAINER", raising=False)
    metric = RosettaDevelopabilityMetric()
    with pytest.raises(OSError, match="PyRosetta container not set"):
        metric.compute(AtomArray(0))


def test_rosetta_metric_assembles_command_and_reads_json(tmp_path, monkeypatch):
    """The metric shells the apptainer command with the right flags and parses its JSON."""
    container = tmp_path / "pyrosetta.sif"
    score_script = tmp_path / "score.py"
    container.touch()
    score_script.touch()

    def fake_run(command, **kwargs):
        assert {"--cif", "--sasa_cutoff", "--output_json"} <= set(command)
        out_path = command[command.index("--output_json") + 1]
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump({"sap_score": 12.5, "hydrophobic_exposure_sasa": 345.0}, handle)
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "rfproteina.utils.structure_export.write_input_structure_cif",
        lambda _arr, path: path.write_text("data_design\n", encoding="utf-8"),
    )
    monkeypatch.setattr("rfproteina.metrics.physics.developability_metrics.subprocess.run", fake_run)

    metric = RosettaDevelopabilityMetric(
        container=str(container),
        score_script=str(score_script),
        apptainer_bin="apptainer",
        binds=[],
        scratch_dir=str(tmp_path),
    )
    assert metric.compute(_atom_array_from_sequence("ACDE")) == {
        "sap_score": 12.5,
        "hydrophobic_exposure_sasa": 345.0,
    }


# --- Real Rosetta SAP scoring (requires the PyRosetta apptainer; runs on clusters) ---

_ROSETTA_SAP_MAX = 50.0  # configs/metrics/pass_criteria threshold
# Real PDB structures with SAP verified against the container (atomworks parse, full-atom):
# a soluble monomer passes, a hydrophobic membrane protein fails. Values (~): 1pgb 34, 1c3w 334.
_SAP_CASES = {"1pgb": True, "1c3w": False}
_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "developability"
_CONTAINER = os.environ.get("PYROSETTA_CONTAINER")
_HAVE_ROSETTA = shutil.which("apptainer") is not None and _CONTAINER is not None and Path(_CONTAINER).is_file()


@pytest.mark.skipif(not _HAVE_ROSETTA, reason="PyRosetta apptainer/container not available")
@pytest.mark.parametrize("pdb_id, should_pass", _SAP_CASES.items(), ids=_SAP_CASES.keys())
def test_rosetta_sap_pass_fail_on_real_structures(pdb_id: str, should_pass: bool, tmp_path):
    """Soluble monomers pass the SAP threshold; a hydrophobic membrane protein fails.

    Skipped unless the PyRosetta apptainer ($PYROSETTA_CONTAINER) is available.
    """
    atom_array = parse(_DATA_DIR / f"{pdb_id}.cif.gz")["asym_unit"][0]
    metric = RosettaDevelopabilityMetric(scratch_dir=str(tmp_path))
    sap = metric.compute(atom_array)["sap_score"]
    assert (sap <= _ROSETTA_SAP_MAX) == should_pass
