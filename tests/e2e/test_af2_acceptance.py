"""End-to-end acceptance coverage for the launcher-owned AF2 worker pool."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import biotite.structure as struc
import numpy as np
import pytest
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.query import mask as query_mask
from atomworks.ml.conditions import C_CRD
from biotite.structure import AtomArray
from biotite.structure.residues import get_residue_starts

from rfproteina.metrics.oracles.af2_oracle import AF2Oracle
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput
from rfproteina.utils.standard_annotations import SA_CWI
from tests.conftest import TEST_DATA_DIR
from tests.e2e.helpers.gpu_e2e import ROOT, require_gpu_e2e, require_preval_oracle
from tests.e2e.helpers.regression_baselines import deterministic_env, require_sota_ckpt

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

_COMPLETE_CRYSTAL_FIXTURE = TEST_DATA_DIR / "5vht_from_file.cif"
_BINDER_SPEC = ROOT / "data/benchmarks/cleaned_ppi.yaml"
_SEED = 42
_BACKBONE = frozenset({"N", "CA", "C", "O"})
_PHE_ATOMS = frozenset({"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"})
_TARGET, _BINDER = "A", "B"
_PROTEIN_CA = "is_protein & (atom_name == 'CA')"
_PROTEIN_BACKBONE = "is_protein & (atom_name in ['N', 'CA', 'C', 'O'])"

# Seeded Pallas-on bounds with modest deterministic cross-GPU slack.
_QUALITY_BOUNDS = {
    "monomer": ({"overall_plddt": 52.0, "overall_ptm": 29.0}, {"overall_pae": 16.5}),
    "initial_guess_off": ({"overall_plddt": 54.0, "overall_ptm": 32.0}, {"overall_pae": 16.0}),
    "initial_guess_on": ({"overall_plddt": 58.0, "overall_ptm": 37.0}, {"overall_pae": 15.0}),
}
_PPI_METRIC_BOUNDS = (
    {"overall_plddt": 90.0, "overall_ptm": 85.0, "overall_iptm": 85.0},
    {"overall_pae": 5.0, "ipae_coi": 5.0, "min_ipae_coi": 1.5},
)
_GENERATED_METRIC_BOUNDS = (
    {"overall_plddt": 27.0, "overall_ptm": 9.0, "overall_iptm": 5.0, "plddt_coi": 27.0},
    {"overall_pae": 26.0, "ipae_coi": 30.0, "min_ipae_coi": 23.0},
)
_GENERATED_TARGET_RMSD_MAX = 10.0


class RMSDCheck(NamedTuple):
    """One AF3-style structural check with separate alignment and measurement selections."""

    run: str
    label: str
    align: str
    measure: str
    allowable: float


_RMSD_CHECKS = (
    RMSDCheck("monomer", "monomer", _PROTEIN_CA, _PROTEIN_BACKBONE, 6.5),
    RMSDCheck(
        "ppi",
        "PPI target",
        f"(chain_id == '{_TARGET}') & (atom_name == 'CA')",
        f"(chain_id == '{_TARGET}') & ({_PROTEIN_BACKBONE})",
        0.5,
    ),
    RMSDCheck(
        "ppi",
        "PPI complex in target frame",
        f"(chain_id == '{_TARGET}') & (atom_name == 'CA')",
        _PROTEIN_BACKBONE,
        0.9,
    ),
    RMSDCheck(
        "ppi",
        "PPI binder in target frame",
        f"(chain_id == '{_TARGET}') & (atom_name == 'CA')",
        f"(chain_id == '{_BINDER}') & ({_PROTEIN_BACKBONE})",
        1.2,
    ),
    RMSDCheck(
        "ppi",
        "PPI binder internal",
        f"(chain_id == '{_BINDER}') & (atom_name == 'CA')",
        f"(chain_id == '{_BINDER}') & ({_PROTEIN_BACKBONE})",
        0.9,
    ),
    RMSDCheck("initial_guess_off", "initial guess off", _PROTEIN_CA, _PROTEIN_BACKBONE, 5.5),
    RMSDCheck("initial_guess_on", "initial guess on", _PROTEIN_CA, _PROTEIN_BACKBONE, 5.5),
)


class FoldResult(NamedTuple):
    input: AtomArray
    prediction: AtomArray
    metrics: dict
    summary: dict


class AcceptanceRuns(NamedTuple):
    runs: dict[str, FoldResult]
    manifests: list[list[dict]]


def _preserve_artifacts(root: Path) -> None:
    """Copy fresh E2E artifacts for CI upload without making them a cache root."""
    configured = os.environ.get("AF2_E2E_ARTIFACT_DIR")
    if configured:
        shutil.copytree(root, Path(configured) / root.name, dirs_exist_ok=True)


def _load_protein(path: Path) -> AtomArray:
    atom_array = as_atom_array_plus(
        parse(
            str(path),
            config=ParseConfig.from_preset("minimal", load_standard_annotations=True),
        )["asym_unit"][0]
    )
    ensure_annotations(atom_array, "is_protein")
    atom_array = atom_array[atom_array.is_protein & (atom_array.atom_name != "OXT")]
    assert atom_array.array_length() > 0
    return atom_array


def _load_complete_input(path: Path) -> AtomArray:
    """Load a gap-free, full-backbone protein fixture with finite coordinates."""
    atom_array = _load_protein(path)
    pbf = atom_array.res_name == "PBF"
    atom_array = atom_array[~pbf | np.isin(atom_array.atom_name, list(_PHE_ATOMS))]
    atom_array.res_name[atom_array.res_name == "PBF"] = "PHE"
    assert np.all(np.isfinite(atom_array.coord))
    for chain_id in _chain_order(atom_array):
        chain = atom_array[atom_array.chain_id == chain_id]
        bounds = get_residue_starts(chain, add_exclusive_stop=True)
        residue_ids = chain.res_id[bounds[:-1]]
        assert np.all(np.diff(residue_ids) == 1)
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            assert set(chain.atom_name[start:stop]) >= _BACKBONE
    return atom_array


def _chain_order(atom_array: AtomArray) -> list[str]:
    return list(dict.fromkeys(atom_array.chain_id.tolist()))


def _all_residues_selected(atom_array: AtomArray, mask: np.ndarray) -> bool:
    starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    return all(np.any(mask[start:stop]) for start, stop in zip(starts[:-1], starts[1:], strict=True))


def _canonicalize_chains(atom_array: AtomArray, ordered_original_ids: list[str]) -> AtomArray:
    canonical = atom_array.copy()
    for index, original_id in enumerate(ordered_original_ids):
        canonical.chain_id[atom_array.chain_id == original_id] = chr(ord("A") + index)
    return canonical


def _monomer_fixture() -> AtomArray:
    crystal = _load_complete_input(_COMPLETE_CRYSTAL_FIXTURE)
    first_chain = _chain_order(crystal)[0]
    monomer = _canonicalize_chains(crystal[crystal.chain_id == first_chain], [first_chain])
    C_CRD.set_mask(monomer, np.zeros(monomer.array_length(), dtype=bool))
    return monomer


def _ppi_fixture() -> AtomArray:
    complex_gt = _load_complete_input(_COMPLETE_CRYSTAL_FIXTURE)
    original_ids = _chain_order(complex_gt)
    assert len(original_ids) == 2
    canonical = _canonicalize_chains(complex_gt, original_ids)
    C_CRD.set_mask(canonical, canonical.chain_id == _TARGET)
    canonical_mask = C_CRD.mask(canonical)
    target_atoms = canonical.chain_id == _TARGET
    assert _all_residues_selected(canonical[target_atoms], canonical_mask[target_atoms])
    assert not np.any(canonical_mask[canonical.chain_id == _BINDER])
    return canonical


def _interleaved_identical_sequence_fixture(ppi: AtomArray) -> AtomArray:
    """Return A(X), B(Y), C(X), which AF2 assembles internally as A, C, B."""
    chain_a = ppi[ppi.chain_id == "A"].copy()
    chain_b = ppi[ppi.chain_id == "B"].copy()
    starts = get_residue_starts(chain_b, add_exclusive_stop=True)
    phe_index = next(index for index, start in enumerate(starts[:-1]) if chain_b.res_name[start] == "PHE")
    phe_start, phe_stop = starts[phe_index : phe_index + 2]
    chain_b.res_name[phe_start:phe_stop] = "TYR"
    chain_c = chain_a.copy()
    chain_c.chain_id[:] = "C"
    chain_c.coord += np.asarray([40.0, 0.0, 0.0])
    sequence_a = chain_a.res_name[get_residue_starts(chain_a)].tolist()
    sequence_b = chain_b.res_name[get_residue_starts(chain_b)].tolist()
    sequence_c = chain_c.res_name[get_residue_starts(chain_c)].tolist()
    assert sequence_a == sequence_c
    assert sequence_a != sequence_b
    interleaved = struc.concatenate([chain_a, chain_b, chain_c])
    C_CRD.set_mask(interleaved, interleaved.chain_id == "A")
    return interleaved


def _residue_identity(atom_array: AtomArray) -> list[tuple[str, str]]:
    starts = get_residue_starts(atom_array)
    return [(str(atom_array.chain_id[index]), str(atom_array.res_name[index])) for index in starts]


def _query(atom_array: AtomArray, expression: str) -> np.ndarray:
    """Resolve an atomworks selection, failing rather than scoring an empty subset."""
    ensure_annotations(atom_array, "is_protein")
    selected = query_mask(atom_array, expression)
    assert selected.any(), f"selection matched no atoms: {expression!r}"
    return selected


def _rmsd(reference: AtomArray, subject: AtomArray, align: str, measure: str) -> float:
    """Align and measure matching atoms using the same declarative selections as AF3 tests."""
    assert np.array_equal(subject.atom_name, reference.atom_name)
    assert np.array_equal(subject.chain_id, reference.chain_id)
    assert np.array_equal(subject.res_id, reference.res_id)
    align_mask = _query(reference, align)
    measure_mask = _query(reference, measure)
    _, transform = struc.superimpose(reference[align_mask], subject[align_mask])
    return float(struc.rmsd(reference[measure_mask], transform.apply(subject)[measure_mask]))


def _rmsd_failures(runs: dict[str, FoldResult]) -> list[str]:
    failures = []
    for check in _RMSD_CHECKS:
        result = runs[check.run]
        value = _rmsd(result.input, result.prediction, check.align, check.measure)
        if value >= check.allowable:
            failures.append(f"{check.label}: RMSD {value:.3f} >= {check.allowable}")
    return failures


def _assert_bounds(values: dict, minimum: dict[str, float], maximum: dict[str, float]) -> None:
    for key, bound in minimum.items():
        assert math.isfinite(float(values[key])) and float(values[key]) > bound
    for key, bound in maximum.items():
        assert math.isfinite(float(values[key])) and float(values[key]) < bound


def _fold_results(oracle: AF2Oracle, jobs: list[OracleJobInput]) -> dict[str, FoldResult]:
    assert oracle.fold_batch(jobs) == len(jobs)
    results: dict[str, FoldResult] = {}
    for job in jobs:
        cif = oracle.refolded_cif_path(job.out_dir, job.example_id)
        confidence = oracle.confidence_path(job.out_dir, job.example_id)
        raw_summary = oracle.raw_summary_path(job.out_dir, job.example_id)
        assert cif.is_file(), f"no refolded CIF at {cif}"
        assert confidence.is_file(), f"no confidence JSON at {confidence}"
        assert raw_summary.is_file(), f"no raw AF2 summary at {raw_summary}"
        results[job.example_id] = FoldResult(
            job.atom_array,
            _load_protein(cif),
            json.loads(confidence.read_text()),
            json.loads(raw_summary.read_text()),
        )
    return results


@pytest.fixture(scope="module")
def acceptance_runs(tmp_path_factory: pytest.TempPathFactory) -> AcceptanceRuns:
    """Dispatch all crystal cases through one launcher-owned warm worker pool."""
    require_gpu_e2e()
    require_preval_oracle("af2")
    socket_template = os.environ.get("AF2_LAUNCHER_SOCKET_TEMPLATE")
    assert socket_template, "AF2 acceptance requires the launcher-owned AF2_LAUNCHER_SOCKET_TEMPLATE"

    root = tmp_path_factory.mktemp("af2_acceptance")
    monomer = _monomer_fixture()
    ppi = _ppi_fixture()
    interleaved = _interleaved_identical_sequence_fixture(ppi)
    capture_dir = root / "dispatched_manifests"
    previous_capture_dir = os.environ.get("AF2_MANIFEST_CAPTURE_DIR")
    os.environ["AF2_MANIFEST_CAPTURE_DIR"] = str(capture_dir)
    try:
        # One batch is the primary acceptance request. The initial-guess pair is
        # the repeated dispatch through the same launcher-selected socket.
        primary = AF2Oracle(oracle_name="af2_acceptance", template_query="mask_coordinate_1_atom", seed=_SEED)
        runs = _fold_results(
            primary,
            [
                OracleJobInput(monomer, "monomer", root / "monomer"),
                OracleJobInput(ppi, "ppi", root / "ppi", chain_of_interest=_BINDER),
            ],
        )
        runs.update(
            _fold_results(
                AF2Oracle(oracle_name="af2_initial_guess_off", seed=_SEED, initial_guess=False, pct_seq_mask=0.0),
                [OracleJobInput(monomer, "initial_guess_off", root / "initial_guess_off")],
            )
        )
        runs.update(
            _fold_results(
                AF2Oracle(oracle_name="af2_initial_guess_on", seed=_SEED, initial_guess=True, pct_seq_mask=0.0),
                [
                    OracleJobInput(monomer, "initial_guess_on", root / "initial_guess_on"),
                    OracleJobInput(
                        interleaved,
                        "interleaved_identical_sequence",
                        root / "interleaved_identical_sequence",
                        chain_of_interest="B",
                    ),
                ],
            )
        )
        for index, model_name in enumerate(("model_1_ptm", "model_2_ptm"), start=1):
            result = _fold_results(
                AF2Oracle(oracle_name=f"af2_{model_name}", model_name=model_name, seed=_SEED),
                [OracleJobInput(monomer.copy(), "fixed_model", root / "fixed_model")],
            )
            runs[f"fixed_model_{index}"] = result["fixed_model"]
        manifests = [json.loads(path.read_text()) for path in capture_dir.glob("*-manifest.json")]
        return AcceptanceRuns(runs, manifests)
    finally:
        if previous_capture_dir is None:
            os.environ.pop("AF2_MANIFEST_CAPTURE_DIR", None)
        else:
            os.environ["AF2_MANIFEST_CAPTURE_DIR"] = previous_capture_dir
        _preserve_artifacts(root)


def _assert_parseable_finite_identity(result: FoldResult) -> None:
    assert np.all(np.isfinite(result.prediction.coord))
    assert np.array_equal(result.prediction.atom_name, result.input.atom_name)
    assert np.array_equal(result.prediction.chain_id, result.input.chain_id)
    assert np.array_equal(result.prediction.res_id, result.input.res_id)
    assert np.array_equal(result.prediction.res_name, result.input.res_name)


def test_af2_crystal_acceptance(acceptance_runs) -> None:
    """A warm AF2 pool refolds seeded monomer and target-templated PPI inputs."""
    runs = acceptance_runs.runs
    target, binder = _TARGET, _BINDER
    monomer = runs["monomer"]
    ppi = runs["ppi"]
    interleaved = runs["interleaved_identical_sequence"]

    assert len(acceptance_runs.manifests) == 5
    primary_manifest = next(
        manifest
        for manifest in acceptance_runs.manifests
        if {entry["name"] for entry in manifest} == {"monomer", "ppi"}
    )
    dispatched = {entry["name"]: entry for entry in primary_manifest}
    assert "template" not in dispatched["monomer"]
    assert list(dispatched["ppi"]["template"]) == [target]
    assert binder not in dispatched["ppi"]["template"]
    template = dispatched["ppi"]["template"][target]
    target_residue_count = len(get_residue_starts(ppi.input[ppi.input.chain_id == target]))
    assert template["mmcif"]
    assert template["query_indices"] == template["template_indices"] == list(range(target_residue_count))

    _assert_parseable_finite_identity(monomer)
    _assert_bounds(monomer.metrics, *_QUALITY_BOUNDS["monomer"])

    ppi_template_chains = [
        chain for chain in _chain_order(ppi.input) if np.any(C_CRD.mask(ppi.input)[ppi.input.chain_id == chain])
    ]
    assert ppi_template_chains == [target]
    assert binder not in ppi_template_chains
    _assert_parseable_finite_identity(ppi)
    _assert_bounds(ppi.metrics, *_PPI_METRIC_BOUNDS)

    binder_index = ppi.summary["chain_ids"].index(binder)
    target_index = ppi.summary["chain_ids"].index(target)
    assert float(ppi.summary["chain_plddt"][target_index]) > 0.4
    pair_min = ppi.summary["chain_pair_pae_min"]
    expected_min_ipae = (
        float(pair_min[binder_index][target_index]) + float(pair_min[target_index][binder_index])
    ) / 2.0
    assert ppi.metrics["min_ipae_coi"] == pytest.approx(expected_min_ipae)
    pair_iptm = ppi.summary["chain_pair_iptm"]
    expected_iptm = (float(pair_iptm[binder_index][target_index]) + float(pair_iptm[target_index][binder_index])) / 2.0
    assert ppi.metrics["iptm_coi"] == pytest.approx(100.0 * expected_iptm)
    assert float(ppi.metrics["iptm_coi"]) > 0.0
    _assert_parseable_finite_identity(interleaved)
    assert interleaved.summary["chain_ids"] == ["A", "C", "B"]
    assert interleaved.metrics["chain_of_interest"] == "B"
    binder_index = interleaved.summary["chain_ids"].index("B")
    assert interleaved.metrics["plddt_coi"] == pytest.approx(
        100.0 * float(interleaved.summary["chain_plddt"][binder_index])
    )
    assert interleaved.metrics["ptm_coi"] == pytest.approx(
        100.0 * float(interleaved.summary["chain_ptm"][binder_index])
    )
    assert interleaved.metrics["pae_coi"] == pytest.approx(float(interleaved.summary["chain_pae"][binder_index]))
    pair_iptm = np.asarray(interleaved.summary["chain_pair_iptm"])
    other_indices = np.arange(len(pair_iptm)) != binder_index
    expected_iptm = np.mean((pair_iptm[binder_index, other_indices] + pair_iptm[other_indices, binder_index]) / 2.0)
    assert interleaved.metrics["iptm_coi"] == pytest.approx(100.0 * expected_iptm)

    for name in ("initial_guess_off", "initial_guess_on"):
        result = runs[name]
        _assert_parseable_finite_identity(result)
        _assert_bounds(result.metrics, *_QUALITY_BOUNDS[name])

    initial_guess_off = runs["initial_guess_off"]
    initial_guess_on = runs["initial_guess_on"]
    assert initial_guess_off.prediction.coord.shape == initial_guess_on.prediction.coord.shape
    assert np.max(np.abs(initial_guess_on.prediction.coord - initial_guess_off.prediction.coord)) > 1e-6
    assert float(initial_guess_on.metrics["overall_plddt"]) > float(initial_guess_off.metrics["overall_plddt"])
    assert float(initial_guess_on.metrics["overall_pae"]) < float(initial_guess_off.metrics["overall_pae"])
    assert float(initial_guess_on.metrics["overall_ptm"]) > float(initial_guess_off.metrics["overall_ptm"])

    results = [acceptance_runs.runs[f"fixed_model_{index}"] for index in (1, 2)]
    fixed_manifests = [
        manifest for manifest in acceptance_runs.manifests if {entry["name"] for entry in manifest} == {"fixed_model"}
    ]
    assert sorted(manifest[0]["model_name"] for manifest in fixed_manifests) == ["model_1_ptm", "model_2_ptm"]
    assert [result.metrics["af2_model_name"] for result in results] == ["model_1_ptm", "model_2_ptm"]
    assert [result.summary["af2_model_name"] for result in results] == ["model_1_ptm", "model_2_ptm"]
    assert np.max(np.abs(results[0].prediction.coord - results[1].prediction.coord)) > 1e-6
    for result in results:
        _assert_parseable_finite_identity(result)

    assert not (failures := _rmsd_failures(runs)), "\n".join(failures)


@pytest.fixture(scope="module")
def generated_binder_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path, dict]:
    """Generate one 50-residue binder and refold it through the warm AF2 daemon."""
    require_preval_oracle("af2")
    require_sota_ckpt()
    assert os.environ.get(
        "AF2_LAUNCHER_SOCKET_TEMPLATE"
    ), "generated-binder acceptance requires the launcher-owned AF2_LAUNCHER_SOCKET_TEMPLATE"
    root = tmp_path_factory.mktemp("af2_acceptance") / "generated_binder"
    root.mkdir(parents=True, exist_ok=True)
    manifest_capture_dir = root / "dispatched_manifests"
    command = [
        sys.executable,
        str(ROOT / "src/rfproteina/generate.py"),
        f"inputs={_BINDER_SPEC}",
        "subset_to_examples=rfd3_pdl1",
        "+generation.dataset.specification.contig='50,/0,A17-131'",
        "num_replicates=1",
        "batch_size=1",
        f"out_dir={root}",
        "run_name_=af2_task5_generated_binder",
        f"seed={_SEED}",
        "generation.args.nsteps=20",
        "metrics_tags=[ppi_refolding,hotspot_conditioning]",
        "metrics/refolding_oracles=af2",
        "prevalidate_inputs=true",
        "skip_existing=false",
        "compile_lengths=null",
        "num_workers=0",
        "+single=true",
        f"hydra.run.dir={root / 'hydra_outputs'}",
    ]
    env = deterministic_env()
    env["CCD_MIRROR_PATH"] = ""
    env["PDB_MIRROR_PATH"] = ""
    env["AF2_MANIFEST_CAPTURE_DIR"] = str(manifest_capture_dir)
    try:
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, f"generated-binder inference failed:\n{completed.stderr or completed.stdout}"

        generated = list(root.glob("inference_samples/**/*_generated.cif"))
        refolded = list(root.glob("inference_samples/**/*_af2_ppi_refolded.cif"))
        confidences = list(root.glob("inference_samples/**/*_af2_ppi_confidences.json"))
        manifests = list(manifest_capture_dir.glob("*-manifest.json"))
        assert len(generated) == len(refolded) == len(confidences) == len(manifests) == 1
        return generated[0], refolded[0], manifests[0], json.loads(confidences[0].read_text())
    finally:
        _preserve_artifacts(root)


def test_generated_binder_flows_through_af2_metric(generated_binder_run) -> None:
    """The release generator conditions one target and refolds one 50-residue binder."""
    generated_path, refolded_path, manifest_path, metrics = generated_binder_run
    generated = _load_protein(generated_path)
    refolded = _load_protein(refolded_path)
    assert np.all(np.isfinite(generated.coord))
    assert np.all(np.isfinite(refolded.coord))
    coordinate_mask = C_CRD.mask(generated)
    conditioned = [chain for chain in _chain_order(generated) if np.any(coordinate_mask[generated.chain_id == chain])]
    unconditioned = [chain for chain in _chain_order(generated) if chain not in conditioned]
    assert len(conditioned) == len(unconditioned) == 1
    target, binder = conditioned[0], unconditioned[0]
    assert _all_residues_selected(
        generated[generated.chain_id == target], coordinate_mask[generated.chain_id == target]
    )
    assert not np.any(coordinate_mask[generated.chain_id == binder])
    assert len(get_residue_starts(generated[generated.chain_id == binder])) == 50

    dispatched_manifest = json.loads(manifest_path.read_text())
    assert len(dispatched_manifest) == 1
    assert list(dispatched_manifest[0]["template"]) == [target]
    assert binder not in dispatched_manifest[0]["template"]

    assert _chain_order(refolded) == _chain_order(generated)
    assert _residue_identity(refolded) == _residue_identity(generated)
    conditioned_reference = generated.copy()
    conditioned_coords = SA_CWI.annotation(generated)
    if conditioned_coords.ndim == 3:
        conditioned_coords = conditioned_coords[0]
    target_atoms = generated.chain_id == target
    assert np.all(np.isfinite(conditioned_coords[target_atoms]))
    conditioned_reference.coord[target_atoms] = conditioned_coords[target_atoms]
    target_backbone_rmsd = _rmsd(
        conditioned_reference,
        refolded,
        f"(chain_id == '{target}') & (atom_name == 'CA')",
        f"(chain_id == '{target}') & ({_PROTEIN_BACKBONE})",
    )
    assert target_backbone_rmsd < _GENERATED_TARGET_RMSD_MAX

    for key in ("overall_plddt", "overall_ptm", "overall_iptm", "plddt_coi"):
        assert 0.0 <= float(metrics[key]) <= 100.0
    for key in ("overall_pae", "ipae_coi", "min_ipae_coi"):
        assert float(metrics[key]) >= 0.0
    assert metrics["chain_of_interest"] == f"{binder}_1"
    _assert_bounds(metrics, *_GENERATED_METRIC_BOUNDS)
