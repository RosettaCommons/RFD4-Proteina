"""Tests for the TMol block-pair metric (ligand params prepared from AtomArray)."""

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from biotite.structure import AtomArray

# TMol is optional; skip rather than fail collection for the whole suite when absent.
pytest.importorskip("tmol")

from rfproteina.metrics.physics import tmol_block_pair_metric  # noqa: E402
from rfproteina.metrics.physics.tmol_block_pair_metric import TMolBlockPairMetric  # noqa: E402
from rfproteina.metrics.physics.tmol_ddg_core import atom_sel_to_block_mask, create_score_function  # noqa: E402

SAM_BINDER_CIF = Path(__file__).resolve().parents[1] / "data" / "sam_binder_generated_minimized.cif"


@pytest.fixture
def sam_binder_atom_array():
    if not SAM_BINDER_CIF.is_file():
        pytest.skip(f"Missing fixture: {SAM_BINDER_CIF}")
    config = ParseConfig(model=1, build_assembly=None, add_missing_atoms=False)
    return parse(str(SAM_BINDER_CIF), config=config)["asym_unit"][0]


def test_selected_dropped_residue_raises():
    atom_array = AtomArray(2)
    atom_array.chain_id = np.array(["A", "A"])
    atom_array.res_id = np.array([1, 2])
    atom_array.res_name = np.array(["ALA", "SAM"])
    atom_array.atom_name = np.array(["CA", "C1"])

    pose_stack = SimpleNamespace(
        block_type_ind=torch.tensor([[0]]),
        max_n_blocks=1,
        pdb_info=SimpleNamespace(
            chain_labels=np.array([["A"]]),
            residue_labels=np.array([[1]]),
            residue_insertion_codes=np.array([[""]]),
        ),
    )

    with pytest.raises(ValueError, match=r"1 selected residue\(s\) dropped"):
        atom_sel_to_block_mask(
            pose_stack,
            atom_array,
            np.ones(2, dtype=bool),
            torch.device("cpu"),
        )


def test_score_function_requires_parameter_database():
    with pytest.raises(ValueError, match="param_db is required"):
        create_score_function(torch.device("cpu"), param_db=None)


def test_cuda_device_requires_available_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="set device='cpu'"):
        TMolBlockPairMetric(device="cuda")


@pytest.mark.parametrize(("cuda_version", "extra"), [("12.9", "cu12"), ("13.2", "cu13")])
def test_cuda_device_requires_matching_tmol_operators(monkeypatch, cuda_version, extra):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(tmol_block_pair_metric, "_precompiled_tmol_scoring_available", lambda: False)

    with pytest.raises(RuntimeError, match=rf"tmol\[{extra}\]"):
        TMolBlockPairMetric(device="cuda")


def test_cuda_device_reports_tmol_loader_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(
        tmol_block_pair_metric,
        "_precompiled_tmol_scoring_available",
        Mock(side_effect=RuntimeError("wrong CUDA major")),
    )

    with pytest.raises(RuntimeError, match=r"tmol\[cu13\]"):
        TMolBlockPairMetric(device="cuda")


def test_cuda_device_normalizes_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(tmol_block_pair_metric, "_precompiled_tmol_scoring_available", lambda: True)

    assert TMolBlockPairMetric(device="cuda").device == torch.device("cuda:2")


def test_explicit_cpu_does_not_probe_cuda_operators(monkeypatch):
    unexpected = Mock(side_effect=AssertionError("explicit CPU must not load CUDA operators"))
    monkeypatch.setattr(tmol_block_pair_metric, "_precompiled_tmol_scoring_available", unexpected)

    metric = TMolBlockPairMetric(device="cpu")

    assert metric.device == torch.device("cpu")
    unexpected.assert_not_called()


@pytest.mark.gpu
def test_sam_binder_block_pair_interface_metrics(sam_binder_atom_array):
    metric = TMolBlockPairMetric(
        region_a="res_name == 'SAM'",
        region_b="chain_id == 'A'",
        device="cuda",
        no_opt_h=False,
        term_names=["hbond"],
        term_cutoffs={"n_interface_hbonds": ("hbond", -0.3)},
    )
    metric.setup()
    out = metric.compute(predicted_atom_array=sam_binder_atom_array)
    metric.teardown()

    # Ligand prep can vary protonation/conformer details; assert scoring invariants only.
    assert math.isfinite(out["interface_ddg"])
    assert out["n_interface_hbonds"] >= 1
    assert out["hbond"] < 0.0


@pytest.mark.parametrize(
    ("has_cuda", "operators", "expected"),
    [
        (False, SimpleNamespace(), False),
        (True, SimpleNamespace(ljlk_pose_scores=Mock(), ljlk_rotamer_scores=Mock()), True),
        (True, SimpleNamespace(ljlk_pose_scores=Mock(), ljlk_rotamer_scores=None), False),
    ],
)
def test_precompiled_tmol_scoring_probe(monkeypatch, has_cuda, operators, expected):
    cpp_lib = SimpleNamespace(_ensure_loaded=Mock(), has_cuda=lambda: has_cuda)
    import_module = Mock(side_effect=[cpp_lib, operators])
    monkeypatch.setattr(tmol_block_pair_metric.importlib, "import_module", import_module)

    assert tmol_block_pair_metric._precompiled_tmol_scoring_available() is expected
    assert import_module.call_args_list[0].args == ("tmol._cpp_lib",)
    assert import_module.call_count == (1 if not has_cuda else 2)


@pytest.mark.gpu
def test_sam_binder_fast_relax_block_pair_interface_metrics(sam_binder_atom_array):
    metric = TMolBlockPairMetric(
        region_a="res_name == 'SAM'",
        region_b="chain_id == 'A'",
        device="cuda",
        relax="fast_relax",
        no_opt_h=False,
        term_names=["hbond"],
        term_cutoffs={"n_interface_hbonds": ("hbond", -0.3)},
    )
    metric.setup()
    out = metric.compute(predicted_atom_array=sam_binder_atom_array)
    metric.teardown()

    assert math.isfinite(out["interface_ddg"])
    assert out["n_interface_hbonds"] >= 1
    assert out["hbond"] < 0.0
