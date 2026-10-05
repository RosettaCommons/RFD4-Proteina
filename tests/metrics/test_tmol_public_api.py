"""Public TMol API contracts and preservation of constrained minimization."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

pytest.importorskip("tmol")

from tmol.io import build_context_from_biotite  # noqa: E402
from tmol.optimization import run_cart_min, run_kin_min  # noqa: E402

from rfproteina.metrics.physics import tmol_block_pair_metric as metric_module  # noqa: E402
from rfproteina.metrics.physics import tmol_ddg_core as core  # noqa: E402


@pytest.mark.parametrize("mode", ["cartesian", "torsions", "shell"])
def test_relax_preserves_selected_degrees_of_freedom(monkeypatch, mode):
    pose, score, forest, mask, result = (object() for _ in range(5))
    moves = SimpleNamespace(coord_mask=mask)
    calls = []
    selected_shell = []
    moves.set_move_all_sc_tors_for_blocks = selected_shell.append
    monkeypatch.setattr(core, "CartesianMoveMap", lambda: moves)
    monkeypatch.setattr(core.MoveMap, "from_pose_stack", lambda _: moves)
    monkeypatch.setattr(core.FoldForest, "reasonable_fold_forest", lambda _: forest)
    monkeypatch.setattr(core, "PackerPalette", object)

    def cartesian(*args, **kwargs):
        inspect.signature(run_cart_min).bind(*args, **kwargs)
        assert args == (pose, score)
        assert kwargs["coord_mask"] is mask
        calls.append("cartesian")
        return result

    def torsions(*args, **kwargs):
        inspect.signature(run_kin_min).bind(*args, **kwargs)
        assert args == (pose, score, forest, moves)
        calls.append("torsions")
        return result

    def relax(actual_pose, actual_score, palette, move_map, fold_forest, *, min_fn, **kwargs):
        assert actual_pose is pose and actual_score is score
        assert move_map is moves and fold_forest is forest
        return min_fn(pose, score, fold_forest=forest, move_map=moves, verbose=False)

    monkeypatch.setattr(core, "run_cart_min", cartesian)
    monkeypatch.setattr(core, "run_kin_min", torsions)
    monkeypatch.setattr(core, "fast_relax", relax)
    if mode == "shell":
        actual = core.relax_shell(pose, score, mask)
        assert selected_shell == [mask]
        assert not hasattr(moves, "move_all_jumps")
    else:
        actual = core.relax_pose_stack(pose, score, cartesian=mode == "cartesian")
        if mode == "torsions":
            assert moves.move_all_jumps and moves.move_all_named_torsions
    assert actual is result
    assert calls == ["cartesian" if mode == "cartesian" else "torsions"]


def test_ligand_context_matches_public_constructor(monkeypatch):
    atoms = AtomArray(1)
    atoms.res_name = np.array(["ALA"])
    database, scorer = object(), object()
    context = SimpleNamespace(parameter_database=database)
    calls = []

    def build(*args, **kwargs):
        bound = inspect.signature(build_context_from_biotite).bind(*args, **kwargs)
        assert args[0] is not atoms
        np.testing.assert_array_equal(args[0].coord, atoms.coord)
        assert args[0].get_annotation_categories() == atoms.get_annotation_categories()
        for name in atoms.get_annotation_categories():
            np.testing.assert_array_equal(args[0].get_annotation(name), atoms.get_annotation(name))
        assert args[1] == torch.device("cpu")
        assert bound.arguments["prepare_ligands"] is True
        assert bound.arguments["strict_ligands"] is True
        assert bound.arguments["ligand_seed"] == 0
        calls.append(bound)
        return context

    def score_function(device, *, use_soft_rep, param_db):
        assert device == torch.device("cpu")
        assert param_db is database
        return scorer

    monkeypatch.setattr(metric_module, "build_context_from_biotite", build)
    monkeypatch.setattr(metric_module, "create_score_function", score_function)
    metric = metric_module.TMolBlockPairMetric(device="cpu", reuse_context=False, ligand_seed=0)
    metric._ensure_context_and_sfxn(atoms)
    assert metric._context is context and metric._sfxn is scorer
    assert len(calls) == 1


def test_relax_builds_outside_inference_mode(monkeypatch):
    metric = metric_module.TMolBlockPairMetric(device="cpu", relax="fast_relax")

    def compute(atoms):
        assert torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        assert not torch.zeros(1).is_inference()
        return {"interface_ddg": 0.0}

    monkeypatch.setattr(metric, "_compute", compute)
    with torch.inference_mode():
        assert metric.compute(object()) == {"interface_ddg": 0.0}
