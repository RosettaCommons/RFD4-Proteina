"""Unit tests for Proteina.recycle_depths, including partial-diffusion trajectories."""

from types import SimpleNamespace

import pytest

from rfproteina.proteina import Proteina

# midpeak6_50: [width, depth] bands summing to 50 steps.
SCHEDULE = [[5, 1], [5, 2], [5, 3], [5, 5], [10, 6], [5, 5], [5, 3], [5, 2], [5, 1]]


def _model(inf_cfg: dict, recycle_enabled: bool = True) -> Proteina:
    model = Proteina.__new__(Proteina)
    model.inf_cfg = inf_cfg
    model.nn = SimpleNamespace(recycle_enabled=recycle_enabled)
    return model


def test_full_trajectory_expands_whole_schedule():
    depths = _model({"n_recycle_schedule": SCHEDULE}).recycle_depths(50, full_nsteps=50)
    assert len(depths) == 50
    assert (depths[0], depths[20], depths[-1]) == (1, 6, 1)  # first band, peak band, last band


def test_partial_diffusion_takes_schedule_tail():
    """Partial diffusion runs the final `nsteps` of the trajectory, so recycle_depths returns the
    tail of the full schedule -- it must NOT raise on the length mismatch (the bug this fixes)."""
    model = _model({"n_recycle_schedule": SCHEDULE})
    full = model.recycle_depths(50, full_nsteps=50)
    partial = model.recycle_depths(15, full_nsteps=50)
    assert len(partial) == 15
    assert partial == full[-15:]


def test_full_nsteps_defaults_to_nsteps():
    model = _model({"n_recycle_schedule": SCHEDULE})
    assert model.recycle_depths(50) == model.recycle_depths(50, full_nsteps=50)


def test_schedule_length_must_match_full_nsteps():
    with pytest.raises(ValueError, match="expands to 50 steps; expected 40"):
        _model({"n_recycle_schedule": SCHEDULE}).recycle_depths(30, full_nsteps=40)


def test_no_schedule_uses_constant_depth_per_step_run():
    assert _model({"n_recycle": 2}).recycle_depths(15) == (2,) * 15


def test_schedule_requires_recycling_checkpoint():
    model = _model({"n_recycle_schedule": SCHEDULE}, recycle_enabled=False)
    with pytest.raises(ValueError, match="requires a recycling checkpoint"):
        model.recycle_depths(50, full_nsteps=50)
