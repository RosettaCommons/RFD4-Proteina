"""The partial-diffusion schedule must start where the state was actually noised."""

import pytest
import torch

from rfproteina.flow_matching.product_space_flow_matcher import get_schedule
from rfproteina.search.vanilla import build_partial_schedule, resolve_partial_t

# Matches configs/generation/inference_base.yaml: bb_ca is a log schedule, local_latents a power one.
SCHEDULES = {"bb_ca": ("log", 2.0), "local_latents": ("power", 2.0)}


class _FlowMatcher:
    """Just enough ProductSpaceFlowMatcher to build schedules."""

    data_modes = list(SCHEDULES)

    def sample_schedule(self, nsteps: int, sampling_model_args: dict | None = None) -> tuple[dict, dict]:
        ts = {name: get_schedule(mode, nsteps, p1=p) for name, (mode, p) in SCHEDULES.items()}
        return ts, {name: torch.ones(nsteps) for name in SCHEDULES}


@pytest.fixture
def flow_matcher():
    return _FlowMatcher()


@pytest.fixture
def sampling_args():
    return {name: {"schedule": {"mode": mode, "p": p}} for name, (mode, p) in SCHEDULES.items()}


# Small fractions are the regression: `actual_nsteps = round(frac * nsteps)`, so frac=0.05 runs 5
# steps, and the old `min(pt, 1 - 1/nsteps)` clamp moved the schedule start to 0.80 while
# `corrupt_batch` had noised the state to t=0.997. The integrator was then told the state held 20%
# noise when it held 0.26%, and flattened the coordinate-conditioned regions on the first update.
@pytest.mark.parametrize("frac", [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00])
def test_schedule_starts_at_the_noised_time(flow_matcher, sampling_args, frac: float):
    partial_t = resolve_partial_t({"partial_trajectory_fraction_steps": frac}, flow_matcher, sampling_args)
    ts, gt = build_partial_schedule(partial_t, max(round(frac * 100), 1), flow_matcher, sampling_args)

    for data_mode, start in partial_t.items():
        assert ts[data_mode][0] == pytest.approx(start, abs=1e-6), (
            f"{data_mode}: integration starts at {float(ts[data_mode][0])} but the state was noised " f"to {start}"
        )
        assert ts[data_mode][-1] == pytest.approx(1.0, abs=1e-6), "a partial trajectory must end at t=1"
        assert len(gt[data_mode]) == len(ts[data_mode]) - 1
