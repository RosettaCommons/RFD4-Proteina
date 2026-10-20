"""The partial-diffusion schedule must start where the state was actually noised."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher, get_schedule
from rfproteina.search.beam_search import BeamSearchStrategy
from rfproteina.search.vanilla import VanillaStrategy, build_partial_schedule, resolve_partial_t

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


@pytest.mark.parametrize("frac", [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00])
@pytest.mark.parametrize("full_nsteps", [None, 100])
def test_schedule_starts_at_the_noised_time(flow_matcher, sampling_args, frac: float, full_nsteps):
    partial_t = resolve_partial_t(
        {"partial_trajectory_fraction_steps": frac, "nsteps": 100}, flow_matcher, sampling_args
    )
    ts, gt = build_partial_schedule(partial_t, round(frac * 100), flow_matcher, sampling_args, full_nsteps=full_nsteps)

    for data_mode, start in partial_t.items():
        assert ts[data_mode][0] == pytest.approx(start, abs=1e-6), (
            f"{data_mode}: integration starts at {float(ts[data_mode][0])} but the state was noised " f"to {start}"
        )
        assert ts[data_mode][-1] == pytest.approx(1.0, abs=1e-6), "a partial trajectory must end at t=1"
        assert len(gt[data_mode]) == len(ts[data_mode]) - 1


class _ConfiguredFlowMatcher:
    data_modes = ["bb_ca", "local_latents"]
    sample_schedule = ProductSpaceFlowMatcher.sample_schedule


@pytest.fixture
def generation_config():
    return OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/generation/inference_base.yaml")


@pytest.mark.parametrize("nsteps", [60, 61, 100, 200])
@pytest.mark.parametrize("frac", [1 / 60, 1 / 3, 0.3333, 0.375, 0.5, 1.0])
def test_fraction_corruption_uses_configured_grid(generation_config, nsteps, frac):
    fm = _ConfiguredFlowMatcher()
    generation_config.args.nsteps = nsteps
    generation_config.args.partial_trajectory_fraction_steps = frac
    ts, _ = fm.sample_schedule(nsteps, generation_config.model)
    partial_t = resolve_partial_t(generation_config.args, fm, generation_config.model)
    start_step = nsteps - round(frac * nsteps)

    for dm in fm.data_modes:
        assert partial_t[dm] == ts[dm][start_step].item()


@pytest.mark.parametrize("tail_steps", [0, 10])
@pytest.mark.parametrize("noise_power", [1.0, 1.5])
@pytest.mark.parametrize("remaining_steps", [1, 2, 20, 21, 28, 60])
def test_fraction_preserves_full_schedule_tail(generation_config, tail_steps, noise_power, remaining_steps):
    fm = _ConfiguredFlowMatcher()
    for args in generation_config.model.values():
        args.schedule.tail_steps = tail_steps
        args.gt.p = noise_power
    full_ts, full_gt = fm.sample_schedule(60, generation_config.model)
    partial_t = {dm: full_ts[dm][-remaining_steps - 1].item() for dm in fm.data_modes}
    ts, gt = build_partial_schedule(partial_t, remaining_steps, fm, generation_config.model, full_nsteps=60)

    for dm in fm.data_modes:
        torch.testing.assert_close(ts[dm], full_ts[dm][-remaining_steps - 1 :], rtol=0, atol=0)
        torch.testing.assert_close(gt[dm], full_gt[dm][-remaining_steps:], rtol=0, atol=0)
        assert ts[dm][0].item() == partial_t[dm]
        assert ts[dm][-1].item() == 1.0
        assert (ts[dm].diff() > 0).all()
        assert torch.isfinite(gt[dm]).all()


def test_fraction_rejects_zero_remaining_steps(generation_config):
    generation_config.args.partial_trajectory_fraction_steps = 0.001
    with pytest.raises(ValueError, match="zero denoising steps"):
        resolve_partial_t(generation_config.args, _ConfiguredFlowMatcher(), generation_config.model)


def test_fraction_rejects_mismatched_corruption_time(generation_config):
    with pytest.raises(ValueError, match="partial corruption time must match"):
        build_partial_schedule(
            {"bb_ca": 0.5, "local_latents": 0.5},
            20,
            _ConfiguredFlowMatcher(),
            generation_config.model,
            full_nsteps=60,
        )


@pytest.mark.parametrize("remaining_steps", [0, 61])
def test_fraction_rejects_invalid_step_count(generation_config, remaining_steps):
    with pytest.raises(ValueError, match="Partial step count must be in"):
        build_partial_schedule({}, remaining_steps, _ConfiguredFlowMatcher(), generation_config.model, full_nsteps=60)


@pytest.mark.parametrize("strategy_name", ["vanilla", "beam"])
def test_fractional_generation_corrupts_at_first_integration_time(generation_config, strategy_name):
    generation_config.args.partial_trajectory_fraction_steps = 1 / 3
    fm = _ConfiguredFlowMatcher()
    full_ts, full_gt = fm.sample_schedule(60, generation_config.model)
    expected_t = {dm: full_ts[dm][40] for dm in fm.data_modes}
    clean = {dm: torch.ones(1, 2, 3) for dm in fm.data_modes}

    def corrupt_batch(batch, t):
        batch["x_t"] = {dm: t[dm][:, None, None] * clean[dm] for dm in fm.data_modes}
        return batch

    def partial_simulation(*, x, ts, gt, nsteps, recycle_depths, **_kwargs):
        assert nsteps == 20
        assert recycle_depths == tuple(range(40, 60))
        for dm in fm.data_modes:
            assert torch.all(x[dm] == expected_t[dm])
            torch.testing.assert_close(ts[dm], full_ts[dm][40:], rtol=0, atol=0)
            torch.testing.assert_close(gt[dm], full_gt[dm][40:], rtol=0, atol=0)
        raise RuntimeError("checked partial initialization")

    fm.corrupt_batch = corrupt_batch
    fm.partial_simulation = partial_simulation
    fm.sample_noise = lambda *_args, **_kwargs: {dm: torch.zeros_like(value) for dm, value in clean.items()}
    fm.apply_flow_emit_masks = lambda x, _batch: x
    model = SimpleNamespace(
        fm=fm,
        inf_cfg=generation_config,
        device=torch.device("cpu"),
        preprocess_batch_for_generation=lambda batch: batch,
        add_clean_samples=lambda batch: {**batch, "x_1": clean},
        predict_for_sampling=lambda *_args, **_kwargs: None,
        recycle_depths=lambda count, full_nsteps: tuple(range(full_nsteps - count, full_nsteps)),
    )
    batch = {
        "coords": torch.zeros(1, 2, 37, 3),
        "coords_nm": torch.zeros(1, 2, 37, 3),
        "padding_mask": torch.ones(1, 2, dtype=torch.bool),
        "supports_local_latent": torch.ones(1, 2, dtype=torch.bool),
        "is_in_context_conditioning": torch.zeros(1, 2, dtype=torch.bool),
    }
    strategy = (
        VanillaStrategy()
        if strategy_name == "vanilla"
        else BeamSearchStrategy(checkpoints={20: {}}, beam_width=1, n_branch=1)
    )
    with pytest.raises(RuntimeError, match="checked partial initialization"):
        strategy.generate(model, batch)
