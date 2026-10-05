"""Per-design noise streams match serial sampling without changing global RNG."""

import pytest
import torch

from rfproteina.flow_matching.rdn_flow_matcher import RDNFlowMatcher
from rfproteina.utils.sample_rng import independent_sample_rng, sample_randn, sample_randn_like, sample_rng_states


def state_after_preprocessing(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.randn(7)
        return torch.get_rng_state()


def rollout(group_ids):
    batch_size, length = group_ids.shape
    mask = torch.ones(batch_size, length, dtype=torch.bool)
    # Encoder latents consume random numbers even when their emit mask is empty.
    scale = torch.zeros(batch_size, length, 32)[..., 16:]
    encoded = sample_randn_like(scale)
    matcher = RDNFlowMatcher(
        dim=3,
        zero_com_noise=True,
        stochastic_centering_mode="x_0",
        stochastic_centering_scale=0.5,
        stochastic_centering_scope="per_group",
        stochastic_centering_group_keys=["chain_iid"],
    )
    x = matcher.sample_noise(
        length,
        torch.device("cpu"),
        shape=(batch_size,),
        emit_mask=mask,
        training=False,
        groups={"chain_iid": group_ids},
    )
    latent = RDNFlowMatcher(dim=16).sample_noise(length, torch.device("cpu"), shape=(batch_size,), emit_mask=mask)
    params = {
        "sampling_mode": "vf_tsr",
        "sc_scale_noise": 0.3,
        "sc_scale_score": 1.0,
        "t_lim_ode": 0.9,
        "t_lim_ode_below": 0.1,
        "center_every_step": True,
    }
    for time in (0.3, 0.6):
        x = matcher.simulation_step(
            x, {"v": torch.ones_like(x)}, torch.full((batch_size,), time), 0.05, 0.2, mask, params
        )
    return encoded, x, latent


@pytest.mark.parametrize("order,chunks", [([0, 1, 2], [3]), ([0, 1, 2], [2, 1]), ([2, 0, 1], [1, 2])])
def test_encoder_centering_and_tsr_match_serial_streams(order, chunks):
    states = [state_after_preprocessing(seed) for seed in (0, 41, 99)]
    groups = torch.tensor([[0] * 8, [1] * 4 + [2] * 4, [3] * 2 + [4] * 6])
    expected = []
    for i in range(3):
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(states[i])
            expected.append(rollout(groups[i : i + 1]))
    offset = 0
    global_before = torch.get_rng_state()
    for size in chunks:
        ids = order[offset : offset + size]
        with independent_sample_rng([states[i] for i in ids], "cpu"):
            actual = rollout(groups[ids])
        for row, original in enumerate(ids):
            for got, want in zip(actual, expected[original], strict=True):
                torch.testing.assert_close(got[row : row + 1], want, rtol=0, atol=0)
        offset += size
    assert torch.equal(torch.get_rng_state(), global_before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_randn_like_matches_native_singleton_layout(dtype, strided):
    tensor = torch.zeros(1, 8, 32, dtype=dtype)
    tensor = tensor[..., 16:] if strided else tensor
    state = state_after_preprocessing(12)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(state)
        expected = torch.randn_like(tensor)
    with independent_sample_rng([state], "cpu"):
        actual = sample_randn_like(tensor)
    assert torch.equal(actual, expected)


def test_scope_restores_after_error_and_nested_context():
    before = torch.get_rng_state()
    state = state_after_preprocessing(9)
    with pytest.raises(RuntimeError, match="intentional"), independent_sample_rng([state], "cpu"):
        sample_randn((1, 8, 3), device="cpu")
        raise RuntimeError("intentional")
    assert torch.equal(torch.get_rng_state(), before)
    with independent_sample_rng([state], "cpu"):
        first = sample_randn((1, 8, 3), device="cpu")
        with independent_sample_rng([state], "cpu"):
            assert torch.equal(sample_randn((1, 8, 3), device="cpu"), first)
        second = sample_randn((1, 8, 3), device="cpu")
    with independent_sample_rng([state], "cpu"):
        sample_randn((1, 8, 3), device="cpu")
        assert torch.equal(sample_randn((1, 8, 3), device="cpu"), second)


def test_mismatched_rows_fail_before_consuming_noise():
    with (
        independent_sample_rng([state_after_preprocessing(1)], "cpu"),
        pytest.raises(ValueError, match="leading dimension"),
    ):
        sample_randn((2, 8, 3), device="cpu")


@pytest.mark.parametrize("states", [[], [torch.zeros(3)], [torch.zeros(2, 3, dtype=torch.uint8)]])
def test_invalid_states_fail(states):
    with pytest.raises(ValueError), independent_sample_rng(states, "cpu"):
        pass


def test_final_states_continue_reward_noise_and_are_independent_snapshots():
    states = [state_after_preprocessing(seed) for seed in (5, 7)]
    with independent_sample_rng(states, "cpu"):
        sample_randn((2, 16, 3), device="cpu")
        final_states = sample_rng_states()
        frozen = [state.clone() for state in final_states]
        sample_randn((2, 16, 3), device="cpu")
        assert all(torch.equal(a, b) for a, b in zip(final_states, frozen, strict=True))
    for initial, final in zip(states, final_states, strict=True):
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initial)
            torch.randn(1, 16, 3)
            expected_reward_noise = torch.randn(5)
            torch.set_rng_state(final)
            assert torch.equal(torch.randn(5), expected_reward_noise)
    with pytest.raises(RuntimeError, match="require independent_sample_rng"):
        sample_rng_states()
