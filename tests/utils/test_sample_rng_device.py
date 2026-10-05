"""CUDA aliases resolve once before constructing per-design generators."""

import pytest
import torch

from rfproteina.utils.sample_rng import independent_sample_rng, sample_randn_like, sample_rng_states


@pytest.mark.parametrize("selector", ["cuda", torch.device("cuda")])
def test_unindexed_cuda_is_resolved_before_generator_construction(monkeypatch, selector):
    cpu_generator = torch.Generator
    calls = []
    current_calls = []

    def current_device():
        current_calls.append(True)
        return 2

    def generator(*, device):
        calls.append(device)
        # Real CPU state validation, without requiring a GPU for argument routing.
        return cpu_generator(device="cpu")

    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch, "Generator", generator)
    with independent_sample_rng([torch.get_rng_state(), torch.get_rng_state()], selector):
        assert len(sample_rng_states()) == 2
    assert calls == [torch.device("cuda:2"), torch.device("cuda:2")]
    assert len(current_calls) == 1


def test_explicit_cuda_index_is_preserved(monkeypatch):
    cpu_generator = torch.Generator
    calls = []

    def unexpected_current_device():
        pytest.fail("An explicit CUDA device must not resolve the current device")

    def generator(*, device):
        calls.append(device)
        return cpu_generator(device="cpu")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected_current_device)
    monkeypatch.setattr(torch, "Generator", generator)
    with independent_sample_rng([torch.get_rng_state()], "cuda:3"):
        pass
    assert calls == [torch.device("cuda:3")]


def test_cpu_rng_does_not_query_cuda_and_preserves_global_state(monkeypatch):
    def unexpected_current_device():
        pytest.fail("CPU random streams must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected_current_device)
    before = torch.get_rng_state().clone()
    with independent_sample_rng([before], "cpu"):
        first = sample_randn_like(torch.empty(1, 8))
    with independent_sample_rng([before], torch.device("cpu")):
        second = sample_randn_like(torch.empty(1, 8))
    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA generator devices")
def test_cuda_alias_and_explicit_device_have_identical_streams():
    device = torch.device("cuda", torch.cuda.current_device())
    before = torch.cuda.get_rng_state(device).clone()
    tensor = torch.empty(2, 8, device=device)
    with independent_sample_rng([before, before], "cuda"):
        implicit = sample_randn_like(tensor)
        implicit_states = sample_rng_states()
    with independent_sample_rng([before, before], device):
        explicit = sample_randn_like(tensor)
        explicit_states = sample_rng_states()
    assert torch.equal(implicit, explicit)
    assert all(torch.equal(a, b) for a, b in zip(implicit_states, explicit_states, strict=True))
    assert torch.equal(torch.cuda.get_rng_state(device), before)
