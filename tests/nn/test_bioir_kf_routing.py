"""CPU routing contracts; these do not qualify KF numerics or GPU dispatch."""

import pytest
import torch
from torch import nn

from rfproteina.nn import bioir, execution
from rfproteina.nn.bioir import _KFTriangle, _select_triangle, _SharedTriangle
from rfproteina.nn.execution import ALIGNED_LENGTHS, BIOIR_LENGTHS, inference_padding_buckets, resolve_backends


class Native(nn.Module):
    def forward(self, pair, mask):
        return pair + 2


class Candidate(nn.Module):
    def __init__(self, available=True, decline=False, error=False):
        super().__init__()
        self.kf_available = available
        self.decline = decline
        self.error = error

    def set_token_padding(self, enabled):
        assert enabled

    def try_kf_forward(self, pair, mask, trimul_metadata, residual=False):
        assert not residual
        if self.error:
            raise RuntimeError("kernel failure")
        return None if self.decline else pair + 3


def policy():
    return {
        "inference": {name: "bioir" for name in ("flow", "encoder", "decoder")},
        "bioir": {},
    }


@pytest.fixture(autouse=True)
def avoid_compiling_mock_modules(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: module)


def test_bioir_flow_automatically_uses_aligned_padding():
    assert execution.BIOIR_AUTOMATIC_KF_SUPPORTED
    assert inference_padding_buckets(policy()) == ALIGNED_LENGTHS
    ae_only = {"inference": {"flow": "eager", "encoder": "bioir", "decoder": "bioir"}}
    assert inference_padding_buckets(ae_only) == BIOIR_LENGTHS
    for option in ("triangle_backend", "compile_shared_triangles"):
        with pytest.raises(ValueError, match="automatically"):
            resolve_backends({"bioir": {option: "shared"}}, "inference")
    with pytest.raises(ValueError, match="prefix masks"):
        resolve_backends({"bioir": {"flow_prefix_masks": False}}, "inference")


def test_old_runtime_rejected_for_flow_but_ae_does_not_require_kf():
    assert isinstance(_select_triangle(nn.Identity(), Native(), torch.bfloat16, False), _SharedTriangle)
    with pytest.raises(RuntimeError, match="try-KF API"):
        _select_triangle(nn.Identity(), Native(), torch.bfloat16, True)


@pytest.mark.parametrize("available,decline,expected", [(True, False, 3), (True, True, 2), (False, False, 2)])
def test_declined_kf_uses_shared_native_triangle(available, decline, expected, caplog):
    candidate = Candidate(available=available, decline=decline)
    selected = _select_triangle(candidate, Native(), torch.bfloat16, True)
    pair = torch.zeros(1, 8, 8, 4, dtype=torch.bfloat16)
    result = selected(pair, torch.ones(1, 8, 8, dtype=torch.bool), trimul_metadata=None)
    assert torch.equal(result, torch.full_like(pair, expected))
    if decline or not available:
        assert "shared native triangle" in caplog.text
    else:
        assert isinstance(selected, _KFTriangle)
        assert selected.readiness == "eligible_not_profiled"
    assert not selected.can_fuse_residual(pair)


def test_kernel_errors_are_not_hidden_as_fallback():
    selected = _select_triangle(Candidate(error=True), Native(), torch.bfloat16, True)
    with pytest.raises(RuntimeError, match="kernel failure"):
        selected(torch.zeros(1), torch.ones(1, dtype=torch.bool), trimul_metadata=None)


def test_kf_routes_only_to_flow(monkeypatch):
    trunks = {name: nn.Identity() for name in ("flow", "encoder", "decoder")}
    monkeypatch.setattr(execution, "execution_components", lambda model: trunks)
    monkeypatch.setattr(execution, "configure_triangle_backends", lambda model, backend: None)
    monkeypatch.setattr(bioir, "_require_bioir", lambda: None)
    execution.configure_execution(nn.Identity(), policy(), "inference")
    assert trunks["flow"]._execution.bioir_options["prefix_masks"]
    for name in ("encoder", "decoder"):
        assert "triangle_backend" not in trunks[name]._execution.bioir_options
        assert not trunks[name]._execution.bioir_options["prefix_masks"]


def test_kf_api_signature_guard():
    class Incompatible(Candidate):
        def try_kf_forward(self, pair):
            return pair

    with pytest.raises(RuntimeError, match="signature"):
        _select_triangle(Incompatible(), Native(), torch.bfloat16, True)


def test_observations_distinguish_construction_kf_and_fallback():
    selected = _select_triangle(Candidate(), Native(), torch.bfloat16, True)
    status = bioir.triangle_execution_status(selected)
    assert not status["kf_observed"] and not status["shared_observed"]
    pair = torch.zeros(1)
    selected(pair, pair.bool(), trimul_metadata=None)
    assert bioir.triangle_execution_status(selected)["kf_observed"]
    selected.bioir.decline = True
    selected(pair, pair.bool(), trimul_metadata=None)
    status = bioir.triangle_execution_status(selected)
    assert status["kf_observed"] and status["fallback_observed"]
    assert status["fallback_reason"] == "input_declined"
    assert bioir.triangle_execution_status(nn.Identity()) is None


def test_float32_uses_observable_shared_fallback():
    selected = _select_triangle(Candidate(), Native(), torch.float32, True)
    assert isinstance(selected, _SharedTriangle)
    value = torch.zeros(1)
    selected(value, value.bool(), trimul_metadata=None)
    status = bioir.triangle_execution_status(selected)
    assert status["fallback_observed"] and not status["kf_observed"]
    assert status["fallback_reason"] == "unsupported_device_dtype_or_width"
