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


def policy(triangle_backend="shared", prefix=True):
    return {
        "inference": {name: "bioir" for name in ("flow", "encoder", "decoder")},
        "bioir": {"triangle_backend": triangle_backend, "flow_prefix_masks": prefix},
    }


def test_kf_padding_is_opt_in():
    assert inference_padding_buckets(policy()) == BIOIR_LENGTHS
    assert inference_padding_buckets(policy("kf")) == ALIGNED_LENGTHS
    with pytest.raises(ValueError, match="flow_prefix_masks"):
        resolve_backends(policy("kf", False), "inference")
    with pytest.raises(ValueError, match="triangle_backend"):
        resolve_backends(policy("invalid"), "inference")


@pytest.mark.parametrize("selection", ["shared", "kf"])
def test_old_runtime_only_rejected_when_kf_requested(selection):
    if selection == "shared":
        assert isinstance(_select_triangle(nn.Identity(), Native(), torch.bfloat16, selection, True), _SharedTriangle)
    else:
        with pytest.raises(RuntimeError, match="MR229"):
            _select_triangle(nn.Identity(), Native(), torch.bfloat16, selection, True)


@pytest.mark.parametrize("available,decline,expected", [(True, False, 3), (True, True, 2), (False, False, 2)])
def test_declined_kf_uses_shared_native_triangle(available, decline, expected, caplog):
    candidate = Candidate(available=available, decline=decline)
    selected = _select_triangle(candidate, Native(), torch.bfloat16, "kf", True)
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
    selected = _select_triangle(Candidate(error=True), Native(), torch.bfloat16, "kf", True)
    with pytest.raises(RuntimeError, match="kernel failure"):
        selected(torch.zeros(1), torch.ones(1, dtype=torch.bool), trimul_metadata=None)


def test_kf_routes_only_to_flow(monkeypatch):
    trunks = {name: nn.Identity() for name in ("flow", "encoder", "decoder")}
    monkeypatch.setattr(execution, "execution_components", lambda model: trunks)
    monkeypatch.setattr(execution, "configure_triangle_backends", lambda model, backend: None)
    monkeypatch.setattr(bioir, "_require_bioir", lambda: None)
    execution.configure_execution(nn.Identity(), policy("kf"), "inference")
    assert trunks["flow"]._execution.bioir_options["triangle_backend"] == "kf"
    assert trunks["flow"]._execution.bioir_options["prefix_masks"]
    for name in ("encoder", "decoder"):
        assert trunks[name]._execution.bioir_options["triangle_backend"] == "shared"
        assert not trunks[name]._execution.bioir_options["prefix_masks"]


def test_kf_api_signature_guard():
    class Incompatible(Candidate):
        def try_kf_forward(self, pair):
            return pair

    with pytest.raises(RuntimeError, match="signature"):
        _select_triangle(Incompatible(), Native(), torch.bfloat16, "kf", True)
