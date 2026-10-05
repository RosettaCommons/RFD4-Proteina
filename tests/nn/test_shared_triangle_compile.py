"""Contract for the internal compiled shared fallback."""

import torch
from torch import nn

from rfproteina.nn.bioir import _SharedTriangle


class NativeTriangle(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.configured_backend = "cuequivariance"

    def forward(self, pair, mask):
        return pair * self.weight * mask[..., None]


def test_compile_policy_disables_nested_graphs(monkeypatch):
    calls = []

    def compile_(module, **kwargs):
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr(torch, "compile", compile_)
    source = NativeTriangle().train()
    original = source.weight.detach().clone()
    result = _SharedTriangle(source, torch.bfloat16)
    assert len(calls) == 1
    assert calls[0][1] == {"fullgraph": True, "dynamic": False, "options": {"triton.cudagraphs": False}}
    assert result.native is not source
    assert result.native.configured_backend == "cuequivariance"
    assert not result.native.training and source.training
    assert source.weight.dtype == torch.float32
    assert result.native.weight.dtype == torch.bfloat16
    assert source.weight.data_ptr() != result.native.weight.data_ptr()
    torch.testing.assert_close(source.weight, original)
    assert list(source.state_dict()) == ["weight"]


def test_forward_preserves_mask_and_native_result(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: module)
    source = NativeTriangle()
    result = _SharedTriangle(source, torch.bfloat16)
    pair = torch.ones((1, 2, 2, 2), dtype=torch.bfloat16)
    mask = torch.tensor([[[True, False], [True, True]]])
    expected = result.native(pair, mask)
    torch.testing.assert_close(result(pair, mask, trimul_metadata=object()), expected)
    assert not result.can_fuse_residual(pair)


def test_compiler_failure_is_not_silently_downgraded(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(torch, "compile", unavailable)
    try:
        _SharedTriangle(NativeTriangle(), torch.bfloat16)
    except RuntimeError as error:
        assert str(error) == "compiler unavailable"
    else:
        raise AssertionError("Compiler error was hidden")


def test_actual_cpu_compiler_handles_multiple_shapes_without_cache_limit_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    before_limit = torch._dynamo.config.cache_size_limit
    source = NativeTriangle().eval()
    compiled = _SharedTriangle(source, torch.bfloat16)
    first = None
    with torch.inference_mode():
        for batch, length in [(1, 128), (4, 250), (2, 512), (1, 128)]:
            pair = torch.ones(batch, length, 2, dtype=torch.bfloat16)
            mask = torch.ones(batch, length, dtype=torch.bool)
            actual = compiled(pair, mask, trimul_metadata=None)
            expected = pair * source.weight.to(torch.bfloat16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if first is None:
                first = actual.clone()
            elif (batch, length) == (1, 128):
                torch.testing.assert_close(actual, first, rtol=0, atol=0)
    assert torch._dynamo.config.cache_size_limit == before_limit
