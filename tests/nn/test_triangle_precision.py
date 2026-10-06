"""Public packed-call contract; actual cuEquivariance kernels need GPU tests."""

import pytest
import torch

import rfproteina.nn.modules.openfold_modules.triangular_multiplicative_update as trimul


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("triangle", [trimul.TriangleMultiplicationIncoming, trimul.TriangleMultiplicationOutgoing])
def test_packed_bf16_preserves_parameters_and_gradients(monkeypatch, bias, triangle):
    module = triangle(32, 16, bias=bias, backend="torch")
    original = {name: value.detach().clone() for name, value in module.named_parameters()}
    calls = []

    def packed(pair, **kwargs):
        calls.append(kwargs)
        # Exercise differentiable packing/casting without pretending CPU runs CUDA kernels.
        operands = [value for value in kwargs.values() if isinstance(value, torch.Tensor) and value.is_floating_point()]
        coefficient = sum(value.to(torch.bfloat16).float().mean() for value in operands)
        return pair * coefficient.to(torch.bfloat16)

    def split(*args, **kwargs):
        pytest.fail("The split API cannot receive the explicit BF16 precision request")

    monkeypatch.setattr(trimul, "_CUEQ_PACKED", packed)
    monkeypatch.setattr(trimul, "_CUEQ_SPLIT", split)
    pair = torch.ones(1, 4, 4, 32, dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = module._cueq_forward(pair, mask, module._cueq_parameters())
    output.float().sum().backward()

    assert len(calls) == 1
    call = calls[0]
    assert call["precision"] == "bfloat16"
    assert call["direction"] == module._direction
    assert call["mask"] is mask
    for name in ("p_in_bias", "g_in_bias", "p_out_bias", "g_out_bias"):
        assert (call[name] is not None) == bias
    assert call["norm_in_bias"].shape == (32,)
    assert call["norm_out_bias"].shape == (16,)
    if not bias:
        assert torch.count_nonzero(call["norm_in_bias"]) == 0
        assert torch.count_nonzero(call["norm_out_bias"]) == 0
    assert call["p_in_weight"].shape == (32, 32)
    assert call["g_in_weight"].shape == (32, 32)
    assert output.dtype == pair.grad.dtype == torch.bfloat16
    for name, parameter in module.named_parameters():
        assert parameter.dtype == torch.float32
        assert torch.equal(parameter, original[name])
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
