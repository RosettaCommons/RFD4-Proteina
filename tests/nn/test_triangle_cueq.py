"""Triangle multiplication: grouped Torch projections and the cuEquivariance backend."""

import pytest
import torch
from peft import LoraConfig, inject_adapter_in_model

import rfproteina.nn.modules.openfold_modules.triangular_multiplicative_update as trimul
from rfproteina.nn.modules.openfold_modules.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming as Incoming,
)
from rfproteina.nn.modules.openfold_modules.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing as Outgoing,
)

EQUATIONS = {Outgoing: "bikc,bjkc->bijc", Incoming: "bkic,bkjc->bijc"}


def _randomized(module: torch.nn.Module) -> torch.nn.Module:
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(1.0 if p.ndim == 1 else 0.0, 0.1)
    return module


@pytest.mark.parametrize("grad,bias,lora", [(True, True, False), (False, False, True)])
def test_grouped_projections_match_reference(grad: bool, bias: bool, lora: bool) -> None:
    for triangle, equation in EQUATIONS.items():
        m = _randomized(triangle(32, 16, bias=bias))
        if lora:  # Wrapped projections must still run their own forward.
            inject_adapter_in_model(LoraConfig(r=2, target_modules=["linear_b_g"]), m)
            torch.nn.init.normal_(m.linear_b_g.lora_B["default"].weight)
        z, mask = torch.randn(2, 12, 12, 32), torch.rand(2, 12, 12, 1).round()
        zn = m.layer_norm_in(z)
        a = m.linear_a_p(zn) * torch.sigmoid(m.linear_a_g(zn)) * mask
        b = m.linear_b_p(zn) * torch.sigmoid(m.linear_b_g(zn)) * mask
        expected = m.linear_z(m.layer_norm_out(torch.einsum(equation, a, b))) * torch.sigmoid(m.linear_g(zn))
        with torch.set_grad_enabled(grad):
            torch.testing.assert_close(m(z, mask[..., 0]), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reason", [ImportError("missing kernel"), "", "\n", "Traceback\nRuntimeError: ABI mismatch"])
def test_unavailable_cueq_keeps_torch_backend(monkeypatch, caplog, reason):
    monkeypatch.setattr(trimul, "_CUEQ_UNAVAILABLE", reason)
    trimul._warn_once.cache_clear()
    module = Outgoing(16, 8, backend="cuequivariance")
    assert module.backend == "torch"
    assert "cuEquivariance is unavailable; using torch" in caplog.text
    assert str(reason) in caplog.text
    reference = Outgoing(16, 8, backend="torch")
    reference.load_state_dict(module.state_dict())
    x, mask = torch.randn(1, 4, 4, 16), torch.ones(1, 4, 4)
    torch.testing.assert_close(module(x, mask), reference(x, mask))


@pytest.mark.gpu
@pytest.mark.skipif(trimul._CUEQ_UNAVAILABLE is not None, reason="requires cuequivariance-ops-torch>=0.13")
@pytest.mark.parametrize(
    "triangle,c_hidden,bias", [(Incoming, 256, False), (Outgoing, 256, True), (Outgoing, 384, False)]
)
@pytest.mark.parametrize("compiled", [False, True])
def test_cueq_backend_matches_torch(triangle: type, c_hidden: int, bias: bool, compiled: bool) -> None:
    """Pair 384 with hidden 256 is fused on H200; hidden 384 falls back to the torch path."""
    torch._dynamo.reset()
    ref = _randomized(triangle(384, c_hidden, bias=bias)).cuda()
    fused = triangle(384, c_hidden, bias=bias, backend="cuequivariance").cuda()
    fused.load_state_dict(ref.state_dict())
    x = torch.randn(2, 256, 256, 384, device="cuda")
    mask = torch.ones(2, 256, 256, device="cuda", dtype=torch.bool)
    mask[:, -8:] = mask[:, :, -8:] = False

    results = []
    for source in (ref, fused):
        # Compare identical reductions even when preceding tests enable compiler tuning.
        module = (
            torch.compile(source, fullgraph=True, dynamic=False, options={"deterministic": True})
            if compiled
            else source
        )
        xi = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = module(xi, mask)
            with torch.no_grad():
                no_grad_out = module(x, mask)
        out.float().square().mean().backward()
        results.append([out, no_grad_out, xi.grad, *(p.grad for p in source.parameters())])
    fused_route = trimul._CUEQ_SUPPORTED[(2, 256, 384, c_hidden, x.device.index, True, True)]
    assert fused_route == (c_hidden == 256 and torch.cuda.get_device_capability() == (9, 0))
    for ref_t, fused_t in zip(*results, strict=True):
        if fused_route:
            cosine = torch.nn.functional.cosine_similarity(ref_t.flatten().float(), fused_t.flatten().float(), dim=0)
            assert cosine > 0.99, f"cosine {cosine:.4f}"
        else:
            torch.testing.assert_close(fused_t, ref_t)
