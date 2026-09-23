"""Check the SDPA PairBiasAttention (and the cuEq variant) against a manual-einsum reference."""

import pytest
import torch
from einops import rearrange

from rfproteina.nn.modules.pair_bias_attn import (
    CUET_AVAILABLE,
    NEG_INF,
    PairBiasAttention,
)

TF32_MAX_ABS_DIFF = 1e-3


@pytest.fixture(autouse=True)
def enable_sdpa_backends():
    """Enable the efficient SDPA backends on GPU (pair bias precludes flash-2)."""
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _reference_forward(
    attn: PairBiasAttention,
    node_feats: torch.Tensor,
    pair_feats: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Manual einsum-softmax ground truth mirroring ``PairBiasAttention.forward``.

    Uses ``attn``'s own submodules so the only difference vs. the module is the
    attention kernel (SDPA on GPU / einsum fallback on CPU) — catches SDPA,
    bias, masking, and gating mistakes.
    """
    h = attn.heads
    x = attn.node_norm(node_feats)
    q, k, v = attn.to_qkv(x).chunk(3, dim=-1)
    q, k = attn.q_layer_norm(q), attn.k_layer_norm(k)
    g = attn.to_g(x)
    q, k, v, g = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v, g))

    b, n = node_feats.shape[0], node_feats.shape[1]
    if pair_feats is not None and attn.to_bias is not None:
        bias = rearrange(attn.to_bias(attn.pair_norm(pair_feats)), "b i j h -> b h i j")
    else:
        bias = torch.zeros(b, h, n, n, device=node_feats.device, dtype=node_feats.dtype)
    if mask is not None:
        bias = bias.masked_fill(~rearrange(mask, "b i j -> b () i j"), NEG_INF)

    scale = attn.dim_head**-0.5
    sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale + bias
    o = torch.einsum("b h i j, b h j d -> b h i d", torch.softmax(sim, dim=-1), v)
    o = torch.sigmoid(g) * o
    return attn.to_out_node(rearrange(o, "b h n d -> b n (h d)", h=h))


def _make_attn(cls, device, *, bias=True, qkln=True):
    return cls(node_dim=64, dim_head=16, heads=4, bias=bias, dim_out=64, qkln=qkln, pair_dim=32).to(device)


@pytest.mark.parametrize("batch_size,seq_len", [(2, 32), (1, 64)])
def test_sdpa_matches_einsum_reference(batch_size: int, seq_len: int):
    """The SDPA PairBiasAttention matches the manual-einsum reference."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = _make_attn(PairBiasAttention, device)

    node = torch.randn(batch_size, seq_len, 64, device=device)
    pair = torch.randn(batch_size, seq_len, seq_len, 32, device=device)
    mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)

    with torch.no_grad():
        out = attn(node, pair, mask)
        ref = _reference_forward(attn, node, pair, mask)

    assert out.shape == ref.shape
    max_diff = (out - ref).abs().max().item()
    assert max_diff < TF32_MAX_ABS_DIFF, f"Max diff: {max_diff:.3e} (expected < {TF32_MAX_ABS_DIFF:.0e})"


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance_torch not installed")
@pytest.mark.parametrize("batch_size,seq_len", [(2, 32), (1, 64)])
@pytest.mark.parametrize("bias,qkln", [(True, True), (False, False)])
@pytest.mark.parametrize("padded", [False, True])
def test_cuequivariance_matches_reference(batch_size: int, seq_len: int, bias: bool, qkln: bool, padded: bool):
    """The cuEquivariance fused kernel matches the manual-einsum reference."""
    from rfproteina.nn.modules.pair_bias_attn import CuEqPairBiasAttention

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_attn = _make_attn(PairBiasAttention, device, bias=bias, qkln=qkln)
    cuet_attn = _make_attn(CuEqPairBiasAttention, device, bias=bias, qkln=qkln)
    cuet_attn.load_state_dict(ref_attn.state_dict(), strict=True)

    node = torch.randn(batch_size, seq_len, 64, device=device)
    pair = torch.randn(batch_size, seq_len, seq_len, 32, device=device)
    token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)  # cuEq: [b, n]
    if padded:
        token_mask[:, -4:] = False
    pair_mask = token_mask[:, :, None] * token_mask[:, None, :]  # reference: [b, n, n]

    with torch.no_grad():
        ref = _reference_forward(ref_attn, node, pair, pair_mask)
        out_cuet = cuet_attn(node, pair, token_mask)

    assert out_cuet.shape == ref.shape
    max_diff = (out_cuet - ref)[token_mask].abs().max().item()
    assert max_diff < TF32_MAX_ABS_DIFF, f"Max diff: {max_diff:.3e} (expected < {TF32_MAX_ABS_DIFF:.0e})"
