"""Test that PairBiasAttention variants (Flash, CuEq) produce similar outputs to standard PyTorch."""

import pytest
import torch

from proteinfoundation.nn.modules.pair_bias_attn import (
    CUET_AVAILABLE,
    FlashPairBiasAttention,
    PairBiasAttention,
    is_transformer_engine_available,
)

TE_AVAILABLE = is_transformer_engine_available()


@pytest.fixture(autouse=True)
def enable_flash_backends():
    """Enable Flash Attention backends for all tests."""
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@pytest.mark.parametrize("batch_size,seq_len", [(2, 32), (1, 64)])
def test_flash_vs_torch_attention_similar(batch_size: int, seq_len: int):
    """Verify Flash and Torch attention produce similar outputs."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config
    node_dim, pair_dim, heads, dim_head = 64, 32, 4, 16

    # Create both attention modules with same weights
    torch_attn = PairBiasAttention(
        node_dim=node_dim, dim_head=dim_head, heads=heads,
        bias=True, dim_out=node_dim, qkln=True, pair_dim=pair_dim,
    ).to(device)

    flash_attn = FlashPairBiasAttention(
        node_dim=node_dim, dim_head=dim_head, heads=heads,
        bias=True, dim_out=node_dim, qkln=True, pair_dim=pair_dim,
    ).to(device)

    # Copy weights from torch to flash
    flash_attn.load_state_dict(torch_attn.state_dict(), strict=False)

    # Input tensors
    node_feats = torch.randn(batch_size, seq_len, node_dim, device=device)
    pair_feats = torch.randn(batch_size, seq_len, seq_len, pair_dim, device=device)
    mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)

    # Forward pass
    with torch.no_grad():
        out_torch = torch_attn(node_feats, pair_feats, mask)
        out_flash = flash_attn(node_feats, pair_feats, mask)

    # Check similarity
    assert out_torch.shape == out_flash.shape
    max_diff = (out_torch - out_flash).abs().max().item()
    assert max_diff < 1e-4, f"Max diff: {max_diff:.6f} (expected < 1e-4)"


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance_torch not installed")
@pytest.mark.parametrize("batch_size,seq_len", [(2, 32), (1, 64)])
def test_cuequivariance_vs_torch_attention_similar(batch_size: int, seq_len: int):
    """Verify cuEquivariance and Torch attention produce similar outputs."""
    from proteinfoundation.nn.modules.pair_bias_attn import CuEqPairBiasAttention

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config
    node_dim, pair_dim, heads, dim_head = 64, 32, 4, 16

    # Create both attention modules with same weights
    torch_attn = PairBiasAttention(
        node_dim=node_dim, dim_head=dim_head, heads=heads,
        bias=True, dim_out=node_dim, qkln=True, pair_dim=pair_dim,
    ).to(device)

    cuet_attn = CuEqPairBiasAttention(
        node_dim=node_dim, dim_head=dim_head, heads=heads,
        bias=True, dim_out=node_dim, qkln=True, pair_dim=pair_dim,
    ).to(device)

    # Copy weights from torch to cuequivariance
    cuet_attn.load_state_dict(torch_attn.state_dict(), strict=False)

    # Input tensors
    node_feats = torch.randn(batch_size, seq_len, node_dim, device=device)
    pair_feats = torch.randn(batch_size, seq_len, seq_len, pair_dim, device=device)
    # 1D token mask [b, n] - cuequivariance computes pair mask internally
    token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    # 2D pair mask [b, n, n] - for PairBiasAttention
    pair_mask = token_mask[:, :, None] * token_mask[:, None, :]

    # Forward pass
    with torch.no_grad():
        out_torch = torch_attn(node_feats, pair_feats, pair_mask)  # Uses [b, n, n] mask
        out_cuet = cuet_attn(node_feats, pair_feats, token_mask)  # Uses [b, n] mask

    # Check similarity (cuequivariance may have larger numerical differences)
    assert out_torch.shape == out_cuet.shape
    max_diff = (out_torch - out_cuet).abs().max().item()
    assert max_diff < 1e-4, f"Max diff: {max_diff:.6f} (expected < 0.1)"


@pytest.mark.skipif(not TE_AVAILABLE, reason="transformer_engine not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TE attention")
@pytest.mark.parametrize("batch_size,seq_len", [(2, 32), (1, 64)])
def test_te_vs_torch_attention_similar(batch_size: int, seq_len: int):
    """Verify Transformer Engine and Torch attention produce similar outputs."""
    from proteinfoundation.nn.modules.pair_bias_attn import TEPairBiasAttention

    torch.manual_seed(42)
    device = "cuda"

    # Config
    node_dim, pair_dim, heads, dim_head = 64, 32, 4, 16

    # Create both attention modules with same weights
    torch_attn = PairBiasAttention(
        node_dim=node_dim, dim_head=dim_head, heads=heads,
        bias=True, dim_out=node_dim, qkln=True, pair_dim=pair_dim,
    ).to(device)

    te_attn = TEPairBiasAttention(
        node_dim=node_dim, dim_head=dim_head, heads=heads,
        bias=True, dim_out=node_dim, qkln=True, pair_dim=pair_dim,
    ).to(device)

    # Copy weights from torch to TE
    te_attn.load_state_dict(torch_attn.state_dict(), strict=False)

    # Input tensors
    node_feats = torch.randn(batch_size, seq_len, node_dim, device=device)
    pair_feats = torch.randn(batch_size, seq_len, seq_len, pair_dim, device=device)
    mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)

    # Forward pass
    with torch.no_grad():
        out_torch = torch_attn(node_feats, pair_feats, mask)
        out_te = te_attn(node_feats, pair_feats, mask)

    # Check similarity (TE may have slightly different numerical behavior)
    assert out_torch.shape == out_te.shape
    max_diff = (out_torch - out_te).abs().max().item()
    assert max_diff < 1e-3, f"Max diff: {max_diff:.6f} (expected < 1e-3)"
