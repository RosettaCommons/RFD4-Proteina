# MIT License

# Copyright (c) 2022 MattMcPartlon

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn

from rfproteina.nn.modules.adaptive_ln_scale import (
    AdaptiveLayerNorm,
    AdaptiveOutputScale,
)
from rfproteina.nn.modules.norm import make_norm
from rfproteina.utils.common import default, exists

# Large negative value for masking (not -inf to avoid gradient issues)
NEG_INF = -1e4


@contextmanager
def sdpa_training_context() -> Iterator[None]:
    """Exclude cuDNN during an eager training step, including its backward pass."""
    if not torch.is_grad_enabled():
        yield
        return
    enabled = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
        yield
    finally:
        torch.backends.cuda.enable_cudnn_sdp(enabled)


# Try to import cuequivariance for fused attention; import the ops module explicitly since the
# torch wrapper can import even when the compiled extension (e.g., libnvrtc.so) fails to load.
try:
    import cuequivariance_ops_torch  # noqa: F401
    from cuequivariance_torch import attention_pair_bias as cuet_attention_pair_bias

    CUET_AVAILABLE = True
except (ImportError, OSError):
    CUET_AVAILABLE = False
    cuet_attention_pair_bias = None


class PairBiasAttention(nn.Module):
    """Pair-bias multi-head attention via ``F.scaled_dot_product_attention``.

    The pair bias is passed as SDPA's ``attn_mask``. An additive ``attn_mask``
    precludes the FlashAttention-2 kernel, so SDPA dispatches to the
    memory-efficient backend (still a large speed/memory win over a materialized
    einsum attention). Falls back to an explicit einsum on CPU (CI / local dev).
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads = heads
        self.dim_head = dim_head

        # Combined projection for Q, K, V, and Gate
        self.node_norm = make_norm(node_dim)
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)

        self.q_layer_norm = make_norm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = make_norm(inner_dim) if qkln else nn.Identity()

        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim), bias=bias)

        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = make_norm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """Multi-head attention with the SDPA kernel.

        Args:
            node_feats: Scalar features of shape (b, n, d_s).
            pair_feats: Pair features of shape (b, n, n, d_e).
            mask: Boolean tensor of node adjacencies (b, n, n).

        Returns:
            Updated node features of shape (b, n, d_s).
        """
        # Get target dtype for the attention kernel (handles autocast)
        target_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else node_feats.dtype

        node_feats = self.node_norm(node_feats)
        h = self.heads

        # Compute Q, K, V
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)

        # Reshape for attention: [b, n, (h d)] -> [b, h, n, d]
        q, k, v, g = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v, g))

        # Compute pair bias: [b, n, n, pair_dim] -> [b, h, n, n]
        if exists(pair_feats) and exists(self.to_bias):
            pair_feats = self.pair_norm(pair_feats)
            attn_bias = rearrange(self.to_bias(pair_feats), "b i j h -> b h i j")
            attn_bias = attn_bias.to(target_dtype)
        else:
            attn_bias = None

        # Apply mask to bias
        if exists(mask):
            # mask is [b, n, n] boolean
            mask_expanded = rearrange(mask, "b i j -> b () i j")  # [b, 1, n, n]
            if attn_bias is not None:
                attn_bias = torch.masked_fill(attn_bias, ~mask_expanded, NEG_INF)
            else:
                # Create mask-only bias
                attn_bias = torch.zeros(
                    node_feats.shape[0],
                    h,
                    node_feats.shape[1],
                    node_feats.shape[1],
                    device=node_feats.device,
                    dtype=target_dtype,
                )
                attn_bias = torch.masked_fill(attn_bias, ~mask_expanded, NEG_INF)

            # Avoid crashes for fully masked attention rows.
            attn_bias = attn_bias.masked_fill(~mask_expanded.any(dim=-1, keepdim=True), 0.0)

        # SDPA on GPU; explicit einsum fallback on CPU (CI / local dev only, not
        # intended for production CPU inference).
        if not torch.cuda.is_available():
            scale = self.dim_head**-0.5
            sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
            if attn_bias is not None:
                sim = sim + attn_bias
            attn = torch.softmax(sim, dim=-1)
            o = einsum("b h i j, b h j d -> b h i d", attn, v)
        else:
            o = F.scaled_dot_product_attention(
                q.contiguous().to(target_dtype),
                k.contiguous().to(target_dtype),
                v.contiguous().to(target_dtype),
                attn_mask=attn_bias.contiguous() if attn_bias is not None else None,
            )

        # Apply gating
        o_gated = torch.sigmoid(g) * o

        # Reshape and project out: [b, h, n, d] -> [b, n, (h d)]
        out = rearrange(o_gated, "b h n d -> b n (h d)", h=h)
        return self.to_out_node(out)


class CuEqPairBiasAttention(nn.Module):
    """
    Pair bias attention using cuEquivariance fused kernel.

    Uses cuequivariance_torch.attention_pair_bias for optimized computation.
    This is a drop-in replacement for PairBiasAttention that uses fused CUDA kernels.

    Note: The cuequivariance kernel handles:
    - Pair feature normalization (via w_ln_z, b_ln_z)
    - Pair bias projection (via w_proj_z)
    - Gate computation (via w_proj_g, b_proj_g)
    - Output projection (via w_proj_o, b_proj_o)
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        if not CUET_AVAILABLE:
            raise ImportError(
                "cuequivariance_torch is required for CuEqPairBiasAttention. "
                "Install it or use attention_type='torch'."
            )

        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads = heads
        self.dim_head = dim_head

        self.node_norm = make_norm(node_dim)
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim), bias=bias)

        self.q_layer_norm = make_norm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = make_norm(inner_dim) if qkln else nn.Identity()

        if exists(pair_dim):
            self.pair_norm = make_norm(pair_dim)
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
        else:
            self.pair_norm, self.to_bias = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """Multi-head attention with cuEquivariance fused kernel.

        Args:
            node_feats: Scalar features of shape (b, n, d_s).
            pair_feats: Pair features of shape (b, n, n, d_e).
            mask: Token mask of shape (b, n) - cuequivariance computes pair mask internally.

        Returns:
            Updated node features of shape (b, n, d_s).
        """
        q_weight, k_weight, v_weight = self.to_qkv.weight.chunk(3)
        q_bias, k_bias, v_bias = self.to_qkv.bias.chunk(3) if self.to_qkv.bias is not None else (None, None, None)
        result = cuet_attention_pair_bias(
            single_repr=node_feats,
            pair_repr=pair_feats,
            mask=mask,
            num_heads=self.heads,
            w_ln_a=self.node_norm.weight,
            b_ln_a=getattr(self.node_norm, "bias", None),
            w_proj_q=q_weight,
            b_proj_q=q_bias,
            w_proj_k=k_weight,
            b_proj_k=k_bias,
            w_proj_v=v_weight,
            b_proj_v=v_bias,
            w_ln_q=getattr(self.q_layer_norm, "weight", None),
            b_ln_q=getattr(self.q_layer_norm, "bias", None),
            w_ln_k=getattr(self.k_layer_norm, "weight", None),
            b_ln_k=getattr(self.k_layer_norm, "bias", None),
            w_proj_z=self.to_bias.weight if self.to_bias is not None else None,
            w_ln_z=getattr(self.pair_norm, "weight", None),
            b_ln_z=getattr(self.pair_norm, "bias", None),
            w_proj_g=self.to_g.weight,
            b_proj_g=self.to_g.bias,
            w_proj_o=self.to_out_node.weight,
            b_proj_o=self.to_out_node.bias,
            inf=-NEG_INF,
            eps=self.node_norm.eps,
            norm_kind="rms_norm" if isinstance(self.node_norm, nn.RMSNorm) else "layer_norm",
        )
        return result[0] if isinstance(result, tuple) else result


class MultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair-biased multi-head self-attention with adaptive layer norm on the input
    and adaptive scaling on the output (SDPA backend)."""

    def __init__(self, dim_token: int, dim_pair: int | None, nheads: int, dim_cond: int, use_qkln: bool) -> None:
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,  # ...None builds no bias projection; attention keeps only the mask
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token, dim_cond=dim_cond)

    def forward(self, x: Tensor, pair_rep: Tensor, cond: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class CuEqMultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention using cuEquivariance fused kernel.

    Uses CuEqPairBiasAttention for optimized computation.
    """

    def __init__(self, dim_token: int, dim_pair: int, nheads: int, dim_cond: int, use_qkln: bool) -> None:
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = CuEqPairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token, dim_cond=dim_cond)

    def forward(self, x: Tensor, pair_rep: Tensor, cond: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        # Note: CuEqPairBiasAttention expects 1D token mask [b, n],
        # cuequivariance kernel computes pair mask internally
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=mask)  # Pass 1D mask
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]
