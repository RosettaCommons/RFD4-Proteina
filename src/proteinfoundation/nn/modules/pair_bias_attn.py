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


import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn

from proteinfoundation.nn.modules.adaptive_ln_scale import (
    AdaptiveLayerNorm,
    AdaptiveOutputScale,
)

# Large negative value for masking (not -inf to avoid gradient issues)
NEG_INF = -1e4

# Try to import cuequivariance for fused attention; import the ops module explicitly since the
# torch wrapper can import even when the compiled extension (e.g., libnvrtc.so) fails to load.
try:
    import cuequivariance_ops_torch  # noqa: F401
    from cuequivariance_torch import attention_pair_bias as cuet_attention_pair_bias

    CUET_AVAILABLE = True
except (ImportError, OSError):
    CUET_AVAILABLE = False
    cuet_attention_pair_bias = None

# Lazy import Transformer Engine for optimized attention
TE_AVAILABLE = None
te = None


def _load_transformer_engine() -> bool:
    """Load Transformer Engine on demand."""
    global TE_AVAILABLE, te
    if TE_AVAILABLE is None:
        try:
            import transformer_engine.pytorch as te_module
        except ImportError:
            TE_AVAILABLE = False
            te = None
        else:
            TE_AVAILABLE = True
            te = te_module
    return TE_AVAILABLE


def is_transformer_engine_available() -> bool:
    """Return whether Transformer Engine can be imported."""
    return _load_transformer_engine()


def exists(val: object) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x: object, y: object) -> object:
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


def max_neg_value(x: torch.Tensor) -> float:
    return torch.finfo(x.dtype).min


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
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
        **kawrgs: object,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param mask: boolean tensor of node adjacencies
        :return:
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        node_feats, h = self.node_norm(node_feats), self.heads
        pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)
        b = rearrange(self.to_bias(pair_feats), "b ... h -> b h ...") if exists(pair_feats) else 0
        q, k, v, g = (rearrange(t, "b ... (h d) -> b h ... d", h=h) for t in (q, k, v, g))
        attn_feats = self._attn(q, k, v, b, mask)
        attn_feats = rearrange(torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h)
        return self.to_out_node(attn_feats)

    def _attn(self, q: Tensor, k: Tensor, v: Tensor, b: Tensor, mask: Tensor | None) -> Tensor:
        """Perform attention update"""
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)


class FlashPairBiasAttention(nn.Module):
    """
    Pair bias attention using PyTorch's scaled_dot_product_attention for Flash Attention.

    Uses the flash attention kernel via F.scaled_dot_product_attention, which provides
    significant memory and speed improvements. The pair bias is passed through the
    attn_mask parameter.
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
        self.node_norm = nn.LayerNorm(node_dim)
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)

        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()

        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim), bias=bias)

        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """Multi-head attention with flash attention kernel.

        Args:
            node_feats: Scalar features of shape (b, n, d_s).
            pair_feats: Pair features of shape (b, n, n, d_e).
            mask: Boolean tensor of node adjacencies (b, n, n).

        Returns:
            Updated node features of shape (b, n, d_s).
        """
        # Get target dtype for flash attention (handles autocast)
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

        # Flash attention via scaled_dot_product_attention; CPU CI / local dev only
        # when CUDA is unavailable (not intended for production CPU inference).
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

        # Apply gating (matches PairBiasAttention)
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
                "Install it or use attention_type='torch' or 'flash'."
            )

        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        self.node_norm = nn.LayerNorm(node_dim)
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))

        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()

        if exists(pair_dim):
            self.pair_norm = nn.LayerNorm(pair_dim)
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
        h = self.heads
        node_feats_normed = self.node_norm(node_feats)

        # Compute Q, K, V (matching original PairBiasAttention flow)
        q, k, v = self.to_qkv(node_feats_normed).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)

        # Reshape for cuequivariance: [b, n, (h d)] -> [b, h, n, d]
        # This matches the original commented code: "b ... (h d) -> b h ... d"
        q, k, v = (rearrange(t, "b ... (h d) -> b h ... d", h=h) for t in (q, k, v))

        # Use cuequivariance fused kernel
        # Note: cuet expects raw pair_feats (pre-normalization) and applies LayerNorm internally
        # Note: cuet expects 1D token mask [b, n] and computes pair mask internally
        # Returns tuple (output, z_proj) - we only need the output
        result = cuet_attention_pair_bias(
            s=node_feats_normed,  # Normalized node features
            q=q,
            k=k,
            v=v,
            z=pair_feats,  # Raw pair features (cuet applies pair_norm internally)
            mask=mask,  # [b, n] token mask
            num_heads=self.heads,
            w_proj_z=self.to_bias.weight if self.to_bias is not None else None,
            w_proj_g=self.to_g.weight,
            w_proj_o=self.to_out_node.weight,
            w_ln_z=self.pair_norm.weight if self.pair_norm is not None else None,
            b_ln_z=self.pair_norm.bias if self.pair_norm is not None else None,
            b_proj_z=None,  # to_bias has no bias
            b_proj_g=self.to_g.bias if self.to_g.bias is not None else None,
            b_proj_o=self.to_out_node.bias if self.to_out_node.bias is not None else None,
        )
        # cuet returns (output, z_proj) tuple - extract the output tensor
        out = result[0] if isinstance(result, tuple) else result
        return out


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


class MultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token: int, dim_pair: int, nheads: int, dim_cond: int, use_qkln: bool) -> None:
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
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class FlashMultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention using Flash Attention kernel.

    Uses FlashPairBiasAttention which leverages F.scaled_dot_product_attention
    for improved memory efficiency and speed.
    """

    def __init__(self, dim_token: int, dim_pair: int, nheads: int, dim_cond: int, use_qkln: bool) -> None:
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = FlashPairBiasAttention(
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
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class TEPairBiasAttention(nn.Module):
    """
    Pair bias attention using NVIDIA Transformer Engine's DotProductAttention.

    Uses TE's optimized attention kernel with support for:
    - FP8 precision when used with te.autocast
    - Custom attention bias via core_attention_bias parameter
    - Fused kernels for improved performance

    This is a drop-in replacement for PairBiasAttention/FlashPairBiasAttention.
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
        if not _load_transformer_engine():
            raise ImportError(
                "transformer_engine is required for TEPairBiasAttention. "
                "Install it via: pip install transformer_engine[pytorch] "
                "or use attention_type='torch' or 'flash'."
            )

        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads = heads
        self.dim_head = dim_head

        # Projections for Q, K, V, and Gate
        self.node_norm = nn.LayerNorm(node_dim)
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)

        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()

        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim), bias=bias)

        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

        # TE DotProductAttention - uses optimized kernels
        self.dpa = te.DotProductAttention(
            num_attention_heads=heads,
            kv_channels=dim_head,
            attention_dropout=0.0,
            qkv_format="bshd",  # [batch, seq, heads, dim]
            attn_mask_type="arbitrary",  # Allows custom attention bias
        )

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """Multi-head attention with Transformer Engine kernel.

        Args:
            node_feats: Scalar features of shape (b, n, d_s).
            pair_feats: Pair features of shape (b, n, n, d_e).
            mask: Boolean tensor of node adjacencies (b, n, n).

        Returns:
            Updated node features of shape (b, n, d_s).
        """
        node_feats = self.node_norm(node_feats)
        h = self.heads

        # Compute Q, K, V
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)

        # Reshape for TE: [b, n, (h d)] -> [b, n, h, d] (bshd format)
        q = rearrange(q, "b n (h d) -> b n h d", h=h)
        k = rearrange(k, "b n (h d) -> b n h d", h=h)
        v = rearrange(v, "b n (h d) -> b n h d", h=h)

        # Ensure q, k, v have the same dtype (required by Transformer Engine)
        k = k.to(q.dtype)
        v = v.to(q.dtype)

        # Compute pair bias: [b, n, n, pair_dim] -> [b, h, n, n]
        if exists(pair_feats) and exists(self.to_bias):
            pair_feats_normed = self.pair_norm(pair_feats)
            attn_bias = rearrange(self.to_bias(pair_feats_normed), "b i j h -> b h i j")
        else:
            attn_bias = None

        # Apply mask to bias (TE expects bias with mask already applied)
        if exists(mask):
            mask_expanded = rearrange(mask, "b i j -> b () i j")  # [b, 1, n, n]
            if attn_bias is not None:
                attn_bias = torch.masked_fill(attn_bias, ~mask_expanded, NEG_INF)
            else:
                # Create mask-only bias
                b, n = node_feats.shape[:2]
                attn_bias = torch.zeros(b, h, n, n, device=node_feats.device, dtype=node_feats.dtype)
                attn_bias = torch.masked_fill(attn_bias, ~mask_expanded, NEG_INF)

        # TE attention with pair bias
        # Note: core_attention_bias is added to attention scores before softmax
        out = self.dpa(
            q,
            k,
            v,
            core_attention_bias=attn_bias,
            core_attention_bias_type="post_scale_bias" if attn_bias is not None else "no_bias",
        )

        # TE DotProductAttention returns [b, n, h*d] (flattened), reshape to [b, n, h, d]
        out = rearrange(out, "b n (h d) -> b n h d", h=h)

        # Apply gating (matches PairBiasAttention)
        g = rearrange(g, "b n (h d) -> b n h d", h=h)
        o_gated = torch.sigmoid(g) * out

        # Reshape and project out: [b, n, h, d] -> [b, n, (h d)]
        out = rearrange(o_gated, "b n h d -> b n (h d)", h=h)
        return self.to_out_node(out)


class TEMultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention using Transformer Engine kernel.

    Uses TEPairBiasAttention for optimized computation with FP8 support.
    """

    def __init__(self, dim_token: int, dim_pair: int, nheads: int, dim_cond: int, use_qkln: bool) -> None:
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = TEPairBiasAttention(
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
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]
