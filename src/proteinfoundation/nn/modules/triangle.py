import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from opt_einsum import contract as einsum


# from foundry import SHOULD_USE_CUEQUIVARIANCE
# from foundry.training.checkpoint import activation_checkpointing

def init_lecun_normal(module: nn.Module, scale: float = 1.0) -> nn.Module:
    """Re-initialize ``module.weight`` from a truncated (|z| < 2) LeCun normal."""
    stddev = math.sqrt(scale / module.weight.shape[-1]) / 0.87962566103423978

    normal = torch.distributions.normal.Normal(0, 1)
    lo, hi = normal.cdf(torch.tensor(-2.0)), normal.cdf(torch.tensor(2.0))
    p = lo + (hi - lo) * torch.rand(module.weight.shape)
    v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
    x = torch.clamp(math.sqrt(2) * torch.erfinv(v), -2, 2)

    module.weight = nn.Parameter(stddev * x)
    return module


# Fused kernels are optional: disable via DISABLE_CUEQUIVARIANCE, or simply don't install
# cuequivariance-torch. Either way we fall back to the vanilla torch path below.
_disable_env = os.environ.get("DISABLE_CUEQUIVARIANCE", "").strip().lower()
SHOULD_USE_CUEQUIVARIANCE = _disable_env not in ("true", "1", "yes")

if SHOULD_USE_CUEQUIVARIANCE:
    try:
        import cuequivariance_torch as cuet
    except ImportError:
        SHOULD_USE_CUEQUIVARIANCE = False
        logging.getLogger(__name__).info("cuequivariance-torch not installed — using vanilla triangle ops")
else:
    logging.getLogger(__name__).info("cuEquivariance disabled via DISABLE_CUEQUIVARIANCE env var")


class TriangleAttention(nn.Module):
    """Implementation of Triangle Attention from AlphaFold3.

    Routes to either cuEquivariance or vanilla implementation based on configuration.

    Args:
        pair: Pair representation tensor of shape (B, L, L, d_pair)

    Returns:
        Updated pair representation tensor of shape (B, L, L, d_pair)

    """

    def __init__(
        self,
        d_pair: int,
        n_head: int,
        d_hidden: int,
        start_node: bool = True,
        use_cuequivariance: bool = True,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(d_pair)

        self.to_q = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_pair, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_pair)

        self.scaling = 1 / math.sqrt(d_hidden)

        self.h = n_head
        self.dim = d_hidden
        self.start_node = start_node

        self.use_cuequivariance = use_cuequivariance

        self.reset_parameter()

    def reset_parameter(self) -> None:
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual
        # operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair: torch.Tensor, mask_I: torch.Tensor) -> torch.Tensor:
        """Forward pass of triangle attention.

        Args:
            pair: Pair representation, shape (B, L, L, d_pair).
            mask_I: Boolean mask where True = valid token, shape (B, L).
        """
        pair = self.norm(pair)
        bias = self.to_b(pair)  # (B, L, L, h)

        if not self.start_node:
            pair = rearrange(pair, "b i j d -> b j i d")

        # Route to appropriate implementation
        if self.use_cuequivariance and SHOULD_USE_CUEQUIVARIANCE:
            out = self._forward_cuequivariance(pair, bias, mask_I)
        else:
            out = self._forward_vanilla(pair, bias, mask_I)

        if not self.start_node:
            out = rearrange(out, "b i j d -> b j i d")

        # output projection
        out = self.to_out(out)
        return out

    def _forward_cuequivariance(self, pair: torch.Tensor, bias: torch.Tensor, mask_I: torch.Tensor) -> torch.Tensor:
        """cuEquivariance triangle attention implementation."""
        # Handle autocast conversion
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            pair = pair.to(dtype=dtype)
            bias = bias.to(dtype=dtype)

        assert (
            pair.dtype == torch.bfloat16 and bias.dtype == torch.bfloat16
        ), f"cuEquivariance requires bfloat16 inputs (got pair={pair.dtype}, bias={bias.dtype})"

        # Gate computation
        gate = torch.sigmoid(self.to_g(pair))  # (B, L, L, h*dim)

        # Project and reshape to cuEquivariance format: (B, L, H, L, D)
        query = rearrange(self.to_q(pair), "b i j (h d) -> b i h j d", h=self.h)
        key = rearrange(self.to_k(pair), "b i k (h d) -> b i h k d", h=self.h)
        value = rearrange(self.to_v(pair), "b i k (h d) -> b i h k d", h=self.h)

        # Bias: (B, L, L, H) -> (B, 1, H, L, L)
        bias_cueq = rearrange(bias, "b i j h -> b 1 h i j")

        # Mask: (B, I) -> (B, I, 1, 1, I) for cuEquivariance (1 = valid, 0 = padding)
        mask_II = mask_I.unsqueeze(-1) & mask_I.unsqueeze(-2)  # (B, L, L)
        mask = rearrange(mask_II, "b i j -> b i 1 1 j")

        # Call cuEquivariance triangle attention
        out_cueq = cuet.triangle_attention(query, key, value, bias=bias_cueq, scale=self.scaling, mask=mask)

        # Reshape back: (B, L, H, L, D) -> (B, L, L, H*D)
        out = rearrange(out_cueq, "b i h j d -> b i j (h d)")
        out = gate * out  # gated attention
        return out

    def _forward_vanilla(self, pair: torch.Tensor, bias: torch.Tensor, mask_I: torch.Tensor) -> torch.Tensor:
        """Vanilla PyTorch triangle attention implementation."""
        B, L = pair.shape[:2]

        # Gate computation
        gate = torch.sigmoid(self.to_g(pair))  # (B, L, L, h*dim)

        # Project and reshape to vanilla format: (B, L, L, H, D)
        query = self.to_q(pair).reshape(B, L, L, self.h, -1)
        key = self.to_k(pair).reshape(B, L, L, self.h, -1)
        value = self.to_v(pair).reshape(B, L, L, self.h, -1)

        query = query * self.scaling

        attn = einsum("bijhd,bikhd->bijkh", query, key)
        attn = attn + bias.unsqueeze(1).expand(-1, L, -1, -1, -1)  # (bijkh)

        # Mask out positions where any index is padding (fill with -inf before softmax)
        valid_III = mask_I[..., None, None] & mask_I[:, None, :, None] & mask_I[:, None, None, :]
        attn = attn.masked_fill(~valid_III[..., None], -1e35)
        attn = F.softmax(attn, dim=-2)

        out = einsum("bijkh,bikhd->bijhd", attn, value).reshape(B, L, L, -1)
        out = gate * out  # gated attention

        return out


class TriangleMultiplication(nn.Module):
    """Implementation of Triangle Multiplicative Update from AlphaFold3.

    Routes to either cuEquivariance or naive implementation based on configuration.

    Args:
        d_pair: Pair representation dimension (must equal d_hidden for cuEquivariance)
        d_hidden: Hidden dimension (must equal d_pair for cuEquivariance)
        direction: "outgoing" or "incoming" triangle multiplication direction
        bias: Whether to use bias in normalization layers
        use_cuequivariance: Whether to use cuEquivariance fused kernel when available
    """

    def __init__(
        self,
        d_pair: int,
        d_hidden: int | None = None,
        direction: str = "outgoing",
        bias: bool = True,
        use_cuequivariance: bool = True,
    ) -> None:
        super().__init__()

        # Set d_hidden to d_pair if not specified
        if d_hidden is None:
            d_hidden = d_pair

        self.d_pair = d_pair
        self.d_hidden = d_hidden

        # Validate direction parameter
        if direction not in ["outgoing", "incoming"]:
            raise ValueError(f"direction must be 'outgoing' or 'incoming', got '{direction}'")
        self.direction = direction

        self.use_cuequivariance = use_cuequivariance

        if self.use_cuequivariance:
            # cuEquivariance kernel requires d_pair == d_hidden...
            assert d_pair == d_hidden, "cuEquivariance triangle multiplication requires d_pair == d_hidden"
            # ... and d_pair must be a multiple of 32
            assert d_pair % 32 == 0, "cuEquivariance triangle multiplication requires d_pair to be a multiple of 32"

        # Input normalization (optional bias)
        self.norm_in = nn.LayerNorm(d_pair, bias=bias)

        # Input projections: combine left and right projections (2*d_hidden, d_pair) (no bias)
        self.p_in = nn.Linear(d_pair, 2 * d_hidden, bias=False)

        # Input gating: combine left and right gates (2*d_hidden, d_pair) (no bias)
        self.g_in = nn.Linear(d_pair, 2 * d_hidden, bias=False)

        # Output normalization (optional bias)
        self.norm_out = nn.LayerNorm(d_hidden, bias=bias)

        # Output projection (no bias)
        self.p_out = nn.Linear(d_hidden, d_pair, bias=False)

        # Output gating (no bias)
        self.g_out = nn.Linear(d_pair, d_pair, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Input projections: lecun normal distribution for regular linear weights
        self.p_in = init_lecun_normal(self.p_in)

        # We use default PyTorch initialization for the other parameters, as in AF-3 they do not specify their
        # weight initialization schemes. Without bias, e.g., the gate initialization from AF-2 is not correct.

    def forward(
        self,
        pair: Float[torch.Tensor, "B N N D"],
        mask_I: Float[torch.Tensor, "B N"],
    ) -> Float[torch.Tensor, "B N N D"]:
        """Forward pass of triangle multiplication.

        Args:
            pair: Pair representation, shape (B, N, N, D).
            mask_I: Boolean mask where True = valid token, shape (B, N).
        """
        # Route to appropriate implementation
        if self.use_cuequivariance and SHOULD_USE_CUEQUIVARIANCE:
            return self._forward_cuequivariance(pair, mask_I)
        else:
            return self._forward_vanilla(pair, mask_I)

    def _forward_vanilla(
        self,
        pair: Float[torch.Tensor, "B N N D"],
        mask_I: Float[torch.Tensor, "B N"],
        eps: float = 1e-3,
    ) -> Float[torch.Tensor, "B N N D"]:
        """Vanilla PyTorch triangle multiplication implementation."""
        B, L = pair.shape[:2]

        # Input normalization
        pair_norm = self.norm_in(pair)

        # Input projections: get combined output and split
        p_combined = self.p_in(pair_norm)  # (B, L, L, 2*d_hidden)
        left = p_combined[..., : self.d_hidden]  # (B, L, L, d_hidden)
        right = p_combined[..., self.d_hidden :]  # (B, L, L, d_hidden)

        # Input gating: get combined output and split
        g_combined = self.g_in(pair_norm)  # (B, L, L, 2*d_hidden)
        left_gate = torch.sigmoid(g_combined[..., : self.d_hidden])
        right_gate = torch.sigmoid(g_combined[..., self.d_hidden :])

        # Apply gating
        left = left_gate * left
        right = right_gate * right

        # Zero out padding positions
        mask_II = mask_I.unsqueeze(-1) & mask_I.unsqueeze(-2)  # (B, L, L)
        left = left * mask_II.unsqueeze(-1)
        right = right * mask_II.unsqueeze(-1)

        # Divide by number of valid elements to get mean
        denom = mask_I.sum(dim=-1)
        right = right / (denom[:, None, None, None] + eps)  # NOTE: avoid division by zero

        # Triangle multiplication based on direction
        if self.direction == "outgoing":
            out = torch.einsum("bikd,bjkd->bijd", left, right)
        else:  # incoming
            out = torch.einsum("bkid,bkjd->bijd", left, right)

        # Output normalization
        out = self.norm_out(out)

        # Output projection
        out = self.p_out(out)

        # Output gating
        gate = torch.sigmoid(self.g_out(pair_norm))
        out = gate * out

        return out

    def _forward_cuequivariance(
        self,
        pair: Float[torch.Tensor, "B N N D"],
        mask_I: Float[torch.Tensor, "B N"],
    ) -> Float[torch.Tensor, "B N N D"]:
        """cuEquivariance triangle multiplication implementation."""
        # Handle autocast conversion
        # (Use bfloat16 for optimal performance)
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            pair = pair.to(dtype=dtype)

        assert pair.dtype == torch.bfloat16, "cuEquivariance requires bfloat16 inputs for optimal performance"
        mask_II = (mask_I.unsqueeze(-1) & mask_I.unsqueeze(-2)).bool()  # (B, I, I)

        output = cuet.triangle_multiplicative_update(
            x=pair,
            direction=self.direction,
            mask=mask_II,
            norm_in_weight=self.norm_in.weight,
            norm_in_bias=self.norm_in.bias,
            p_in_weight=self.p_in.weight,  # (2*d_hidden, d_pair)
            g_in_weight=self.g_in.weight,  # (2*d_hidden, d_pair)
            norm_out_weight=self.norm_out.weight,
            norm_out_bias=self.norm_out.bias,
            p_out_weight=self.p_out.weight,  # (d_pair, d_pair) since d_hidden == d_pair
            g_out_weight=self.g_out.weight,  # (d_pair, d_pair)
            eps=1e-5,
        )

        return output
