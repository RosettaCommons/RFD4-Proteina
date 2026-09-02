import torch

from rfproteina.nn.modules.pair_bias_attn import (
    CuEqMultiHeadBiasedAttentionADALN_MM,
    MultiHeadBiasedAttentionADALN_MM,
)
from rfproteina.nn.modules.seq_transition_af3 import TransitionADALN


class MultiheadAttnAndTransition(torch.nn.Module):
    """Layer that applies mha and transition to a sequence representation. Both layers are their adaptive versions
    which rely on conditining variables (see above).

    Args:
        dim_token: Token dimension in sequence representation.
        dim_pair: Dimension of pair representation.
        nheads: Number of attention heads.
        dim_cond: Dimension of conditioning variables.
        residual_mha: Whether to use a residual connection in the mha layer.
        residual_transition: Whether to use a residual connection in the transition layer.
        parallel_mha_transition: Whether to run mha and transition in parallel or sequentially.
        use_attn_pair_bias: Whether to use a pair represnetation to bias attention.
        use_qkln: Whether to use layer norm on keyus and queries for attention.
        dropout: droput use in the self-attention layer.
        attention_type: "torch" (SDPA, default) or "cuequivariance" (fused kernel).
    """

    def __init__(
        self,
        dim_token: int,
        dim_pair: int,
        nheads: int,
        dim_cond: int,
        residual_mha: bool,
        residual_transition: bool,
        parallel_mha_transition: bool,
        use_attn_pair_bias: bool,
        use_qkln: bool,
        dropout: float = 0.0,
        expansion_factor: int = 4,
        attention_type: str = "torch",
    ) -> None:
        super().__init__()
        self.parallel = parallel_mha_transition
        self.use_attn_pair_bias = use_attn_pair_bias

        # If parallel do not allow both layers to have a residual connection since it leads to adding x twice
        if self.parallel and residual_mha and residual_transition:
            residual_transition = False

        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        # Choose attention implementation: cuequivariance fused kernel, else SDPA.
        if attention_type == "cuequivariance":
            if not use_attn_pair_bias:
                raise ValueError("cuequivariance attention has no unbiased path; use attention_type='torch'")
            self.mhba = CuEqMultiHeadBiasedAttentionADALN_MM(
                dim_token=dim_token,
                dim_pair=dim_pair,
                nheads=nheads,
                dim_cond=dim_cond,
                use_qkln=use_qkln,
            )
        else:
            self.mhba = MultiHeadBiasedAttentionADALN_MM(
                dim_token=dim_token,
                dim_pair=dim_pair if use_attn_pair_bias else None,
                nheads=nheads,
                dim_cond=dim_cond,
                use_qkln=use_qkln,
            )

        self.transition = TransitionADALN(dim=dim_token, dim_cond=dim_cond, expansion_factor=expansion_factor)

    def _apply_mha(
        self,
        x: torch.Tensor,
        pair_rep: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x_attn = self.mhba(x, pair_rep if self.use_attn_pair_bias else None, cond, mask)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]

    def forward(self, x: torch.Tensor, pair_rep: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]
            pair_rep: Pair representation (if provided, if no bias will be ignored), shape [b, n, n, dim_pair] or None

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        x = x * mask[..., None]
        if self.parallel:
            x = self._apply_mha(x, pair_rep, cond, mask) + self._apply_transition(x, cond, mask)
        else:
            x = self._apply_mha(x, pair_rep, cond, mask)
            x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]
