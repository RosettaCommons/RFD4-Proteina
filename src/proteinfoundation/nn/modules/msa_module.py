from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from foundry.model.layers.blocks import Dropout
from proteinfoundation.nn.modules.outer_product_mean import (
    OuterProductMean_AF3,
)
from proteinfoundation.nn.modules.triangle import (
    TriangleAttention,
    TriangleMultiplication,
)
# from foundry.training.checkpoint import activation_checkpointing


class Transition(nn.Module):
    """SwiGLU feed-forward transition with expansion factor ``n``."""

    def __init__(self, n: int, c: int):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(c)
        self.linear_1 = nn.Linear(c, n * c, bias=False)
        self.linear_2 = nn.Linear(c, n * c, bias=False)
        self.linear_3 = nn.Linear(n * c, c, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.layer_norm_1(X)
        return self.linear_3(F.silu(self.linear_1(X)) * self.linear_2(X))


class MSAModule(nn.Module):
    def __init__(
        self,
        n_block: int,
        c_m: int,
        p_drop_msa: float,
        p_drop_pair: float,
        msa_subsample_embedder: dict,
        outer_product: dict,
        msa_pair_weighted_averaging: dict,
        msa_transition: dict,
        triangle_multiplication_outgoing: dict,
        triangle_multiplication_incoming: dict,
        triangle_attention_starting: dict,
        triangle_attention_ending: dict,
        pair_transition: dict,
        use_cuequivariance: bool = True,
        use_triangle_attention: bool = True,
    ) -> None:
        super().__init__()
        self.n_block = n_block
        self.msa_embedder = MSAEmbedder(**msa_subsample_embedder)
        # Final block does not update MSA features, so we use a different class
        self.blocks = nn.ModuleList(
            [
                MSAModuleBlock(
                    outer_product=outer_product,
                    msa_pair_weighted_averaging=msa_pair_weighted_averaging,
                    msa_transition=msa_transition,
                    triangle_multiplication_outgoing=triangle_multiplication_outgoing,
                    triangle_multiplication_incoming=triangle_multiplication_incoming,
                    triangle_attention_starting=triangle_attention_starting,
                    triangle_attention_ending=triangle_attention_ending,
                    pair_transition=pair_transition,
                    p_drop_msa=p_drop_msa,
                    p_drop_pair=p_drop_pair,
                    use_cuequivariance=use_cuequivariance,
                    use_triangle_attention=use_triangle_attention,
                )
                for _ in range(n_block - 1)
            ]
        )

        self.final_block = MSAModuleFinalBlock(
            outer_product=outer_product,
            msa_pair_weighted_averaging=msa_pair_weighted_averaging,
            msa_transition=msa_transition,
            triangle_multiplication_outgoing=triangle_multiplication_outgoing,
            triangle_multiplication_incoming=triangle_multiplication_incoming,
            triangle_attention_starting=triangle_attention_starting,
            triangle_attention_ending=triangle_attention_ending,
            pair_transition=pair_transition,
            p_drop_msa=p_drop_msa,
            p_drop_pair=p_drop_pair,
            use_cuequivariance=use_cuequivariance,
            use_triangle_attention=use_triangle_attention,
        )

    def forward(
        self,
        f: dict,
        Z_II: torch.Tensor,
        S_inputs_I: torch.Tensor,
        mask_I: torch.Tensor,
        mask_S: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of MSA module.

        Args:
            f: Dict containing 'msa_final' tensor.
            Z_II: Pair representation, shape (B, I, I, d_pair).
            S_inputs_I: Sequence inputs, shape (B, I, token_dim).
            mask_I: Boolean mask where True = valid token, shape (B, I).
            mask_S: Boolean mask where True = valid MSA row, shape (B, S).
        """
        msa = f["msa_final"]
        msa_SI = self.msa_embedder(msa, S_inputs_I)

        for i in range(self.n_block - 1):
            msa_SI, Z_II = self.blocks[i](
                msa_SI,
                Z_II,
                mask_I=mask_I,
                mask_S=mask_S,
            )
        Z_II = self.final_block(
            msa_SI,
            Z_II,
            mask_I=mask_I,
            mask_S=mask_S,
        )
        return Z_II


class MSAModuleBlock(nn.Module):
    def __init__(
        self,
        outer_product: dict,
        msa_pair_weighted_averaging: dict,
        msa_transition: dict,
        triangle_multiplication_outgoing: dict,
        triangle_multiplication_incoming: dict,
        triangle_attention_starting: dict,
        triangle_attention_ending: dict,
        pair_transition: dict,
        p_drop_msa: float,
        p_drop_pair: float,
        use_cuequivariance: bool = True,
        use_triangle_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_triangle_attention = use_triangle_attention
        self.outer_product = OuterProductMean_AF3(**outer_product)
        self.msa_pair_weighted_averaging = MSAPairWeightedAverage(**msa_pair_weighted_averaging)
        self.msa_transition = Transition(**msa_transition)

        self.drop_row_msa = Dropout(broadcast_dim=-2, p_drop=p_drop_msa)
        self.drop_row_pair = Dropout(broadcast_dim=-2, p_drop=p_drop_pair)
        self.drop_col_pair = Dropout(broadcast_dim=-3, p_drop=p_drop_pair)

        self.tri_mult_outgoing = TriangleMultiplication(
            d_pair=triangle_multiplication_outgoing["d_pair"],
            d_hidden=triangle_multiplication_outgoing["d_hidden"],
            direction="outgoing",
            bias=False,
            use_cuequivariance=use_cuequivariance,
        )
        self.tri_mult_incoming = TriangleMultiplication(
            d_pair=triangle_multiplication_incoming["d_pair"],
            d_hidden=triangle_multiplication_incoming["d_hidden"],
            direction="incoming",
            bias=False,
            use_cuequivariance=use_cuequivariance,
        )
        if self.use_triangle_attention:
            self.tri_attn_start = TriangleAttention(
                **triangle_attention_starting, start_node=True, use_cuequivariance=use_cuequivariance
            )
            self.tri_attn_end = TriangleAttention(
                **triangle_attention_ending, start_node=False, use_cuequivariance=use_cuequivariance
            )
        self.pair_transition = Transition(**pair_transition)

    def forward(
        self,
        msa_SI: torch.Tensor,
        Z_II: torch.Tensor,
        mask_I: torch.Tensor,
        mask_S: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # update MSA features
        Z_II = Z_II + self.outer_product(msa_SI, mask_S)
        msa_SI = msa_SI + self.drop_row_msa(self.msa_pair_weighted_averaging(msa_SI, Z_II, mask_I, mask_S))

        msa_SI = msa_SI + self.msa_transition(msa_SI)
        # update pair features
        Z_II = Z_II + self.drop_row_pair(self.tri_mult_outgoing(Z_II, mask_I))
        Z_II = Z_II + self.drop_row_pair(self.tri_mult_incoming(Z_II, mask_I))

        if self.use_triangle_attention:
            Z_II = Z_II + self.drop_row_pair(self.tri_attn_start(Z_II, mask_I))
            Z_II = Z_II + self.drop_col_pair(self.tri_attn_end(Z_II, mask_I))
        Z_II = Z_II + self.pair_transition(Z_II)

        return msa_SI, Z_II


class MSAModuleFinalBlock(MSAModuleBlock):
    """
    Final block of MSA module that does not update MSA features
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.msa_pair_weighted_averaging = None
        self.msa_transition = None
        self.drop_row_msa = None

    def forward(
        self,
        msa_SI: torch.Tensor,
        Z_II: torch.Tensor,
        mask_I: torch.Tensor,
        mask_S: torch.Tensor,
    ) -> torch.Tensor:
        # update pair features only
        Z_II = Z_II + self.drop_row_pair(self.outer_product(msa_SI, mask_S))
        Z_II = Z_II + self.drop_row_pair(self.tri_mult_outgoing(Z_II, mask_I))
        Z_II = Z_II + self.drop_row_pair(self.tri_mult_incoming(Z_II, mask_I))
        if self.use_triangle_attention:
            Z_II = Z_II + self.drop_row_pair(self.tri_attn_start(Z_II, mask_I))
            Z_II = Z_II + self.drop_col_pair(self.tri_attn_end(Z_II, mask_I))
        Z_II = Z_II + self.pair_transition(Z_II)
        return Z_II


class MSAEmbedder(nn.Module):
    def __init__(self, num_sequences: int, dim_raw_msa: int, c_msa_embed: int, c_s_inputs: int) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.emb_msa = nn.Linear(dim_raw_msa, c_msa_embed, bias=False)
        self.emb_S_inputs = nn.Linear(c_s_inputs, c_msa_embed, bias=False)

    def forward(
        self,
        msa_SI: torch.Tensor,  # (S, I, 34) (32 tokens + has_deletion + deletion value)
        S_inputs: torch.Tensor,  # (L, S_dim)
    ) -> torch.Tensor:
        # Embed the subsampled MSA
        # (NOTE: We subsample in the data loader to avoid memory issues)
        msa_SI = self.emb_msa(msa_SI)
        msa_SI = msa_SI + self.emb_S_inputs(S_inputs)[:, None]
        return msa_SI


class MSAPairWeightedAverage(nn.Module):
    """MSA pair weighted averaging."""

    def __init__(
        self,
        c_weighted_average: int,
        n_heads: int,
        c_msa_embed: int,
        c_z: int,
    ) -> None:
        super().__init__()
        self.weighted_average_channels = c_weighted_average
        self.n_heads = n_heads
        self.msa_channels = c_msa_embed
        self.pair_channels = c_z
        self.norm_msa = nn.LayerNorm(self.msa_channels, eps=1e-3)
        self.to_v = nn.Linear(self.msa_channels, self.n_heads * self.weighted_average_channels, bias=False)
        self.norm_pair = nn.LayerNorm(self.pair_channels)
        self.to_bias = nn.Linear(self.pair_channels, self.n_heads, bias=False)

        self.to_gate = nn.Linear(
            self.msa_channels,
            self.weighted_average_channels * self.n_heads,
            bias=False,
        )

        self.to_out = nn.Linear(self.weighted_average_channels * self.n_heads, self.msa_channels, bias=False)

    def forward(
        self,
        msa_SI: torch.Tensor,
        pair_II: torch.Tensor,
        mask_I: torch.Tensor,
        mask_S: torch.Tensor,
    ) -> torch.Tensor:
        """MSA pair-weighted averaging (Algorithm 10 from AF3 paper).

        Masks use True = valid convention throughout:
        1. Values at padding positions are zeroed out so they don't contribute.
        2. Softmax logits at padding positions get -inf so they have zero weight.

        Args:
            msa_SI: MSA embedding, shape (B, S, I, c_msa_embed).
            pair_II: Pair representation, shape (B, I, I, c_z).
            mask_I: Boolean mask where True = valid token, shape (B, I).
            mask_S: Boolean mask where True = valid MSA row, shape (B, S).
        """
        B, S, I = msa_SI.shape[:3]

        # normalize inputs
        msa_norm_SI = self.norm_msa(msa_SI)

        # construct values, bias and weights
        v_SIH = self.to_v(msa_norm_SI).reshape(B, S, I, self.n_heads, self.weighted_average_channels)

        # zero out contribution from padding tokens on length dimension and sequence dimension
        mask_SI = mask_S.unsqueeze(-1) & mask_I.unsqueeze(-2)  # (B, S, I) True = valid
        v_SIH = v_SIH * mask_SI[..., None, None].to(v_SIH.dtype)

        # construct biases
        bias_IIH = self.to_bias(self.norm_pair(pair_II))

        # padding tokens do not contribute to the Softmax (valid token probabilities should add to 1.0)
        mask_II = mask_I[..., None] & mask_I[..., None, :]  # (B, I, I) True = valid
        bias_IIH = bias_IIH - 1e9 * (~mask_II)[..., None].to(bias_IIH.dtype)

        # compute probabilities for weighted average
        w_IIH = F.softmax(bias_IIH, dim=-2)

        # construct gate
        gate_SIH = torch.sigmoid(self.to_gate(msa_SI))
        gate_SIH = gate_SIH * mask_SI[..., None].to(gate_SIH.dtype)

        # compute weighted average & apply gate
        weights = torch.einsum("bijh,bsjhc->bsihc", w_IIH, v_SIH).reshape(B, S, I, -1)
        o_SIH = gate_SIH * weights

        # concatenate heads and project
        msa_update_SI = self.to_out(o_SIH.reshape(B, S, I, -1))
        return msa_update_SI
