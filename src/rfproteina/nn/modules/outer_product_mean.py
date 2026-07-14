import torch
import torch.nn as nn

# from rfproteina.utils.checkpoint import activation_checkpointing


class OuterProductMean_AF3(nn.Module):
    def __init__(self, c_msa_embed: int, c_outer_product: int, c_out: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c_msa_embed)
        self.proj_left = nn.Linear(c_msa_embed, c_outer_product, bias=False)
        self.proj_right = nn.Linear(c_msa_embed, c_outer_product, bias=False)
        self.proj_out = nn.Linear(c_outer_product * c_outer_product, c_out, bias=False)

    def forward(self, msa: torch.Tensor, mask_S: torch.Tensor) -> torch.Tensor:
        """Compute outer product mean of MSA representation.

        Args:
            msa: MSA embedding, shape (B, S, I, c_msa_embed).
            mask_S: Boolean mask where True = valid MSA row, shape (B, S).
        """
        B, N, I = msa.shape[:3]
        msa_norm = self.norm(msa)
        left = self.proj_left(msa_norm)
        right = self.proj_right(msa_norm)

        # Mask out padding sequences in the MSA sequence dimension
        left = left * mask_S[..., None, None].to(left.dtype)
        right = right * mask_S[..., None, None].to(right.dtype)

        # NOTE: dividing by N here makes the sum in the einsum a mean
        # Looks like a bug but is mathematically equivalent to dividing after the sum in the einsum
        num_valid_sequences = mask_S.sum(dim=-1).clamp(min=1)[..., None, None, None]
        right = right / num_valid_sequences.to(right.dtype)

        out = torch.einsum("bsli,bsmj->blmij", left, right).reshape(B, I, I, -1)
        out = self.proj_out(out)

        return out
