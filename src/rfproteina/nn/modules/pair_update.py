import torch
from jaxtyping import Bool, Float
from torch.utils.checkpoint import checkpoint

from rfproteina.nn.modules.openfold_modules.pair_transition import PairTransition
from rfproteina.nn.modules.openfold_modules.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

PAIR_WRITE_MODES = ("sum", "hadamard")


class PairReprUpdate(torch.nn.Module):
    """Update pair representations with an additive or Hadamard single-to-pair write."""

    def __init__(
        self,
        token_dim: int,
        pair_dim: int,
        expansion_factor_transition: int = 2,
        use_tri_mult: bool = False,
        tri_mult_c: int = 196,
        tri_mult_bias: bool = True,
        pair_write: str = "sum",
        checkpoint_tri_mult: bool = True,
        tri_mult_backend: str = "torch",
    ) -> None:
        super().__init__()
        assert pair_write in PAIR_WRITE_MODES, f"Unknown pair_write {pair_write!r}; expected {PAIR_WRITE_MODES}"

        self.use_tri_mult = use_tri_mult
        self.pair_write = pair_write
        self.checkpoint_tri_mult = checkpoint_tri_mult
        self.layer_norm_in = torch.nn.LayerNorm(token_dim)
        self.linear_x = torch.nn.Linear(token_dim, int(2 * pair_dim), bias=False)

        if use_tri_mult:
            kwargs = {"c_z": pair_dim, "c_hidden": tri_mult_c, "bias": tri_mult_bias, "backend": tri_mult_backend}
            self.tri_mult_out = TriangleMultiplicationOutgoing(**kwargs)
            self.tri_mult_in = TriangleMultiplicationIncoming(**kwargs)
        self.transition_out = PairTransition(c_z=pair_dim, n=expansion_factor_transition)

    def _apply_mask(self, pair_rep: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """
        pair_rep has shape [b, n, n, pair_dim]
        pair_mask has shape [b, n, n]
        """
        return pair_rep * pair_mask[..., None]

    def _write_single(
        self, x: Float[torch.Tensor, "b n d"], pair_rep: Float[torch.Tensor, "b n n p"]
    ) -> Float[torch.Tensor, "b n n p"]:
        """Add the configured single-to-pair write to ``pair_rep``."""
        pi, pj = self.linear_x(self.layer_norm_in(x)).chunk(2, dim=-1)
        if self.pair_write == "sum":
            return pair_rep + pi[:, None, :, :] + pj[:, :, None, :]
        return pair_rep + pi[:, None, :, :] * pj[:, :, None, :]

    def forward(
        self,
        x: Float[torch.Tensor, "b n d"],
        pair_rep: Float[torch.Tensor, "b n n p"],
        mask: Bool[torch.Tensor, "b n"],
    ) -> Float[torch.Tensor, "b n n p"]:
        """
        Args:
            x: Input sequence, shape [b, n, token_dim]
            pair_rep: Input pair representation, shape [b, n, n, pair_dim]
            mask: binary mask, shape [b, n]

        Returns:
            Updated pair representation, shape [b, n, n, pair_dim].
        """
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
        x = x * mask[..., None]  # [b, n, token_dim]
        pair_rep = self._write_single(x, pair_rep)
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        run = (
            (lambda module, *args: checkpoint(module, *args, use_reentrant=False))
            if self.checkpoint_tri_mult
            else (lambda module, *args: module(*args))
        )
        if self.use_tri_mult:
            pair_rep = pair_rep + run(self.tri_mult_out, pair_rep, pair_mask)
            pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
            pair_rep = pair_rep + run(self.tri_mult_in, pair_rep, pair_mask)
            pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        pair_rep = pair_rep + run(self.transition_out, pair_rep, pair_mask)
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        return pair_rep
