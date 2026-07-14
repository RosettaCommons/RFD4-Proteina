import torch

from rfproteina.nn.feature_factory import FeatureFactory
from rfproteina.nn.modules.adaptive_ln_scale import AdaptiveLayerNorm


class PairReprBuilder(torch.nn.Module):
    """Builds initial pair representation with optional adaptive layer norm conditioning.

    Args:
        feats_repr: List of feature names for pair representation.
        feats_cond: List of feature names for conditioning (adaptive layernorm).
        dim_feats_out: Output dimension for pair representation.
        dim_cond_pair: Output dimension for conditioning features.
        **kwargs: Additional arguments passed to feature factory.
    """

    def __init__(
        self,
        feats_repr: list,
        feats_cond: list | None,
        dim_feats_out: int,
        dim_cond_pair: int,
        **kwargs: object,
    ) -> None:
        super().__init__()

        self.init_repr_factory = FeatureFactory(
            feats=feats_repr,
            dim_feats_out=dim_feats_out,
            use_ln_out=True,
            mode="pair",
            **kwargs,
        )

        self.cond_factory = None
        if feats_cond is not None and len(feats_cond) > 0:
            self.cond_factory = FeatureFactory(
                feats=feats_cond,
                dim_feats_out=dim_cond_pair,
                use_ln_out=True,
                mode="pair",
                **kwargs,
            )
            self.adaln = AdaptiveLayerNorm(dim=dim_feats_out, dim_cond=dim_cond_pair)

    def forward(self, batch_nn: dict) -> torch.Tensor:
        mask = batch_nn["padding_mask"]  # [b, n]
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        repr = self.init_repr_factory(batch_nn)  # [b, n, n, dim_feats_out]
        if self.cond_factory is not None:
            cond = self.cond_factory(batch_nn)  # [b, n, n, dim_cond]
            repr = self.adaln(repr, cond, pair_mask)
        return repr
