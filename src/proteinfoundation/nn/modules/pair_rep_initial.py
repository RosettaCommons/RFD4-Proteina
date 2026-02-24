import torch

from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.adaptive_ln_scale import AdaptiveLayerNorm


class PairReprBuilder(torch.nn.Module):
    """
    Builds initial pair representation. Essentially the pair feature factory, but potentially with
    an adaptive layer norm layer as well.

    Args:
        feats_repr: List of feature names for pair representation
        feats_cond: List of feature names for conditioning (adaptive layernorm)
        dim_feats_out: Output dimension for pair representation
        dim_cond_pair: Output dimension for conditioning features
        use_atom37_factory: If True, use FeatureFactoryAtom37 (supports bond features).
                           If False, use old FeatureFactory (for autoencoder compatibility).
                           Defaults to False for backward compatibility.
        **kwargs: Additional arguments passed to feature factory
    """

    def __init__(self, feats_repr, feats_cond, dim_feats_out, dim_cond_pair, use_atom37_factory=False, **kwargs):
        super().__init__()

        # Choose which feature factory to use
        if use_atom37_factory:
            from proteinfoundation.nn.feature_factory_atom37 import FeatureFactory as FeatureFactoryAtom37

            factory_class = FeatureFactoryAtom37
        else:
            factory_class = FeatureFactory

        self.init_repr_factory = factory_class(
            feats=feats_repr,
            dim_feats_out=dim_feats_out,
            use_ln_out=True,
            mode="pair",
            **kwargs,
        )

        self.cond_factory = None  # Build a pair feature for conditioning and use it for adaln the pair representation
        if feats_cond is not None:
            if len(feats_cond) > 0:
                self.cond_factory = factory_class(
                    feats=feats_cond,
                    dim_feats_out=dim_cond_pair,
                    use_ln_out=True,
                    mode="pair",
                    **kwargs,
                )
                self.adaln = AdaptiveLayerNorm(dim=dim_feats_out, dim_cond=dim_cond_pair)

    def forward(self, batch_nn):
        mask = batch_nn["padding_mask"]  # [b, n]
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        repr = self.init_repr_factory(batch_nn)  # [b, n, n, dim_feats_out]
        if self.cond_factory is not None:
            cond = self.cond_factory(batch_nn)  # [b, n, n, dim_cond]
            repr = self.adaln(repr, cond, pair_mask)
        return repr
