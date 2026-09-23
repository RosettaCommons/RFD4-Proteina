import torch

from rfproteina.model_masks import compute_ae_masks
from rfproteina.nn.execution import execution_blocks
from rfproteina.nn.feature_factory import FeatureFactory
from rfproteina.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from rfproteina.nn.modules.norm import set_norm_type
from rfproteina.nn.modules.pair_update import PairReprUpdate
from rfproteina.nn.modules.seq_transition_af3 import Transition
from rfproteina.utils.torch_compile import torch_compile_kwargs


class EncoderTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super().__init__()
        self._compile_kwargs = torch_compile_kwargs(kwargs)
        # Set the process-global norm type before building submodules (make_norm reads it).
        set_norm_type(kwargs["encoder"].get("norm_type"))
        self.nlayers = kwargs["encoder"]["nlayers"]
        self.token_dim = kwargs["encoder"]["token_dim"]
        self.pair_repr_dim = kwargs["encoder"]["pair_repr_dim"]
        self.update_pair_repr = kwargs["encoder"]["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["encoder"]["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["encoder"]["use_tri_mult"]
        # Off drops the pair track entirely: no factory, no bias projection, no pair parameters.
        self.use_attn_pair_bias = kwargs["encoder"].get("use_attn_pair_bias", True)
        self.normalize_latent = kwargs["encoder"]["normalize_latent"]
        self.attention_type = kwargs["encoder"].get("attention_type", "torch")

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["encoder"]["feats_seq"],
            dim_feats_out=kwargs["encoder"]["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs["encoder"],
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["encoder"]["feats_cond_seq"],
            dim_feats_out=kwargs["encoder"]["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs["encoder"],
        )

        self.transition_c_1 = Transition(kwargs["encoder"]["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["encoder"]["dim_cond"], expansion_factor=2)

        # To get pair representation
        self.pair_rep_factory = (
            FeatureFactory(
                feats=kwargs["encoder"]["feats_pair_repr"],
                dim_feats_out=kwargs["encoder"]["pair_repr_dim"],
                use_ln_out=False,
                mode="pair",
                **kwargs["encoder"],
            )
            if self.use_attn_pair_bias
            else None
        )

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=kwargs["encoder"]["token_dim"],
                    dim_pair=kwargs["encoder"]["pair_repr_dim"],
                    nheads=kwargs["encoder"]["nheads"],
                    dim_cond=kwargs["encoder"]["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=self.use_attn_pair_bias,
                    use_qkln=kwargs["encoder"]["use_qkln"],
                    attention_type=self.attention_type,
                )
                for _ in range(self.nlayers)
            ]
        )

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["encoder"]["token_dim"],
                            pair_dim=kwargs["encoder"]["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        self.latent_decoder_mean_n_log_scale = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, int(2 * kwargs["latent_z_dim"]), bias=False),
        )

        self.ln_z = torch.nn.Identity()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Runs the network.

        Args:
            batch: batch from dataset

        Returns:
            Dictionary
            ...
        """
        # Two token masks, derived from the regular batch (see :py:func:`compute_ae_masks`):
        #   attn_mask - tokens the encoder attends over; emit_mask - tokens it emits a latent for.
        attn_mask, emit_mask = compute_ae_masks(batch)

        # Conditioning variables
        c = self.cond_factory(batch)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, attn_mask), attn_mask)  # [b, n, dim_cond]

        # Iinitial sequence representation from features
        seq_f_repr = self.init_repr_factory(batch)  # [b, n, token_dim]
        seqs = seq_f_repr * attn_mask[..., None]  # [b, n, token_dim]

        pair_rep = self.pair_rep_factory(batch) if self.pair_rep_factory is not None else None  # [b, n, n, pair_dim]

        # Run trunk
        attn_layers, pair_layers = execution_blocks(self, attn_mask)
        for i in range(self.nlayers):
            seqs = attn_layers[i](seqs, pair_rep, c, attn_mask)  # [b, n, token_dim]

            if self.update_pair_repr and i < self.nlayers - 1 and pair_layers[i] is not None:
                pair_rep = pair_layers[i](seqs, pair_rep, attn_mask)  # [b, n, n, pair_dim]

        # Emit only meaningful local latents; occupancy plays no part here.
        flat_out = self.latent_decoder_mean_n_log_scale(seqs) * emit_mask[..., None]  # [b, n, 2 * latent_dim]
        mean, log_scale = torch.chunk(flat_out, chunks=2, dim=-1)  # [b, n, latent_dim] each

        z = mean + torch.randn_like(log_scale) * torch.exp(log_scale)  # [b, n, latent_dim]

        z_pre_ln = z * emit_mask[..., None]
        z = self.ln_z(z) * emit_mask[..., None]  # [b, n, latent_dim]
        output = {
            "mean": mean,
            "log_scale": log_scale,
            "z_latent": z,
            "z_latent_pre_ln": z_pre_ln,
            "emit_mask": emit_mask,
        }
        return output
