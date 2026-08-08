import logging

import torch

from rfproteina.nn.feature_factory import FeatureFactory
from rfproteina.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from rfproteina.nn.modules.msa_module import MSAModule
from rfproteina.nn.modules.norm import make_norm, set_norm_type
from rfproteina.nn.modules.pair_rep_initial import PairReprBuilder
from rfproteina.nn.modules.pair_update import PairReprUpdate
from rfproteina.nn.modules.seq_transition_af3 import Transition

logger = logging.getLogger(__name__)


def _configure_flash_attention_backends() -> None:
    """Configure PyTorch backends for optimal Flash Attention performance."""
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class LocalLatentsTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super().__init__()
        # Set the process-global norm type before building submodules (make_norm reads it).
        set_norm_type(kwargs.get("norm_type"))
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_pair_repr = kwargs["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["use_tri_mult"]
        self.use_qkln = kwargs["use_qkln"]
        self.output_param = kwargs["output_parameterization"]
        self.attention_type = kwargs.get("attention_type", "torch")

        # Enable the efficient SDPA backends for the SDPA ("torch") attention path.
        if self.attention_type != "cuequivariance":
            _configure_flash_attention_backends()

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        # To get pair representation
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr"],
            feats_cond=kwargs["feats_pair_cond"],
            dim_feats_out=kwargs["pair_repr_dim"],
            dim_cond_pair=kwargs["dim_cond"],
            **kwargs,
        )

        # MSA module is optional
        self.use_msa_module = kwargs.get("use_msa_module", False)
        if self.use_msa_module:
            self.msa_module = MSAModule(**kwargs["msa_module"])
        else:
            self.msa_module = None

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=self.use_qkln,
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
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                            tri_mult_c=kwargs["tri_mult_c"],
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        self.local_latents_linear = torch.nn.Sequential(
            make_norm(self.token_dim),
            torch.nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
        )
        self.ca_linear = torch.nn.Sequential(
            make_norm(self.token_dim),
            torch.nn.Linear(self.token_dim, 3, bias=False),
        )

        # Insertion prediction head: bilinear [b, n, n] outer product over all token
        # pairs.  Only unindexed-query x indexed-key pairs contribute to the CE loss.
        self.insertion_q_proj = torch.nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.insertion_k_proj = torch.nn.Linear(self.token_dim, self.token_dim, bias=False)
        self._insertion_head_scale = self.token_dim**-0.5

        # Wrap the training forward with torch.compile(fullgraph=True). Val/inference
        # runs under no_grad; keeping a separate eager impl for those paths avoids
        # grad_mode-triggered dynamo recompiles of the shared compiled graph.
        self.use_torch_compile = kwargs.get("use_torch_compile", False)
        self._forward_eval = self._forward_impl  # eager unless enable_inference_compile() is called
        if self.use_torch_compile:
            logger.info("Enabling torch.compile(fullgraph=True) for LocalLatentsTransformer training forward.")
            self._forward_compiled = torch.compile(self._forward_impl, fullgraph=True)
        else:
            self._forward_compiled = self._forward_impl

    def enable_inference_compile(self, lengths: list[int]) -> None:
        """Compile the eval forward with one static graph per padded token count.

        Args:
          lengths: Padded token counts to compile for, i.e. the length buckets.
        """
        n_shapes = 2 * len(lengths) + 2  # one graph per bucket, plus the short final batch
        logger.info(
            f"Enabling torch.compile(dynamic=False) for LocalLatentsTransformer "
            f"inference forward (recompile budget: {n_shapes} shapes)."
        )
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, n_shapes)
        self._forward_eval = torch.compile(self._forward_impl, dynamic=False)

    def forward(self, input: dict) -> dict[str, dict[str, torch.Tensor]]:
        """
        Runs the network. Delegates to _forward_impl, which may be torch.compiled: the
        training path when ``use_torch_compile`` is set, the eval path only after
        :py:meth:`enable_inference_compile` (both are eager otherwise).

        Args:
            input: {
                # Sampling and training
                "x_t": Dict[str, torch.Tensor[b, n, dim]]
                "t": Dict[str, torch.Tensor[b]]
                "padding_mask": boolean torch.Tensor[b, n]

                # Only training (other batch elements)
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "residue_mask": boolean torch.Tensor(b, n)
                ...
            }

        Returns:
            Dictionary:
            {
                "coors_nm": all atom coordinates, shape [b, n, 37, 3]
                "seq_logits": logits for the residue types, shape [b, n, 20]
                "residue_mask": boolean [b, n]
                "aatype_max": residue type by taking the most likely logit, shape [b, n],
                    with integer values {0, ..., 19}
                "atom_mask": boolean [b, n, 37], atom37 mask corresponding to aatype_max
            }
        """
        if self.use_torch_compile and self.training:
            # torch.compile miscompiles the layer-norm backward into a misaligned CUDA access
            # unless the token dim is 16-aligned. Fail fast instead of crashing deep in a kernel.
            n_tokens = input["padding_mask"].shape[1]
            assert n_tokens % 16 == 0, (
                f"nn.use_torch_compile requires a 16-aligned token dim (got n={n_tokens}); set the train "
                "collate_fn's fixed_dimensions to an aligned value, e.g. {I: 256}."
            )
        impl = self._forward_compiled if self.training else self._forward_eval
        return impl(input)

    def _forward_impl(self, input: dict) -> dict[str, dict[str, torch.Tensor]]:
        """Core forward logic. May be wrapped by torch.compile."""
        seq_mask = input["padding_mask"]  # [b, n] boolean, True = valid token
        msa_mask = input.get("msa_row_mask")  # [b, m] boolean or None, True = valid row

        # Conditioning variables
        c = self.cond_factory(input)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, seq_mask), seq_mask)  # [b, n, dim_cond]
        # Initial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * seq_mask[..., None]  # [b, n, token_dim]

        pair_rep = self.pair_repr_builder(input)  # [b, n, n, pair_dim]

        if self.use_msa_module:
            # All masks use True = valid convention (same as rest of codebase)
            pair_rep = self.msa_module(input, pair_rep, seqs, seq_mask, msa_mask)

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](seqs, pair_rep, c, seq_mask)  # [b, n, token_dim]

            if self.update_pair_repr and i < self.nlayers - 1 and self.pair_update_layers[i] is not None:
                pair_rep = self.pair_update_layers[i](seqs, pair_rep, seq_mask)  # [b, n, n, pair_dim]

        # Get outputs
        local_latents_out = self.local_latents_linear(seqs) * seq_mask[..., None]  # [b, n, latent_dim]
        ca_nm_out = self.ca_linear(seqs) * seq_mask[..., None]  # [b, n, 3]
        nn_out = {}
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: ca_nm_out}
        nn_out["local_latents"] = {self.output_param["local_latents"]: local_latents_out}

        # Detached probe for insertion head
        insertion_seqs = seqs.detach()
        q = self.insertion_q_proj(insertion_seqs)  # [b, n, d]
        k = self.insertion_k_proj(insertion_seqs)  # [b, n, d]
        nn_out["insertion"] = {"logits": torch.bmm(q, k.transpose(1, 2)) * self._insertion_head_scale}

        return nn_out
