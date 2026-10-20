import logging
from collections.abc import Sequence

import torch
from torch.utils.checkpoint import checkpoint

from rfproteina.encoding import UNIFIED_ATOM37_ENCODING
from rfproteina.model_masks import compute_flow_emit_masks
from rfproteina.nn.feature_factory import RECYCLE_GRAD_DEPTHS, FeatureFactory
from rfproteina.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from rfproteina.nn.modules.norm import make_norm, set_norm_type
from rfproteina.nn.modules.pair_rep_initial import PairReprBuilder
from rfproteina.nn.modules.pair_update import PairReprUpdate
from rfproteina.nn.modules.seq_transition_af3 import Transition
from rfproteina.utils.torch import raise_dynamo_cache_for
from rfproteina.utils.torch_compile import report_compile_progress, torch_compile_kwargs

CHECKPOINT_LEVELS = {
    "none": (False, False),
    "low": (False, True),
    "medium": (True, True),
}

# Dynamo caches one graph per (grad-mode, shape)
_TRAIN_COMPILE_CACHE_SLOTS = 16

logger = logging.getLogger(__name__)


def _configure_flash_attention_backends() -> None:
    """Configure PyTorch backends for optimal Flash Attention performance."""
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback


class LocalLatentsTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    #: Settings of this module's config that may change after training without touching the weights.
    #: ``rfd4-generate`` accepts exactly these as ``nn.<key>`` overrides; any other key would change
    #: the architecture and break strict weight loading.
    RUNTIME_OPTIONS = ("torch_compile_mode", "torch_compile_fullgraph", "tri_mult_backend", "attention_type")

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
        self.condition_on_recycle_remaining = "recycle_remaining" in (kwargs["feats_cond_seq"] or [])
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

        checkpointing = kwargs.get("checkpointing", "medium")
        try:
            self.recycle_checkpoint, self.checkpoint_pair_update = CHECKPOINT_LEVELS[checkpointing]
        except KeyError as exc:
            raise ValueError(
                f"nn.checkpointing must be one of {sorted(CHECKPOINT_LEVELS)}, got {checkpointing!r}"
            ) from exc

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
                            tri_mult_bias=kwargs.get("tri_mult_bias", True),
                            # TODO: Remove fallback to sum when checkpoint is updated
                            pair_write=kwargs.get("pair_write", "sum"),
                            checkpoint_tri_mult=self.checkpoint_pair_update,
                            tri_mult_backend=kwargs.get("tri_mult_backend", "cuequivariance"),
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        # Recycle single and pair states through zero-initialized projections
        recycle_cfg = kwargs.get("recycle", {}) or {}
        self.recycle_enabled = bool(recycle_cfg.get("enabled", False))
        self.recycle_grad = recycle_cfg.get("grad", "last")
        self.recycle_condition_probability = float(recycle_cfg.get("condition_probability", 0.5))
        if not 0 <= self.recycle_condition_probability <= 1:
            raise ValueError("nn.recycle.condition_probability must be between 0 and 1")

        if self.recycle_enabled:
            # Validate recycle backpropagation settings
            if self.recycle_grad != "all" and self.recycle_grad not in RECYCLE_GRAD_DEPTHS:
                raise ValueError(f"nn.recycle.grad must be one of last|last2|all, got {self.recycle_grad!r}")

            self.s_cycle_norm = make_norm(self.token_dim)
            self.s_cycle_proj = torch.nn.Linear(self.token_dim, self.token_dim, bias=False)
            torch.nn.init.zeros_(self.s_cycle_proj.weight)
            self.z_cycle_norm = make_norm(self.pair_repr_dim)
            self.z_cycle_proj = torch.nn.Linear(self.pair_repr_dim, self.pair_repr_dim, bias=False)
            torch.nn.init.zeros_(self.z_cycle_proj.weight)

        self.local_latents_linear = torch.nn.Sequential(
            make_norm(self.token_dim),
            torch.nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
        )
        self.ca_linear = torch.nn.Sequential(
            make_norm(self.token_dim),
            torch.nn.Linear(self.token_dim, 3, bias=False),
        )

        # Optional sequence readout. Keeping the probe behind an architecture config
        # preserves strict loading of older checkpoints, whose saved config has no head.
        sequence_readout_cfg = kwargs.get("sequence_readout", {}) or {}
        self.sequence_readout_enabled = sequence_readout_cfg.get("enabled", False)
        self.sequence_readout_detach = sequence_readout_cfg.get("detach", True)
        if self.sequence_readout_enabled:
            self.sequence_readout = torch.nn.Sequential(
                make_norm(self.token_dim),
                torch.nn.Linear(
                    self.token_dim,
                    sequence_readout_cfg.get("num_classes", UNIFIED_ATOM37_ENCODING.n_tokens),
                    bias=False,
                ),
            )

        # Insertion prediction head: bilinear [b, n, n] outer product over all token
        # pairs.  Only unindexed-query x indexed-key pairs contribute to the CE loss.
        self.insertion_q_proj = torch.nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.insertion_k_proj = torch.nn.Linear(self.token_dim, self.token_dim, bias=False)
        self._insertion_head_scale = self.token_dim**-0.5

        # Each trunk block is torch.compiled into off-tree lists (loop stays eager)
        self.use_torch_compile = kwargs.get("use_torch_compile", False)
        self._compiled_attn_blocks: list = []
        self._compiled_pair_blocks: list = []
        self._trunk_compiled = False
        self._checked_compile_shapes: set[tuple[int, int]] = set()

        # Compile only on shapes we know are fixed/bucketed (to avoid recompilation/thrashing)
        self._shapes_reserved = False
        self._compile_kwargs = torch_compile_kwargs(kwargs, default_fullgraph=True)
        if self.use_torch_compile:
            self._setup_compiled_trunk()

    def _setup_compiled_trunk(self) -> None:
        """Compile each trunk block into off-tree lists; the trunk loop stays eager."""
        logger.info("Enabling torch.compile per trunk block.")
        self._compiled_attn_blocks = [torch.compile(layer, **self._compile_kwargs) for layer in self.transformer_layers]
        if self.update_pair_repr:
            self._compiled_pair_blocks = [
                torch.compile(layer, **self._compile_kwargs) if layer is not None else None
                for layer in self.pair_update_layers
            ]
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, _TRAIN_COMPILE_CACHE_SLOTS)
        self._trunk_compiled = True

    def reserve_shape_cache(self, lengths: list[int]) -> None:
        """Declare fixed length buckets so eval may compile, and size the dynamo cache for them.

        Args:
          lengths: Padded token counts to reserve cache slots for, i.e. the length buckets.
        """
        if not self._trunk_compiled:
            # Compile the trunk blocks if not already done (e.g. if use_torch_compile was False at init).
            self._setup_compiled_trunk()

        # Reserve the cache for the given lengths, so eval can compile them without thrashing the training shapes
        self._shapes_reserved = True
        n_shapes = 2 * len(lengths) + 2  # one graph per bucket per grad-mode, plus the short final batch
        raise_dynamo_cache_for(n_shapes)
        logger.info(f"Reserved dynamo cache for {len(lengths)} length buckets ({n_shapes} shapes).")

    def forward(self, input: dict) -> dict[str, dict[str, torch.Tensor]]:
        """Runs the network.

        NOTE: The compiled unit is the per-block trunk pass (``_trunk_step``), used
        identically in training, validation, and inference; the surrounding feature and head code
        runs eagerly.

        Args:
            input: {
                # Sampling and training
                "x_t": Dict[str, torch.Tensor[b, n, dim]]
                "t": Dict[str, torch.Tensor[b]]
                "padding_mask": boolean torch.Tensor[b, n]

                # Only training (other batch elements)
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                ...
            }

        Returns:
            Dictionary:
            {
                "coors_nm": all atom coordinates, shape [b, n, 37, 3]
                "seq_logits": logits for the residue types, shape [b, n, 20]
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

        return self._forward_impl(input)

    def _forward_impl(self, input: dict) -> dict[str, dict[str, torch.Tensor]]:
        """Core forward logic. May be wrapped by torch.compile."""
        seq_mask = input["padding_mask"]  # [b, n] boolean, True = valid token

        # Conditioning variables
        c = self.cond_factory({**input, "recycle_remaining": None})  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, seq_mask), seq_mask)  # [b, n, dim_cond]

        # Initial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        s_init = seq_f_repr * seq_mask[..., None]  # [b, n, token_dim]

        z_init = self.pair_repr_builder(input)  # [b, n, n, pair_dim]

        # Run the trunk.
        s, _z = self._run_trunk(input, s_init, z_init, c, seq_mask)

        # Get outputs
        emit_masks = compute_flow_emit_masks(input)
        local_latents_out = self.local_latents_linear(s) * emit_masks["local_latents"][..., None]
        ca_nm_out = self.ca_linear(s) * emit_masks["bb_ca"][..., None]
        nn_out = {}
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: ca_nm_out}
        nn_out["local_latents"] = {self.output_param["local_latents"]: local_latents_out}

        if self.sequence_readout_enabled:
            sequence_seqs = s.detach() if self.sequence_readout_detach else s
            nn_out["sequence"] = {"logits": self.sequence_readout(sequence_seqs) * seq_mask[..., None]}

        # Detached probe for insertion head
        insertion_seqs = s.detach()
        q = self.insertion_q_proj(insertion_seqs)  # [b, n, d]
        k = self.insertion_k_proj(insertion_seqs)  # [b, n, d]
        nn_out["insertion"] = {"logits": torch.bmm(q, k.transpose(1, 2)) * self._insertion_head_scale}

        return nn_out

    def _apply_trunk(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        c: torch.Tensor,
        seq_mask: torch.Tensor,
        attn_layers: Sequence,
        pair_layers: Sequence,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one trunk pass over the given attention/pair block lists (compiled or registered)."""
        for i in range(self.nlayers):
            s = attn_layers[i](s, z, c, seq_mask)  # [b, n, token_dim]

            if self.update_pair_repr and i < self.nlayers - 1 and pair_layers[i] is not None:
                z = pair_layers[i](s, z, seq_mask)  # [b, n, n, pair_dim]

        return s, z

    def _trunk_step(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        c: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one trunk pass.

        Runs compiled if the model has compiled blocks and the pass is over
        a fixed shape (training or reserved eval buckets); otherwise, runs the registered eager layers.
        """
        use_compiled = bool(self._compiled_attn_blocks) and (self.training or self._shapes_reserved)
        attn = self._compiled_attn_blocks if use_compiled else self.transformer_layers

        if not self.update_pair_repr:
            pair: Sequence = []
        else:
            pair = self._compiled_pair_blocks if use_compiled else self.pair_update_layers

        shape = (s.shape[0], s.shape[1])
        if use_compiled and not self.training and shape not in self._checked_compile_shapes:
            with report_compile_progress(*shape):
                result = self._apply_trunk(s, z, c, seq_mask, attn, pair)
            self._checked_compile_shapes.add(shape)
            return result
        return self._apply_trunk(s, z, c, seq_mask, attn, pair)

    def _run_trunk(
        self,
        input: dict,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        c: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the configured number of trunk cycles.

        Each cycle reads the previous single and pair states through zero-initialized
        projections. Remaining-pass conditioning is absent on the first pass, and
        supplied for all later passes together with ``recycle_condition_probability``
        during training. ``recycle_grad`` sets gradient depth independently of conditioning width.
        """
        # (Routes to either eager or compiled trunk blocks, depending
        # on the model's compilation state and the input shape.)
        trunk = self._trunk_step

        if not self.recycle_enabled:
            return trunk(s_init, z_init, c, seq_mask)

        n_recycle = max(1, int(input.get("n_recycle", 1)))
        grad_k = n_recycle if self.recycle_grad == "all" else RECYCLE_GRAD_DEPTHS[self.recycle_grad]
        condition_remaining = self.condition_on_recycle_remaining and n_recycle > 1
        if condition_remaining:
            condition_remaining = (
                bool(torch.rand(()) < self.recycle_condition_probability)
                if self.training
                else input.get("condition_on_recycle_remaining", True)
            )

        # Keep both projections in the graph for every recycle depth.
        s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)
        for cycle in range(n_recycle):
            # SDPA requires a contiguous pair bias.
            z_in = (z_init + self.z_cycle_proj(self.z_cycle_norm(z))).contiguous()
            s_in = s_init + self.s_cycle_proj(self.s_cycle_norm(s))

            grad_on = torch.is_grad_enabled() and cycle >= n_recycle - grad_k
            with torch.set_grad_enabled(grad_on):
                if cycle > 0 and condition_remaining:
                    c = self.cond_factory({**input, "recycle_remaining": n_recycle - cycle - 1})
                    c = self.transition_c_2(self.transition_c_1(c, seq_mask), seq_mask)
                if grad_on and self.recycle_checkpoint:
                    s, z = checkpoint(trunk, s_in, z_in, c, seq_mask, use_reentrant=False)
                else:
                    s, z = trunk(s_in, z_in, c, seq_mask)
            if not grad_on:
                s, z = s.detach(), z.detach()

        return s, z
