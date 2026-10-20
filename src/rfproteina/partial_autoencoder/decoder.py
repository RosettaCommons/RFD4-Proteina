import einops
import torch

from rfproteina.datapipes.schema.proteina import NUM_TOKEN_TYPES, TokenType
from rfproteina.encoding import RESIDUES_BY_TOKEN_TYPE, TOKEN_TYPE_IDS, UNIFIED_ATOM37_ENCODING
from rfproteina.model_masks import compute_ae_masks
from rfproteina.nn.feature_factory import FeatureFactory
from rfproteina.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from rfproteina.nn.modules.norm import set_norm_type
from rfproteina.nn.modules.pair_update import PairReprUpdate
from rfproteina.nn.modules.seq_transition_af3 import Transition
from rfproteina.utils.constants import _build_unified_atom37_mask


def get_allowed_residue_type_mask(device: torch.device | None = None) -> torch.Tensor:
    """Get per-token-type allowed residue classes for AE decoding.

    Token type convention:
    - 0: protein
    - 1: RNA
    - 2: DNA
    - 3: atomized
    - 4: unknown
    - 5: glycan
    - 6: NCAA/PTM
    """
    allowed = torch.zeros((NUM_TOKEN_TYPES, UNIFIED_ATOM37_ENCODING.n_tokens), dtype=torch.bool, device=device)
    for group, names in RESIDUES_BY_TOKEN_TYPE.items():
        if TOKEN_TYPE_IDS[group] != TokenType.UNKNOWN:
            allowed[TOKEN_TYPE_IDS[group], [UNIFIED_ATOM37_ENCODING.token_to_idx[name] for name in names]] = True
    allowed[TokenType.UNKNOWN] = allowed.any(dim=0)
    return allowed


def get_atom_mask(device: torch.device | None = None) -> torch.Tensor:
    """Return legal residue-by-Atom37 slots; export applies chain-dependent cleanup."""
    return torch.from_numpy(_build_unified_atom37_mask()).to(dtype=torch.bool, device=device)


class DecoderTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super().__init__()
        # Set the process-global norm type before building submodules (make_norm reads it).
        set_norm_type(kwargs["decoder"].get("norm_type"))
        self.n_residue_types = kwargs.get("n_residue_types")
        if self.n_residue_types != UNIFIED_ATOM37_ENCODING.n_tokens:
            raise ValueError(
                f"AE vocabulary requires {UNIFIED_ATOM37_ENCODING.n_tokens} classes, " f"got {self.n_residue_types}"
            )
        self.register_buffer(
            "allowed_residue_types",
            get_allowed_residue_type_mask(),
            persistent=False,
        )
        self.register_buffer("residue_atom_mask", get_atom_mask(), persistent=False)

        self.nlayers = kwargs["decoder"]["nlayers"]
        self.token_dim = kwargs["decoder"]["token_dim"]
        self.pair_repr_dim = kwargs["decoder"]["pair_repr_dim"]
        self.update_pair_repr = kwargs["decoder"]["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["decoder"]["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["decoder"]["use_tri_mult"]
        # Off drops the pair track entirely: no factory, no bias projection, no pair parameters.
        self.use_attn_pair_bias = kwargs["decoder"].get("use_attn_pair_bias", True)
        self.use_qkln = kwargs["decoder"]["use_qkln"]
        self.attention_type = kwargs["decoder"].get("attention_type", "torch")
        self.abs_coors = kwargs["decoder"].get("abs_coors", True)

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["decoder"]["feats_seq"],
            dim_feats_out=kwargs["decoder"]["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs["decoder"],
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["decoder"]["feats_cond_seq"],
            dim_feats_out=kwargs["decoder"]["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs["decoder"],
        )

        self.transition_c_1 = Transition(kwargs["decoder"]["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["decoder"]["dim_cond"], expansion_factor=2)

        # To get pair representation
        self.pair_rep_factory = (
            FeatureFactory(
                feats=kwargs["decoder"]["feats_pair_repr"],
                dim_feats_out=kwargs["decoder"]["pair_repr_dim"],
                use_ln_out=False,
                mode="pair",
                **kwargs["decoder"],
            )
            if self.use_attn_pair_bias
            else None
        )

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=kwargs["decoder"]["nheads"],
                    dim_cond=kwargs["decoder"]["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=self.use_attn_pair_bias,
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
                            token_dim=kwargs["decoder"]["token_dim"],
                            pair_dim=kwargs["decoder"]["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        # Sequence prediction head (uses self.n_residue_types set above)
        self.logit_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, self.n_residue_types, bias=False),
        )
        self.struct_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, (37 * 3), bias=False),
        )
        self.atom_presence_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 37, bias=False),
        )

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Runs the network.

        Args:
            input: {
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "padding_mask", "is_in_context_conditioning", "supports_local_latent": boolean torch.Tensor(b, n)
            }

        Returns:
            Dictionary:
            {
                "coors_nm": all atom coordinates, shape [b, n, 37, 3]
                "seq_logits": logits for the residue types, shape [b, n, n_residue_types]
                "emit_mask": boolean [b, n]
                "aatype_max": residue type by taking the most likely logit, shape [b, n],
                    with integer values {0, ..., n_residue_types - 1}
                "atom_presence_logits": presence logits [b, n, 37]
                "atom_mask": predicted presence within aatype_max's allowed slots [b, n, 37]
            }
        """
        ca_coors_nm = input["ca_coors_nm"]  # [b, n, 3]
        # Two token masks, derived from the regular batch (see :py:func:`compute_ae_masks`):
        #   attn_mask - tokens the decoder attends over; emit_mask - tokens it reconstructs.
        attn_mask, emit_mask = compute_ae_masks(input)

        # Conditioning variables
        c = self.cond_factory(input)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, attn_mask), attn_mask)  # [b, n, dim_cond]

        # Iinitial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * attn_mask[..., None]  # [b, n, token_dim]

        pair_rep = self.pair_rep_factory(input) if self.pair_rep_factory is not None else None

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](seqs, pair_rep, c, attn_mask)  # [b, n, token_dim]

            if self.update_pair_repr and i < self.nlayers - 1 and self.pair_update_layers[i] is not None:
                pair_rep = self.pair_update_layers[i](seqs, pair_rep, attn_mask)  # [b, n, n, pair_dim]

        # Get logits
        logits_out = self.logit_linear(seqs) * emit_mask[..., None]  # [b, n, n_residue_types]

        if "token_type" in input:
            allowed_res_types = self.allowed_residue_types
            token_type = input["token_type"].long()
            max_token_type = allowed_res_types.shape[0] - 1
            if not torch.compiler.is_compiling():
                invalid_token_type = (token_type < 0) | (token_type > max_token_type)
                if torch.any(invalid_token_type):
                    bad_values = torch.unique(token_type[invalid_token_type]).detach().cpu().tolist()
                    raise ValueError(
                        f"token_type must be in [0, {max_token_type}] for AE decoding; got invalid values {bad_values}"
                    )
            allowed_per_token = allowed_res_types[token_type] | ~emit_mask[..., None]
            logits_out = logits_out.masked_fill(~allowed_per_token, torch.finfo(logits_out.dtype).min)

        # Get coordinates
        coors_flat_nm = self.struct_linear(seqs) * emit_mask[..., None]  # [b, n, 37 * 3]
        coors_a37_nm = einops.rearrange(coors_flat_nm, "b n (a t) -> b n a t", a=37, t=3)  # [b, n, 37, 3]

        if self.abs_coors:
            coors_a37_nm[..., 1, :] = coors_a37_nm[..., 1, :] * 0.0 + ca_coors_nm
        else:
            coors_a37_nm[..., 1, :] = coors_a37_nm[..., 1, :] * 0.0
            coors_a37_nm = coors_a37_nm + ca_coors_nm[:, :, None, :]  # [b, n, 37, 3]
        coors_a37_nm = coors_a37_nm * emit_mask[..., None, None]

        # Get sequence
        aatype_max = torch.argmax(logits_out, dim=-1)  # [b, n]
        aatype_max = aatype_max * emit_mask  # [b, n]

        atom_mask = self.residue_atom_mask[aatype_max] & emit_mask[..., None]
        atom_presence_logits = self.atom_presence_linear(seqs)
        predicted_mask = atom_presence_logits > 0
        predicted_mask[..., 1] = True  # Every emitted token requires its reference atom.
        atom_mask = atom_mask & predicted_mask

        output = {
            "coors_nm": coors_a37_nm,  # [b, n, 37, 3]
            "seq_logits": logits_out,  # [b, n, n_residue_types]
            "emit_mask": emit_mask,  # [b, n]
            "aatype_max": aatype_max,  # [b, n]
            "atom_mask": atom_mask,  # [b, n, 37]
        }
        output["atom_presence_logits"] = atom_presence_logits
        return output
