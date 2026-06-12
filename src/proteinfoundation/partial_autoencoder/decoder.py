import einops
import torch

from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition
from proteinfoundation.utils.constants import RESTYPE_ATOM37_MASK, UNIFIED_RESTYPE_ATOM37_MASK


def get_allowed_residue_type_mask(device: torch.device | None = None, n_residue_types: int = 20) -> torch.Tensor | None:
    """Get per-token-type allowed residue classes for AE decoding.

    Token type convention:
    - 0: protein
    - 1: RNA
    - 2: DNA
    - 3: atomized
    - 4: unknown
    """
    if n_residue_types != 33:
        return None

    allowed = torch.zeros((5, n_residue_types), dtype=torch.bool, device=device)
    allowed[0, 1:22] = True  # 20 AA + UNK
    allowed[1, 22:27] = True  # RNA + unknown RNA
    allowed[2, 27:32] = True  # DNA + unknown DNA
    allowed[3, 32] = True  # atomized token only
    allowed[4, 1:33] = True  # unknown token-type: allow any non-mask class
    return allowed


def get_atom_mask(device: torch.device | None = None, n_residue_types: int = 20) -> torch.Tensor:
    """Get atom37 mask for residue types.

    Args:
        device: torch device
        n_residue_types: Number of residue types (20 for legacy, 33 for unified)

    Returns:
        Atom37 mask tensor of shape [21, 37] or [33, 37]
    """
    if n_residue_types == 20:
        # Legacy: use hardcoded protein-only masks [21, 37]
        return torch.from_numpy(RESTYPE_ATOM37_MASK).to(dtype=torch.bool, device=device)
    elif n_residue_types == 33:
        # Unified: use precomputed masks for all UNIFIED_ATOM37_ENCODING types [33, 37]
        return torch.from_numpy(UNIFIED_RESTYPE_ATOM37_MASK).to(dtype=torch.bool, device=device)
    else:
        raise ValueError(f"Unsupported n_residue_types={n_residue_types}. Must be 20 or 33.")


class DecoderTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super().__init__()
        self.nlayers = kwargs["decoder"]["nlayers"]
        self.token_dim = kwargs["decoder"]["token_dim"]
        self.pair_repr_dim = kwargs["decoder"]["pair_repr_dim"]
        self.update_pair_repr = kwargs["decoder"]["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["decoder"]["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["decoder"]["use_tri_mult"]
        self.use_qkln = kwargs["decoder"]["use_qkln"]
        self.attention_type = kwargs["decoder"].get("attention_type", "torch")
        self.abs_coors = kwargs["decoder"].get("abs_coors", True)

        # Support both legacy (20 AAs) and unified (33 types) encodings
        # Check both top-level and decoder-level kwargs
        self.n_residue_types = kwargs.get("n_residue_types") or kwargs["decoder"].get("n_residue_types", 20)

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
        self.pair_rep_factory = FeatureFactory(
            feats=kwargs["decoder"]["feats_pair_repr"],
            dim_feats_out=kwargs["decoder"]["pair_repr_dim"],
            use_ln_out=False,
            mode="pair",
            **kwargs["decoder"],
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

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Runs the network.

        Args:
            input: {
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "residue_mask": boolean torch.Tensor(b, n)
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
        ca_coors_nm = input["ca_coors_nm"]  # [b, n, 3]
        mask = input["residue_mask"]  # [b, n] boolean

        # Conditioning variables
        c = self.cond_factory(input)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Iinitial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]

        pair_rep = self.pair_rep_factory(input)  # [b, n, n, pair_dim]

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)  # [b, n, token_dim]

            if self.update_pair_repr and i < self.nlayers - 1 and self.pair_update_layers[i] is not None:
                pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask)  # [b, n, n, pair_dim]

        # Get logits
        logits_out = self.logit_linear(seqs) * mask[..., None]  # [b, n, 20]

        if "token_type" in input:
            allowed_res_types = get_allowed_residue_type_mask(
                device=logits_out.device,
                n_residue_types=self.n_residue_types,
            )
            if allowed_res_types is not None:
                token_type = input["token_type"].long()
                max_token_type = allowed_res_types.shape[0] - 1
                invalid_token_type = (token_type < 0) | (token_type > max_token_type)
                if torch.any(invalid_token_type):
                    bad_values = torch.unique(token_type[invalid_token_type]).detach().cpu().tolist()
                    raise ValueError(
                        f"token_type must be in [0, {max_token_type}] for unified AE decoding; "
                        f"got invalid values {bad_values}"
                    )
                allowed_per_token = allowed_res_types[token_type] | ~mask[..., None]
                logits_out = logits_out.masked_fill(~allowed_per_token, torch.finfo(logits_out.dtype).min)

        # Get coordinates
        coors_flat_nm = self.struct_linear(seqs) * mask[..., None]  # [b, n, 37 * 3]
        coors_a37_nm = einops.rearrange(coors_flat_nm, "b n (a t) -> b n a t", a=37, t=3)  # [b, n, 37, 3]

        if self.abs_coors:
            coors_a37_nm[..., 1, :] = coors_a37_nm[..., 1, :] * 0.0 + ca_coors_nm
        else:
            coors_a37_nm[..., 1, :] = coors_a37_nm[..., 1, :] * 0.0
            coors_a37_nm = coors_a37_nm + ca_coors_nm[:, :, None, :]  # [b, n, 37, 3]

        # Get sequence
        aatype_max = torch.argmax(logits_out, dim=-1)  # [b, n]
        aatype_max = aatype_max * mask  # [b, n]

        # Get atom_mask - use appropriate size for legacy (20) or unified (33) encoding
        aa_a37_mask = get_atom_mask(
            device=logits_out.device, n_residue_types=self.n_residue_types
        )  # [n_residue_types, 37]
        atom_mask = aa_a37_mask[aatype_max, :]  # [b, n, 37] boolean
        atom_mask = atom_mask * mask[..., None]  # [b, n, 37] boolean

        output = {
            "coors_nm": coors_a37_nm,  # [b, n, 37, 3]
            "seq_logits": logits_out,  # [b, n, 20]
            "residue_mask": mask,  # [b, n]
            "aatype_max": aatype_max,  # [b, n]
            "atom_mask": atom_mask,  # [b, n, 37]
        }
        return output
