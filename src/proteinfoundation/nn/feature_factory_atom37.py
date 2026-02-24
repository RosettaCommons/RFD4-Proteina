import math
import os
from typing import Literal
import numpy as np

import einops
import torch
from jaxtyping import Float
from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)
from atomworks.ml.conditions import C_SEQ
from torch.nn import functional as F

from proteinfoundation.conditions import RasaEnum, C_CATH
from proteinfoundation.utils.angle_utils import bond_angles, signed_dihedral_angle

################################
# # Some auxiliary functions # #
################################


# From frameflow code
def get_index_embedding(indices, edim, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of type integer, shape either [n] or [b, n].
        edim: dimension of the embeddings to create.
        max_len: maximum length.

    Returns:
        positional embedding of shape either [n, edim] or [b, n, edim]
    """
    # indices [n] of [b, n]
    K = torch.arange(edim // 2, device=indices.device)  # [edim / 2]

    if len(indices.shape) == 1:  # [n]
        K = K[None, ...]
    elif len(indices.shape) == 2:  # [b, n]
        K = K[None, None, ...]

    pos_embedding_sin = torch.sin(indices[..., None] * math.pi / (max_len ** (2 * K / edim))).to(indices.device)
    # [n, 1] / [1, edim/2] -> [n, edim/2] or [b, n, 1] / [1, 1, edim/2] -> [b, n, edim/2]
    pos_embedding_cos = torch.cos(indices[..., None] * math.pi / (max_len ** (2 * K / edim))).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)  # [n, edim]
    return pos_embedding


def get_time_embedding(
    t: Float[torch.Tensor, "b"], edim: int, max_positions: int = 2000
) -> Float[torch.Tensor, "b embdim"]:
    """
    Code from Frameflow, which got it from
    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

    Creates embedding for a given vector of times t.

    Args:
        t: vector of times (float) of shape [b].
        edim: dimension of the embeddings.
        max_positions: ...

    Returns:
        Embedding for the vector t of shape [b, edim]
    """
    assert len(t.shape) == 1
    t = t * max_positions
    half_dim = edim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if edim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (t.shape[0], edim)
    return emb


def bin_pairwise_distances(x, min_dist, max_dist, dim):
    """
    Takes coordinates and bins the pairwise distances.

    Args:
        x: Coordinates of shape [b, n, 3]
        min_dist: Right limit of first bin
        max_dist: Left limit of last bin
        dim: Dimension of the final one hot vectors

    Returns:
        Tensor of shape [b, n, n, dim] consisting of one-hot vectors
    """
    pair_dists_nm = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # [b, n, n]
    bin_limits = torch.linspace(min_dist, max_dist, dim - 1, device=x.device)  # Open left and right
    return bin_and_one_hot(pair_dists_nm, bin_limits)  # [b, n, n, pair_dist_dim]


def bin_and_one_hot(tensor, bin_limits):
    """
    Converts a tensor of shape [*] to a tensor of shape [*, d] using the given bin limits.

    Args:
        tensor (Tensor): Input tensor of shape [*]
        bin_limits (Tensor): bin limits [l1, l2, ..., l_{d-1}]. d-1 limits define
            d-2 bins, and the first one is <l1, the last one is >l_{d-1}, giving a total of d bins.

    Returns:
        torch.Tensor: Output tensor of shape [*, d] where d = len(bin_limits) + 1
    """
    bin_indices = torch.bucketize(tensor, bin_limits)
    return torch.nn.functional.one_hot(bin_indices, len(bin_limits) + 1) * 1.0


def indices_force_start_w_one(pdb_idx, mask):
    """
    Takes a tensor with pdb indices for a batch and forces them all to start with the index 1.
    Masked elements are still assigned the index -1.

    Args:
        pdb_idx: tensor of increasing integers (except masked ones fixed to -1), shape [b, n]
        mask: binary tensor, shape [b, n]

    Returns:
        pdb_idx but now all rows start at 1, masked elements are still set to -1.
    """
    pdb_idx_with_masked_as_inf = torch.masked_fill(pdb_idx, ~mask, torch.iinfo(pdb_idx.dtype).max)
    minimal_non_masked_val = torch.min(pdb_idx_with_masked_as_inf, dim=-1, keepdim=True).values
    pdb_idx = pdb_idx - minimal_non_masked_val + 1
    pdb_idx = torch.masked_fill(pdb_idx, ~mask, -1)  # set masked elements to -1
    return pdb_idx


################################
# # Classes for each feature # #
################################


class Feature(torch.nn.Module):
    """Base class for features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def get_dim(self):
        return self.dim

    def forward(self, batch: dict):
        pass  # Implemented by each class

    def extract_bs_and_n(self, batch: dict):
        """
        Extracts batch size and n from input batch.
        Once we add more modalities just try modaliites until you hit one.
        Other option is to pass arguments to nn which modalities are used.
        """
        if "x_t" in batch:
            if "bb_ca" in batch["x_t"]:
                v = batch["x_t"]["bb_ca"]  # [b, n, 3]
        elif "coords" in batch:
            v = batch["coords"]  # [b, n, 37, 3]
        elif "z_latent" in batch:
            v = batch["z_latent"]  # [b, n, latent_dim]
        else:
            raise OSError("Don't know how to extract batch size and n from batch...")
        bs, n = v.shape[0], v.shape[1]
        return bs, n

    def extract_device(self, batch: dict):
        """Extracts device from input batch."""
        if "x_t" in batch:
            if "bb_ca" in batch["x_t"]:
                v = batch["x_t"]["bb_ca"]  # [b, n, 3]
        elif "coords" in batch:
            v = batch["coords"]  # [b, n, 37, 3]
        elif "z_latent" in batch:
            v = batch["z_latent"]  # [b, n, latent_dim]
        else:
            raise OSError("Don't know how to extract device from batch...")
        return v.device

    def assert_defaults_allowed(self, batch: dict, ftype: str):
        """Raises error if default features should not be used to fill-up missing features in the current batch."""
        if "strict_feats" in batch:
            if batch["strict_feats"]:
                raise OSError(
                    f"{ftype} feature requested but no appropriate feature provided. "
                    "Make sure to include the relevant transform in the data config."
                )


class ZeroFeat(Feature):
    """Computes empty feature (zero) of shape [b, n, dim] or [b, n, n, dim],
    depending on sequence or pair features."""

    def __init__(
        self,
        dim_feats_out=128,
        mode: Literal["seq", "pair"] = "seq",
        name=None,
        **kwargs,
    ):
        super().__init__(dim=128)
        self.mode = mode

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if self.mode == "seq":
            return torch.zeros((b, n, self.dim), device=device)
        elif self.mode == "pair":
            torch.zeros((b, n, n, self.dim_feats_out), device=device)
        else:
            raise OSError(f"Mode {self.mode} wrong for zero feature")


class CroppedFlagSeqFeat(Feature):
    """Computes feature of shape [b, n, 1] indicating if protein is cropped
    (1s if it is, 0s if it isn't)."""

    def __init__(self):
        super().__init__(dim=1)

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if "cropped" in batch:
            ones = torch.ones((b, n, self.dim), device=device)
            cropped = batch["cropped"]  # boolean [b]
            return ones * cropped[..., None, None]  # [b, n, dim(=1)]
        else:
            return torch.zeros((b, n, self.dim), device=device)


class FoldEmbeddingSeqFeat(Feature):
    """Computes fold class embedding and returns as sequence feature of shape [b, n, fold_emb_dim * 3]."""

    def __init__(
        self,
        cath_emb_dim=32,
        **kwargs,
    ):
        """
        Args:
            cath_emb_dim (int): Dimension of the embedding
        """
        super().__init__(dim=cath_emb_dim * 3)

        self.register_buffer("_device_param", torch.tensor(0), persistent=False)

        # We add an extra class for null values
        self.embedding_C = torch.nn.Embedding(5 + 1, cath_emb_dim)
        self.embedding_A = torch.nn.Embedding(43 + 1, cath_emb_dim)
        self.embedding_T = torch.nn.Embedding(1472 + 1, cath_emb_dim)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, batch):
        # Add 1 because padding/null values are set to -1
        cath_codes = batch[C_CATH.full_name] + 1  # [b, n, 3]

        cath_embeddings = torch.cat(
            [
                self.embedding_C(cath_codes[:, :, 0]),
                self.embedding_A(cath_codes[:, :, 1]),
                self.embedding_T(cath_codes[:, :, 2]),
            ],
            dim=-1,
        )  # [b, n, fold_emb_dim * 3]

        return cath_embeddings  # [b, n, fold_emb_dim * 3]


class TimeEmbeddingSeqFeat(Feature):
    """Computes time embedding and returns as sequence feature of shape [b, n, t_emb_dim]."""

    def __init__(self, data_mode_use, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use

    def forward(self, batch):
        t = batch["t"][self.data_mode_use]  # [b]
        _, n = self.extract_bs_and_n(batch)
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, :]  # [b, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, t_emb.shape[2]))  # [b, n, t_emb_dim]


class TimeEmbeddingPairFeat(Feature):
    """Computes time embedding and returns as pair feature of shape [b, n, n, t_emb_dim]."""

    def __init__(self, data_mode_use, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use

    def forward(self, batch):
        t = batch["t"][self.data_mode_use]  # [b]
        _, n = self.extract_bs_and_n(batch)
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, None, :]  # [b, 1, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, n, t_emb.shape[3]))  # [b, n, t_emb_dim]


class IdxEmbeddingSeqFeat(Feature):
    """Computes index embedding and returns sequence feature of shape [b, n, idx_emb]."""

    def __init__(self, idx_emb_dim, **kwargs):
        super().__init__(dim=idx_emb_dim)

    def forward(self, batch):
        if "within_index_group_res_idx" in batch:
            inds = batch["within_index_group_res_idx"]  # [b, n]
            inds = indices_force_start_w_one(inds, batch["padding_mask"])
        else:
            self.assert_defaults_allowed(batch, "Residue index sequence")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(device)  # [b, n]
        return get_index_embedding(inds, edim=self.dim)  # [b, n, idx_embed_dim]


class ChainBreakPerResidueSeqFeat(Feature):
    """Computes a 1D sequence feature indicating if a residue is followed by a chain break, shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "chain_breaks_per_residue" in batch:
            chain_breaks = batch["chain_breaks_per_residue"] * 1.0  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Chain break sequence")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            chain_breaks = torch.zeros((b, n), device=device) * 1.0  # [b, n]
        return chain_breaks[..., None]  # [b, n, 1]


class XscBBCASeqFeat(Feature):
    """Computes feature from backbone CA self conditining coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, mode_key="x_sc", **kwargs):
        super().__init__(dim=3)
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert (
                "bb_ca" in data_modes_avail
            ), f"`bb_ca` sc/recycle seq feature requested but key not available in data modes {data_modes_avail}"
            return batch[self.mode_key]["bb_ca"]  # [b, n, 3]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(f"No {self.mode_key} in batch, returning zeros for XscBBCASeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, 3, device=device)


class XscLocalLatentsSeqFeat(Feature):
    """Computes feature from local latents self conditining, seq feature of shape [b, n, dim]."""

    def __init__(self, latent_dim, mode_key="x_sc", **kwargs):
        super().__init__(dim=latent_dim)
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert (
                "local_latents" in data_modes_avail
            ), f"`local_latents` sc/recycle seq feature requested but key not available in data modes {data_modes_avail}"
            return batch[self.mode_key]["local_latents"]  # [b, n, latent_dim]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(f"No {self.mode_key} in batch, returning zeros for XscLocalLatentsSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class XtBBCASeqFeat(Feature):
    """Computes feature from backbone CA x_t coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert (
            "bb_ca" in data_modes_avail
        ), f"`bb_ca` seq feat feature requested but key not available in data modes {data_modes_avail}"
        return batch["x_t"]["bb_ca"]  # [b, n, 3]


class XtLocalLatentsSeqFeat(Feature):
    """Computes feature from backbone CA x_t coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, latent_dim, **kwargs):
        super().__init__(dim=latent_dim)

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert (
            "local_latents" in data_modes_avail
        ), f"`local_latents` seq feat feature requested but key not available in data modes {data_modes_avail}"
        return batch["x_t"]["local_latents"]  # [b, n, latent_dim]


class CaCoorsNanometersSeqFeat(Feature):
    """Computes feature from ca coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        assert (
            "ca_coors_nm" in batch or "coords_nm" in batch
        ), "`ca_coors_nm` nor `coords_nm` in batch, cannot compute CaCoorsNanometersSeqFeat"
        if "ca_coors_nm" in batch:
            return batch["ca_coors_nm"]  # [b, n, 3]
        else:
            return batch["coords_nm"][:, :, 1, :]  # [b, n, 3]


class TryCaCoorsNanometersSeqFeat(CaCoorsNanometersSeqFeat):
    """
    If `ca_coors_nm` in batch, returns sequence feature with CA coordinates (in nm) of shape [b, n, 3].

    If `ca_coors_nm` not in batch return zero feature.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch):
        if "ca_coors_nm" in batch or "coords_nm" in batch:
            return super().forward(batch)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No ca_coors_nm or coords_nm in batch, returning zeros for TryCaCoorsNanometersSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class ResidueTypeSeqFeat(Feature):
    """
    Computes feature from residue type, feature of shape [b, n, 20].

    Residue type is an integer in {0, 1, ..., 19}, coorsponding to the 20 aa types.
    Feature is a one-hot vector of dimension 20.

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=20)

    def forward(self, batch):
        assert "residue_type" in batch, "`residue_type` not in batch, cannot compute ResidueTypeSeqFeat"
        rtype = batch["residue_type"]  # [b, n]
        rpadmask = batch["padding_mask"]  # [b, n] binary
        rtype = rtype * rpadmask  # [b, n], the -1 padding becomes 0
        rtype_onehot = F.one_hot(rtype, num_classes=20)  # [b, n, 20]
        rtype_onehot = rtype_onehot * rpadmask[..., None]
        return rtype_onehot * 1.0


class ConditionalResidueTypeSeqFeat(Feature):
    """
    Computes feature from conditional residue type, feature of shape [b, n, 33].

    Uses `C_SEQ.full_name` if available, otherwise returns zeros.
    This prevents leakage of ground truth sequence information during training.

    The feature is masked by `C_SEQ.mask_name` which indicates which tokens
    have sequence conditioning applied (zeros for to-be-generated tokens).

    Supports 33 classes:
    0: mask token
    1-20: standard amino acids
    21: unknown amino acid
    22-25: 4 RNA nucleotides
    26: unknown RNA nucleotide
    27-30: 4 DNA nucleotides
    31: unknown DNA nucleotide
    32: atomised token
    """

    def __init__(self, num_classes=33, learnable=False, **kwargs):
        super().__init__(dim=num_classes)
        self._has_logged = False
        self.num_classes = num_classes

        # Create embedding layer initialized as identity matrix (one-hot behavior)
        self.embedding = torch.nn.Embedding(num_classes, num_classes)
        if not learnable:
            # Initialize as identity matrix for one-hot behavior
            self.embedding.weight.data = torch.eye(num_classes)
            self.embedding.weight.requires_grad = False

    def forward(self, batch):
        if C_SEQ.full_name in batch and C_SEQ.mask_name in batch:
            rtype = batch[C_SEQ.full_name]  # [b, n] - contains <M> for unconditioned tokens
            seq_cond_mask = batch[C_SEQ.mask_name]  # [b, n] - True for sequence-conditioned tokens

            # Apply sequence conditioning mask
            rtype = rtype * seq_cond_mask  # [b, n]

            # Validate indices before embedding lookup
            invalid_mask = (rtype < 0) | (rtype >= self.num_classes)
            if invalid_mask.any():
                invalid_indices = torch.where(invalid_mask)
                invalid_values = rtype[invalid_mask]
                example_ids = batch.get("example_id_batch", ["unknown"] * rtype.shape[0])
                batch_indices = invalid_indices[0].tolist()
                token_indices = invalid_indices[1].tolist()
                raise ValueError(
                    f"ConditionalResidueTypeSeqFeat: Invalid residue type indices for embedding "
                    f"(expected 0-{self.num_classes - 1}, got values: {invalid_values.tolist()}). "
                    f"Batch indices: {batch_indices[:10]}{'...' if len(batch_indices) > 10 else ''}, "
                    f"Token indices: {token_indices[:10]}{'...' if len(token_indices) > 10 else ''}, "
                    f"Example IDs: {[example_ids[i] for i in set(batch_indices[:5])]}"
                )

            # Use embedding layer (acts like one-hot if initialized as identity)
            rtype_embedded = self.embedding(rtype)  # [b, n, num_classes]
            rtype_embedded = rtype_embedded * seq_cond_mask[..., None]
            return rtype_embedded
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    f"No {C_SEQ.full_name} or {C_SEQ.mask_name} in batch, "
                    f"returning zeros for ConditionalResidueTypeSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.num_classes, device=device)


class TokenTypeSeqFeat(Feature):
    """Token type embedding (protein/RNA/DNA/atomized/unknown).

    Reads token_type from batch (populated by collate) and embeds it.

    Token types:
        0: protein (AAs 1-21)
        1: RNA (22-26)
        2: DNA (27-31)
        3: atomized (32)
        4: unknown (mask/padding)

    Args:
        token_type_emb_dim: Embedding dimension. Defaults to 32.
    """

    def __init__(self, token_type_emb_dim: int = 32, **kwargs):
        super().__init__(dim=token_type_emb_dim)
        self.token_type_emb_dim = token_type_emb_dim
        self.embedding = torch.nn.Embedding(5, token_type_emb_dim)

    def forward(self, batch):
        assert "token_type" in batch, "`token_type` not in batch"
        token_type = batch["token_type"]  # [b, n] integers 0-4

        # Validate indices before embedding lookup
        invalid_mask = (token_type < 0) | (token_type >= 5)
        if invalid_mask.any():
            invalid_indices = torch.where(invalid_mask)
            invalid_values = token_type[invalid_mask]
            example_ids = batch.get("example_id_batch", ["unknown"] * token_type.shape[0])
            batch_indices = invalid_indices[0].tolist()
            raise ValueError(
                f"TokenTypeSeqFeat: Invalid token_type indices for embedding "
                f"(expected 0-4, got values: {invalid_values.tolist()[:20]}). "
                f"Example IDs: {[example_ids[i] for i in set(batch_indices[:5])]}"
            )

        return self.embedding(token_type)  # [b, n, token_type_emb_dim]


class Atom37NanometersCoorsSeqFeat(Feature):
    """
    Computes feature from the atom37 representation (in Å), feature of shape [b, n, 37 * 4].

    Atom37 has shape [b, n, 37, 3], and the appropriate mask (for the residue type) has shape
    [b, n, 37]. This feature concatenates the flattened mask (shape [b, n, 37]) with the flattened coordinates (of shape
    [b, n, 37 * 3])

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, rel=False, **kwargs):
        super().__init__(dim=int(37 * 4))
        # 37 * 4, 37 * 3 for the coordinates, + 37 for the mask
        self.rel = rel  # whether to get features relative to CA or absolute

    def forward(self, batch):
        assert "coords_nm" in batch, "`coords_nm` not in batch, cannot compute Atom37NanometersCoorsSeqFeat"
        assert "coord_mask" in batch, "`coord_mask` not in batch, cannot compute Atom37NanometersCoorsSeqFeat"
        coors = batch["coords_nm"]  # [b, n, 37, 3]
        coors_mask = batch["coord_mask"]  # [b, n, 37]
        coors = coors * coors_mask[..., None]  # Zero-out non-atoms

        if self.rel:
            # If relative remove CA coordinates
            ca_coors = coors[:, :, 1, :]  # [b, n, 3]
            coors = coors - ca_coors[:, :, None, :]  # [b, n, 37, 3]
            coors = coors * coors_mask[..., None]

        coors_flat = einops.rearrange(coors, "b n a t -> b n (a t)")  # [b, n, 37, 3] -> [b, n, 37 * 3]
        feat = torch.cat([coors_flat, coors_mask], dim=-1)  # [b, n, 37 * 4]
        return feat


class ConditionalAtom37NanometersCoorsSeqFeat(Feature):
    """
    Computes feature from conditional atom37 coordinates, feature of shape [b, n, 37 * 4].

    Uses `C_CRD.full_name` and `C_CRD.mask_name` if available, otherwise returns zeros.
    This prevents leakage of ground truth coordinate information during training.

    The feature is masked by `C_CRD.mask_name` which indicates which atoms have
    absolute coordinate conditioning applied (zeros for internal or no coordinate conditioning).

    Note: `C_CRD.full_name` is pre-masked by the pipeline as `coords * C_CRD.mask_name`.
    For unconditional generation, this is all zeros.
    """

    def __init__(self, rel=False, **kwargs):
        super().__init__(dim=int(37 * 4))
        self.rel = rel  # whether to get features relative to CA or absolute
        self._has_logged = False

    def forward(self, batch):
        from atomworks.ml.conditions import C_CRD

        if C_CRD.full_name in batch and C_CRD.mask_name in batch:
            # C_CRD.full_name coordinates are in Angstroms from collate, convert to nm
            # Already pre-masked by pipeline, but we apply mask explicitly for clarity
            coors = batch[C_CRD.full_name] / 10.0  # [b, n, 37, 3] - convert Angstroms to nm
            coors_mask = batch[C_CRD.mask_name]  # [b, n, 37] - CONDITIONING mask (not atom validity)
            coors = coors * coors_mask[..., None]  # Zero-out unconditioned atoms

            if self.rel:
                # If relative remove CA coordinates
                ca_coors = coors[:, :, 1, :]  # [b, n, 3]
                coors = coors - ca_coors[:, :, None, :]  # [b, n, 37, 3]
                coors = coors * coors_mask[..., None]

            coors_flat = einops.rearrange(coors, "b n a t -> b n (a t)")  # [b, n, 37, 3] -> [b, n, 37 * 3]
            feat = torch.cat([coors_flat, coors_mask], dim=-1)  # [b, n, 37 * 4]
            return feat
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    f"No {C_CRD.full_name} or {C_CRD.mask_name} in batch, "
                    f"returning zeros for ConditionalAtom37NanometersCoorsSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class ConditionalAtom37NanometersInternalCoorsSeqFeat(Feature):
    """
    Computes feature from conditional atom37 coordinates, feature of shape [b, n, 37 * 4].

    Uses `C_INTCRD.full_name` and `C_INTCRD.mask_name` if available, otherwise returns zeros.
    This prevents leakage of ground truth coordinate information during training.

    The feature is masked by `C_INTCRD.mask_name` which indicates which atoms have
    internal coordinate conditioning applied (zeros for absolute or no coordinate conditioning).

    For unconditional generation, this is all zeros.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(37 * 4))
        self._has_logged = False

    def forward(self, batch):
        from proteinfoundation.conditions import C_INTCRD

        if C_INTCRD.full_name in batch and C_INTCRD.mask_name in batch:
            # Ensure valid inputs
            internal_coords_mask = batch[C_INTCRD.mask_name]  # [b, n, 37] - CONDITIONING mask (not atom validity)

            # C_INTCRD.full_name coordinates are in Angstroms from collate, convert to nm
            # Already pre-masked by pipeline, but we apply mask explicitly for clarity
            internal_coords = batch[C_INTCRD.full_name] / 10.0  # [b, n, 37, 3] - convert Angstroms to nm
            internal_coords = torch.nan_to_num(internal_coords)  # Replace NaN values  with zeros
            internal_coords = internal_coords * internal_coords_mask[..., None]  # Zero-out unconditioned atoms

            internal_coords_flat = einops.rearrange(
                internal_coords, "b n a t -> b n (a t)"
            )  # [b, n, 37, 3] -> [b, n, 37 * 3]
            feat = torch.cat([internal_coords_flat, internal_coords_mask], dim=-1)  # [b, n, 37 * 4]
            return feat
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    f"No {C_INTCRD.full_name} or {C_INTCRD.mask_name} in batch, "
                    f"returning zeros for ConditionalAtom37NanometersInternalCoorsSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)

class CoordGroupSeqFeat(Feature):
    """Binary feature for whether two atoms are in the same coordinate group."""

    def __init__(self, **kwargs):
        super().__init__(dim=37)

    def forward(self, batch):
        from proteinfoundation.conditions import C_CRDGRP

        coord_group = batch[C_CRDGRP.full_name].long().clone()  # [b, n, 37]

        # Assign a random value to the non-padding index groups, to avoid the model learning spurious correlations
        non_padding_mask = coord_group != C_CRDGRP.UNKNOWN_COORDS
        unique_groups, inverse_indices = torch.unique(coord_group[non_padding_mask], return_inverse=True)
        permuted_index_groups = torch.randperm(len(unique_groups), device=coord_group.device)
        coord_group[non_padding_mask] = permuted_index_groups[inverse_indices]

        return coord_group.float()  # [b, n, 37]

class BackboneTorsionAnglesSeqFeat(Feature):
    """
    Computes torsion angle and featurizes it, with binning and 1-hot.

    TODO: Add mask?
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(3 * 21))

    def forward(self, batch):
        bb_torsion = self._get_bb_torsion_angles(batch)  # [b, n, 3]
        bb_torsion_feats = bin_and_one_hot(
            bb_torsion,
            torch.linspace(-torch.pi, torch.pi, 20, device=bb_torsion.device),
        )  # [b, n, 3, nbins], nbins in 20+1
        bb_torsion_feats = einops.rearrange(bb_torsion_feats, "b n t d -> b n (t d)")  # [b, n, 3 * nbins]
        return bb_torsion_feats

    def _get_bb_torsion_angles(self, batch):
        a37 = batch["coords"]  # [b, n, 37, 3]
        if "within_index_group_res_idx" in batch and batch["within_index_group_res_idx"] is not None:
            idx = batch["within_index_group_res_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            idx = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(device)  # [b, n]
        N = a37[:, :, 0, :]  # [b, n, 3]
        CA = a37[:, :, 1, :]  # [b, n, 3]
        C = a37[:, :, 2, :]  # [b, n, 3]

        psi = signed_dihedral_angle(N[:, :-1, :], CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :])  # [b, n-1]
        omega = signed_dihedral_angle(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :])  # [b, n-1]
        phi = signed_dihedral_angle(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :], C[:, 1:, :])  # [b, n-1]
        bb_angles = torch.stack([psi, omega, phi], dim=-1)  # [b, n-1, 3]

        good_pair = idx[:, 1:] - idx[:, :-1] == 1  # boolean [b, n-1]
        bb_angles = bb_angles * good_pair[..., None]  # [b, n-1, 3]

        zero_pad = torch.zeros((a37.shape[0], 1, 3), device=bb_angles.device)
        bb_angles = torch.cat([bb_angles, zero_pad], dim=1)  # [b, n, 3]
        return bb_angles


class BackboneBondAnglesSeqFeat(Feature):
    """
    Computes bond angle and featurizes it, with binning and 1-hot.

    TODO: Add mask?
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(3 * 21))

    def forward(self, batch):
        bb_bond_angle = self._get_bb_bond_angles(batch)  # [b, n, 3]
        bb_bond_angle_feats = bin_and_one_hot(
            bb_bond_angle,
            torch.linspace(-torch.pi, torch.pi, 20, device=bb_bond_angle.device),
        )  # [b, n, 3, nbins]
        bb_bond_angle_feats = einops.rearrange(bb_bond_angle_feats, "b n t d -> b n (t d)")  # [b, n, 3 * nbins]
        return bb_bond_angle_feats

    def _get_bb_bond_angles(self, batch):
        a37 = batch["coords"]  # [b, n, 37, 3]
        mask = batch["padding_mask"]  # [b, n]

        if "within_index_group_res_idx" in batch and batch["within_index_group_res_idx"] is not None:
            idx = batch["within_index_group_res_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            idx = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(device)  # [b, n]
        b = a37.shape[0]

        N = a37[:, :, 0, :]  # [b, n, 3]
        CA = a37[:, :, 1, :]  # [b, n, 3]
        C = a37[:, :, 2, :]  # [b, n, 3]
        theta_1 = bond_angles(N[:, :, :], CA[:, :, :], C[:, :, :])  # [b, n]
        theta_2 = bond_angles(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :])  # [b, n-1]
        theta_3 = bond_angles(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :])  # [b, n-1]

        # Account for chain breaks in theta_2 and theta_3
        good_pair = idx[:, 1:] - idx[:, :-1] == 1  # boolean [b, n-1]
        theta_2 = theta_2 * good_pair  # [b, n-1]
        theta_3 = theta_3 * good_pair  # [b, n-1]

        # Add a zero at the end of theta_2 and theta_3 to get shape [b, n]
        zero_pad = torch.zeros((b, 1), device=theta_2.device)  # [b, 1]
        theta_2 = torch.cat([theta_2, zero_pad], dim=-1)  # [b, n]
        theta_3 = torch.cat([theta_3, zero_pad], dim=-1)  # [b, n]

        bb_angles = torch.stack([theta_1, theta_2, theta_3], dim=-1)  # [b, n, 3]
        return bb_angles


class OpenfoldSideChainAnglesSeqFeat(Feature):
    """Computes sequence features from side chain angles."""

    def __init__(self, **kwargs):
        super().__init__(dim=int(4 * 21 + 4))  # 88

    def forward(self, batch):
        _, angles, torsion_angles_mask = self._get_sidechain_angles(batch)
        # _, [b, n, 4] and [b, n, 4]
        angles_feat = bin_and_one_hot(
            angles, torch.linspace(-torch.pi, torch.pi, 20, device=angles.device)
        )  # [b, n, 4, nbins]
        angles_feat = angles_feat * torsion_angles_mask[..., None]
        angles_feat = einops.rearrange(angles_feat, "b n s d -> b n (s d)")  # [b, n, 4 * nbins]
        feat = torch.cat([angles_feat, torsion_angles_mask], dim=-1)  # [b, n, 4 * nbins + 4]
        return feat

    def _get_sidechain_angles(self, batch):
        orig_dtype = batch["coords"].dtype
        aatype = batch["residue_type"]  # [b, n]
        coords = batch["coords"].double()  # [b, n, 37, 3]
        atom_mask = batch["coord_mask"].double()  # [b, n, 37]
        p = {
            "aatype": aatype,
            "all_atom_positions": coords,
            "all_atom_mask": atom_mask,
        }
        from proteinfoundation.utils.openfold_utils import atom37_to_torsion_angles

        p = atom37_to_torsion_angles(p, prefix="")
        torsion_angles_sin_cos = p["torsion_angles_sin_cos"]  # [b, n, 7, 2]
        alt_torsion_angles_sin_cos = p["alt_torsion_angles_sin_cos"]  # [b, n, 7, 2]
        # Normalize, all these vectors should have norm 1
        torsion_angles_sin_cos = torsion_angles_sin_cos / (
            torch.linalg.norm(torsion_angles_sin_cos, dim=-1, keepdim=True) + 1e-10
        )  # [b, n, 7, 2]
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos / (
            torch.linalg.norm(alt_torsion_angles_sin_cos, dim=-1, keepdim=True) + 1e-10
        )  # [b, n, 7, 2]
        torsion_angles_mask = p["torsion_angles_mask"]  # [b, n, 7]
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[..., None]
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask[..., None]
        angles = torch.atan2(torsion_angles_sin_cos[..., 0], torsion_angles_sin_cos[..., 1])  # [b, n, 7]
        angles = angles * torsion_angles_mask
        # Keep only sidechain
        torsion_angles_sin_cos = torsion_angles_sin_cos[..., -4:, :]  # [b, n, 4, 2]
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos[..., -4:, :]  # [b, n, 4, 2]
        angles = angles[..., -4:]  # [b, n, 4]
        torsion_angles_mask = torsion_angles_mask[..., -4:]  # [b, n, 4]
        return (
            torsion_angles_sin_cos.to(dtype=orig_dtype),
            angles.to(dtype=orig_dtype),
            torsion_angles_mask.bool(),
        )  # [b, n, 4, 2], [b, n, 4] and [b, n, 4]


class LatentVariableSeqFeat(Feature):
    """Returns sequence feature from latent variable."""

    def __init__(self, latent_z_dim, **kwargs):
        super().__init__(dim=latent_z_dim)

    def forward(self, batch):
        assert "z_latent" in batch, "`z_latent` not in batch, cannot compute LatentVariableSeqFeat"
        return batch["z_latent"]  # [b, n, latent_dim]


class RasaBinnedSeqFeat(Feature):
    """RASA (surface accessibility) binned feature: 0=BURIED, 1=INTERMEDIATE, 2=EXPOSED, 3=UNSPECIFIED.

    Returns atom37 representation [b, n, 37] as feature.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=37)

    def forward(self, batch):
        from proteinfoundation.conditions import C_RAS

        if C_RAS.full_name not in batch:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, 37, device=device)

        rasa_atom37 = batch[C_RAS.full_name].float()  # [b, n, 37]
        atom37_mask = batch["atom37_mask"]  # [b, n, 37]

        # Set masked atoms to UNSPECIFIED
        rasa_atom37[~atom37_mask.bool()] = RasaEnum.UNSPECIFIED
        return rasa_atom37


class SecondaryStructureSeqFeat(Feature):
    """Secondary structure: 0=NONE, 1=HELIX, 2=SHEET, 3=LOOP."""

    def __init__(self, embed_dim=16, **kwargs):
        super().__init__(dim=embed_dim)
        self.ss_embedding = torch.nn.Embedding(4, embed_dim, padding_idx=0)

    def forward(self, batch):
        from proteinfoundation.conditions import C_SEC

        if C_SEC.full_name not in batch:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, self.dim, device=device)

        ss_type = batch[C_SEC.full_name].long()  # [b, n] - already token-level

        # Validate indices before embedding lookup
        invalid_mask = (ss_type < 0) | (ss_type >= 4)
        if invalid_mask.any():
            invalid_indices = torch.where(invalid_mask)
            invalid_values = ss_type[invalid_mask]
            example_ids = batch.get("example_id_batch", ["unknown"] * ss_type.shape[0])
            batch_indices = invalid_indices[0].tolist()
            raise ValueError(
                f"SecondaryStructureSeqFeat: Invalid ss_type indices for embedding "
                f"(expected 0-3, got values: {invalid_values.tolist()[:20]}). "
                f"Example IDs: {[example_ids[i] for i in set(batch_indices[:5])]}"
            )

        return self.ss_embedding(ss_type)  # [b, n, embed_dim]


class HydrogenBondSeqFeat(Feature):
    """Hydrogen bonds: 0=NONE, 1=DONOR, 2=ACCEPTOR.

    Returns atom37 representation [b, n, 37] as feature.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=37)

    def forward(self, batch):
        from proteinfoundation.conditions import C_HB

        if C_HB.full_name not in batch:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, 37, device=device)

        hb_atom37 = batch[C_HB.full_name].float()  # [b, n, 37]
        atom37_mask = batch["atom37_mask"]  # [b, n, 37]

        # Zero out invalid atoms
        hb_feat = hb_atom37 * atom37_mask.float()  # [b, n, 37]
        return hb_feat


class HotspotSeqFeat(Feature):
    """Hotspot annotation: 0=not hotspot, 1=hotspot.

    Returns atom37 representation [b, n, 37] as feature.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=37)

    def forward(self, batch):
        from proteinfoundation.conditions import C_HOT

        if C_HOT.full_name not in batch:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, 37, device=device)

        hotspot_atom37 = batch[C_HOT.full_name].float()  # [b, n, 37] boolean
        atom37_mask = batch["atom37_mask"]  # [b, n, 37]

        # Zero out invalid atoms
        hotspot_feat = hotspot_atom37 * atom37_mask.float()  # [b, n, 37]
        return hotspot_feat


class IndexGroupSeqFeat(Feature):
    """Index group embedding for leaked index classes."""

    def __init__(self, max_groups=25, **kwargs):
        super().__init__(dim=max_groups)
        # Group 0 = globally indexed, 1+ = different unindexed groups
        self.max_groups = max_groups

    def forward(self, batch):
        from proteinfoundation.conditions import C_IDXGRP

        if C_IDXGRP.full_name not in batch:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, self.dim, device=device)

        idx_group = batch[C_IDXGRP.full_name].long().clone()  # [b, n] - already token-level
        idx_group = idx_group + 1  # Shift by 1, since padding positions are -1
        if idx_group.max() >= self.max_groups:
            raise ValueError(
                f"IndexGroupSeqFeat: C_IDXGRP has value {idx_group.max().item()}, maximum is {self.max_groups - 1}. "
                f"If {self.max_groups - 1} index groups is limiting, the max_groups parameter will need to be increased."
            )
        
        # Assign a random value to the non-padding index groups, to avoid the model learning spurious correlations
        permuted_index_groups = torch.randperm(self.max_groups, device=idx_group.device)
        non_padding_indices = torch.where(idx_group != 0)
        idx_group[non_padding_indices] = permuted_index_groups[idx_group[non_padding_indices]]

        embedded_idx_groups = F.one_hot(idx_group, num_classes=self.max_groups).float()  # [b, n, max_groups]

        return embedded_idx_groups  # [b, n, max_groups]

class IndexGroupPairFeat(Feature):
    """Binary feature for whether two tokens are in the same index group."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        from proteinfoundation.conditions import C_IDXGRP

        idx_group = batch[C_IDXGRP.full_name].long().clone()  # [b, n] - already token-level
        idx_group_pair_feat = idx_group[:, :, None] == idx_group[:, None, :]  # [b, n, n] boolean
        return idx_group_pair_feat.unsqueeze(-1).float()  # [b, n, n, 1]

class GraphLaplacianPESeqFeat(Feature):
    """Graph Laplacian positional encoding for atomized molecular components.

    Retrieves pre-computed Laplacian eigenvector encodings from the batch.
    These capture topological graph structure of small molecules/ligands.
    Non-atomized tokens (e.g., polymer residues) have zero vectors.

    Args:
        graph_laplacian_pe_dim: Dimensionality of the positional encoding.
    """

    def __init__(self, graph_laplacian_pe_dim: int = 32, **kwargs):
        super().__init__(dim=graph_laplacian_pe_dim)
        self._has_logged = False

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        # Check if pre-computed graph Laplacian PE is available (collate puts it at top level)
        if "graph_laplacian_pe_token_level" in batch:
            graph_lap_pe = batch["graph_laplacian_pe_token_level"]  # [b, n, pe_dim]

            # Handle dimension mismatch (pad or truncate)
            pe_dim = graph_lap_pe.shape[-1]
            if pe_dim < self.dim:
                padding = torch.zeros(b, n, self.dim - pe_dim, device=device, dtype=graph_lap_pe.dtype)
                graph_lap_pe = torch.cat([graph_lap_pe, padding], dim=-1)
            elif pe_dim > self.dim:
                graph_lap_pe = graph_lap_pe[..., : self.dim]

            return graph_lap_pe.to(device)  # [b, n, graph_laplacian_pe_dim]
        else:
            if not self._has_logged:
                logger.warning(
                    "No graph_laplacian_pe_token_level in batch, "
                    "returning zeros for GraphLaplacianPESeqFeat. "
                    "Ensure FeaturizeBondsAsLaplacianPositionalEncoding transform is in the pipeline."
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class AtomChargeSeqFeat(Feature):
    """Atomic charge for atomized tokens (ligands, ions) ONLY.

    For atomized tokens: Returns charge from first atom slot (no aggregation needed).
    For non-atomized tokens (proteins): Returns zero.
    Shape: [b, n, 1]

    Note: Atomized tokens have only ONE atom per token, so we just take slot 0.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        # Initialize with zeros
        charge = torch.zeros(b, n, 1, device=device, dtype=torch.float32)

        # Only populate for atomized tokens (ligands, ions)
        if "charge" in batch and "is_atomized" in batch:
            # For atomized tokens, take charge from first atom slot (slot 0)
            # [b, n, 37] -> [b, n] (take slot 0)
            charge_atom37 = batch["charge"].float()  # [b, n, 37]
            is_atomized = batch["is_atomized"]  # [b, n]

            # Take first slot for atomized tokens
            charge_slot0 = charge_atom37[:, :, 0:1]  # [b, n, 1]

            # Mask: only keep for atomized tokens
            charge = torch.where(
                is_atomized.unsqueeze(-1),  # [b, n, 1]
                charge_slot0,
                torch.zeros_like(charge_slot0),
            )
        elif "charge" not in batch and not self._has_logged:
            logger.warning("No charge in batch, returning zeros for AtomChargeSeqFeat")
            self._has_logged = True

        return charge


class AtomicElementSeqFeat(Feature):
    """Atomic element one-hot encoding for atomized tokens ONLY.

    For atomized tokens: Returns element from first atom slot (no aggregation needed).
    For non-atomized tokens (proteins): Returns zero vector.
    Shape: [b, n, 128] (one-hot up to atomic number 128)

    Note: Atomized tokens have only ONE atom per token, so we just take slot 0.
    """

    def __init__(self, max_atomic_num: int = 128, **kwargs):
        super().__init__(dim=max_atomic_num)
        self._has_logged = False
        self.max_atomic_num = max_atomic_num

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        # Initialize with zeros
        element_one_hot = torch.zeros(b, n, self.max_atomic_num, device=device, dtype=torch.float32)

        # Only populate for atomized tokens (ligands, ions)
        if "element" in batch and "is_atomized" in batch:
            # For atomized tokens, take element from first atom slot (slot 0)
            element_atom37 = batch["element"].long()  # [b, n, 37]
            is_atomized = batch["is_atomized"]  # [b, n]

            # Take first slot for atomized tokens
            element_slot0 = element_atom37[:, :, 0]  # [b, n]

            # Clamp to valid range and one-hot encode
            element_slot0 = element_slot0.clamp(0, self.max_atomic_num - 1)
            element_one_hot_all = F.one_hot(element_slot0, num_classes=self.max_atomic_num).float()  # [b, n, 128]

            # Mask: only keep for atomized tokens
            element_one_hot = element_one_hot_all * is_atomized.unsqueeze(-1)  # [b, n, 128]
        elif "element" not in batch and not self._has_logged:
            logger.warning("No element in batch, returning zeros for AtomicElementSeqFeat")
            self._has_logged = True

        return element_one_hot


class NhydSeqFeat(Feature):
    """Number of hydrogens embedding for atomized tokens.

    Values: 0-4 for actual hydrogen count, -1 for unspecified.
    Unspecified (-1) is mapped to class 5.

    For non-atomized tokens (proteins): Returns zero vector.
    Note: Uses slot 1 (CA slot) for consistency with coordinate handling.
    """

    def __init__(self, embed_dim: int = 16, num_nhyd_classes: int = 6, **kwargs):
        super().__init__(dim=embed_dim)
        self._has_logged = False
        self.num_nhyd_classes = num_nhyd_classes  # 0,1,2,3,4 + unknown
        self.embedding = torch.nn.Embedding(num_nhyd_classes, embed_dim)

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        # Initialize with zeros
        nhyd_emb = torch.zeros(b, n, self.dim, device=device, dtype=torch.float32)

        # Only populate for atomized tokens (ligands, ions)
        if "nhyd" in batch and "token_type" in batch:
            nhyd_atom37 = batch["nhyd"].long()  # [b, n, 37]
            token_type = batch["token_type"]  # [b, n]
            is_atomized = token_type == 3  # TokenType.ATOMIZED

            # Take slot 1 (CA slot) for atomized tokens, consistent with coordinates
            nhyd_slot1 = nhyd_atom37[:, :, 1]  # [b, n]

            # Map -1 (unspecified) to class 5, others stay 0-4
            nhyd_slot1 = torch.where(
                nhyd_slot1 == -1,
                torch.tensor(5, device=device, dtype=nhyd_slot1.dtype),
                nhyd_slot1,
            )

            # Validate indices before embedding lookup
            invalid_mask = (nhyd_slot1 < 0) | (nhyd_slot1 >= self.num_nhyd_classes)
            if invalid_mask.any():
                invalid_indices = torch.where(invalid_mask)
                invalid_values = nhyd_slot1[invalid_mask]
                example_ids = batch.get("example_id_batch", ["unknown"] * nhyd_slot1.shape[0])
                batch_indices = invalid_indices[0].tolist()
                raise ValueError(
                    f"NhydSeqFeat: Invalid nhyd indices for embedding "
                    f"(expected 0-{self.num_nhyd_classes - 1}, got values: {invalid_values.tolist()[:20]}). "
                    f"Example IDs: {[example_ids[i] for i in set(batch_indices[:5])]}"
                )

            # Embed
            nhyd_emb_all = self.embedding(nhyd_slot1)  # [b, n, embed_dim]

            # Mask: only keep for atomized tokens
            nhyd_emb = nhyd_emb_all * is_atomized.unsqueeze(-1)  # [b, n, embed_dim]
        elif "nhyd" not in batch and not self._has_logged:
            logger.warning("No nhyd in batch, returning zeros for NhydSeqFeat")
            self._has_logged = True

        return nhyd_emb


class ChainIdxSeqFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "chain_iid" in batch:
            mask = batch["chain_iid"].unsqueeze(-1).float()  # [b, n, 1]
        elif "chains" in batch:
            mask = batch["chains"].unsqueeze(-1).float()  # [b, n, 1] (legacy key)
        else:
            raise ValueError("Neither 'chain_iid' nor 'chains' found in batch")
        return mask


class SequenceSeparationPairFeat(Feature):
    """Computes sequence separation and returns feature of shape [b, n, n, seq_sep_dim]."""

    def __init__(self, seq_sep_dim, **kwargs):
        super().__init__(dim=seq_sep_dim)

    def forward(self, batch):
        if "within_index_group_res_idx" in batch:
            # no need to force 1 since taking difference
            inds = batch["within_index_group_res_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(device)  # [b, n]

        seq_sep = inds[:, :, None] - inds[:, None, :]  # [b, n, n]

        # Dimension should be odd, bins limits [-(dim/2-1), ..., -1.5, -0.5, 0.5, 1.5, ..., dim/2-1]
        # gives dim-2 bins, and the first and last for values beyond the bin limits
        assert self.dim % 2 == 1, "Relative seq separation feature dimension must be odd and > 3"

        # Create bins limits [..., -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, ...]
        # Equivalent to binning relative sequence separation
        low = -(self.dim / 2.0 - 1)
        high = self.dim / 2.0 - 1
        bin_limits = torch.linspace(low, high, self.dim - 1, device=inds.device)

        return bin_and_one_hot(seq_sep, bin_limits)  # [b, n, n, seq_sep_dim]


class XtBBCAPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, xt_pair_dist_dim, xt_pair_dist_min, xt_pair_dist_max, **kwargs):
        super().__init__(dim=xt_pair_dist_dim)
        self.min_dist = xt_pair_dist_min
        self.max_dist = xt_pair_dist_max

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert (
            "bb_ca" in data_modes_avail
        ), f"`bb_ca` pair dist feature requested but key not available in data modes {data_modes_avail}"
        return bin_pairwise_distances(
            x=batch["x_t"]["bb_ca"],
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class CaCoorsNanometersPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, **kwargs):
        super().__init__(dim=30)
        self.min_dist = 0.1
        self.max_dist = 3.0

    def forward(self, batch):
        assert (
            "ca_coors_nm" in batch or "coords_nm" in batch
        ), "`ca_coors_nm` pair dist feature requested but key `ca_coors_nm` nor `coords_nm` not available"
        if "ca_coors_nm" in batch:
            ca_coors = batch["ca_coors_nm"]
        else:
            ca_coors = batch["coords_nm"][:, :, 1, :]
        return bin_pairwise_distances(
            x=ca_coors,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class XscBBCAPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(
        self,
        x_sc_pair_dist_dim,
        x_sc_pair_dist_min,
        x_sc_pair_dist_max,
        mode_key="x_sc",
        **kwargs,
    ):
        super().__init__(dim=x_sc_pair_dist_dim)
        self.min_dist = x_sc_pair_dist_min
        self.max_dist = x_sc_pair_dist_max
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert (
                "bb_ca" in data_modes_avail
            ), f"`bb_ca` sc/recycle pair dist feature requested but key not available in data modes {data_modes_avail}"
            return bin_pairwise_distances(
                x=batch[self.mode_key]["bb_ca"],
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                dim=self.dim,
            )  # [b, n, n, pair_dist_dim]
        else:
            # If we do not provide self-conditioning as input to the nn
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(f"No {self.mode_key} in batch, returning zeros for XscBBCAPairwiseDistancesPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, self.dim, device=device)


class RelativeResidueOrientationPairFeat(Feature):
    """Computes pair feature with pairwise residue orientations.

    See paper "Improved protein structure prediction using
    predicted inter-residue orientations".

    TODO: Impute beta carbon for Glycine
    TODO: 20 as argument
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(5 * 21))  # 105

    def forward(self, batch):
        aatype = batch["residue_type"]  # [b, n]
        coords = batch["coords"]  # [b, n, 37, 3]
        atom_mask = batch["coord_mask"]  # [b, n, 37]
        mask = atom_mask[:, :, 1]  # [b, n]
        has_cb = atom_mask[:, :, 3]  # [b, n] boolean, indicates if corresponding
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n] boolean
        beta_carbon_pair_mask = has_cb[:, :, None] * has_cb[:, :, None]  # [b, n, n] boolean
        pair_mask = pair_mask * beta_carbon_pair_mask  # [b, n, n]

        N = coords[:, :, 0, :]  # [b, n, 3]
        CA = coords[:, :, 1, :]  # [b, n, 3]
        CB = coords[:, :, 3, :]  # [b, n, 3]

        N_p1, CA_p1, CB_p1 = map(lambda v: v[:, :, None, :], (N, CA, CB))  # Each [b, n, 1, 3]
        N_p2, CA_p2, CB_p2 = map(lambda v: v[:, None, :, :], (N, CA, CB))  # Each [b, 1, n, 3]

        theta_12 = signed_dihedral_angle(N_p1, CA_p1, CB_p1, CB_p2)  # [b, n, n]
        theta_21 = signed_dihedral_angle(N_p2, CA_p2, CB_p2, CB_p1)  # [b, n, n]
        phi_12 = bond_angles(CA_p1, CB_p1, CB_p2)  # [b, n, n]
        phi_21 = bond_angles(CA_p2, CB_p2, CB_p1)  # [b, n, n]
        w = signed_dihedral_angle(CA_p1, CB_p1, CB_p2, CA_p2)  # [b, n, n]
        angles = torch.stack([theta_12, theta_21, phi_12, phi_21, w], dim=-1)  # [b, n, n, 5]

        angles_feat = bin_and_one_hot(
            angles, torch.linspace(-torch.pi, torch.pi, 20, device=angles.device)
        )  # [b, n, n, 5, nbins]
        angles_feat = einops.rearrange(angles_feat, "b n m f d -> b n m (f d)")  # [b, n, n, 5 * nbins]
        angles_feat = angles_feat * pair_mask[..., None]  # Mask padding
        return angles_feat


class BackbonePairDistancesNanometerPairFeat(Feature):
    """
    Computes pairwise distances between backbone atoms.

    Position (i, j) encodes the distance between CA_i and
    {N_j, CA_j, C_j, CB_j}.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(4 * 21))  # 84

    def forward(self, batch):
        assert "coords_nm" in batch, "`coords_nm` not in batch, cannot comptue BackbonePairDistancesNanometerPairFeat"
        coords = batch["coords_nm"]
        atom_mask = batch["coord_mask"]  # [b, n, 37]
        mask = atom_mask[:, :, 1]  # [b, n]
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
        has_cb = atom_mask[:, :, 3]  # [b, n] boolean, indicates if corresponding

        N = coords[:, :, 0, :]  # [b, n, 3]
        CA = coords[:, :, 1, :]  # [b, n, 3]
        C = coords[:, :, 2, :]  # [b, n, 3]
        CB = coords[:, :, 3, :]  # [b, n, 3]

        CA_i = CA[:, :, None, :]  # [b, n, 1, 3]
        N_j, CA_j, C_j, CB_j = map(lambda v: v[:, None, :, :], (N, CA, C, CB))  # Each [b, 1, n, 3]

        CA_N, CA_CA, CA_C, CA_CB = map(
            lambda v: torch.linalg.norm(v[0] - v[1], dim=-1),
            ((CA_i, N_j), (CA_i, CA_j), (CA_i, C_j), (CA_i, CB_j)),
        )  # Each shape [b, n, n]
        # CA_X[..., i, j] has distance (nm) between CA[..., i] and X[..., j]

        # Accomodate residues without CB
        CA_CB = CA_CB * has_cb[:, None, :]  # [b, n, n]

        # Fix for mask
        CA_N, CA_CA, CA_C, CA_CB = map(
            lambda v: v * pair_mask,
            (CA_N, CA_CA, CA_C, CA_CB),
        )  # Each shape [b, n, n]

        bin_limits = torch.linspace(0.1, 2, 20, device=coords.device)
        CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat = map(
            lambda v: bin_and_one_hot(v, bin_limits=bin_limits),
            (CA_N, CA_CA, CA_C, CA_CB),
        )  # Each [b, n, n, 21]

        feat = torch.cat([CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat], dim=-1)  # [b, n, n, 4 * 21]
        feat = feat * pair_mask[..., None]
        return feat


class RelativeResidueOrientationPairFeat(Feature):
    """Computes relative residue orientation and returns feature of shape [b, n, n, 3]."""

    def __init__(self, **kwargs):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if not self._has_logged:
            logger.warning("No x_motif in batch, returning zeros for XmotifPairwiseDistancesPairFeat")
            self._has_logged = True
        return torch.zeros(b, n, n, self.dim, device=device)


class ChainIdxPairFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "chain_iid" in batch:
            seq_mask = batch["chain_iid"].float()  # [b, n]
        elif "chains" in batch:
            seq_mask = batch["chains"].float()  # [b, n] (legacy key)
        else:
            raise ValueError("Neither 'chain_iid' nor 'chains' found in batch")
        mask = torch.einsum("bi,bj->bij", seq_mask, seq_mask).unsqueeze(-1)  # [b, n, n, 1]
        return mask


class BondMaskPairFeat(Feature):
    """Binary bond existence mask for atomized molecules.

    Non-atomized token pairs get zeros.
    Shape: [b, n, n, 1]
    """

    def __init__(self, **kwargs):
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        if "bond_mask" in batch:
            # [b, n, n] -> [b, n, n, 1]
            return batch["bond_mask"][..., None].float()
        else:
            if not self._has_logged:
                logger.warning("No bond_mask in batch, returning zeros for BondMaskPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, 1, device=device)


class BondOrderPairFeat(Feature):
    """Bond order one-hot encoding for atomized molecules.

    Bond types: 0=none, 1=single, 2=double, 3=triple, 4=aromatic
    Non-atomized token pairs get zeros.
    Shape: [b, n, n, 5]
    """

    def __init__(self, num_bond_orders=5, **kwargs):
        super().__init__(dim=num_bond_orders)
        self._has_logged = False
        self.num_bond_orders = num_bond_orders

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        if "bond_order" in batch:
            # [b, n, n] int -> [b, n, n, 5] one-hot
            bond_order = batch["bond_order"].long()
            return F.one_hot(bond_order, num_classes=self.num_bond_orders).float()
        else:
            if not self._has_logged:
                logger.warning("No bond_order in batch, returning zeros for BondOrderPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, self.num_bond_orders, device=device)


class StochasticTranslationSeqFeat(Feature):
    """Gets stochastic translation from centering transform and returns feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)
        self._has_logged = False

    def forward(self, batch):
        if "stochastic_translation" in batch:
            b, n = self.extract_bs_and_n(batch)
            translation = batch["stochastic_translation"]  # [b, 3]
            # Broadcast translation to all residues
            mask = translation[:, None, :].expand(b, n, -1)  # [b, n, 3]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No stochastic_translation in batch, returning zeros for StochasticTranslationSeqFeat")
                self._has_logged = True
            mask = torch.zeros((b, n, 3), device=device)
        return mask


####################################
# # Class that produces features # #
####################################


class FeatureFactory(torch.nn.Module):
    def __init__(
        self,
        feats: list[str],
        dim_feats_out: int,
        use_ln_out: bool,
        mode: Literal["seq", "pair"],
        **kwargs,
    ):
        """
        Feature factory for creating sequence and pair features.

        Sequence features include:
            Time embeddings:
                - "time_emb_bb_ca": Time embedding for backbone CA atoms
                - "time_emb_local_latents": Time embedding for local latents

            Position and structure:
                - "res_seq_pdb_idx": Residue sequence position (requires ResidueSequencePositionPdbTransform)
                - "chain_break_per_res": Chain break per residue (requires ChainBreakPerResidueTransform)
                - "fold_emb": Fold embedding

            Coordinates and angles:
                - "x_sc_bb_ca": Self-conditioning backbone CA coordinates
                - "x_recycle_bb_ca": Recycled backbone CA coordinates
                - "x_sc_local_latents": Self-conditioning local latents
                - "x_recycle_local_latents": Recycled local latents
                - "xt_bb_ca": Target backbone CA coordinates
                - "xt_local_latents": Target local latents
                - "ca_coors_nm": CA coordinates in nanometers
                - "ca_coors_nm_try": Try CA coordinates in nanometers
                - "optional_ca_coors_nm_seq_feat": Optional CA coordinates in nanometers
                - "x1_bb_angles": Backbone torsion angles
                - "x1_bond_angles": Backbone bond angles
                - "x1_sidechain_angles": Sidechain angles

            Residue information:
                - "x1_aatype": Residue type
                - "optional_res_type_seq_feat": Optional residue type
                - "x1_a37coors_nm": Atom37 coordinates in nanometers
                - "x1_a37coors_nm_rel": Relative atom37 coordinates in nanometers
                - "z_latent_seq": Latent variable sequence

            Graph structure (atomized molecules):
                - "graph_laplacian_pe": Graph Laplacian positional encoding for atomized molecules

        Pair features include:
            Distance features:
                - "xt_bb_ca_pair_dists": Target backbone CA pairwise distances
                - "x_sc_bb_ca_pair_dists": Self-conditioning backbone CA pairwise distances
                - "x_recycle_bb_ca_pair_dists": Recycled backbone CA pairwise distances
                - "ca_coors_nm_pair_dists": CA coordinates pairwise distances in nanometers
                - "x1_bb_pair_dists_nm": Backbone pairwise distances in nanometers
                - "optional_ca_pair_dist": Optional CA pairwise distances

            Sequence and time:
                - "rel_seq_sep": Relative sequence separation
                - "time_emb_bb_ca": Time embedding for backbone CA atoms
                - "time_emb_local_latents": Time embedding for local latents

            Structure and orientation:
                - "x1_bb_pair_orientation": Relative residue orientation
                - "chain_idx_pair": Chain index pairwise feature

        """
        super().__init__()
        self.mode = mode
        self.ret_zero = True if (feats is None or len(feats) == 0) else False
        if self.ret_zero:
            logger.info("No features requested")
            self.zero_creator = ZeroFeat(dim_feats_out=dim_feats_out, mode=mode)
            return

        self.feat_creators = torch.nn.ModuleList([self.get_creator(f, **kwargs) for f in feats])
        self.ln_out = torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        self.linear_out = torch.nn.Linear(sum([c.get_dim() for c in self.feat_creators]), dim_feats_out, bias=False)

    def get_creator(self, f, **kwargs):
        """Returns the right class for the requested feature f (a string)."""

        if self.mode == "seq":
            # Time embeddings
            if f == "time_emb_bb_ca":
                return TimeEmbeddingSeqFeat(data_mode_use="bb_ca", **kwargs)
            elif f == "time_emb_local_latents":
                return TimeEmbeddingSeqFeat(data_mode_use="local_latents", **kwargs)

            # Position and indexing
            elif f == "res_seq_pdb_idx":
                return IdxEmbeddingSeqFeat(**kwargs)
            elif f == "chain_break_per_res":
                return ChainBreakPerResidueSeqFeat(**kwargs)
            elif f == "chain_idx_seq":
                return ChainIdxSeqFeat(**kwargs)
            elif f == "cropped_flag_seq":
                return CroppedFlagSeqFeat()

            # Basic residue information
            elif f == "x1_aatype":
                return ResidueTypeSeqFeat(**kwargs)
            elif f == "token_type":
                return TokenTypeSeqFeat(**kwargs)

            # Raw coordinate features
            elif f == "ca_coors_nm":
                return CaCoorsNanometersSeqFeat(**kwargs)
            elif f == "ca_coors_nm_try":
                return TryCaCoorsNanometersSeqFeat(**kwargs)
            elif f == "x1_a37coors_nm":
                return Atom37NanometersCoorsSeqFeat(**kwargs)
            elif f == "x1_a37coors_nm_rel":
                return Atom37NanometersCoorsSeqFeat(rel=True, **kwargs)

            # Diffusion/sampling coordinates
            elif f == "xt_bb_ca":
                return XtBBCASeqFeat(**kwargs)
            elif f == "xt_local_latents":
                return XtLocalLatentsSeqFeat(**kwargs)
            elif f == "x_sc_bb_ca":
                return XscBBCASeqFeat(**kwargs)
            elif f == "x_recycle_bb_ca":
                return XscBBCASeqFeat(mode_key="x_recycle", **kwargs)
            elif f == "x_sc_local_latents":
                return XscLocalLatentsSeqFeat(**kwargs)
            elif f == "x_recycle_local_latents":
                return XscLocalLatentsSeqFeat(mode_key="x_recycle", **kwargs)

            # Structural features (angles)
            elif f == "x1_bb_angles":
                return BackboneTorsionAnglesSeqFeat(**kwargs)
            elif f == "x1_bond_angles":
                return BackboneBondAnglesSeqFeat(**kwargs)
            elif f == "x1_sidechain_angles":
                return OpenfoldSideChainAnglesSeqFeat(**kwargs)

            # Latent variables
            elif f == "z_latent_seq":
                return LatentVariableSeqFeat(**kwargs)

            # Centering
            elif f == "stochastic_translation":
                return StochasticTranslationSeqFeat(**kwargs)

            # 1D Condition annotations
            elif f == "rasa_binned":
                return RasaBinnedSeqFeat(**kwargs)
            elif f == "secondary_structure":
                return SecondaryStructureSeqFeat(**kwargs)
            elif f == "hbond":
                return HydrogenBondSeqFeat(**kwargs)
            elif f == "hotspot":
                return HotspotSeqFeat(**kwargs)
            elif f == "index_group_seq":
                return IndexGroupSeqFeat(**kwargs)
            elif f == "coord_group_seq":
                return CoordGroupSeqFeat(**kwargs)
            elif f == "graph_laplacian_pe":
                return GraphLaplacianPESeqFeat(**kwargs)
            elif f == "cath_code":
                return FoldEmbeddingSeqFeat(**kwargs)

            # Ligand/atomized features
            elif f == "atom_charge":
                return AtomChargeSeqFeat(**kwargs)
            elif f == "atomic_element":
                return AtomicElementSeqFeat(**kwargs)
            elif f == "nhyd":
                return NhydSeqFeat(**kwargs)

            # Conditional features (prevent leakage)
            elif f == "cond_residue_type":
                return ConditionalResidueTypeSeqFeat(**kwargs)
            elif f == "cond_a37coors_nm":
                return ConditionalAtom37NanometersCoorsSeqFeat(**kwargs)
            elif f == "cond_a37coors_nm_rel":
                return ConditionalAtom37NanometersCoorsSeqFeat(rel=True, **kwargs)
            elif f == "cond_a37coors_internal_nm":
                return ConditionalAtom37NanometersInternalCoorsSeqFeat(**kwargs)

            # Special/utility features
            elif f == "zero_feat_seq":
                return ZeroFeat(**kwargs)
            else:
                raise OSError(f"Sequence feature {f} not implemented.")

        elif self.mode == "pair":
            # Time embeddings
            if f == "time_emb_bb_ca":
                return TimeEmbeddingPairFeat(data_mode_use="bb_ca", **kwargs)
            elif f == "time_emb_local_latents":
                return TimeEmbeddingPairFeat(data_mode_use="local_latents", **kwargs)

            # Sequence separation
            elif f == "rel_seq_sep":
                return SequenceSeparationPairFeat(**kwargs)

            # Distance features
            elif f == "xt_bb_ca_pair_dists":
                return XtBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "x_sc_bb_ca_pair_dists":
                return XscBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "x_recycle_bb_ca_pair_dists":
                return XscBBCAPairwiseDistancesPairFeat(mode_key="x_recycle", **kwargs)
            elif f == "ca_coors_nm_pair_dists":
                return CaCoorsNanometersPairwiseDistancesPairFeat(**kwargs)
            elif f == "x1_bb_pair_dists_nm":
                return BackbonePairDistancesNanometerPairFeat(**kwargs)

            # Structural and orientation features
            elif f == "x1_bb_pair_orientation":
                return RelativeResidueOrientationPairFeat(**kwargs)

            # Chain and indexing features
            elif f == "chain_idx_pair":
                return ChainIdxPairFeat(**kwargs)
            elif f == "index_group_pair":
                return IndexGroupPairFeat(**kwargs)

            # Bond features (for atomized molecules/ligands)
            elif f == "bond_mask":
                return BondMaskPairFeat(**kwargs)
            elif f == "bond_order":
                return BondOrderPairFeat(**kwargs)

            else:
                raise OSError(f"Pair feature {f} not implemented.")

        else:
            raise OSError(f"Wrong feature mode (creator): {self.mode}. Should be 'seq' or 'pair'.")

    def apply_padding_mask(self, feature_tensor, mask):
        """
        Applies mask to features.

        Args:
            feature_tensor: tensor with requested features, shape [b, n, d] of [b, n, n, d] depending on self.mode ('seq' or 'pair')
            mask: Binary mask, shape [b, n]

        Returns:
            Masked features, same shape as input tensor.
        """
        if self.mode == "seq":
            return feature_tensor * mask[..., None]  # [b, n, d]
        elif self.mode == "pair":
            mask_pair = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
            return feature_tensor * mask_pair[..., None]  # [b, n, n, d]
        else:
            raise OSError(f"Wrong feature mode (pad mask): {self.mode}. Should be 'seq' or 'pair'.")

    def forward(self, batch):
        """Returns masked features, shape depends on mode, either 'seq' or 'pair'."""
        # If no features requested just return the zero tensor of appropriate dimensions
        if self.ret_zero:
            return self.zero_creator(batch)

        # Compute requested features
        feature_tensors = []
        for fcreator in self.feat_creators:
            feature_tensors.append(fcreator(batch))  # [b, n, dim_f] or [b, n, n, dim_f] if seq or pair mode

        # Concatenate features and mask
        features = torch.cat(feature_tensors, dim=-1)  # [b, n, dim_f] or [b, n, n, dim_f]
        features = self.apply_padding_mask(features, batch["padding_mask"])  # [b, n, dim_f] or [b, n, n, dim_f]

        # Linear layer and mask
        features_proc = self.ln_out(self.linear_out(features))  # [b, n, dim_f] or [b, n, n, dim_f]
        return self.apply_padding_mask(features_proc, batch["padding_mask"])  # [b, n, dim_f] or [b, n, n, dim_f]
