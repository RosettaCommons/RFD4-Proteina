import math
from functools import lru_cache
from typing import Any, Literal

import einops
import torch
from atomworks.ml.conditions import C_CRD, C_SEQ
from jaxtyping import Float
from torch.nn import functional as F

from foundry.utils.ddp import RankedLogger
from proteinfoundation.conditions import C_CATH, C_CRDGRP, C_HB, C_HOT, C_IDXGRP, C_INTCRD, C_RAS, C_SEC, RasaEnum
from proteinfoundation.transforms.chirality_feature import MAX_OVERLAPPING_CENTERS
from proteinfoundation.utils.angle_utils import bond_angles, signed_dihedral_angle

logger = RankedLogger(__name__, rank_zero_only=True)


@lru_cache(maxsize=128)
def _warn_once(msg: str) -> None:
    """Log ``msg`` at WARNING level, once per unique string per process (bounded cache)."""
    logger.warning(msg)


################################
# # Some auxiliary functions # #
################################


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


def bin_pairwise_distances(x: torch.Tensor, min_dist: float, max_dist: float, dim: int) -> torch.Tensor:
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


def bin_and_one_hot(tensor: torch.Tensor, bin_limits: torch.Tensor) -> torch.Tensor:
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


################################
# # Classes for each feature # #
################################


class Feature(torch.nn.Module):
    """Base class for features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def forward(self, batch: dict) -> torch.Tensor:
        pass  # Implemented by each class

    def extract_bs_and_n(self, batch: dict) -> tuple[int, int]:
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

    def extract_device(self, batch: dict) -> torch.device:
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

    def assert_defaults_allowed(self, batch: dict, ftype: str) -> None:
        """Raises error if default features should not be used to fill-up missing features in the current batch."""
        if batch.get("strict_feats"):
            raise OSError(
                f"{ftype} feature requested but no appropriate feature provided. "
                "Make sure to include the relevant transform in the data config."
            )


class ZeroFeat(Feature):
    """Computes empty feature (zero) of shape [b, n, dim] or [b, n, n, dim],
    depending on sequence or pair features."""

    def __init__(
        self,
        dim_feats_out: int = 128,
        mode: Literal["seq", "pair"] = "seq",
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(dim=128)
        self.mode = mode

    def forward(self, batch: dict) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if self.mode == "seq":
            return torch.zeros((b, n, self.dim), device=device)
        elif self.mode == "pair":
            torch.zeros((b, n, n, self.dim_feats_out), device=device)
        else:
            raise OSError(f"Mode {self.mode} wrong for zero feature")


class FoldEmbeddingSeqFeat(Feature):
    """Computes fold class embedding and returns as sequence feature of shape [b, n, fold_emb_dim * 3]."""

    def __init__(
        self,
        cath_emb_dim: int = 32,
        **kwargs: Any,
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
    def device(self) -> torch.device:
        return next(self.buffers()).device

    def forward(self, batch: dict) -> torch.Tensor:
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

    def __init__(self, data_mode_use: str, t_emb_dim: int, **kwargs: Any) -> None:
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use

    def forward(self, batch: dict) -> torch.Tensor:
        t = batch["t"][self.data_mode_use]  # [b]
        _, n = self.extract_bs_and_n(batch)
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, :]  # [b, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, t_emb.shape[2]))  # [b, n, t_emb_dim]


class TimeEmbeddingPairFeat(Feature):
    """Computes time embedding and returns as pair feature of shape [b, n, n, t_emb_dim]."""

    def __init__(self, data_mode_use: str, t_emb_dim: int, **kwargs: Any) -> None:
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use

    def forward(self, batch: dict) -> torch.Tensor:
        t = batch["t"][self.data_mode_use]  # [b]
        _, n = self.extract_bs_and_n(batch)
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, None, :]  # [b, 1, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, n, t_emb.shape[3]))  # [b, n, t_emb_dim]


class ChainBreakPerResidueSeqFeat(Feature):
    """Computes a 1D sequence feature indicating if a residue is followed by a chain break, shape [b, n, 1]."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
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

    def __init__(self, mode_key: str = "x_sc", **kwargs: Any) -> None:
        super().__init__(dim=3)
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
        if self.mode_key in batch:
            data_modes_avail = list(batch[self.mode_key])
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

    def __init__(self, latent_dim: int, mode_key: str = "x_sc", **kwargs: Any) -> None:
        super().__init__(dim=latent_dim)
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
        if self.mode_key in batch:
            data_modes_avail = list(batch[self.mode_key])
            assert "local_latents" in data_modes_avail, (
                f"`local_latents` sc/recycle seq feature requested but key not available "
                f"in data modes {data_modes_avail}"
            )
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=3)

    def forward(self, batch: dict) -> torch.Tensor:
        data_modes_avail = list(batch["x_t"])
        assert (
            "bb_ca" in data_modes_avail
        ), f"`bb_ca` seq feat feature requested but key not available in data modes {data_modes_avail}"
        return batch["x_t"]["bb_ca"]  # [b, n, 3]


class XtLocalLatentsSeqFeat(Feature):
    """Computes feature from backbone CA x_t coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(dim=latent_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        data_modes_avail = list(batch["x_t"])
        assert (
            "local_latents" in data_modes_avail
        ), f"`local_latents` seq feat feature requested but key not available in data modes {data_modes_avail}"
        return batch["x_t"]["local_latents"]  # [b, n, latent_dim]


class CaCoorsNanometersSeqFeat(Feature):
    """Computes feature from ca coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=3)

    def forward(self, batch: dict) -> torch.Tensor:
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
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
    """Computes one-hot residue type feature of shape [b, n, n_residue_types].

    Supports both standard 20-class encoding and unified 33-class encoding
    (used by the autoencoder with ``n_residue_types=33``).
    """

    def __init__(self, n_residue_types: int = 20, **kwargs: Any) -> None:
        super().__init__(dim=n_residue_types)
        self.n_residue_types = n_residue_types

    def forward(self, batch: dict) -> torch.Tensor:
        assert "residue_type" in batch, "`residue_type` not in batch, cannot compute ResidueTypeSeqFeat"
        rtype = batch["residue_type"]  # [b, n]
        if "can_be_encoded" in batch:
            rpadmask = batch["can_be_encoded"].long()
        else:
            rpadmask = batch["padding_mask"]
        rtype = torch.clamp(rtype, min=0, max=self.n_residue_types - 1)
        rtype = rtype * rpadmask
        rtype_onehot = F.one_hot(rtype, num_classes=self.n_residue_types)
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

    def __init__(self, num_classes: int = 33, learnable: bool = False, **kwargs: Any) -> None:
        super().__init__(dim=num_classes)
        self._has_logged = False
        self.num_classes = num_classes

        # Create embedding layer initialized as identity matrix (one-hot behavior)
        self.embedding = torch.nn.Embedding(num_classes, num_classes)
        if not learnable:
            # Initialize as identity matrix for one-hot behavior
            self.embedding.weight.data = torch.eye(num_classes)
            self.embedding.weight.requires_grad = False

    def forward(self, batch: dict) -> torch.Tensor:
        if C_SEQ.full_name in batch and C_SEQ.mask_name in batch:
            rtype = batch[C_SEQ.full_name]  # [b, n] - contains <M> for unconditioned tokens
            seq_cond_mask = batch[C_SEQ.mask_name]  # [b, n] - True for sequence-conditioned tokens

            # Apply sequence conditioning mask
            rtype = rtype * seq_cond_mask  # [b, n]

            # Validate indices before embedding lookup
            if not torch.compiler.is_compiling():
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

    def forward(self, batch: dict) -> torch.Tensor:
        assert "token_type" in batch, "`token_type` not in batch"
        token_type = batch["token_type"]  # [b, n] integers 0-4

        # Validate indices before embedding lookup
        if not torch.compiler.is_compiling():
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

    def __init__(self, rel: bool = False, **kwargs: Any) -> None:
        super().__init__(dim=int(37 * 4))
        # 37 * 4, 37 * 3 for the coordinates, + 37 for the mask
        self.rel = rel  # whether to get features relative to CA or absolute

    def forward(self, batch: dict) -> torch.Tensor:
        assert "coords_nm" in batch, "`coords_nm` not in batch, cannot compute Atom37NanometersCoorsSeqFeat"
        coors = batch["coords_nm"]  # [b, n, 37, 3]
        if "atom37_mask" in batch:
            coors_mask = batch["atom37_mask"]  # [b, n, 37]
        elif "coord_mask" in batch:
            coors_mask = batch["coord_mask"]  # [b, n, 37]
        else:
            raise ValueError("Neither 'atom37_mask' nor 'coord_mask' found in batch")
        coors = coors * coors_mask[..., None]  # Zero-out non-atoms

        if self.rel:
            # If relative remove CA coordinates
            ca_coors = coors[:, :, 1, :]  # [b, n, 3]
            coors = coors - ca_coors[:, :, None, :]  # [b, n, 37, 3]
            coors = coors * coors_mask[..., None]

        coors_flat = einops.rearrange(coors, "b n a t -> b n (a t)")  # [b, n, 37, 3] -> [b, n, 37 * 3]
        feat = torch.cat([coors_flat, coors_mask.float()], dim=-1)  # [b, n, 37 * 4]
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

    def __init__(self, rel: bool = False, **kwargs: Any) -> None:
        super().__init__(dim=int(37 * 4))
        self.rel = rel  # whether to get features relative to CA or absolute
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
        if C_CRD.full_name in batch and C_CRD.mask_name in batch:
            # C_CRD.full_name coordinates are in Angstroms from collate, convert to nm
            # Already pre-masked by pipeline, but we apply mask explicitly for clarity
            # turn nans into 0s in coors
            coors = torch.nan_to_num(batch[C_CRD.full_name], nan=0.0) / 10.0  # [b, n, 37, 3] - convert Angstroms to nm
            coors_mask = batch[C_CRD.mask_name]  # [b, n, 37] - CONDITIONING mask (not atom validity)
            coors = coors * coors_mask[..., None]  # Zero-out unconditioned atoms

            if self.rel:
                # If relative remove CA coordinates
                ca_coors = coors[:, :, 1, :]  # [b, n, 3]
                coors = coors - ca_coors[:, :, None, :]  # [b, n, 37, 3]
                coors = coors * coors_mask[..., None]

            coors_flat = einops.rearrange(coors, "b n a t -> b n (a t)")  # [b, n, 37, 3] -> [b, n, 37 * 3]
            feat = torch.cat([coors_flat, coors_mask.float()], dim=-1)  # [b, n, 37 * 4]
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


# TODO: Remove once AE is retrained without this
class OpenfoldSideChainAnglesSeqFeat(Feature):
    """Computes sequence features from side chain angles."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=int(4 * 21 + 4))  # 88

    def forward(self, batch: dict) -> torch.Tensor:
        _, angles, torsion_angles_mask = self._get_sidechain_angles(batch)
        # _, [b, n, 4] and [b, n, 4]
        angles_feat = bin_and_one_hot(
            angles, torch.linspace(-torch.pi, torch.pi, 20, device=angles.device)
        )  # [b, n, 4, nbins]
        angles_feat = angles_feat * torsion_angles_mask[..., None]
        angles_feat = einops.rearrange(angles_feat, "b n s d -> b n (s d)")  # [b, n, 4 * nbins]
        feat = torch.cat([angles_feat, torsion_angles_mask], dim=-1)  # [b, n, 4 * nbins + 4]
        return feat

    def _get_sidechain_angles(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


# TODO: Remove once AE is retrained without this
class BackboneTorsionAnglesSeqFeat(Feature):
    """
    Computes torsion angle and featurizes it, with binning and 1-hot.

    TODO: Add mask?
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=int(3 * 21))

    def forward(self, batch: dict) -> torch.Tensor:
        bb_torsion = self._get_bb_torsion_angles(batch)  # [b, n, 3]
        bb_torsion_feats = bin_and_one_hot(
            bb_torsion,
            torch.linspace(-torch.pi, torch.pi, 20, device=bb_torsion.device),
        )  # [b, n, 3, nbins], nbins in 20+1
        bb_torsion_feats = einops.rearrange(bb_torsion_feats, "b n t d -> b n (t d)")  # [b, n, 3 * nbins]
        return bb_torsion_feats

    def _get_bb_torsion_angles(self, batch: dict) -> torch.Tensor:
        a37 = batch["coords"]  # [b, n, 37, 3]
        b, n = a37.shape[0], a37.shape[1]
        device = a37.device

        # Get mask for encodable tokens (proteins only, not small molecules)
        if "can_be_encoded" in batch:
            can_be_encoded = batch["can_be_encoded"].bool()  # [b, n]
        elif "padding_mask" in batch:
            can_be_encoded = batch["padding_mask"].bool()  # [b, n]
        else:
            can_be_encoded = torch.ones(b, n, dtype=torch.bool, device=device)

        if "residue_pdb_idx" in batch and batch["residue_pdb_idx"] is not None:
            idx = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            idx = torch.arange(1, n + 1, device=device).unsqueeze(0).expand(b, -1).float()  # [b, n]

        # Extract backbone atoms
        N = a37[:, :, 0, :]  # [b, n, 3]
        CA = a37[:, :, 1, :]  # [b, n, 3]
        C = a37[:, :, 2, :]  # [b, n, 3]

        # Compute dihedral angles (these will have NaN for non-protein tokens)
        psi = signed_dihedral_angle(N[:, :-1, :], CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :])  # [b, n-1]
        omega = signed_dihedral_angle(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :])  # [b, n-1]
        phi = signed_dihedral_angle(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :], C[:, 1:, :])  # [b, n-1]
        bb_angles = torch.stack([psi, omega, phi], dim=-1)  # [b, n-1, 3]

        # Mask: both residues in pair must be encodable and sequential
        good_pair = idx[:, 1:] - idx[:, :-1] == 1  # boolean [b, n-1]
        valid_pair = can_be_encoded[:, :-1] & can_be_encoded[:, 1:] & good_pair  # [b, n-1]

        # Zero out invalid pairs (this handles NaN from small molecules)
        bb_angles = torch.where(valid_pair[..., None], bb_angles, torch.zeros_like(bb_angles))

        zero_pad = torch.zeros((b, 1, 3), device=device)
        bb_angles = torch.cat([bb_angles, zero_pad], dim=1)  # [b, n, 3]
        return bb_angles


class ConditionalAtom37NanometersInternalCoorsSeqFeat(Feature):
    """
    Computes feature from conditional atom37 coordinates, feature of shape [b, n, 37 * 4].

    Uses `C_INTCRD.full_name` and `C_INTCRD.mask_name` if available, otherwise returns zeros.
    This prevents leakage of ground truth coordinate information during training.

    The feature is masked by `C_INTCRD.mask_name` which indicates which atoms have
    internal coordinate conditioning applied (zeros for absolute or no coordinate conditioning).

    For unconditional generation, this is all zeros.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=int(37 * 4))
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
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
            feat = torch.cat([internal_coords_flat, internal_coords_mask.float()], dim=-1)  # [b, n, 37 * 4]
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


class ChiralitySeqFeat(Feature):
    """Raw chirality block-diagonal feature.

    Output: ``[b, n, MAX_OVERLAPPING_CENTERS * 3]``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=MAX_OVERLAPPING_CENTERS * 3)

    def forward(self, batch: dict) -> torch.Tensor:
        if "chirality" in batch:
            return batch["chirality"].float()

        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        _warn_once("No chirality in batch, returning zeros")
        return torch.zeros(b, n, self.dim, device=device)


class CoordGroupSeqFeat(Feature):
    """Binary feature for whether two atoms are in the same coordinate group."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37)

    def forward(self, batch: dict) -> torch.Tensor:
        coord_group = batch[C_CRDGRP.full_name].long().clone()  # [b, n, 37]

        # Assign a random value to the non-padding index groups, to avoid the model learning spurious correlations
        non_padding_mask = coord_group != C_CRDGRP.UNKNOWN_COORDS
        unique_groups, inverse_indices = torch.unique(coord_group[non_padding_mask], return_inverse=True)
        permuted_index_groups = torch.randperm(len(unique_groups), device=coord_group.device)
        coord_group[non_padding_mask] = permuted_index_groups[inverse_indices]

        return coord_group.float()  # [b, n, 37]


class LatentVariableSeqFeat(Feature):
    """Returns sequence feature from latent variable."""

    def __init__(self, latent_z_dim: int, **kwargs: Any) -> None:
        super().__init__(dim=latent_z_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        assert "z_latent" in batch, "`z_latent` not in batch, cannot compute LatentVariableSeqFeat"
        return batch["z_latent"]  # [b, n, latent_dim]


class RasaBinnedSeqFeat(Feature):
    """RASA (surface accessibility) binned feature: 0=BURIED, 1=INTERMEDIATE, 2=EXPOSED, 3=UNSPECIFIED.

    Returns atom37 representation [b, n, 37] as feature.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37)

    def forward(self, batch: dict) -> torch.Tensor:
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

    def __init__(self, embed_dim: int = 16, **kwargs: Any) -> None:
        super().__init__(dim=embed_dim)
        self.ss_embedding = torch.nn.Embedding(4, embed_dim, padding_idx=0)

    def forward(self, batch: dict) -> torch.Tensor:
        if C_SEC.full_name not in batch:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            # Dummy forward so embedding participates in DDP backward pass
            zeros = torch.zeros(b, n, self.dim, device=device)
            return zeros + 0.0 * self.ss_embedding.weight.sum()

        ss_type = batch[C_SEC.full_name].long()  # [b, n] - already token-level

        # Validate indices before embedding lookup
        if not torch.compiler.is_compiling():
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37)

    def forward(self, batch: dict) -> torch.Tensor:
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37)

    def forward(self, batch: dict) -> torch.Tensor:
        if C_HOT.full_name not in batch:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, 37, device=device)

        hotspot_atom37 = batch[C_HOT.full_name].float()  # [b, n, 37] boolean
        atom37_mask = batch["atom37_mask"]  # [b, n, 37]

        # Zero out invalid atoms
        hotspot_feat = hotspot_atom37 * atom37_mask.float()  # [b, n, 37]
        return hotspot_feat


class IndexGroupPairFeat(Feature):
    """Binary feature for whether two tokens are in the same index group."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        idx_group = batch[C_IDXGRP.full_name].long().clone()  # [b, n] - already token-level
        idx_group_pair_feat = idx_group[:, :, None] == idx_group[:, None, :]  # [b, n, n] boolean
        return idx_group_pair_feat.unsqueeze(-1).float()  # [b, n, n, 1]


class AtomChargeSeqFeat(Feature):
    """Learned embedding for formal charge on atomized tokens.

    Formal charges are mapped to classes: -4→0, -3→1, ..., 0→4, ..., +6→10.
    Out-of-range values are clamped. Non-atomized tokens get zeros.

    The embedding table is initialized from sin/cos of actual charge values so
    that nearby charges start with similar representations, but the weights are
    learnable and can specialize during training.

    .. note::
       For atomized residues each atom occupies one token, and the per-atom
       scalars (charge, element, atom-name, ...) are written into ``slot 1`` of
       the atom37 channel by ``ConvertToAtom37Format``. Slot 0 is reserved for
       the residue-level CA position (zero / unused for atomized tokens), so
       reading ``batch["charge"][:, :, 1]`` is the atom37 convention for
       fetching the per-atom value of an atomized token.

    Shape: [b, n, embed_dim].
    """

    _CHARGE_OFFSET = 4  # maps charge -4..+6 to index 0..10
    _NUM_CLASSES = 11

    def __init__(self, charge_emb_dim: int = 16, **kwargs):
        super().__init__(dim=charge_emb_dim)
        self._has_logged = False
        self.embedding = torch.nn.Embedding(self._NUM_CLASSES, charge_emb_dim)
        self._init_embedding_from_charge_values(charge_emb_dim)

    def _init_embedding_from_charge_values(self, dim: int) -> None:
        """Initialize embedding rows from sin/cos of actual charge values.

        Args:
          dim: Embedding dimension (must be even).
        """
        charges = torch.arange(
            -self._CHARGE_OFFSET,
            self._NUM_CLASSES - self._CHARGE_OFFSET,
            dtype=torch.float32,
        )
        with torch.no_grad():
            half = dim // 2
            i = torch.arange(half, dtype=torch.float32)
            freqs = 1.0 / (10.0 ** (2 * i / dim))
            for idx, c in enumerate(charges):
                self.embedding.weight[idx, 0::2] = torch.sin(c * freqs)
                self.embedding.weight[idx, 1::2] = torch.cos(c * freqs)

    def forward(self, batch: dict) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        if "charge" in batch and "is_atomized" in batch:
            charge_slot1 = batch["charge"][:, :, 1].long()  # [b, n]
            cls_idx = (charge_slot1 + self._CHARGE_OFFSET).clamp(0, self._NUM_CLASSES - 1)
            emb = self.embedding(cls_idx)  # [b, n, charge_emb_dim]
            return emb * batch["is_atomized"].unsqueeze(-1)
        if not self._has_logged:
            logger.warning("No charge in batch, returning zeros for AtomChargeSeqFeat")
            self._has_logged = True
        # Dummy forward so embedding participates in DDP backward pass
        zeros = torch.zeros(b, n, self.dim, device=device)
        return zeros + 0.0 * self.embedding.weight.sum()


class AtomicElementSeqFeat(Feature):
    """Learned embedding for atomic element on atomized tokens.

    For atomized tokens: Embeds element from atom37 slot 1 (atomized atoms
    are placed in slot 1 by ConvertToAtom37Format).
    For non-atomized tokens (proteins): Returns zero vector.
    Shape: [b, n, embed_dim].
    """

    def __init__(self, max_atomic_num: int = 128, element_emb_dim: int = 16, **kwargs):
        super().__init__(dim=element_emb_dim)
        self._has_logged = False
        self.max_atomic_num = max_atomic_num
        self.embedding = torch.nn.Embedding(max_atomic_num, element_emb_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        if "element" in batch and "is_atomized" in batch:
            # Atomized tokens place their atom in slot 1
            element_slot1 = batch["element"][:, :, 1].long()  # [b, n]
            element_slot1 = element_slot1.clamp(0, self.max_atomic_num - 1)
            emb = self.embedding(element_slot1)  # [b, n, embed_dim]
            return emb * batch["is_atomized"].unsqueeze(-1)
        if not self._has_logged:
            logger.warning("No element in batch, returning zeros for AtomicElementSeqFeat")
            self._has_logged = True

        # Dummy forward so embedding participates in DDP backward pass
        zeros = torch.zeros(b, n, self.dim, device=device)
        return zeros + 0.0 * self.embedding.weight.sum()


class ChainIdxSeqFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        if "chain_iid" in batch:
            mask = batch["chain_iid"].unsqueeze(-1).float()  # [b, n, 1]
        elif "chains" in batch:
            mask = batch["chains"].unsqueeze(-1).float()  # [b, n, 1] (legacy key)
        else:
            raise ValueError("Neither 'chain_iid' nor 'chains' found in batch")
        return mask


class SequenceSeparationPairFeat(Feature):
    """Computes sequence separation and returns feature of shape [b, n, n, seq_sep_dim]."""

    def __init__(self, seq_sep_dim: int, **kwargs: Any) -> None:
        super().__init__(dim=seq_sep_dim)

    def forward(self, batch: dict) -> torch.Tensor:
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

    def __init__(self, xt_pair_dist_dim: int, xt_pair_dist_min: float, xt_pair_dist_max: float, **kwargs: Any) -> None:
        super().__init__(dim=xt_pair_dist_dim)
        self.min_dist = xt_pair_dist_min
        self.max_dist = xt_pair_dist_max

    def forward(self, batch: dict) -> torch.Tensor:
        data_modes_avail = list(batch["x_t"])
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=30)
        self.min_dist = 0.1
        self.max_dist = 3.0

    def forward(self, batch: dict) -> torch.Tensor:
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
        x_sc_pair_dist_dim: int,
        x_sc_pair_dist_min: float,
        x_sc_pair_dist_max: float,
        mode_key: str = "x_sc",
        **kwargs: Any,
    ) -> None:
        super().__init__(dim=x_sc_pair_dist_dim)
        self.min_dist = x_sc_pair_dist_min
        self.max_dist = x_sc_pair_dist_max
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
        if self.mode_key in batch:
            data_modes_avail = list(batch[self.mode_key])
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


# TODO: Remove once AE is retrained without this
class RelativeResidueOrientationPairFeat(Feature):
    """Computes pair feature with pairwise residue orientations.

    See paper "Improved protein structure prediction using
    predicted inter-residue orientations".

    TODO: Impute beta carbon for Glycine
    TODO: 20 as argument
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=int(5 * 21))  # 105

    def forward(self, batch: dict) -> torch.Tensor:
        _aatype = batch["residue_type"]  # [b, n]
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

        N_p1, CA_p1, CB_p1 = (v[:, :, None, :] for v in (N, CA, CB))  # Each [b, n, 1, 3]
        N_p2, CA_p2, CB_p2 = (v[:, None, :, :] for v in (N, CA, CB))  # Each [b, 1, n, 3]

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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=int(4 * 21))  # 84

    def forward(self, batch: dict) -> torch.Tensor:
        assert "coords_nm" in batch, "`coords_nm` not in batch, cannot comptue BackbonePairDistancesNanometerPairFeat"
        coords = batch["coords_nm"]
        if "atom37_mask" in batch:
            atom_mask = batch["atom37_mask"]  # [b, n, 37]
        elif "coord_mask" in batch:
            atom_mask = batch["coord_mask"]  # [b, n, 37]
        else:
            raise ValueError("Neither 'atom37_mask' nor 'coord_mask' found in batch")
        mask = atom_mask[:, :, 1]  # [b, n]
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
        has_cb = atom_mask[:, :, 3]  # [b, n] boolean, indicates if corresponding

        N = coords[:, :, 0, :]  # [b, n, 3]
        CA = coords[:, :, 1, :]  # [b, n, 3]
        C = coords[:, :, 2, :]  # [b, n, 3]
        CB = coords[:, :, 3, :]  # [b, n, 3]

        CA_i = CA[:, :, None, :]  # [b, n, 1, 3]
        N_j, CA_j, C_j, CB_j = (v[:, None, :, :] for v in (N, CA, C, CB))  # Each [b, 1, n, 3]

        CA_N, CA_CA, CA_C, CA_CB = (
            torch.linalg.norm(v[0] - v[1], dim=-1) for v in ((CA_i, N_j), (CA_i, CA_j), (CA_i, C_j), (CA_i, CB_j))
        )  # Each shape [b, n, n]
        # CA_X[..., i, j] has distance (nm) between CA[..., i] and X[..., j]

        # Accomodate residues without CB
        CA_CB = CA_CB * has_cb[:, None, :]  # [b, n, n]

        # Fix for mask
        CA_N, CA_CA, CA_C, CA_CB = (v * pair_mask for v in (CA_N, CA_CA, CA_C, CA_CB))  # Each shape [b, n, n]

        bin_limits = torch.linspace(0.1, 2, 20, device=coords.device)
        CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat = (
            bin_and_one_hot(v, bin_limits=bin_limits) for v in (CA_N, CA_CA, CA_C, CA_CB)
        )  # Each [b, n, n, 21]

        feat = torch.cat([CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat], dim=-1)  # [b, n, n, 4 * 21]
        feat = feat * pair_mask[..., None]
        return feat


class ChainIdxPairFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, n, 1]."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        if "chain_iid" in batch:
            seq_mask = batch["chain_iid"].float()  # [b, n]
        elif "chains" in batch:
            seq_mask = batch["chains"].float()  # [b, n] (legacy key)
        else:
            raise ValueError("Neither 'chain_iid' nor 'chains' found in batch")
        mask = torch.einsum("bi,bj->bij", seq_mask, seq_mask).unsqueeze(-1)  # [b, n, n, 1]
        return mask


class MoleculeIdxPairFeat(Feature):
    """Binary feature for whether two tokens belong to the same molecule instance."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        molecule_iid = batch["molecule_iid"].long()  # [b, n]
        same_molecule = molecule_iid[:, :, None] == molecule_iid[:, None, :]  # [b, n, n] boolean
        return same_molecule.unsqueeze(-1).float()  # [b, n, n, 1]


class BondMaskPairFeat(Feature):
    """Binary bond existence mask for atomized molecules.

    Non-atomized token pairs get zeros.
    Shape: [b, n, n, 1]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
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
    """Bond order one-hot encoding for atomized molecules (Kekulé assignments).

    Bond types: 0=none, 1=single, 2=double, 3=triple, 4=coordination.
    Aromatic bonds receive their Kekulé bond order (single/double/triple).
    Non-atomized token pairs get zeros.
    Shape: [b, n, n, 5]
    """

    def __init__(self, num_bond_orders: int = 5, **kwargs: Any) -> None:
        super().__init__(dim=num_bond_orders)
        self._has_logged = False
        self.num_bond_orders = num_bond_orders

    def forward(self, batch: dict) -> torch.Tensor:
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


class IsAromaticPairFeat(Feature):
    """Binary aromaticity flag for atomized molecule bonds.

    Dropout is handled upstream by :class:`DropoutAromaticFlag` transform.
    Shape: [b, n, n, 1]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        if "is_aromatic" in batch:
            return batch["is_aromatic"][..., None].float()  # [b, n, n, 1]
        else:
            if not self._has_logged:
                logger.warning("No is_aromatic in batch, returning zeros for IsAromaticPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, 1, device=device)


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
        self.ret_zero = bool(feats is None or len(feats) == 0)
        if self.ret_zero:
            logger.info("No features requested")
            self.zero_creator = ZeroFeat(dim_feats_out=dim_feats_out, mode=mode)
            return

        self.feat_creators = torch.nn.ModuleList([self.get_creator(f, **kwargs) for f in feats])
        self.ln_out = torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        self.linear_out = torch.nn.Linear(sum([c.get_dim() for c in self.feat_creators]), dim_feats_out, bias=False)

    def get_creator(self, f: str, **kwargs: Any) -> Feature:
        """Returns the right class for the requested feature f (a string)."""

        if self.mode == "seq":
            # Time embeddings
            if f == "time_emb_bb_ca":
                return TimeEmbeddingSeqFeat(data_mode_use="bb_ca", **kwargs)
            elif f == "time_emb_local_latents":
                return TimeEmbeddingSeqFeat(data_mode_use="local_latents", **kwargs)

            # Position and indexing
            elif f == "chain_break_per_res":
                return ChainBreakPerResidueSeqFeat(**kwargs)
            elif f == "chain_idx_seq":
                return ChainIdxSeqFeat(**kwargs)

            # Basic residue information
            elif f == "x1_aatype":
                return ResidueTypeSeqFeat(**kwargs)
            elif f == "token_type":
                return TokenTypeSeqFeat(**kwargs)

            # Raw coordinate features
            elif f == "ca_coors_nm":
                return CaCoorsNanometersSeqFeat(**kwargs)
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
            elif f == "x_sc_local_latents":
                return XscLocalLatentsSeqFeat(**kwargs)

            # Latent variables
            elif f == "z_latent_seq":
                return LatentVariableSeqFeat(**kwargs)

            # 1D Condition annotations
            elif f == "rasa_binned":
                return RasaBinnedSeqFeat(**kwargs)
            elif f == "secondary_structure":
                return SecondaryStructureSeqFeat(**kwargs)
            elif f == "hbond":
                return HydrogenBondSeqFeat(**kwargs)
            elif f == "hotspot":
                return HotspotSeqFeat(**kwargs)
            elif f == "coord_group_seq":
                return CoordGroupSeqFeat(**kwargs)
            elif f == "cath_code":
                return FoldEmbeddingSeqFeat(**kwargs)

            # Structural features (angles)
            elif f == "x1_bb_angles":
                return BackboneTorsionAnglesSeqFeat(**kwargs)
            elif f == "x1_sidechain_angles":
                return OpenfoldSideChainAnglesSeqFeat(**kwargs)

            # Ligand/atomized features
            elif f == "atom_charge":
                return AtomChargeSeqFeat(**kwargs)
            elif f == "atomic_element":
                return AtomicElementSeqFeat(**kwargs)
            # Conditional features (prevent leakage)
            elif f == "cond_residue_type":
                return ConditionalResidueTypeSeqFeat(**kwargs)
            elif f == "cond_a37coors_nm":
                return ConditionalAtom37NanometersCoorsSeqFeat(**kwargs)
            elif f == "cond_a37coors_internal_nm":
                return ConditionalAtom37NanometersInternalCoorsSeqFeat(**kwargs)

            # Chirality
            elif f == "chirality":
                return ChiralitySeqFeat(**kwargs)

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
            elif f == "molecule_idx_pair":
                return MoleculeIdxPairFeat(**kwargs)
            elif f == "index_group_pair":
                return IndexGroupPairFeat(**kwargs)

            # Bond features (for atomized molecules/ligands)
            elif f == "bond_mask":
                return BondMaskPairFeat(**kwargs)
            elif f == "bond_order":
                return BondOrderPairFeat(**kwargs)
            elif f == "is_aromatic":
                return IsAromaticPairFeat(**kwargs)

            else:
                raise OSError(f"Pair feature {f} not implemented.")

        else:
            raise OSError(f"Wrong feature mode (creator): {self.mode}. Should be 'seq' or 'pair'.")

    def apply_padding_mask(self, feature_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies mask to features.

        Args:
            feature_tensor: tensor with requested features, shape [b, n, d] or [b, n, n, d]
                depending on self.mode ('seq' or 'pair')
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

    def forward(self, batch: dict) -> torch.Tensor:
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
