import math

import numpy as np
import torch
from atomworks.io.utils.atom_array_plus import (
    AtomArrayPlus,
    as_atom_array_plus,
)
from atomworks.ml.conditions import C_DIS
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import (
    apply_and_spread_token_wise,
    get_token_starts,
)
from beartype.typing import Any
from biotite.structure import AtomArray
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor

from foundry.common import true_with_prob
from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)

Array = np.ndarray
"""Type alias for numpy arrays."""


def stochastic_pair_knockout(
    eligible_pair_indices: tuple[np.ndarray, np.ndarray],
    min_valid_elements: int = 1,
    low: float = 0.1,
    high: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly knocks out pairs with probability.

    Args:
        eligible_pair_indices: tuple of arrays where eligible_pair_indices[0][i] and
            eligible_pair_indices[1][i] are the two indices of the i-th eligible pair.
            This must be upper triangle indices (i < j) and must not include the diagonal!
        min_valid_elements: The minimum number of valid elements that must remain after
            dropout.
        low: The lower bound for the random dropout percentage. If 0.1, then a minimum
            of 10% of the valid elements can be dropped.
        high: The upper bound for the random dropout percentage. If 0.9, then a maximum
            of 90% of the valid elements can be dropped.

    Returns:
        tuple of arrays where the i-th pair is knocked out with probability.
    """

    # make sure eligible_pair_indices is upper triangle indices
    assert (
        eligible_pair_indices[0] < eligible_pair_indices[1]
    ).all(), "eligible_pair_indices must be upper triangle indices"

    if len(eligible_pair_indices[0]) > min_valid_elements:
        percent_to_remove = np.random.uniform(low, high)
        n_knockout = min(
            int(len(eligible_pair_indices[0]) * percent_to_remove),  # amount we would remove
            len(eligible_pair_indices[0]) - min_valid_elements,  # max amount we can remove
        )

        rng = np.random.default_rng()
        knockout_indices = rng.choice(len(eligible_pair_indices[0]), size=n_knockout, replace=False)

        eligible_pair_indices = (
            eligible_pair_indices[0][knockout_indices],
            eligible_pair_indices[1][knockout_indices],
        )  # (n_knockout, 2)

    return eligible_pair_indices


def compute_distances(atom_array: AtomArray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # ... build edge list
    # idxs = np.where(mask)[0]
    # m = len(idxs)
    if mask.ndim == 1:
        assert mask.shape == (
            atom_array.array_length(),
        ), f"mask.shape: {mask.shape} != (atom_array.array_length(),): {atom_array.array_length()}"

        # # Upper triangle (i < j)
        # row_u, col_u = np.triu_indices(m, k=1)
        # pairs = np.column_stack((idxs[row_u], idxs[col_u]))  # <- upper triangle
        mask_2d = np.outer(mask, mask)  # (L, L)
    else:
        assert mask.ndim == 2, f"mask.ndim: {mask.ndim} != 1 or 2"
        assert (
            mask.shape[0] == mask.shape[1] == atom_array.array_length()
        ), f"mask.shape: {mask.shape} != (L, L). (L={atom_array.array_length()})"
        mask_2d = mask

    mask_2d[np.diag_indices(mask_2d.shape[0])] = False  # always do this just in case
    pairs = np.array(np.where(mask_2d)).T  # (n_dist, 2)
    # ... compute distances
    distances = np.linalg.norm(atom_array.coord[pairs[:, 0]] - atom_array.coord[pairs[:, 1]], axis=-1)  # (n_dist,)
    return pairs, distances


def sample_distance_condition(
    atom_array: AtomArray | AtomArrayPlus,
    can_have_condition: Bool[Array, "L"],
    knockout_p: float = 0.8,
    dropout_min_fraction: float = 0.1,
    dropout_max_fraction: float = 0.9,
    additive: bool = False,
) -> AtomArrayPlus:
    # Ensure we are working with AtomArrayPlus for 2D annotation support
    atom_array = as_atom_array_plus(atom_array)

    can_have_condition = can_have_condition & ~np.isnan(atom_array.coord).any(
        axis=-1
    )  # We do not support nan coordinates
    can_have_condition_positions = np.where(can_have_condition)[0]

    #  Generate set of `eligible` pairs
    # A pair is eligible if both atoms can have condition (e.g., outer product of 1-body mask)
    # eligible_pair_indices is a tuple of arrays where eligible_pair_indices[0][i] and
    # eligible_pair_indices[1][i] are the two indices of the i-th eligible pair
    # using can_have_condition_positions. So the i-th eligible pair is:
    # (can_have_condition_positions[eligible_pair_indices[0][i]],
    #  can_have_condition_positions[eligible_pair_indices[1][i]])
    eligible_pair_indices = np.triu_indices(len(can_have_condition_positions), k=1)  # k=1 to exclude the diagonal

    # ... knockout some pairs with probability
    if true_with_prob(knockout_p):
        eligible_pair_indices = stochastic_pair_knockout(
            eligible_pair_indices,
            min_valid_elements=1,
            low=dropout_min_fraction,
            high=dropout_max_fraction,
        )

    # ... get real pair indices and distance values. Still only upper triangle!!!
    upper_triangle_pairs = np.stack(
        [
            can_have_condition_positions[eligible_pair_indices[0]],
            can_have_condition_positions[eligible_pair_indices[1]],
        ],
        axis=1,
    )  # (n_dist, 2)

    upper_triangle_distances = np.linalg.norm(
        atom_array.coord[upper_triangle_pairs[:, 0]] - atom_array.coord[upper_triangle_pairs[:, 1]],
        axis=-1,
    )  # (n_dist,)

    # ... symmetrize!
    pairs = np.concatenate([upper_triangle_pairs, upper_triangle_pairs[:, ::-1]])
    dist_values = np.concatenate([upper_triangle_distances, upper_triangle_distances])

    # check for nans
    assert not np.isnan(dist_values).any(), "dist_values has NaNs"

    # Annotate the atom array
    if additive:
        existing_annotation = C_DIS.annotation(atom_array, default="generate")
        existing_pairs, existing_dist_values = (
            existing_annotation.pairs,
            existing_annotation.values,
        )
        C_DIS.set_annotation(
            atom_array,
            pairs=np.concatenate([existing_pairs, pairs]),
            values=np.concatenate([existing_dist_values, dist_values]),
        )
    else:
        C_DIS.set_annotation(atom_array, pairs=pairs, values=dist_values)

    return atom_array


def fourier_embed_scales(
    x: Float[Tensor, "... L"],
    scales: Float[Tensor, "d"],
) -> Float[Tensor, "... L 2*d"]:
    """Applies Fourier feature embedding to input tensor using provided scales.

    Formula (scales s = [s1, s2, ...]):
        ϕ(x) = [sin(2πx / s), cos(2πx / s)]

    Args:
        x: Input tensor to embed, with arbitrary leading dimensions.
        scales: Tensor of shape (d,) where d is half the desired embedding dimension.
            Indicates the scales of interest (i.e. resolution).

    Returns:
        Tensor with shape (*batch, t, 2*d) containing sin/cos embeddings.
    """
    proj = einsum(x, 1.0 / scales, "... t, d -> ... t d")
    proj = 2 * math.pi * proj
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def log_embed_scales(
    x: Float[Tensor, "... L"],
    scales: Float[Tensor, "d"],
) -> Float[Tensor | Array, "... L d"]:
    """Applies Log embedding to input tensor using provided scales."""
    return torch.log1p(x / scales)


def exponential_decay_embed_scales(
    x: Float[Tensor, "... L"],
    scales: Float[Tensor, "d"],
) -> Float[Tensor, "... L d"]:
    """Applies Exponential Decay embedding to input tensor using provided scales.

    Formula (scales s = [s1, s2, ...]):
        ϕ(x) = exp(-x / s)

    Args:
        x: Input tensor to embed, with arbitrary leading dimensions (distances >=0).
        scales: Tensor of shape (d,) indicating the tau scales for each embedding dimension.
            Smaller taus provide faster decay for small distances.

    Returns:
        Tensor with shape (*batch, t, d) containing exponential decay embeddings.
    """
    proj = einsum(x, 1.0 / scales, "... t, d -> ... t d")
    return torch.exp(-proj)


def bessel_radial_embed_scales(
    x: Float[Tensor | Array, "... L"],
    scales: Float[Tensor | Array, "d"],
    max_scale: float = 128.0,
    eps: float = 1e-4,
) -> Float[Tensor, "... L d"]:
    """Applies Bessel radial embedding to input tensor using provided scales.

    Formula (s = [s1, s2, ...]):
        R_s(r) = sqrt(2 / s_max) * sin(πsr / s_max) / r

    Args:
        x: Input tensor to embed, with arbitrary leading dimensions (distances >=0).
        scales: Tensor of shape (d,) indicating the n indices (typically integers 1,2,...d)
            for each embedding dimension. Larger n provides higher frequency resolution.

    Returns:
        Tensor with shape (*batch, t, d) containing Bessel radial embeddings.
    """
    arg = x.unsqueeze(-1) * (scales * torch.pi / max_scale)
    sin_term = torch.sin(arg)
    denom = x.unsqueeze(-1) + eps
    embed = torch.sqrt(2 / torch.as_tensor(max_scale)) * sin_term / denom
    return embed


def featurize_distances_conditioning_via_fourier(atom_array: AtomArrayPlus, nbins: int = 64) -> dict[str, Any]:
    """Featurizes distance conditioning using Fourier embeddings.

    Args:
        atom_array: AtomArrayPlus to featurize.
        nbins: Number of Fourier bins (each phase will include nbins/2 sine/cosine pairs).

    Returns:
        dict: Features for 2D distance conditioning.
    """
    L = atom_array.array_length()
    device = torch.device("cpu")

    ## 1. Target distances
    # ... get the distance values
    distance_annotation = C_DIS.annotation(atom_array, default="generate")
    dists = distance_annotation.as_dense_array(default=0.0).astype(float)  # (L, L)

    # Compute all-by-all pairwise distances (L, L)
    dists_tensor = torch.from_numpy(dists).to(dtype=torch.float32, device=device)  # (L, L)

    # ... ensure no nans
    assert not torch.isnan(dists_tensor).any(), "dists has NaNs"
    # ... ensure dists is symmetric
    assert torch.allclose(
        dists_tensor.T, dists_tensor
    ), f"dists must be symmetric. Got {distance_annotation.as_array()}"

    ## 2. Mask
    has_dist_condition_2d = dists_tensor > 0.0

    ## 3. Fourier embedding
    # Fourier embedding only where is_dist_visible is True
    # For full (L, L, D) output, set to 0 where not visible
    D = nbins
    # scales = torch.linspace(0.1, float(nbins), nbins // 2)
    scales = torch.exp(torch.linspace(np.log(0.1), np.log(nbins), nbins // 2))
    fourier_feats = torch.zeros((L, L, D), dtype=torch.float32, device=device)  # (L, L, D)
    if has_dist_condition_2d.any():
        emb = fourier_embed_scales(dists_tensor, scales)  # (L, L, D)
        fourier_feats[has_dist_condition_2d] = emb[has_dist_condition_2d]

    # ... set (1-body, atom) mask  (= Does this atom receive any distance condition?)
    atom_has_condition = has_dist_condition_2d.any(axis=0).numpy()
    C_DIS.set_mask(atom_array, atom_has_condition, 1, "atom")  # (L)

    # ... set (1-body, token) mask  (= Does any atom in this token receive any distance condition?)
    token_starts = get_token_starts(atom_array)
    token_segments = np.concatenate([token_starts, [atom_array.array_length()]])
    token_has_condition = apply_and_spread_token_wise(
        atom_array,
        C_DIS.mask(atom_array, 1, "atom", default="raise"),
        np.any,
        token_starts=token_segments,
    )
    C_DIS.set_mask(atom_array, token_has_condition, 1, "token")  # (L)

    # ... convert to tensors & fetch from annotation to ensure they were correctly set
    atom_has_condition = torch.from_numpy(C_DIS.mask(atom_array, 1, "atom", default="raise")).bool()
    token_has_condition = torch.from_numpy(C_DIS.mask(atom_array, 1, "token", default="raise")[token_starts]).bool()

    # Convert fourier_feats to sparse COO format
    fourier_feats_sparse = fourier_feats.to_sparse_coo()

    return {
        C_DIS.full_name: dists,  # (L, L)
        C_DIS.get_feature_name(2, "atom"): fourier_feats_sparse,  # (L, L, D) - sparse COO
        C_DIS.get_mask_name(2, "atom"): has_dist_condition_2d.bool(),  # (L, L) - bool
        C_DIS.get_mask_name(1, "atom"): atom_has_condition,  # (L) - bool
        C_DIS.get_mask_name(1, "token"): token_has_condition,  # (I) - bool
    }


class FeaturizeDistanceConditioningViaFourier(Transform):
    """Featurizes distance conditioning using Fourier embeddings."""

    def __init__(self, nbins: int = 64):
        """
        Args:
            nbins (int): Number of Fourier bins (each phase will include nbins/2 sine/cosine pairs).
        """
        self.nbins = nbins

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        feats = featurize_distances_conditioning_via_fourier(atom_array, nbins=self.nbins)
        if "feats" not in data:
            data["feats"] = {}
        data["feats"].update(feats)
        return data
