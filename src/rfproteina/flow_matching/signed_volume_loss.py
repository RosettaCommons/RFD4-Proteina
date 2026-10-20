"""Reference-normalized handedness loss for atomized ligand tetrahedra."""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def signed_volume_loss(
    predicted_coords: Float[Tensor, "b n 3"],
    reference_coords: Float[Tensor, "b n 3"],
    tetrahedron_indices: Int[Tensor, "b n 4"],
    supervised_tokens: Bool[Tensor, "b n"],
) -> Float[Tensor, "b"]:
    """Penalize incorrect handedness using reference-normalized signed volumes.

    Args:
        predicted_coords: Predicted clean CA/atomized coordinates in nm.
        reference_coords: Ground-truth coordinates in nm; gradients are detached.
        tetrahedron_indices: Four vertex-token indices per center; -1 means absent.
            Use four heavy neighbors, or three plus the center as the fourth vertex.
        supervised_tokens: True for tokens eligible for coordinate supervision;
            the center and all four vertices must qualify.

    Returns:
        Mean squared-hinge loss per example, or zero without eligible tetrahedra.
        Reference edge lengths normalize volumes; absolute normalized reference
        volume must be at least 0.1, with margin min(0.2, half that volume).
    """
    bi = torch.arange(predicted_coords.shape[0], device=predicted_coords.device)[:, None, None]
    has_tetrahedron = tetrahedron_indices[..., 0] >= 0
    tetrahedron_indices = tetrahedron_indices.clamp_min(0)
    p = predicted_coords.float()[bi, tetrahedron_indices]
    r = reference_coords.detach().float()[bi, tetrahedron_indices]
    eligible_tetrahedra = has_tetrahedron & supervised_tokens & supervised_tokens[bi, tetrahedron_indices].all(-1)
    eligible_tetrahedra = eligible_tetrahedra & torch.isfinite(r).all((-1, -2))
    if not torch.isfinite(p[eligible_tetrahedra]).all():
        raise FloatingPointError("Nonfinite predicted coordinates at supervised chiral centers")
    p = torch.where(eligible_tetrahedra[..., None, None], p, 0.0)
    r = torch.where(eligible_tetrahedra[..., None, None], r, 0.0)
    pv = p[..., :3, :] - p[..., 3:4, :]
    rv = r[..., :3, :] - r[..., 3:4, :]
    normalizer = torch.linalg.vector_norm(rv, dim=-1).clamp_min(0.01).prod(-1)
    ref_volume = (rv[..., 0, :] * torch.linalg.cross(rv[..., 1, :], rv[..., 2, :])).sum(-1)
    pred_volume = (pv[..., 0, :] * torch.linalg.cross(pv[..., 1, :], pv[..., 2, :])).sum(-1)
    normalized_reference = ref_volume / normalizer
    eligible_tetrahedra = eligible_tetrahedra & (normalized_reference.abs() >= 0.1)
    score = ref_volume.sign() * pred_volume / normalizer
    margin = (normalized_reference.abs() * 0.5).clamp_max(0.2)
    penalty = torch.relu(margin - score).square()
    return torch.where(eligible_tetrahedra, penalty, 0.0).sum(-1) / eligible_tetrahedra.sum(-1).clamp_min(1)
