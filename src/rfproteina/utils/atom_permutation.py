"""Permute symmetric ground-truth atoms onto a prediction.

Informed by Protenix's atom permutation: https://github.com/bytedance/Protenix
"""

import logging
from collections.abc import Sequence

import numpy as np
import torch
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.ml.transforms.symmetry import find_automorphisms_with_networkx
from biotite.structure import AtomArray, AtomArrayStack
from jaxtyping import Bool, Float, Int

from rfproteina.utils.align import _find_rot_alignment

logger = logging.getLogger(__name__)


def _superimpose_on(
    x_pred: Float[torch.Tensor, "n_atoms 3"],
    x_gt: Float[torch.Tensor, "n_atoms 3"],
    frame_mask: Bool[torch.Tensor, "n_atoms"],
) -> Float[torch.Tensor, "n_atoms 3"]:
    """Fit the superposition on ``frame_mask`` and apply it to every atom."""
    pred_centre = x_pred[frame_mask].mean(dim=0)
    gt_centre = x_gt[frame_mask].mean(dim=0)
    pred_centred = (x_pred - pred_centre) * frame_mask[:, None]  # [n_atoms, 3]
    gt_centred = (x_gt - gt_centre) * frame_mask[:, None]  # [n_atoms, 3]
    rotation = _find_rot_alignment(pred_centred[None], gt_centred[None], frame_mask[None])[0]  # [3, 3]
    return (x_pred - pred_centre) @ rotation.transpose(-1, -2) + gt_centre


def _pack_groups(
    automorphisms: Sequence[np.ndarray | torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Stack ragged automorphism groups into one padded batch of atom indices.

    Groups are bucketed by shape so each bucket stacks in a single call.

    Returns:
        ``(destinations, sources, valid_atom, valid_cand)`` of shape ``[n_groups, n_atoms]``,
        ``[n_groups, n_cand, n_atoms]``, ``[n_groups, n_atoms]`` and ``[n_groups, n_cand]``.
        ``valid_atom`` marks the atoms a group may exchange, ``valid_cand`` the real candidate
        rows. ``None`` if no group can exchange anything.
    """
    by_shape: dict[tuple[int, int], list[np.ndarray]] = {}
    for automorphism in automorphisms:
        group = np.asarray(automorphism)
        if group.ndim != 2 or group.shape[0] <= 1 or group.shape[1] == 0:
            continue
        by_shape.setdefault(group.shape, []).append(group)
    if not by_shape:
        return None

    n_groups = sum(len(g) for g in by_shape.values())
    n_cand = max(shape[0] for shape in by_shape)
    n_atoms = max(shape[1] for shape in by_shape)
    destinations = np.zeros((n_groups, n_atoms), dtype=np.int64)
    sources = np.zeros((n_groups, n_cand, n_atoms), dtype=np.int64)
    valid_atom = np.zeros((n_groups, n_atoms), dtype=bool)
    valid_cand = np.zeros((n_groups, n_cand), dtype=bool)

    at = 0
    for (cand, atoms), groups in by_shape.items():
        stacked = np.stack(groups)  # [k, n_cand, n_atoms]
        k = stacked.shape[0]
        destinations[at : at + k, :atoms] = stacked[:, 0]
        sources[at : at + k, :cand, :atoms] = stacked
        sources[at : at + k, cand:, :atoms] = stacked[:, :1]  # ...pad with identity, masked anyway
        valid_atom[at : at + k, :atoms] = np.any(stacked != stacked[:, :1], axis=1)
        valid_cand[at : at + k, :cand] = True
        at += k
    return destinations, sources, valid_atom, valid_cand


def atom_permutation(
    x_pred: Float[torch.Tensor, "n_atoms 3"],
    x_gt: Float[torch.Tensor, "n_atoms 3"],
    gt_mask: Bool[torch.Tensor, "n_atoms"],
    automorphisms: Sequence[np.ndarray | torch.Tensor],
    *,
    atom_classes: Int[torch.Tensor, "n_atoms"] | None = None,
    frame_mask: Bool[torch.Tensor, "n_atoms"] | None = None,
    align: bool = True,
) -> Int[torch.Tensor, "n_atoms"]:
    """Choose the ground-truth permutation that best matches a prediction.

    Args:
        x_pred: Predicted coordinates.
        x_gt: Ground-truth coordinates to permute.
        gt_mask: True where the ground truth has a usable coordinate.  An atom may only be
            permuted into a position that is True here, and only from one that is.
        automorphisms: Automorphism groups; each is [n_candidates, n_group_atoms] of global
            atom indices, with row 0 the identity, so an unpermuted group wins ties.
        atom_classes: Independent of ``gt_mask``: a metric's scope rather than the data's
            validity.  Atoms may only swap within one class, never across, and class -1 marks
            an atom to leave alone entirely (e.g. outside a metric's ``subset_mask``).
        frame_mask: Atoms to superimpose on when ranking.  Must be the frame the caller will
            score in -- a metric that aligns on one selection and scores another has to pass its
            alignment selection, or the labelling is chosen in a frame nothing is reported in.
            Defaults to the atoms this call may rank.
        align: Rank in a superimposed frame.  Off ranks in the raw frame, which is what an
            unaligned loss actually minimises -- the labelling then costs the least the loss
            can be made to cost, rather than the least it would cost after superposition.

    Returns:
        Indices ``perm`` such that ``x_gt[perm]`` is the resolved ground truth, and
        ``anything_per_atom[perm]`` follows the same permutation.

    Examples:
        >>> perm = atom_permutation(x_pred, x_gt, gt_mask, automorphisms)
        >>> resolved_coords = x_gt[perm]
    """
    permutation = torch.arange(x_gt.shape[0], device=x_gt.device)  # [n_atoms]
    if frame_mask is None:  # ...default to the atoms this call may rank
        frame_mask = gt_mask if atom_classes is None else gt_mask & (atom_classes >= 0)
    frame_mask = frame_mask & gt_mask  # [n_atoms]
    if align and int(frame_mask.sum()) >= 3:  # ...Kabsch needs three points; below that leave the frame alone
        x_pred = _superimpose_on(x_pred, x_gt, frame_mask)  # [n_atoms, 3]

    packed = _pack_groups(automorphisms)
    if packed is None:  # ...nothing exchangeable
        return permutation
    dest, src, valid_atom, valid_cand = (torch.as_tensor(a, device=x_gt.device) for a in packed)

    active = gt_mask[dest] & valid_atom  # [n_groups, n_mov]
    if atom_classes is not None:
        active &= atom_classes[dest] >= 0

    # Reject candidates that change the validity mask or cross a class boundary.
    legal = torch.where(active[:, None, :], gt_mask[src], True).all(dim=-1) & valid_cand  # [n_groups, n_cand]
    if atom_classes is not None:
        same_class = atom_classes[src] == atom_classes[dest][:, None, :]
        legal &= torch.where(active[:, None, :], same_class, True).all(dim=-1)

    error = torch.square(x_pred[dest][:, None, :, :] - x_gt[src]).sum(dim=-1)  # [n_groups, n_cand, n_mov]
    error = torch.where(active[:, None, :], error, 0.0).sum(dim=-1) / active.sum(dim=-1, keepdim=True).clamp(min=1)
    error = error.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=torch.inf).masked_fill(~legal, torch.inf)

    chosen = src[torch.arange(src.shape[0], device=src.device), error.argmin(dim=-1)]  # [n_groups, n_mov]
    # Groups partition the atoms (`build_automorphism_units` splits by argsort), so writing them all
    # at once matches applying them in sequence -- no group can read another's write.
    permutation[dest[active]] = permutation[chosen[active]]
    return permutation  # [n_atoms]


def _observed_mask(stack: AtomArrayStack) -> np.ndarray:
    """True where the ground truth actually observed an atom, shape [depth, n_atoms]."""
    observed = np.isfinite(stack.coord).all(axis=-1)  # [depth, n_atoms]
    if "occupancy" in stack.get_annotation_categories():
        occupancy = np.asarray(stack.occupancy) > 0.0
        observed &= np.broadcast_to(occupancy[None, :] if occupancy.ndim == 1 else occupancy, observed.shape)
    return observed


def permute_gt_atoms(
    predicted: AtomArray | AtomArrayStack,
    ground_truth: AtomArray | AtomArrayStack,
    *,
    atom_classes: np.ndarray | None = None,
    frame_mask: np.ndarray | None = None,
) -> AtomArrayStack:
    """Return a copy of the ground truth with its symmetric atoms permuted.

    Args:
        predicted: Predicted structure(s); determines the resolution depth.
        ground_truth: Ground-truth structure(s) to permute.
        atom_classes: Per-atom integer class labels of shape [n_atoms].
            Permutations across class boundaries or into class -1 are rejected.
        frame_mask: Atoms to superimpose on when ranking; pass the caller's alignment
            selection when it differs from ``atom_classes``.

    Returns:
        Permuted copy of the ground truth as an AtomArrayStack.

    Raises:
        ValueError: If atom counts differ, or if GT depth > 1 and does not match
            the prediction depth.
    """
    predicted_stack = ensure_atom_array_stack(predicted)
    ground_truth_stack = ensure_atom_array_stack(ground_truth)
    if predicted_stack.array_length() != ground_truth_stack.array_length():
        raise ValueError(
            f"Atom count mismatch: prediction has {predicted_stack.array_length()}, "
            f"ground truth has {ground_truth_stack.array_length()}"
        )

    predicted_depth = predicted_stack.stack_depth()
    ground_truth_depth = ground_truth_stack.stack_depth()
    if atom_classes is not None and atom_classes.shape != (ground_truth_stack.array_length(),):
        raise ValueError(
            f"atom_classes must have shape ({ground_truth_stack.array_length()},), " f"got {atom_classes.shape}"
        )
    if predicted_depth != ground_truth_depth:
        if ground_truth_depth != 1:
            raise ValueError(
                f"Cannot broadcast ground truth depth {ground_truth_depth} to prediction depth {predicted_depth}"
            )
        resolved_stack = ensure_atom_array_stack([ground_truth_stack[0].copy() for _ in range(predicted_depth)])
    else:
        resolved_stack = ground_truth_stack.copy()

    if resolved_stack.bonds is None:
        logger.warning("Skipping atom permutation: ground truth has no bond graph")
        return resolved_stack

    if atom_classes is not None and np.bincount(atom_classes[atom_classes >= 0] + 1).max(initial=0) < 2:
        return resolved_stack  # ...no class holds two atoms, so no permutation is possible
    automorphisms = find_automorphisms_with_networkx(resolved_stack[0])
    if not any(len(group) > 1 for group in automorphisms):
        return resolved_stack

    validity_mask = _observed_mask(resolved_stack)  # [depth, n_atoms]
    if not validity_mask.any():
        logger.warning("Skipping atom permutation: no ground-truth atom is observed")
        return resolved_stack
    classes = None if atom_classes is None else torch.as_tensor(atom_classes, dtype=torch.long)
    frame = None if frame_mask is None else torch.as_tensor(frame_mask, dtype=torch.bool)

    for model_index in range(predicted_depth):
        x_pred = torch.as_tensor(predicted_stack.coord[model_index], dtype=torch.float32)  # [n_atoms, 3]
        x_gt = torch.as_tensor(resolved_stack.coord[model_index], dtype=torch.float32)  # [n_atoms, 3]
        gt_mask = torch.as_tensor(validity_mask[model_index], dtype=torch.bool)  # [n_atoms]
        permutation = atom_permutation(
            x_pred,
            x_gt,
            gt_mask,
            automorphisms,
            atom_classes=classes,
            frame_mask=frame,
        ).numpy()
        # ... only coordinates are permuted; every annotation stays with its original atom
        resolved_stack.coord[model_index] = resolved_stack.coord[model_index][permutation]

    return resolved_stack


def _flat_atom_index(
    token_ids: np.ndarray, slots: np.ndarray, device: torch.device
) -> tuple[Int[torch.Tensor, "n_valid"], Int[torch.Tensor, "n_valid"], Int[torch.Tensor, "n_valid"]]:
    """Token, slot and atom indices for the atoms that occupy an atom37 slot."""
    valid = slots >= 0
    return tuple(
        torch.as_tensor(index, device=device, dtype=torch.long)
        for index in (token_ids[valid], slots[valid], np.flatnonzero(valid))
    )


def _gather_flat_atom_coords(
    coors: Float[torch.Tensor, "n_tokens 37 3"],
    atom_mask: Bool[torch.Tensor, "n_tokens 37"],
    token_ids: np.ndarray,
    slots: np.ndarray,
) -> tuple[Float[torch.Tensor, "n_atoms 3"], Bool[torch.Tensor, "n_atoms"]]:
    """Project atom37 token-slot coordinates into a flat per-atom array.

    Atoms with ``slot == -1`` are left as zeros with mask False.
    """
    n_atoms = token_ids.shape[0]
    flat_coors = coors.new_zeros((n_atoms, 3))  # [n_atoms, 3]
    flat_mask = torch.zeros(n_atoms, dtype=torch.bool, device=coors.device)  # [n_atoms]
    tokens, slot, atoms = _flat_atom_index(token_ids, slots, coors.device)
    flat_coors[atoms] = coors[tokens, slot]
    flat_mask[atoms] = atom_mask[tokens, slot]
    return flat_coors, flat_mask


def _scatter_flat_atom_coords(
    coors: Float[torch.Tensor, "n_tokens 37 3"],
    flat: Float[torch.Tensor, "n_atoms 3"],
    token_ids: np.ndarray,
    slots: np.ndarray,
) -> Float[torch.Tensor, "n_tokens 37 3"]:
    """Write resolved flat atom coordinates back into the atom37 layout."""
    out = coors.clone()  # [n_tokens, 37, 3]
    tokens, slot, atoms = _flat_atom_index(token_ids, slots, coors.device)
    out[tokens, slot] = flat[atoms]
    return out


def permute_gt_atom37(
    coors_pred: Float[torch.Tensor, "*batch n_tokens 37 3"],
    coors_gt: Float[torch.Tensor, "*batch n_tokens 37 3"],
    atom_mask: Bool[torch.Tensor, "*batch n_tokens 37"],
    automorphisms: Sequence | None,
    layout: dict[str, Sequence] | None,
    *,
    align: bool = True,
) -> Float[torch.Tensor, "*batch n_tokens 37 3"]:
    """Permute ground-truth atom37 coordinates onto the prediction, in nanometres.

    A leading batch dimension permutes each sample independently.

    Args:
        coors_pred: Predicted coordinates, [n_tokens, 37, 3] or [batch, n_tokens, 37, 3].
        coors_gt: Ground-truth coordinates to permute; same shape as coors_pred.
        atom_mask: True where an atom37 slot holds a real atom.
        automorphisms: Automorphism groups of global atom indices -- one list for a single
            structure, or one list per sample for a batch.  ``None`` leaves coors_gt unchanged.
        layout: ``{"token_ids", "slots"}`` mapping atoms onto atom37 slots, per sample for a
            batch.
        align: Rank in a superimposed frame; off ranks in the raw frame, minimising the
            unaligned reconstruction loss exactly.

    Returns:
        Permuted ground truth, same shape as coors_gt.

    Raises:
        ValueError: If a sample carries no automorphisms, or they do not cover the batch.
    """
    if automorphisms is None:
        return coors_gt
    batched = coors_pred.ndim == 4
    per_sample = list(automorphisms) if batched else [automorphisms]
    # ... a hole here is the actionable error; collation holes the layout in the same case, so
    # this must be checked before the layout is inspected at all
    if any(groups is None for groups in per_sample):
        raise ValueError(
            f"{sum(groups is None for groups in per_sample)} of {len(per_sample)} samples carry no "
            "automorphisms, so they would be scored against unpermuted ground truth. Every "
            "dataset in the run must set transform.precompute_atom_permutation=true."
        )
    if not isinstance(layout, dict) or not {"token_ids", "slots"} <= set(layout):
        raise ValueError("Automorphisms arrived without a complete atom37_layout to apply them with.")
    token_ids = layout["token_ids"] if batched else [layout["token_ids"]]
    slots = layout["slots"] if batched else [layout["slots"]]

    pred, gt, mask = (tensor if batched else tensor[None] for tensor in (coors_pred, coors_gt, atom_mask))
    if len(per_sample) != pred.shape[0]:
        raise ValueError(
            f"Automorphisms cover {len(per_sample)} samples but the batch has {int(pred.shape[0])}; "
            "a silent mismatch would score part of the batch against unpermuted ground truth."
        )

    resolved = gt.clone()  # [batch, n_tokens, 37, 3]
    for index, groups in enumerate(per_sample):
        if not any(len(group) > 1 for group in groups):
            continue
        tokens, slot = token_ids[index], slots[index]
        x_pred, _ = _gather_flat_atom_coords(pred[index], mask[index], tokens, slot)  # [n_atoms, 3]
        x_gt, gt_mask = _gather_flat_atom_coords(gt[index], mask[index], tokens, slot)  # [n_atoms, 3]
        permutation = atom_permutation(x_pred, x_gt, gt_mask, groups, align=align)  # [n_atoms]
        resolved[index] = _scatter_flat_atom_coords(gt[index], x_gt[permutation], tokens, slot)
    return resolved if batched else resolved[0]
