"""Chiral accuracy metric — tetrahedral centers, split by organic/metal."""

import logging

import numpy as np
import torch
from atomworks.constants import METAL_ELEMENTS
from atomworks.enums import ChiralType
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.io.utils.annotator import ensure_annotations
from beartype.typing import Any
from biotite.structure import AtomArray, AtomArrayStack
from jaxtyping import Bool, Float

from foundry.metrics.metric import Metric

logger = logging.getLogger(__name__)


def identify_tetrahedral_chiral_centers(atom_array: AtomArray) -> list[dict]:
    """Identify tetrahedral chiral centers from the ``chiral_type`` annotation."""
    ensure_annotations(atom_array, "chiral_type")
    results = []
    for idx in np.where(atom_array.chiral_type != 0)[0]:
        ct = ChiralType(int(atom_array.chiral_type[idx]))
        if not ct.is_tetrahedral():
            continue

        bonded, _ = atom_array.bonds.get_bonds(int(idx))
        bonded_sorted = sorted(int(b) for b in bonded)
        if len(bonded_sorted) < 3 or len(bonded_sorted) > 4:
            continue

        results.append(
            {
                "center_idx": int(idx),
                "bonded_idxs": bonded_sorted,
                "chiral_type": ct,
                "is_metal": atom_array.element[idx].strip().upper() in METAL_ELEMENTS,
            }
        )

    return results


def _normalized_triple_product(
    coords: Float[torch.Tensor, "... N 3"],
    cidx: int,
    bidxs: list[int],
) -> Float[torch.Tensor, "..."]:
    """Triple product of three bond vectors, normalized by the product of their lengths."""
    c = coords[..., cidx, :]
    a = coords[..., bidxs[0], :] - c
    b = coords[..., bidxs[1], :] - c
    d = coords[..., bidxs[2], :] - c
    tp = (a * torch.cross(b, d, dim=-1)).sum(dim=-1)
    denom = (a.norm(dim=-1) * b.norm(dim=-1) * d.norm(dim=-1)).clamp(min=1e-8)
    return tp / denom


def _calc_chiral_metrics(
    pred_coords: Float[torch.Tensor, "B N 3"],
    gt_coords: Float[torch.Tensor, "N 3"],
    chiral_centers: list[dict],
    mask: Bool[torch.Tensor, "N"],
) -> dict:
    """Calculate chiral accuracy and similarity via normalized triple product."""
    valid = [
        cc for cc in chiral_centers if all(mask[i] for i in [cc["center_idx"]] + cc["bonded_idxs"])
    ]
    if not valid:
        return {}

    B = pred_coords.shape[0]
    correct = torch.zeros(B, len(valid), dtype=torch.bool, device=pred_coords.device)
    similarity = torch.zeros(B, len(valid), device=pred_coords.device)

    for i, cc in enumerate(valid):
        cidx, bidxs = cc["center_idx"], cc["bonded_idxs"]
        gt_tp_n = _normalized_triple_product(gt_coords, cidx, bidxs)
        pred_tp_n = _normalized_triple_product(pred_coords, cidx, bidxs)

        correct[:, i] = torch.sign(pred_tp_n) == torch.sign(gt_tp_n)

        # Avoid division by zero when gt is degenerate (e.g., coplanar ligands).
        gt_safe = gt_tp_n.masked_fill(gt_tp_n == 0, 1e-8)
        similarity[:, i] = (pred_tp_n / gt_safe).clamp(-1, 1)

    return {
        "percent_correct_chirality": correct.float().mean(dim=-1),
        "chiral_similarity": similarity.mean(dim=-1),
    }


def compute_chiral_metrics(
    predicted_atom_array_stack: AtomArrayStack | AtomArray | list[AtomArray],
    ground_truth_atom_array_stack: AtomArrayStack | AtomArray | list[AtomArray],
) -> dict:
    """Compute tetrahedral chiral metrics split by organic/metal."""
    predicted_atom_array_stack = ensure_atom_array_stack(predicted_atom_array_stack)
    ground_truth_atom_array_stack = ensure_atom_array_stack(ground_truth_atom_array_stack)
    gt = ground_truth_atom_array_stack[0]

    chiral_centers = identify_tetrahedral_chiral_centers(gt)
    if not chiral_centers:
        return {}

    X_pred = torch.from_numpy(predicted_atom_array_stack.coord)
    X_gt = torch.from_numpy(gt.coord)
    valid_mask = ~torch.isnan(X_gt).any(dim=-1)

    organic = [cc for cc in chiral_centers if not cc["is_metal"]]
    metal = [cc for cc in chiral_centers if cc["is_metal"]]

    metrics: dict[str, float] = {}
    for prefix, centers in [("organic", organic), ("metal", metal), ("all", chiral_centers)]:
        result = _calc_chiral_metrics(X_pred, X_gt, centers, mask=valid_mask)
        if result:
            for key in ("percent_correct_chirality", "chiral_similarity"):
                metrics[f"{prefix}_{key}"] = result[key].mean().item()

    return metrics


class ChiralAccuracy(Metric):
    """Tetrahedral chiral accuracy, split by organic/metal."""

    def __init__(self, input_kwarg_names: dict[str, str] | None = None, **kwargs: Any):
        self._input_kwarg_names = input_kwarg_names or {
            "predicted_atom_array_stack": "predicted_atom_array",
            "ground_truth_atom_array_stack": "ground_truth_atom_array",
        }
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return self._input_kwarg_names

    def compute(
        self,
        predicted_atom_array_stack: AtomArrayStack | AtomArray,
        ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
    ):
        return compute_chiral_metrics(predicted_atom_array_stack, ground_truth_atom_array_stack)
