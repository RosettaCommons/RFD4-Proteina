"""Stereochemistry accuracy for tetrahedral centers, coordination centers and E/Z bonds."""

import logging

import numpy as np
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from beartype.typing import Any
from biotite.structure import AtomArray, AtomArrayStack

from rfproteina.metrics.metric import Metric
from rfproteina.utils.chirals import (
    classify_double_bond_stereoisomer,
    classify_stereoisomer,
    identify_stereogenic_centers,
    identify_stereogenic_double_bonds,
    near_tied_templates,
)
from rfproteina.utils.mask_spec import resolve_mask

logger = logging.getLogger(__name__)


def _calc_chiral_metrics(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    chiral_centers: list[dict],
) -> dict:
    """Per-sample fraction of chiral centers whose predicted stereoisomer matches GT.

    Pred is correct iff its single best-match template is among the GT's set of
    near-tied templates — handles near-degenerate geometries (e.g. axial/
    equatorial-ambiguous TBP) without rejecting them.
    """
    scored: list[tuple[dict, set[int]]] = []
    for cc in chiral_centers:
        cidx, bonded, geom = cc["center_idx"], cc["bonded_idxs"], cc["geometry"]
        gt_set = near_tied_templates(gt_coords[bonded] - gt_coords[cidx], geom)
        if gt_set is not None:
            scored.append((cc, gt_set))

    if not scored:
        return {}

    B = pred_coords.shape[0]
    correct = np.zeros((B, len(scored)), dtype=bool)
    for i, (cc, gt_set) in enumerate(scored):
        cidx, bonded, geom = cc["center_idx"], cc["bonded_idxs"], cc["geometry"]
        for b in range(B):
            pred_idx = classify_stereoisomer(pred_coords[b, bonded] - pred_coords[b, cidx], geom)
            correct[b, i] = pred_idx in gt_set

    return {"percent_correct_chirality": correct.mean(axis=-1)}


def compute_chiral_metrics(
    predicted_atom_array_stack: AtomArrayStack | AtomArray | list[AtomArray],
    ground_truth_atom_array_stack: AtomArrayStack | AtomArray | list[AtomArray],
    subset_mask: str | None = None,
) -> dict:
    """Compute center and E/Z accuracy with matching atom order after selection.

    ``subset_mask`` queries each structure independently; select complete bonded neighborhoods.
    """
    predicted_atom_array_stack = ensure_atom_array_stack(predicted_atom_array_stack)
    ground_truth_atom_array_stack = ensure_atom_array_stack(ground_truth_atom_array_stack)

    if subset_mask is not None:
        predicted_atom_array_stack = predicted_atom_array_stack[
            :, resolve_mask(predicted_atom_array_stack, subset_mask)
        ]
        ground_truth_atom_array_stack = ground_truth_atom_array_stack[
            :, resolve_mask(ground_truth_atom_array_stack, subset_mask)
        ]

    # Extract first model
    gt = ground_truth_atom_array_stack[0]

    pred_coords = np.asarray(predicted_atom_array_stack.coord)
    gt_coords = np.asarray(gt.coord)

    if pred_coords.shape[1] != len(gt):
        raise ValueError(
            f"ChiralAccuracy: predicted atom array has {pred_coords.shape[1]} atoms but "
            f"ground-truth atom array has {len(gt)} atoms. Cannot match atoms to compute chirality."
        )

    if len(gt) == 0:
        return {}
    chiral_centers = identify_stereogenic_centers(gt)
    double_bonds = identify_stereogenic_double_bonds(gt)
    if not chiral_centers and not double_bonds:
        return {}

    splits = {
        "tetrahedral": [cc for cc in chiral_centers if cc["geometry"] == "tetrahedral"],
        "non_tetrahedral": [cc for cc in chiral_centers if cc["geometry"] != "tetrahedral"],
    }

    metrics: dict[str, float] = {}
    for prefix, centers in splits.items():
        result = _calc_chiral_metrics(pred_coords, gt_coords, centers)
        if result:
            metrics[f"{prefix}_percent_correct_chirality"] = float(result["percent_correct_chirality"].mean())
    double_bond_correct = []
    for bond in double_bonds:
        indices = list(bond["reference_idxs"])
        reference = classify_double_bond_stereoisomer(gt_coords[indices])
        if reference is not None:
            double_bond_correct.append(
                [classify_double_bond_stereoisomer(coords[indices]) == reference for coords in pred_coords]
            )
    if double_bond_correct:
        metrics["double_bond_percent_correct_stereochemistry"] = float(np.mean(double_bond_correct))
    return metrics


class ChiralAccuracy(Metric):
    """Accuracy of tetrahedral centers, coordination centers and E/Z double bonds."""

    def __init__(
        self,
        input_kwarg_names: dict[str, str] | None = None,
        subset_mask: str | None = None,
        **kwargs: Any,
    ):
        self.subset_mask = subset_mask
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
    ) -> dict[str, float]:
        return compute_chiral_metrics(
            predicted_atom_array_stack, ground_truth_atom_array_stack, subset_mask=self.subset_mask
        )
