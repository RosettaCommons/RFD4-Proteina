"""Chiral accuracy metric — all stereogenic geometries, split by tetrahedral vs non-tetrahedral."""

import logging

import numpy as np
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from beartype.typing import Any
from biotite.structure import AtomArray, AtomArrayStack

from rfproteina.metrics.metric import Metric
from rfproteina.utils.chirals import (
    classify_stereoisomer,
    identify_stereogenic_centers,
    near_tied_templates,
)

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
) -> dict:
    """Compute stereogenic-center accuracy split by tetrahedral vs non-tetrahedral geometry."""
    predicted_atom_array_stack = ensure_atom_array_stack(predicted_atom_array_stack)
    ground_truth_atom_array_stack = ensure_atom_array_stack(ground_truth_atom_array_stack)

    # Extract first model
    gt = ground_truth_atom_array_stack[0]

    # Identify stereocenters (including higher-order chiralities)
    chiral_centers = identify_stereogenic_centers(gt)
    if not chiral_centers:
        return {}

    pred_coords = np.asarray(predicted_atom_array_stack.coord)
    gt_coords = np.asarray(gt.coord)

    if pred_coords.shape[1] != len(gt):
        raise ValueError(
            f"ChiralAccuracy: predicted atom array has {pred_coords.shape[1]} atoms but "
            f"ground-truth atom array has {len(gt)} atoms. Cannot match atoms to compute chirality."
        )

    splits = {
        "tetrahedral": [cc for cc in chiral_centers if cc["geometry"] == "tetrahedral"],
        "non_tetrahedral": [cc for cc in chiral_centers if cc["geometry"] != "tetrahedral"],
    }

    metrics: dict[str, float] = {}
    for prefix, centers in splits.items():
        result = _calc_chiral_metrics(pred_coords, gt_coords, centers)
        if result:
            metrics[f"{prefix}_percent_correct_chirality"] = float(result["percent_correct_chirality"].mean())
    return metrics


class ChiralAccuracy(Metric):
    """Stereogenic-center accuracy, split by tetrahedral vs non-tetrahedral geometry."""

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
    ) -> dict[str, float]:
        return compute_chiral_metrics(predicted_atom_array_stack, ground_truth_atom_array_stack)
