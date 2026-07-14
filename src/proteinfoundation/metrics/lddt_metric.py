"""All-atom LDDT metric using biotite."""

import logging
from typing import Any, TypeAlias

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, AtomArrayPlusStack, as_atom_array_plus_stack
from biotite.structure import AtomArray, AtomArrayStack, lddt

from proteinfoundation.metrics.metric import Metric

logger = logging.getLogger(__name__)

AtomArrayInput: TypeAlias = (
    AtomArray | AtomArrayStack | AtomArrayPlus | AtomArrayPlusStack | list[AtomArray | AtomArrayPlus]
)


def _get_occupancy_mask(atom_array: AtomArray) -> np.ndarray | None:
    """Get occupancy mask if available (occupancy > 0 means atom is present)."""
    if "occupancy" in atom_array.get_annotation_categories():
        return atom_array.occupancy > 0
    return None


def _to_plain_atom_array(arr: AtomArray | AtomArrayPlus) -> AtomArray:
    """Convert AtomArrayPlus to plain AtomArray for compatibility with some Biotite functions."""
    if isinstance(arr, AtomArrayPlus):
        return arr.as_atom_array()
    return arr


class AllAtomLDDT(Metric):
    """Computes all-atom LDDT scores using Biotite."""

    def __init__(
        self,
        predicted_atom_array_key: str = "predicted_atom_array",
        ground_truth_atom_array_key: str = "ground_truth_atom_array",
        inclusion_radius: float = 15.0,
        **kwargs,
    ):
        """Initialize the LDDT metric.

        Args:
            predicted_atom_array_key: Key for predicted AtomArray(s) in kwargs.
            ground_truth_atom_array_key: Key for ground truth AtomArray(s) in kwargs.
            inclusion_radius: Distance cutoff in Angstroms. Defaults to 15.0.
        """
        # Set before super().__init__ because it accesses kwargs_to_compute_args
        self.predicted_atom_array_key = predicted_atom_array_key
        self.ground_truth_atom_array_key = ground_truth_atom_array_key
        self.inclusion_radius = inclusion_radius
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted": self.predicted_atom_array_key,
            "ground_truth": self.ground_truth_atom_array_key,
        }

    def compute(
        self,
        predicted: AtomArrayInput,
        ground_truth: AtomArrayInput,
    ) -> dict[str, float]:
        """Compute all-atom LDDT using Biotite.

        Args:
            predicted: Predicted structure(s).
            ground_truth: Ground truth structure(s).

        Returns:
            Dictionary with 'all_atom_lddt' score.
        """
        # Convert to stacks for uniform handling
        pred_stack = as_atom_array_plus_stack(predicted)
        gt_stack = as_atom_array_plus_stack(ground_truth)

        # Compute LDDT for each pair
        scores = []
        for i in range(len(pred_stack)):
            # Convert to plain AtomArray for compatibility with Biotite lddt function
            pred = _to_plain_atom_array(pred_stack[i])
            gt = _to_plain_atom_array(gt_stack[i])
            # lddt requires matching atom counts
            if len(pred) != len(gt):
                logger.warning(f"Atom count mismatch: predicted={len(pred)}, ground_truth={len(gt)}. Returning NaN.")
                scores.append(float("nan"))
                continue
            # Use occupancy mask if available
            atom_mask = _get_occupancy_mask(gt)
            # ... and score
            score = lddt(gt, pred, atom_mask=atom_mask, inclusion_radius=self.inclusion_radius)
            scores.append(float(score))

        return {"all_atom_lddt": scores[0] if len(scores) == 1 else sum(scores) / len(scores)}
