import numpy as np
from beartype.typing import Any
from biotite.structure import AtomArray, AtomArrayStack

from foundry.metrics.metric import Metric
from foundry.utils.standard_annotations import SA_CWI
from proteinfoundation.metrics.metrics_utils import unwrap_atom_array
from proteinfoundation.transforms.unindexing import (
    JOIN_ATOM_ANNOT_NAME,
    UNINDEXED_INSERTION_ANNOT_NAME,
)


class UnindexedInsertionMetric(Metric):
    """Insertion-RMSD family of metrics for unindexed-region designs.

    Reads the annotations placed by
    :py:func:`~proteinfoundation.transforms.unindexing.annotate_unindexed_insertion`:

    Returns ``insertion_rmsd``, ``insertion_mae``, ``insertion_rmcd``, and
    (when at least one tip-atom token is present) ``join_point_rmsd``.
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"predicted_atom_array": ("predicted_atom_array",)}

    def compute(  # type: ignore[override]  # ty: ignore[invalid-method-override]
        self,
        predicted_atom_array: AtomArray | AtomArrayStack | list,
    ) -> dict[str, Any]:
        # Prepare the AtomArray for metric computation
        arr = predicted_atom_array
        if isinstance(arr, list):
            assert len(arr) == 1, "UnindexedInsertionMetric expects a single design at a time"
            arr = arr[0]
        if isinstance(arr, AtomArrayStack):
            arr = unwrap_atom_array(arr)

        if UNINDEXED_INSERTION_ANNOT_NAME not in arr.get_annotation_categories():
            return {}

        mask = arr.get_annotation(UNINDEXED_INSERTION_ANNOT_NAME)
        if not mask.any():
            return {}

        # Compute per-atom differences for the insertion region
        target = SA_CWI.annotation(arr)[mask]
        actual = arr.coord[mask]
        diff = actual - target  # (n_remapped_atoms, 3)

        sq_dist = (diff**2).sum(-1)
        abs_diff = np.abs(diff)
        l1_per_atom = abs_diff.sum(-1)
        cubed_per_atom = (abs_diff**3).sum(-1)

        # Per-token aggregation by token_id
        _, inverse = np.unique(arr.token_id[mask], return_inverse=True)
        atoms_per_token = np.bincount(inverse)
        per_token_rmsd = np.sqrt(np.bincount(inverse, weights=sq_dist) / atoms_per_token)
        per_token_mae = np.bincount(inverse, weights=l1_per_atom) / atoms_per_token
        per_token_rmcd = np.cbrt(np.bincount(inverse, weights=cubed_per_atom) / atoms_per_token)

        out: dict[str, float] = {
            "insertion_rmsd": float(np.mean(per_token_rmsd)),
            "insertion_mae": float(np.mean(per_token_mae)),
            "insertion_rmcd": float(np.mean(per_token_rmcd)),
        }

        if JOIN_ATOM_ANNOT_NAME in arr.get_annotation_categories():
            join = mask & arr.get_annotation(JOIN_ATOM_ANNOT_NAME)
            if join.any():
                jdiff = arr.coord[join] - SA_CWI.annotation(arr)[join]
                out["join_point_rmsd"] = float(np.linalg.norm(jdiff, axis=-1).mean())

        return out
