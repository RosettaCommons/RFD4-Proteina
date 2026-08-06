from typing import Any

import numpy as np
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.conditions import C_SEQ
from biotite.structure import AtomArray

from rfproteina.metrics.metric import Metric


class SequenceAdherenceMetric(Metric):
    """Fraction of sequence-conditioned residues whose generated identity matches the conditioning.

    The conditioned identity is only recoverable from the *input* array: the generated array's
    ``res_name`` is overwritten with the model's prediction, so ``C_SEQ.annotation`` read off it
    would trivially agree with itself. The generated array is built residue-by-residue from the
    (already expanded) input array, so the two correspond positionally.
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array": ("predicted_atom_array",),
            "ground_truth_atom_array": ("ground_truth_atom_array",),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset(["ground_truth_atom_array"])

    def compute(
        self,
        predicted_atom_array: AtomArray,
        ground_truth_atom_array: AtomArray | None = None,
    ) -> dict[str, float]:
        """Return adherence and the violation count used by the pre-oracle filter."""
        # The offline re-scoring path (`evaluate.py`) has no input array to read the conditioning off
        if ground_truth_atom_array is None:
            return {"sequence_adherence": float(np.nan), "n_violations": 0.0}

        gt_starts = get_residue_starts(ground_truth_atom_array)
        is_conditioned = C_SEQ.mask(ground_truth_atom_array, default="generate")[gt_starts]
        if not is_conditioned.any():
            # Would be misleading to give a numerical value if no conditioning was provided
            return {"sequence_adherence": float(np.nan), "n_violations": 0.0}

        predicted_starts = get_residue_starts(predicted_atom_array)
        assert len(predicted_starts) == len(gt_starts), (
            f"Generated ({len(predicted_starts)}) and input ({len(gt_starts)}) residue counts differ; "
            f"sequence adherence relies on the two arrays corresponding residue-by-residue."
        )
        expected_res_names = ground_truth_atom_array.res_name[gt_starts][is_conditioned]
        generated_res_names = predicted_atom_array.res_name[predicted_starts][is_conditioned]

        n_violations = int((expected_res_names != generated_res_names).sum())
        return {
            "sequence_adherence": 1.0 - n_violations / len(expected_res_names),
            "n_violations": float(n_violations),
        }
