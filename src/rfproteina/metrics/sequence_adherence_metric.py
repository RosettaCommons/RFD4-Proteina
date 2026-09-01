from typing import Any

import numpy as np
from atomworks.constants import MASKED
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.conditions import C_SEQ
from biotite.structure import AtomArray

from rfproteina.metrics.metric import Metric


class SequenceAdherenceMetric(Metric):
    """Fraction of sequence-conditioned residues whose generated identity matches the conditioning.

    The conditioning target is read from ``C_SEQ.annotation`` on the generated array (interpreted as
    "what the sequence was conditioned to be") and compared to the generated residue identity in
    ``res_name`` (interpreted as "what the model actually generated").
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"predicted_atom_array": ("predicted_atom_array",)}

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset()

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Return adherence and the violation count used by the pre-oracle filter."""
        starts = get_residue_starts(predicted_atom_array)
        is_conditioned = C_SEQ.mask(predicted_atom_array, default="generate")[starts]
        if not is_conditioned.any():
            # Would be misleading to give a numerical value if no conditioning was provided.
            return {"sequence_adherence": float(np.nan), "n_violations": 0.0}

        target = C_SEQ.annotation(predicted_atom_array, default="generate")[starts][is_conditioned]
        assert (
            target != MASKED
        ).all(), "C_SEQ is conditioned but its annotation is unpopulated; the freeze must run before scoring."
        generated = predicted_atom_array.res_name[starts][is_conditioned]

        n_violations = int((target != generated).sum())
        return {
            "sequence_adherence": 1.0 - n_violations / len(target),
            "n_violations": float(n_violations),
        }
