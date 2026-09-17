from typing import Any

import numpy as np
from atomworks.io.utils.selection import get_residue_starts
from biotite.structure import AtomArray

from rfproteina.conditions import C_NCYS
from rfproteina.metrics.metric import Metric


class NoCysteineAdherenceMetric(Metric):
    """Fraction of not-cysteine-conditioned residues that are not CYS in output."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"atom_array": ("predicted_atom_array",)}

    def compute(self, atom_array: AtomArray) -> dict[str, float]:
        """Return residue adherence and the violation count used by the pre-oracle filter."""
        residue_starts = get_residue_starts(atom_array)
        is_conditioned = C_NCYS.mask(atom_array, default="generate")[residue_starts]
        if not is_conditioned.any():
            return {"not_cysteine_adherence": float(np.nan), "n_violations": 0.0}

        generated = atom_array.res_name[residue_starts][is_conditioned]
        n_violations = int((generated == "CYS").sum())
        return {
            "not_cysteine_adherence": 1.0 - n_violations / len(generated),
            "n_violations": float(n_violations),
        }
