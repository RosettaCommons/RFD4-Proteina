from typing import Any

import numpy as np
from atomworks.io.utils.selection import get_residue_starts
from biotite.structure import AtomArray

from proteinfoundation.conditions import C_NCYS
from proteinfoundation.metrics.metric import Metric


class NoCysteineAdherenceMetric(Metric):
    """Fraction of not-cysteine-conditioned residues that are not CYS in output."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"atom_array": ("predicted_atom_array",)}

    def compute(self, atom_array: AtomArray) -> dict[str, float]:
        not_cys_conditioning = C_NCYS.annotation(atom_array, default="generate")  # [n_atoms] bool

        residue_starts = get_residue_starts(atom_array)
        has_not_cys_conditioning = not_cys_conditioning[residue_starts]
        res_names = atom_array.res_name[residue_starts]

        if has_not_cys_conditioning.any():
            adherence = (res_names[has_not_cys_conditioning] != "CYS").mean()
            return {"not_cysteine_adherence": float(adherence)}
        else:
            return {"not_cysteine_adherence": float(np.nan)}
