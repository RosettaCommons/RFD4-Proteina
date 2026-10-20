"""Residue-level amino-acid composition of generated structures."""

import numpy as np
from atomworks.constants import STANDARD_AA
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.conditions import C_SEQ
from biotite.structure import AtomArray

from rfproteina.metrics.metric import Metric


class AminoAcidCompositionMetric(Metric):
    """Report generated UNK count and ALA, VAL, and GLY fractions among standard amino acids.

    Count each residue once across all chains, including sequence-conditioned
    residues. Nonstandard residues, ligands, and nucleic acids do not contribute
    unless their residue name is one of the 20 standard amino-acid codes.
    Fractions are in [0, 1], or NaN when no standard amino-acid residues exist.
    The generated UNK count excludes residues sequence-conditioned to UNK in the input.
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, str]:
        return {"predicted_atom_array": "predicted_atom_array"}

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Measure composition by residue identity, independent of atom/token counts."""
        starts = get_residue_starts(predicted_atom_array)
        residue_names = predicted_atom_array.res_name[starts]
        input_unknown = C_SEQ.mask(predicted_atom_array, default="generate")[starts] & (
            C_SEQ.annotation(predicted_atom_array, default="generate")[starts] == "UNK"
        )
        standard_names = residue_names[np.isin(residue_names, STANDARD_AA)]
        n_residues = len(standard_names)
        return {
            "n_generated_unknown_residues": float(np.count_nonzero((residue_names == "UNK") & ~input_unknown)),
            "n_standard_residues": float(n_residues),
            "ala_fraction": float(np.mean(standard_names == "ALA")) if n_residues else float("nan"),
            "val_fraction": float(np.mean(standard_names == "VAL")) if n_residues else float("nan"),
            "gly_fraction": float(np.mean(standard_names == "GLY")) if n_residues else float("nan"),
        }
