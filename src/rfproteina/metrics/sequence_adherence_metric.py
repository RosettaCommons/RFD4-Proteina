from typing import Any

import numpy as np
from atomworks.constants import MASKED
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.conditions import C_SEQ
from biotite.structure import AtomArray

from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS
from rfproteina.metrics.metric import Metric


def sequence_probability_values_masks_and_outcomes(
    predicted_atom_array: AtomArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sequence-probability values, masks, and generated-identity outcomes."""
    residue_starts = get_residue_starts(predicted_atom_array)

    values = np.stack(
        [
            condition.annotation(predicted_atom_array, default="generate")[residue_starts]
            for condition in SEQUENCE_PROBABILITY_CONDITIONS
        ],
        axis=-1,
    )
    masks = np.stack(
        [
            condition.mask(predicted_atom_array, default="generate")[residue_starts]
            for condition in SEQUENCE_PROBABILITY_CONDITIONS
        ],
        axis=-1,
    )
    generated = predicted_atom_array.res_name[residue_starts]
    token_names = np.asarray([condition.token_name for condition in SEQUENCE_PROBABILITY_CONDITIONS])
    outcomes = generated[:, None] == token_names[None, :]
    return values, masks, outcomes


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


class SequenceProbabilityAdherenceMetric(Metric):
    """Check exact zero and one sequence-probability constraints inline."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array": ("predicted_atom_array",),
        }

    def compute(
        self,
        predicted_atom_array: AtomArray,
    ) -> dict[str, float]:
        """Return exact adherence; soft probability errors are pooled from saved CIFs post-hoc."""
        values, masks, outcomes = sequence_probability_values_masks_and_outcomes(predicted_atom_array)

        zero_mask = masks & (values == 0.0)
        one_mask = masks & (values == 1.0)
        n_zero = int(zero_mask.sum())
        n_one = int(one_mask.sum())
        n_zero_violations = int(outcomes[zero_mask].sum())
        n_one_violations = int((~outcomes[one_mask]).sum())

        return {
            "zero_adherence": 1.0 - n_zero_violations / n_zero if n_zero else float(np.nan),
            "one_adherence": 1.0 - n_one_violations / n_one if n_one else float(np.nan),
            "n_zero_conditions": float(n_zero),
            "n_zero_violations": float(n_zero_violations),
            "n_one_conditions": float(n_one),
            "n_one_violations": float(n_one_violations),
        }
