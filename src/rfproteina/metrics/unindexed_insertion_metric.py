from typing import Any

import numpy as np
from atomworks.ml.conditions import C_IDX, C_SEQ
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray

from rfproteina.metrics.metric import Metric
from rfproteina.utils.standard_annotations import SA_UII


class UnindexedInsertionMappingMetric(Metric):
    """Whether the predicted unindexed → generated token mapping was applied in full.

    Two ways the insertion head's mapping can fall short of the conditioning:

    - a token is left unmapped, either because the head produced no target for it or because
      :py:func:`~rfproteina.transforms.unindexing.annotate_unindexed_insertion` rejected the
      mapping (in which case *none* of the unindexed regions are annotated);
    - a mapped token lands on a generated residue of the wrong identity, i.e. the model did not
      reproduce the sequence conditioning attached to the unindexed region.

    Both are counted over tokens (an unmapped token that was also sequence-conditioned counts in
    both); the counts are zero when there is no unindexed conditioning.
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array": ("predicted_atom_array",),
            "full_array_with_unindexed": ("extra_info", "full_array_with_unindexed"),
            "predicted_insertion_mapping": ("extra_info", "predicted_insertion_mapping"),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset(["full_array_with_unindexed", "predicted_insertion_mapping"])

    def compute(
        self,
        predicted_atom_array: AtomArray,
        full_array_with_unindexed: AtomArray | None = None,
        predicted_insertion_mapping: dict[int, int] | None = None,
    ) -> dict[str, float]:
        """Return the violation counts used by the pre-oracle filter."""
        # No unindexed conditioning (or an offline re-scoring path, which never sees the mapping)
        if full_array_with_unindexed is None:
            return {"n_unmapped_tokens": 0.0, "n_sequence_mismatches": 0.0}

        unindexed = full_array_with_unindexed[~C_IDX.mask(full_array_with_unindexed, default="generate")]
        unindexed_starts = get_token_starts(unindexed)

        # SA_UII flags the generated atoms an unindexed token was actually transferred onto, so this
        # counts the mappings that survived annotation rather than the ones that were proposed
        is_insertion_target = SA_UII.mask(predicted_atom_array)
        n_mapped = len(np.unique(predicted_atom_array.token_id[is_insertion_target]))

        mapping = predicted_insertion_mapping or {}
        predicted_starts = get_token_starts(predicted_atom_array)
        generated_res_name = dict(
            zip(
                predicted_atom_array.token_id[predicted_starts],
                predicted_atom_array.res_name[predicted_starts],
                strict=True,
            )
        )
        is_seq_conditioned = C_SEQ.mask(unindexed, default="generate")[unindexed_starts]
        n_sequence_mismatches = sum(
            generated_res_name.get(mapping.get(int(token_id))) != res_name
            for token_id, res_name in zip(
                unindexed.token_id[unindexed_starts][is_seq_conditioned],
                unindexed.res_name[unindexed_starts][is_seq_conditioned],
                strict=True,
            )
        )

        return {
            "n_unmapped_tokens": float(len(unindexed_starts) - n_mapped),
            "n_sequence_mismatches": float(n_sequence_mismatches),
        }
