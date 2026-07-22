from collections import defaultdict
from typing import Any

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

from rfproteina.conditions import C_CENTROID, C_CRD
from rfproteina.metrics.metric import Metric
from rfproteina.metrics.utils.metrics_utils import normalize_to_atom_array_list


def get_centroid_conditioning_mae(atom_array: AtomArray) -> dict[str, float]:
    """Compute centroid-conditioning adherence for a generated structure.

    For each generated (non-coordinate-conditioned) chain with centroid conditioning,
    compares its actual centroid in the predicted structure to the conditioning target.

    Args:
      atom_array: Predicted structure with condition annotations.

    Returns:
      Dict with key ``centroid_conditioning_mae``: mean distance in Angstrom
      between each conditioned generated chain's actual centroid and its target.
      NaN if no generated chain has centroid conditioning.
    """
    categories = atom_array.get_annotation_categories()
    chain_key = "chain_iid" if "chain_iid" in categories else "chain_id"
    chain_identifier = atom_array.get_annotation(chain_key)

    centroid_target = C_CENTROID.annotation(atom_array, default="generate")
    centroid_mask = C_CENTROID.mask(atom_array, default="generate")
    coordinate_conditioned_mask = C_CRD.mask(atom_array, default="generate")

    # Adherence only applies to generated chains; fully-fixed chains
    # always match their own conditioning trivially and would be misleading to include.
    generated_centroid_mask = centroid_mask & ~coordinate_conditioned_mask

    distances = []
    for chain_id in np.unique(chain_identifier[generated_centroid_mask]):
        chain_mask = chain_identifier == chain_id
        target = centroid_target[chain_mask][0]
        actual_centroid = np.nanmean(atom_array.coord[chain_mask], axis=0)
        distances.append(np.linalg.norm(actual_centroid - target))

    if not distances:
        # Would be misleading to give a numerical value if no conditioning was provided
        return {"centroid_conditioning_mae": np.nan}

    return {"centroid_conditioning_mae": float(np.mean(distances))}


class CentroidConditioningAdherenceMetric(Metric):
    """Compute adherence to chain-centroid (``C_CENTROID``) conditioning."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "atom_array_or_stack": ("predicted_atom_array",),
        }

    def compute(self, atom_array_or_stack: AtomArray | AtomArrayStack | list[AtomArray]) -> dict[str, float]:
        out_dict = defaultdict(list)

        atom_array_list = normalize_to_atom_array_list(atom_array_or_stack)

        for atom_array in atom_array_list:
            centroid_metrics = get_centroid_conditioning_mae(atom_array)
            for k, v in centroid_metrics.items():
                out_dict[k].append(v)

        return {k: float(np.nanmean(v)) for k, v in out_dict.items()}
