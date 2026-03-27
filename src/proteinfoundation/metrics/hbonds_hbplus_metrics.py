import logging

import numpy as np
from atomworks.ml.conditions import C_CRD, C_DIS, C_SEQ
from biotite.structure import AtomArray, AtomArrayStack

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger
from proteinfoundation.conditions import C_HB, C_INTCRD, HBondEnum
from proteinfoundation.metrics.metrics_utils import unwrap_atom_array
from proteinfoundation.transforms.design_task_modifiers import (
    calculate_hbonds,
)

logging.basicConfig(level=logging.INFO)
global_logger = RankedLogger(__name__, rank_zero_only=False)


def calculate_hbond_stats(
    output_atom_array: AtomArray,
    cutoff_HA_dist: float = 2.2,
    cutoff_DA_distance: float = 3.2,
) -> tuple[float, float, float] | None:
    """Compute hbond adherence stats for a single structure.

    Returns ``None`` if there are no hbond-conditioned atoms or if hbond
    calculation fails.
    """
    hbond_condition = C_HB.annotation(output_atom_array, default="generate")
    active_donor_mask = hbond_condition == HBondEnum.DONOR.value
    active_acceptor_mask = hbond_condition == HBondEnum.ACCEPTOR.value

    # Skip samples with no donors or acceptors
    if np.sum(active_donor_mask) == 0 and np.sum(active_acceptor_mask) == 0:
        return None

    seq_mask = C_SEQ.mask(output_atom_array, default="generate")
    crd_mask = C_CRD.mask(output_atom_array, default="generate")
    intcrd_mask = C_INTCRD.mask(output_atom_array, default="generate")
    dis_mask = C_DIS.mask(output_atom_array, default="generate").as_dense_array(False).any(axis=0)
    motif_mask = seq_mask | crd_mask | intcrd_mask | dis_mask

    (
        output_atom_array,
        _hbonds,
        motif_diffused_hbond_count,
        hbond_calculation_failed,
    ) = calculate_hbonds(
        output_atom_array,
        cutoff_HA_dist=cutoff_HA_dist,
        cutoff_DA_distance=cutoff_DA_distance,
        raise_on_failure=False,
        motif_mask=motif_mask,
    )

    if hbond_calculation_failed:
        return None

    gt_hbond_condition = output_atom_array.hbond_gt
    gt_donor_mask = gt_hbond_condition == HBondEnum.DONOR.value
    gt_acceptor_mask = gt_hbond_condition == HBondEnum.ACCEPTOR.value

    given_donors = np.sum(gt_donor_mask)
    given_acceptors = np.sum(gt_acceptor_mask)
    correct_donor_pct = np.sum(active_donor_mask & gt_donor_mask) / given_donors if given_donors > 0 else 1.0
    correct_acceptor_pct = (
        np.sum(active_acceptor_mask & gt_acceptor_mask) / given_acceptors if given_acceptors > 0 else 1.0
    )

    return correct_donor_pct, correct_acceptor_pct, float(motif_diffused_hbond_count)


class HbondMetrics(Metric):
    def __init__(
        self,
        cutoff_HA_dist: float = 2.2,
        cutoff_DA_distance: float = 3.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff_HA_dist = cutoff_HA_dist
        self.cutoff_DA_distance = cutoff_DA_distance

    @property
    def kwargs_to_compute_args(self) -> dict:
        return {
            "atom_array": ("predicted_atom_array",),
        }

    def compute(self, *, atom_array: AtomArray | AtomArrayStack) -> dict[str, float]:
        try:
            atom_array = unwrap_atom_array(atom_array)
            result = calculate_hbond_stats(
                output_atom_array=atom_array,
                cutoff_HA_dist=self.cutoff_HA_dist,
                cutoff_DA_distance=self.cutoff_DA_distance,
            )
            if result is None:
                return {}
            d_pct, a_pct, n_hbonds = result
        except Exception as e:
            global_logger.error(f"Error calculating hydrogen bond metrics: {e} | Skipping")
            return {}

        return {
            "mean_correct_donors_percent": float(d_pct),
            "mean_correct_acceptors_percent": float(a_pct),
            "mean_num_hbonds": float(n_hbonds),
        }
