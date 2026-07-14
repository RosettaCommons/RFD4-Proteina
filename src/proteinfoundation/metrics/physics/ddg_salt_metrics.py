"""Charge complementarity metric for protein interfaces.

**ChargeComplementarityMetric** computes a salt bridge bonus score for opposite
charges (D/E with R/K) across the binder-target interface, with distance-weighted
scoring. Binder vs. target chains are identified from AtomArray annotations.

For TMol-based binding ddG, use
:class:`~proteinfoundation.metrics.physics.tmol_block_pair_metric.TMolBlockPairMetric`
(the general region-pair metric; a soft-rep binder/target preset is provided in
``configs/metrics/inline_metrics/tmol_block_pair_metrics.yaml``).

Example:
    >>> charge_metric = ChargeComplementarityMetric(min_dist=7.0, max_dist=9.0)
    >>> results = charge_metric.compute(predicted_atom_array=structure)
    >>> # Returns: {"charge_score": -4.5, "n_salt_bridges": 3}
"""

from functools import cached_property
from typing import Any

import numpy as np
from biotite.structure import AtomArray

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger
from proteinfoundation.conditions import C_CRD

logger = RankedLogger(__name__, rank_zero_only=True)


# ============================================================================
# Helper Functions
# ============================================================================


def get_binder_target_mask(atom_array: AtomArray) -> tuple[np.ndarray, np.ndarray]:
    """Get masks for binder (generated) and target (conditioned) atoms.

    Uses C_CRD (coordinate conditioning) annotation to identify which atoms are
    conditioned (target) vs generated (binder).

    Args:
        atom_array: Input structure with conditioning annotations.

    Returns:
        Tuple of (binder_mask, target_mask) boolean arrays.
    """
    # Try to get coordinate conditioning mask
    try:
        crd_mask = C_CRD.mask(atom_array, default="generate")
        if crd_mask is not None:
            # Conditioned atoms are target, generated atoms are binder
            target_mask = crd_mask.astype(bool)
            binder_mask = ~target_mask
            return binder_mask, target_mask
    except Exception:
        pass

    # Fallback: use chain_iid - assume chain 0 is binder, rest is target
    # This is common convention in binder design
    if hasattr(atom_array, "chain_iid"):
        chain_iids = atom_array.chain_iid
        binder_mask = chain_iids == 0
        target_mask = chain_iids != 0
        return binder_mask, target_mask

    # Final fallback: use chain_id - first chain is binder
    unique_chains = np.unique(atom_array.chain_id)
    if len(unique_chains) >= 2:
        binder_chain = unique_chains[0]
        binder_mask = atom_array.chain_id == binder_chain
        target_mask = ~binder_mask
        return binder_mask, target_mask

    # Single chain - return all as binder, none as target
    binder_mask = np.ones(len(atom_array), dtype=bool)
    target_mask = np.zeros(len(atom_array), dtype=bool)
    return binder_mask, target_mask


def get_charged_residue_positions(atom_array: AtomArray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract charged residue positions and charge values.

    Extracts D/E (negative) and R/K (positive) residues and computes their
    charge center positions (mean of sidechain N/O atoms).

    Args:
        atom_array: Input structure.

    Returns:
        Tuple of (positions, charges, res_ids, chain_ids):
            - positions: [N, 3] array of charge center coordinates
            - charges: [N] array of +1 (R/K) or -1 (D/E)
            - res_ids: [N] array of residue IDs
            - chain_ids: [N] array of chain IDs
    """
    charged_residues = {"ASP", "GLU", "ARG", "LYS", "D", "E", "R", "K"}
    positive_residues = {"ARG", "LYS", "R", "K"}
    _negative_residues = {"ASP", "GLU", "D", "E"}

    # Get unique residues
    unique_residues = set(zip(atom_array.res_id, atom_array.chain_id, strict=False))

    positions = []
    charges = []
    res_ids = []
    chain_ids = []

    for res_id, chain_id in unique_residues:
        mask = (atom_array.res_id == res_id) & (atom_array.chain_id == chain_id)
        residue_atoms = atom_array[mask]

        if len(residue_atoms) == 0:
            continue

        res_name = residue_atoms.res_name[0]

        # Check if charged residue
        if res_name not in charged_residues:
            continue

        # Get sidechain heavy atoms (non-backbone N, O for charged groups)
        # For D/E: carboxyl oxygens (OD1, OD2 for D; OE1, OE2 for E)
        # For R/K: nitrogens (NH1, NH2, NE for R; NZ for K)
        sidechain_mask = np.array(
            [not (name in ["N", "CA", "C", "O", "CB"] or name.startswith("H")) for name in residue_atoms.atom_name]
        )

        # Filter for N/O atoms in sidechain
        element_mask = np.isin(residue_atoms.element, ["N", "O"])
        charge_center_mask = sidechain_mask & element_mask

        charge_center_atoms = residue_atoms[charge_center_mask]

        if len(charge_center_atoms) == 0:
            # Fallback to sidechain center
            sc_atoms = residue_atoms[sidechain_mask]
            if len(sc_atoms) == 0:
                continue
            center = np.mean(sc_atoms.coord, axis=0)
        else:
            center = np.mean(charge_center_atoms.coord, axis=0)

        positions.append(center)
        charges.append(1 if res_name in positive_residues else -1)
        res_ids.append(res_id)
        chain_ids.append(chain_id)

    if len(positions) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.array([], dtype=str),
        )

    return (
        np.array(positions, dtype=np.float32),
        np.array(charges, dtype=np.int32),
        np.array(res_ids, dtype=np.int32),
        np.array(chain_ids),
    )


# ============================================================================
# ChargeComplementarityMetric
# ============================================================================


class ChargeComplementarityMetric(Metric):
    """Salt bridge/charge complementarity metric for protein interfaces.

    Computes a charge complementarity score based on opposite-charge pairs
    (D/E with R/K) across the binder-target interface. Uses distance-weighted
    scoring where closer pairs receive a larger bonus.

    The scoring follows the polar optimizer approach:
    - Full bonus (-max_bonus) at distances <= min_dist
    - Linear/quadratic decay from min_dist to max_dist
    - Zero bonus beyond max_dist

    Args:
        min_dist: Distance for full bonus (Angstroms). Defaults to ``7.0``.
        max_dist: Distance for zero bonus (Angstroms). Defaults to ``9.0``.
        max_bonus: Maximum bonus per favorable pair. Defaults to ``2.0``.
        charge_func: Distance decay function ("linear", "flat", "quadratic").
            Defaults to ``"linear"``.

    Example:
        >>> metric = ChargeComplementarityMetric(min_dist=7.0, max_dist=9.0)
        >>> results = metric.compute(predicted_atom_array=structure)
        >>> print(f"Charge score: {results['charge_score']:.2f}")
    """

    def __init__(
        self,
        min_dist: float = 7.0,
        max_dist: float = 9.0,
        max_bonus: float = 2.0,
        charge_func: str = "linear",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_bonus = max_bonus
        self.charge_func = charge_func

        if charge_func not in ("linear", "flat", "quadratic"):
            raise ValueError(f"charge_func must be 'linear', 'flat', or 'quadratic', got '{charge_func}'")

        logger.info(
            f"ChargeComplementarityMetric initialized (min_dist={min_dist}, max_dist={max_dist}, func={charge_func})"
        )

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map kwargs to compute arguments."""
        return {
            "predicted_atom_array": "predicted_atom_array",
        }

    def _compute_distance_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute distance-based weights for charge pairs.

        Args:
            distances: Pairwise distances between charged residues.

        Returns:
            Weight matrix (same shape as distances).
        """
        if self.charge_func == "flat":
            # Full bonus within max_dist, zero outside
            weights = np.ones_like(distances) * self.max_bonus
            weights[distances > self.max_dist] = 0

        elif self.charge_func == "linear":
            # Linear decay from max_bonus at min_dist to 0 at max_dist
            weights = (self.max_dist - distances) / (self.max_dist - self.min_dist)
            weights = np.clip(weights, 0, 1) * self.max_bonus

        elif self.charge_func == "quadratic":
            # Quadratic decay (1/r^2 normalized)
            weights = 1 / np.clip(distances, self.min_dist, None) ** 2 * self.min_dist**2 * self.max_bonus
            weights[distances > self.max_dist] = 0

        return weights

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Compute charge complementarity score.

        Args:
            predicted_atom_array: Structure to evaluate.

        Returns:
            Dictionary with charge metrics:
                - charge_score: Total charge complementarity score (negative = favorable)
                - n_salt_bridges: Number of favorable opposite-charge pairs
                - n_charge_clashes: Number of same-charge pairs (unfavorable)
                - n_charged_residues_binder: Number of D/E/R/K on binder
                - n_charged_residues_target: Number of D/E/R/K on target
        """
        try:
            # Get binder/target masks
            binder_mask, target_mask = get_binder_target_mask(predicted_atom_array)

            # Get charged residue positions
            positions, charges, res_ids, chain_ids = get_charged_residue_positions(predicted_atom_array)

            if len(positions) == 0:
                return self._get_default_metrics()

            # Determine which charged residues are on binder vs target
            # We need to match res_id/chain_id back to atom masks
            is_binder_residue = np.zeros(len(positions), dtype=bool)

            for i, (res_id, chain_id) in enumerate(zip(res_ids, chain_ids, strict=False)):
                res_mask = (predicted_atom_array.res_id == res_id) & (predicted_atom_array.chain_id == chain_id)
                # Residue is on binder if any of its atoms are in binder_mask
                is_binder_residue[i] = np.any(binder_mask & res_mask)

            is_target_residue = ~is_binder_residue

            n_binder_charged = np.sum(is_binder_residue)
            n_target_charged = np.sum(is_target_residue)

            if n_binder_charged == 0 or n_target_charged == 0:
                return {
                    "charge_score": 0.0,
                    "n_salt_bridges": 0,
                    "n_charge_clashes": 0,
                    "n_charged_residues_binder": int(n_binder_charged),
                    "n_charged_residues_target": int(n_target_charged),
                }

            # Compute pairwise distances
            dist_matrix = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)

            # Compute charge products (+1 for same, -1 for opposite)
            charge_prod = charges[:, None] * charges[None, :]

            # Interface mask: binder residue with target residue
            interface_mask = is_binder_residue[:, None] != is_binder_residue[None, :]

            # Compute distance weights
            dist_weights = self._compute_distance_weights(dist_matrix)

            # Compute charge scores (negative for favorable opposite-charge pairs)
            # charge_prod = -1 for opposite charges (favorable) -> negative score
            # charge_prod = +1 for same charges (unfavorable) -> positive score
            charge_scores = dist_weights * charge_prod * interface_mask

            # Total score (divide by 2 to avoid double counting)
            total_score = float(np.sum(charge_scores) / 2)

            # Count salt bridges (opposite charge pairs within max_dist)
            salt_bridge_mask = (charge_prod == -1) & interface_mask & (dist_matrix <= self.max_dist)
            n_salt_bridges = int(np.sum(salt_bridge_mask) / 2)  # Divide by 2 for unique pairs

            # Count charge clashes (same charge pairs within max_dist)
            clash_mask = (charge_prod == +1) & interface_mask & (dist_matrix <= self.max_dist)
            n_clashes = int(np.sum(clash_mask) / 2)

            return {
                "charge_score": total_score,
                "n_salt_bridges": n_salt_bridges,
                "n_charge_clashes": n_clashes,
                "n_charged_residues_binder": int(n_binder_charged),
                "n_charged_residues_target": int(n_target_charged),
            }

        except Exception as e:
            logger.error(f"ChargeComplementarityMetric computation failed: {e}")
            return self._get_default_metrics()

    def _get_default_metrics(self) -> dict[str, float]:
        """Return default metrics when computation fails."""
        return {
            "charge_score": float("nan"),
            "n_salt_bridges": 0,
            "n_charge_clashes": 0,
            "n_charged_residues_binder": 0,
            "n_charged_residues_target": 0,
        }
