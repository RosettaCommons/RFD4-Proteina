"""DDG and Charge Complementarity metrics using TMol.

This module provides two TMol-based metrics for evaluating protein interfaces:

1. **SoftDDGMetric**: Computes binding ddG = E(complex) - E(binder) - E(target) using 
   TMol's beta2016 score function with configurable soft repulsion (fa_ljrep=0.55).

2. **ChargeComplementarityMetric**: Computes salt bridge bonus score for opposite 
   charges (D/E with R/K) across the interface with distance-weighted scoring.

Both metrics use AtomArray annotations to identify binder vs target chains.

Example:
    >>> # DDG scoring with soft repulsion
    >>> ddg_metric = SoftDDGMetric(use_soft_rep=True)
    >>> results = ddg_metric.compute(predicted_atom_array=structure)
    >>> # Returns: {"ddg": -15.0, "complex_energy": -100.0, "monomer_energy_sum": -85.0}

    >>> # Charge complementarity scoring
    >>> charge_metric = ChargeComplementarityMetric(min_dist=7.0, max_dist=9.0)
    >>> results = charge_metric.compute(predicted_atom_array=structure)
    >>> # Returns: {"charge_score": -4.5, "n_salt_bridges": 3}
"""

import gc
import os
import tempfile
from functools import cached_property
from typing import Any

import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger

from proteinfoundation.conditions import C_CRD

logger = RankedLogger(__name__, rank_zero_only=True)

# TMol path configuration
TMOL_PATH = "/path/to/shared/storage"
if not os.path.exists(TMOL_PATH):
    TMOL_PATH = "/path/to/shared/storage"


# ============================================================================
# Helper Functions
# ============================================================================


def _get_tmol_imports():
    """Lazy import of TMol to avoid dependency issues."""
    import sys
    if TMOL_PATH not in sys.path:
        sys.path.insert(0, TMOL_PATH)
    
    import tmol
    from tmol.score.score_function import ScoreFunction
    from tmol.score.score_types import ScoreType
    from tmol.io.chain_deduction import chain_inds_for_pose_stack
    
    return tmol, ScoreFunction, ScoreType, chain_inds_for_pose_stack


def atom_array_to_pdb_string(atom_array: AtomArray) -> str:
    """Convert AtomArray to PDB format string.
    
    Args:
        atom_array: Input structure.
        
    Returns:
        PDB format string.
    """
    pdb_file = PDBFile()
    pdb_file.set_structure(atom_array)
    
    # Write to string via temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        temp_path = f.name
    
    try:
        pdb_file.write(temp_path)
        with open(temp_path, 'r') as f:
            pdb_string = f.read()
    finally:
        os.unlink(temp_path)
    
    return pdb_string


def atom_array_to_tmol_pose_stack(
    atom_array: AtomArray, 
    device: torch.device
) -> Any:
    """Convert biotite AtomArray to TMol pose_stack for scoring.
    
    Args:
        atom_array: Input structure.
        device: Device for TMol computations.
        
    Returns:
        TMol PoseStack object.
    """
    tmol, _, _, _ = _get_tmol_imports()
    
    # Write to temporary PDB file for TMol loading
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        temp_path = f.name
    
    try:
        pdb_file = PDBFile()
        pdb_file.set_structure(atom_array)
        pdb_file.write(temp_path)
        
        pose_stack = tmol.pose_stack_from_pdb(temp_path, device=device)
    finally:
        os.unlink(temp_path)
    
    return pose_stack


def split_atom_array_by_chain(atom_array: AtomArray) -> dict[str, AtomArray]:
    """Split multi-chain structure into individual chain AtomArrays.
    
    Args:
        atom_array: Input multi-chain structure.
        
    Returns:
        Dictionary mapping chain_id to AtomArray for each chain.
    """
    chain_arrays = {}
    
    # Get unique chains
    unique_chains = np.unique(atom_array.chain_id)
    
    for chain_id in unique_chains:
        chain_mask = atom_array.chain_id == chain_id
        chain_arrays[chain_id] = atom_array[chain_mask].copy()
    
    return chain_arrays


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
    if hasattr(atom_array, 'chain_iid'):
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


def get_charged_residue_positions(
    atom_array: AtomArray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    charged_residues = {'ASP', 'GLU', 'ARG', 'LYS', 'D', 'E', 'R', 'K'}
    positive_residues = {'ARG', 'LYS', 'R', 'K'}
    negative_residues = {'ASP', 'GLU', 'D', 'E'}
    
    # Get unique residues
    unique_residues = set(zip(atom_array.res_id, atom_array.chain_id))
    
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
        sidechain_mask = np.array([
            not (name in ['N', 'CA', 'C', 'O', 'CB'] or name.startswith('H'))
            for name in residue_atoms.atom_name
        ])
        
        # Filter for N/O atoms in sidechain
        element_mask = np.isin(residue_atoms.element, ['N', 'O'])
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
            np.array([], dtype=str)
        )
    
    return (
        np.array(positions, dtype=np.float32),
        np.array(charges, dtype=np.int32),
        np.array(res_ids, dtype=np.int32),
        np.array(chain_ids)
    )


# ============================================================================
# SoftDDGMetric
# ============================================================================


class SoftDDGMetric(Metric):
    """TMol-based binding ddG metric with configurable soft repulsion.
    
    Computes binding energy as ddG = E(complex) - E(binder) - E(target) using
    TMol's beta2016 score function. Supports soft repulsion (fa_ljrep=0.55)
    for more lenient clash handling, which is recommended for AI-generated
    structures.
    
    Args:
        use_soft_rep: Use soft repulsion (fa_ljrep=0.55) vs full (1.0). Defaults to ``True``.
        soft_rep_weight: Weight for fa_ljrep when use_soft_rep=True. Defaults to ``0.55``.
        device: Device for TMol computations. Defaults to CUDA if available.
        
    Example:
        >>> metric = SoftDDGMetric(use_soft_rep=True)
        >>> results = metric.compute(predicted_atom_array=structure)
        >>> print(f"ddG: {results['ddg']:.2f}")
    """
    
    def __init__(
        self,
        use_soft_rep: bool = True,
        soft_rep_weight: float = 0.55,
        device: str = "cuda",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.use_soft_rep = use_soft_rep
        self.soft_rep_weight = soft_rep_weight
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Set CUDA arch for TMol kernel compilation
        os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0"
        
        logger.info(
            f"SoftDDGMetric initialized (soft_rep={use_soft_rep}, "
            f"soft_rep_weight={soft_rep_weight})"
        )
    
    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map kwargs to compute arguments."""
        return {
            "predicted_atom_array": "predicted_atom_array",
        }
    
    def _create_score_function(self, device: torch.device):
        """Create TMol score function with appropriate settings.
        
        Args:
            device: Device for computations.
            
        Returns:
            TMol ScoreFunction object.
        """
        tmol, ScoreFunction, ScoreType, _ = _get_tmol_imports()
        
        if self.use_soft_rep:
            # Create custom score function with soft repulsion
            param_db = tmol.ParameterDatabase.get_default()
            sfxn = ScoreFunction(param_db, device)
            
            # Set beta2016 weights with soft fa_ljrep
            sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
            sfxn.set_weight(ScoreType.fa_ljrep, self.soft_rep_weight)
            sfxn.set_weight(ScoreType.fa_lk, 1.0)
            sfxn.set_weight(ScoreType.fa_elec, 1.0)
            sfxn.set_weight(ScoreType.hbond, 1.0)
            sfxn.set_weight(ScoreType.lk_ball_iso, -0.38)
            sfxn.set_weight(ScoreType.lk_ball, 0.92)
            sfxn.set_weight(ScoreType.lk_bridge, -0.33)
            sfxn.set_weight(ScoreType.lk_bridge_uncpl, -0.33)
            sfxn.set_weight(ScoreType.omega, 0.48)
            sfxn.set_weight(ScoreType.rama, 0.50)
            sfxn.set_weight(ScoreType.disulfide, 1.25)
            sfxn.set_weight(ScoreType.cart_lengths, 0.5)
            sfxn.set_weight(ScoreType.cart_angles, 0.5)
            sfxn.set_weight(ScoreType.cart_torsions, 0.5)
            sfxn.set_weight(ScoreType.cart_impropers, 0.5)
            sfxn.set_weight(ScoreType.cart_hxltorsions, 0.5)
            sfxn.set_weight(ScoreType.dunbrack_rot, 0.76)
            sfxn.set_weight(ScoreType.dunbrack_rotdev, 0.69)
            sfxn.set_weight(ScoreType.dunbrack_semirot, 0.78)
            sfxn.set_weight(ScoreType.ref, 1.0)
        else:
            # Use standard beta2016 score function
            sfxn = tmol.beta2016_score_function(device)
        
        return sfxn
    
    def _score_structure(self, atom_array: AtomArray) -> float:
        """Score a structure using TMol.
        
        Args:
            atom_array: Structure to score.
            
        Returns:
            Total weighted energy.
        """
        try:
            pose_stack = atom_array_to_tmol_pose_stack(atom_array, self.device)
            sfxn = self._create_score_function(self.device)
            scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
            energy = scorer(pose_stack.coords)
            return float(energy.sum().cpu().item())
        except Exception as e:
            logger.warning(f"TMol scoring failed: {e}")
            return float('nan')
    
    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Compute binding ddG for the structure.
        
        Args:
            predicted_atom_array: Structure to evaluate.
            
        Returns:
            Dictionary with ddG metrics:
                - ddg: Binding energy (E_complex - E_binder - E_target)
                - complex_energy: Total complex energy
                - binder_energy: Binder monomer energy
                - target_energy: Target monomer energy
                - monomer_energy_sum: Sum of monomer energies
        """
        try:
            # Get binder/target masks
            binder_mask, target_mask = get_binder_target_mask(predicted_atom_array)
            
            # Score complex
            complex_energy = self._score_structure(predicted_atom_array)
            
            # Extract and score binder
            binder_array = predicted_atom_array[binder_mask].copy()
            binder_energy = self._score_structure(binder_array) if len(binder_array) > 0 else 0.0
            
            # Extract and score target
            target_array = predicted_atom_array[target_mask].copy()
            target_energy = self._score_structure(target_array) if len(target_array) > 0 else 0.0
            
            # Compute ddG
            monomer_energy_sum = binder_energy + target_energy
            ddg = complex_energy - monomer_energy_sum
            
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "ddg": ddg,
                "complex_energy": complex_energy,
                "binder_energy": binder_energy,
                "target_energy": target_energy,
                "monomer_energy_sum": monomer_energy_sum,
            }
            
        except Exception as e:
            logger.error(f"SoftDDGMetric computation failed: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> dict[str, float]:
        """Return default metrics when computation fails."""
        return {
            "ddg": float('nan'),
            "complex_energy": float('nan'),
            "binder_energy": float('nan'),
            "target_energy": float('nan'),
            "monomer_energy_sum": float('nan'),
        }


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
            raise ValueError(
                f"charge_func must be 'linear', 'flat', or 'quadratic', got '{charge_func}'"
            )
        
        logger.info(
            f"ChargeComplementarityMetric initialized "
            f"(min_dist={min_dist}, max_dist={max_dist}, func={charge_func})"
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
            weights = (
                (self.max_dist - distances) / (self.max_dist - self.min_dist)
            )
            weights = np.clip(weights, 0, 1) * self.max_bonus
            
        elif self.charge_func == "quadratic":
            # Quadratic decay (1/r^2 normalized)
            weights = (
                1 / np.clip(distances, self.min_dist, None) ** 2 
                * self.min_dist ** 2 
                * self.max_bonus
            )
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
            positions, charges, res_ids, chain_ids = get_charged_residue_positions(
                predicted_atom_array
            )
            
            if len(positions) == 0:
                return self._get_default_metrics()
            
            # Determine which charged residues are on binder vs target
            # We need to match res_id/chain_id back to atom masks
            is_binder_residue = np.zeros(len(positions), dtype=bool)
            
            for i, (res_id, chain_id) in enumerate(zip(res_ids, chain_ids)):
                res_mask = (
                    (predicted_atom_array.res_id == res_id) & 
                    (predicted_atom_array.chain_id == chain_id)
                )
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
            dist_matrix = np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], 
                axis=-1
            )
            
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
            salt_bridge_mask = (
                (charge_prod == -1) & 
                interface_mask & 
                (dist_matrix <= self.max_dist)
            )
            n_salt_bridges = int(np.sum(salt_bridge_mask) / 2)  # Divide by 2 for unique pairs
            
            # Count charge clashes (same charge pairs within max_dist)
            clash_mask = (
                (charge_prod == +1) & 
                interface_mask & 
                (dist_matrix <= self.max_dist)
            )
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
            "charge_score": float('nan'),
            "n_salt_bridges": 0,
            "n_charge_clashes": 0,
            "n_charged_residues_binder": 0,
            "n_charged_residues_target": 0,
        }


# ============================================================================
# Combined DDG + Salt Metric
# ============================================================================


class SoftDDGWithSaltMetric(Metric):
    """Combined soft ddG + charge complementarity metric.
    
    Computes Brian Coventry's recommended "soft ddG with salt" score:
    - Soft ddG using beta2016 with fa_ljrep=0.55
    - Plus charge complementarity bonus for salt bridges
    
    This is the recommended way to calculate Rosetta-style ddG for AI-generated
    protein designs.
    
    Args:
        use_soft_rep: Use soft repulsion for ddG. Defaults to ``True``.
        soft_rep_weight: Weight for fa_ljrep. Defaults to ``0.55``.
        salt_min_dist: Min distance for salt bridge bonus. Defaults to ``7.0``.
        salt_max_dist: Max distance for salt bridge bonus. Defaults to ``9.0``.
        salt_bonus: Max bonus per salt bridge. Defaults to ``2.0``.
        charge_func: Distance decay ("linear", "flat", "quadratic"). Defaults to ``"linear"``.
        device: Device for TMol. Defaults to ``"cuda"``.
        
    Example:
        >>> metric = SoftDDGWithSaltMetric()
        >>> results = metric.compute(predicted_atom_array=structure)
        >>> print(f"DDG with salt: {results['ddg_with_salt']:.2f}")
    """
    
    def __init__(
        self,
        use_soft_rep: bool = True,
        soft_rep_weight: float = 0.55,
        salt_min_dist: float = 7.0,
        salt_max_dist: float = 9.0,
        salt_bonus: float = 2.0,
        charge_func: str = "linear",
        device: str = "cuda",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        
        self.ddg_metric = SoftDDGMetric(
            use_soft_rep=use_soft_rep,
            soft_rep_weight=soft_rep_weight,
            device=device,
        )
        
        self.charge_metric = ChargeComplementarityMetric(
            min_dist=salt_min_dist,
            max_dist=salt_max_dist,
            max_bonus=salt_bonus,
            charge_func=charge_func,
        )
        
        logger.info("SoftDDGWithSaltMetric initialized")
    
    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map kwargs to compute arguments."""
        return {
            "predicted_atom_array": "predicted_atom_array",
        }
    
    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Compute combined ddG with salt score.
        
        Args:
            predicted_atom_array: Structure to evaluate.
            
        Returns:
            Dictionary with all metrics from both sub-metrics plus:
                - ddg_with_salt: Combined score (ddg + charge_score)
        """
        ddg_results = self.ddg_metric.compute(predicted_atom_array)
        charge_results = self.charge_metric.compute(predicted_atom_array)
        
        # Combine results
        results = {}
        results.update(ddg_results)
        results.update(charge_results)
        
        # Compute combined score
        ddg = ddg_results.get("ddg", float('nan'))
        charge_score = charge_results.get("charge_score", 0.0)
        
        if np.isnan(ddg):
            results["ddg_with_salt"] = float('nan')
        else:
            results["ddg_with_salt"] = ddg + charge_score
        
        return results


