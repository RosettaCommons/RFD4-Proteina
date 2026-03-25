"""TMOL (TorchMOL) energy metric for protein energy evaluation.

This metric uses TMOL to compute interface energies including hydrogen bonds
and electrostatics, particularly useful for evaluating protein-protein interfaces.
"""

import gc
from functools import cached_property
from typing import Any

import torch
from biotite.structure import AtomArray

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class TMOLMetric(Metric):
    """TMOL-based interface energy metric.

    Computes interface energies (H-bonds and/or electrostatics) for protein complexes.
    Particularly useful for evaluating protein-protein interfaces and binder designs.

    Args:
        enable_hbond: Enable hydrogen bond scoring. Defaults to ``True``.
        enable_elec: Enable electrostatic scoring. Defaults to ``False``.
        hbond_weight: Weight for H-bond term. Defaults to ``1.0``.
        elec_weight: Weight for electrostatic term. Defaults to ``1.0``.
        energy_threshold: Threshold for counting significant interactions (kcal/mol). Defaults to ``-0.6``.
        reward_type: "energy_sum" or "interaction_count". Defaults to ``"energy_sum"``.

    Example:
        >>> # H-bonds only
        >>> metric = TMOLMetric(enable_hbond=True, enable_elec=False)
        >>> results = metric.compute(predicted_atom_array=structure)
        >>> # Returns: {"total_interface_energy": -15.0, "n_interface_hbonds": 5, ...}

        >>> # Combined H-bonds + electrostatics
        >>> metric = TMOLMetric(enable_hbond=True, enable_elec=True, elec_weight=0.5)
        >>> results = metric.compute(predicted_atom_array=structure)
    """

    def __init__(
        self,
        enable_hbond: bool = True,
        enable_elec: bool = False,
        hbond_weight: float = 1.0,
        elec_weight: float = 1.0,
        energy_threshold: float = -0.6,
        reward_type: str = "energy_sum",
        **kwargs: Any,
    ):
        """Initialize TMOL metric.

        Args:
            enable_hbond: Enable hydrogen bond scoring
            enable_elec: Enable electrostatic scoring
            hbond_weight: Weight for H-bond term
            elec_weight: Weight for electrostatic term
            energy_threshold: Threshold for significant interactions (kcal/mol)
            reward_type: "energy_sum" or "interaction_count"
            **kwargs: Additional arguments for Metric base class
        """
        super().__init__(**kwargs)

        self.enable_hbond = enable_hbond
        self.enable_elec = enable_elec
        self.hbond_weight = hbond_weight
        self.elec_weight = elec_weight
        self.energy_threshold = energy_threshold
        self.reward_type = reward_type

        if not enable_hbond and not enable_elec:
            raise ValueError("At least one of enable_hbond or enable_elec must be True")

        logger.info(f"TMOLMetric initialized (hbond={enable_hbond}, elec={enable_elec})")

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map kwargs to compute arguments."""
        return {
            "predicted_atom_array": "predicted_atom_array",
        }

    def _atom_array_to_openfold_dict(self, atom_array: AtomArray) -> dict[str, torch.Tensor]:
        """Convert AtomArray to OpenFold format dict for TMOL.

        Args:
            atom_array: Input structure

        Returns:
            Dictionary with positions, aatype, chain_index
        """
        from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
        from atomworks.ml.transforms.encoding import atom_array_to_encoding

        # Convert to encoding
        encoding = atom_array_to_encoding(atom_array, UNIFIED_ATOM37_ENCODING)

        # Get coordinates (N_res, 37, 3)
        positions = torch.from_numpy(encoding.pos).float().unsqueeze(0)  # [1, N_res, 37, 3]

        # Get residue types (0-indexed for standard AAs)
        aatype = torch.from_numpy(encoding.res_type).long().unsqueeze(0)  # [1, N_res]

        # Get chain indices
        chain_ids = []
        seen_res = set()
        for res_id, chain_id in zip(atom_array.res_id, atom_array.chain_id, strict=False):
            key = (res_id, chain_id)
            if key not in seen_res:
                seen_res.add(key)
                chain_ids.append(ord(chain_id) - ord("A"))  # Convert to 0-indexed

        chain_index = torch.tensor([chain_ids], dtype=torch.long)  # [1, N_res]

        return {
            "positions": positions,
            "aatype": aatype,
            "chain_index": chain_index,
        }

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Compute TMOL interface energies.

        Args:
            predicted_atom_array: Structure to evaluate

        Returns:
            Dictionary with TMOL interface energy metrics:
                - total_interface_energy: Combined interface energy
                - n_interface_interactions: Number of significant interactions
                - mean_interface_energy: Mean interface energy
                - n_interface_hbonds: Number of H-bonds (if enable_hbond=True)
                - total_interface_hbond_energy: H-bond energy (if enable_hbond=True)
                - n_interface_elec_interactions: Number of elec interactions (if enable_elec=True)
                - total_interface_elec_energy: Elec energy (if enable_elec=True)
        """
        try:
            # Import TMOL (lazy to avoid dependency for basic usage)
            from proteinfoundation.rewards.tmol_reward_model import TmolRewardModel

            # Convert atom array to OpenFold format
            openfold_dict = self._atom_array_to_openfold_dict(predicted_atom_array)

            # Initialize TMOL model
            tmol_model = TmolRewardModel(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                enable_hbond=self.enable_hbond,
                enable_elec=self.enable_elec,
                hbond_weight=self.hbond_weight,
                elec_weight=self.elec_weight,
                reward_type=self.reward_type,
                energy_threshold=self.energy_threshold,
            )

            # Run TMOL scoring
            result = tmol_model.single_run(
                openfold_dict=openfold_dict,
                require_grad=False,
            )

            # Extract metrics
            metrics = {}
            reward_dict = result["reward"]

            # Core interface metrics
            metrics["total_interface_energy"] = float(reward_dict["total_interface_energy"])
            metrics["n_interface_interactions"] = float(reward_dict["n_interface_interactions"])
            metrics["mean_interface_energy"] = float(reward_dict["mean_interface_energy"])

            # H-bond specific metrics
            if self.enable_hbond and "n_interface_hbonds" in reward_dict:
                metrics["n_interface_hbonds"] = float(reward_dict["n_interface_hbonds"])
                metrics["total_interface_hbond_energy"] = float(reward_dict["total_interface_hbond_energy"])
                metrics["mean_interface_hbond_energy"] = float(reward_dict["mean_interface_hbond_energy"])

            # Electrostatic specific metrics
            if self.enable_elec and "n_interface_elec_interactions" in reward_dict:
                metrics["n_interface_elec_interactions"] = float(reward_dict["n_interface_elec_interactions"])
                metrics["total_interface_elec_energy"] = float(reward_dict["total_interface_elec_energy"])
                metrics["mean_interface_elec_energy"] = float(reward_dict["mean_interface_elec_energy"])

            # Cleanup
            tmol_model.cleanup()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return metrics

        except ImportError as e:
            logger.error(f"TMOL not available: {e}. Install TMOL to use this metric.")
            return self._get_default_metrics()
        except Exception as e:
            logger.error(f"TMOL computation failed: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._get_default_metrics()

    def _get_default_metrics(self) -> dict[str, float]:
        """Return default metrics when TMOL fails or is unavailable."""
        metrics = {
            "total_interface_energy": 0.0,
            "n_interface_interactions": 0.0,
            "mean_interface_energy": 0.0,
        }

        if self.enable_hbond:
            metrics.update(
                {
                    "n_interface_hbonds": 0.0,
                    "total_interface_hbond_energy": 0.0,
                    "mean_interface_hbond_energy": 0.0,
                }
            )

        if self.enable_elec:
            metrics.update(
                {
                    "n_interface_elec_interactions": 0.0,
                    "total_interface_elec_energy": 0.0,
                    "mean_interface_elec_energy": 0.0,
                }
            )

        return metrics
