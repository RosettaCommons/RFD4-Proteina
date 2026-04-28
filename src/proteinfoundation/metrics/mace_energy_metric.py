"""MACE energy metric for comparing predicted vs ground truth structures."""

import gc
import logging
import os
import warnings
from functools import cached_property
from typing import Any, Literal

import numpy as np
import torch
from atomworks.constants import BIOTITE_DEFAULT_ANNOTATIONS, METAL_ELEMENTS
from biotite.structure import AtomArray, AtomArrayStack

from foundry.metrics.metric import Metric
from proteinfoundation.metrics.metrics_utils import normalize_to_atom_array_list

logger = logging.getLogger(__name__)


def compute_mace_energy(
    atom_array: AtomArray,
    calculator: "MACECalculator",
    hydrogen_policy: Literal["infer", "keep"] = "infer",
    infer_charge: Literal["atom_array", "zeros", "raise"] = "atom_array",
) -> float:
    """Compute MACE energy for an AtomArray.

    Args:
      atom_array: Structure to compute energy for.
      calculator: MACE calculator instance.
      hydrogen_policy: How to handle hydrogens.
        - ``"infer"``: Remove existing H, add new H with hydride.
        - ``"keep"``: Keep existing H as-is.
      infer_charge: What to use when charge/spin missing from ase_atoms.info.
        - ``"atom_array"``: Infer by summing per-atom charges (default).
        - ``"zeros"``: Use charge=0, spin=1.
        - ``"raise"``: Raise ValueError if missing.

    Returns:
      Potential energy in eV.

    Raises:
      ValueError: If charge/spin missing and ``infer_charge="raise"``, or if
        ``infer_charge="atom_array"`` but charge annotation not found.

    Examples:
      >>> from mace.calculators import MACECalculator
      >>> calc = MACECalculator(model_paths="model.pt", device="cuda")
      >>> # Use atom charges if ase_atoms.info missing charge/spin
      >>> energy = compute_mace_energy(atom_array, calc, infer_charge="atom_array")
      >>> # Raise error if charge/spin missing
      >>> energy = compute_mace_energy(atom_array, calc, infer_charge="raise")
    """
    from atomworks.io.utils.ase_conversions import atom_array_to_ase

    if hydrogen_policy not in ("infer", "keep"):
        raise ValueError(f"hydrogen_policy must be 'infer' or 'keep', got {hydrogen_policy}")

    if hydrogen_policy == "infer":
        from atomworks.io.transforms.atom_array import add_hydrogen_atom_positions

        # add_hydrogen_atom_positions removes existing H first, then adds new
        # NOTE: hydride requires only specific annotations and can fail on irregular ones
        # ... remove non-essential annotations before processing
        essential_annots = set(BIOTITE_DEFAULT_ANNOTATIONS) | {"charge"}
        for annot in list(atom_array.get_annotation_categories()):
            if annot not in essential_annots:
                atom_array.del_annotation(annot)
        atom_array = add_hydrogen_atom_positions(atom_array)

    ase_atoms = atom_array_to_ase(atom_array)

    # Define default values for charge and spin based on policy
    if infer_charge == "zeros":
        defaults = {"charge": 0.0, "spin": 1.0}
    elif infer_charge == "atom_array":
        # For atom_array: charge from annotations, spin = 1.0
        try:
            charge_sum = float(np.sum(atom_array.get_annotation("charge")))
        except ValueError as e:
            raise ValueError(f"Charge annotation not found in atom_array: {e}") from e
        defaults = {"charge": charge_sum, "spin": 1.0}
    elif infer_charge == "raise":
        defaults = None  # Will raise error if missing values accessed
    else:
        raise ValueError(f"infer_charge must be 'atom_array', 'zeros', or 'raise', got {infer_charge}")

    # Set charge/spin: use existing values in ase_atoms.info, or infer if missing
    for key in ["charge", "spin"]:
        if key in ase_atoms.info and isinstance(ase_atoms.info[key], int | float):
            continue  # Valid value exists, keep it

        # Missing or invalid - apply infer_charge policy
        if infer_charge == "raise":
            raise ValueError(f"{key.capitalize()} not found in ase_atoms.info and infer_charge='raise'")

        ase_atoms.info[key] = defaults[key]
        logger.debug(f"Inferred {key}={ase_atoms.info[key]:.1f} (policy={infer_charge})")

    # Warn if charge magnitude unusually high
    if abs(ase_atoms.info["charge"]) > 10:
        logger.warning(f"Unusually high charge: {ase_atoms.info['charge']:.1f}. Verify structure.")
    ase_atoms.calc = calculator

    # MACE computes forces via autograd, so we need gradients enabled
    # even when called from a validation context (torch.no_grad/inference_mode)
    with torch.enable_grad():
        return ase_atoms.get_potential_energy()


class MACEEnergyMetric(Metric):
    """Compute MACE energies for chains and compare predicted vs ground truth.

    Iterates over chains in the structure, computes MACE energies for chains
    with fewer than ``max_chain_atoms`` atoms, and returns the energy difference
    (predicted - ground_truth) for each chain.
    """

    def __init__(
        self,
        model_path: str | None = None,
        max_chain_atoms: int = 300,
        hydrogen_policy: Literal["infer", "keep"] = "infer",
        infer_charge: Literal["atom_array", "zeros", "raise"] = "atom_array",
        device: str = "cuda",
        split_by_metals: bool = False,
        max_energy_diff: float | None = 1e5,
        **kwargs: Any,
    ) -> None:
        """Initialize MACE energy metric.

        Args:
          model_path: Path to MACE model. Falls back to MACE_MODEL_PATH env var.
          max_chain_atoms: Skip chains larger than this.
          hydrogen_policy: ``"infer"`` to add H with hydride, ``"keep"`` to use existing.
          infer_charge: What to use when charge/spin missing from ase_atoms.info.
            - ``"atom_array"``: Infer by summing per-atom charges (default).
            - ``"zeros"``: Use charge=0, spin=1.
            - ``"raise"``: Raise ValueError if missing.
          device: Device for MACE computation.
          split_by_metals: If True, report separate metrics for chains with/without metals.
          max_energy_diff: Clip energy differences with absolute value larger than this
            to this value (positive). Set to None to disable clipping.
        """
        if hydrogen_policy not in ("infer", "keep"):
            raise ValueError(f"hydrogen_policy must be 'infer' or 'keep', got {hydrogen_policy}")

        # Get MACE model path from argument or environment variable
        if model_path is None:
            model_path = os.environ.get("MACE_MODEL_PATH")

        if model_path is None:
            raise ValueError(
                "MACE model path must be provided via model_path argument or MACE_MODEL_PATH environment variable"
            )

        super().__init__(**kwargs)
        self.model_path = model_path
        self.max_chain_atoms = max_chain_atoms
        self.hydrogen_policy = hydrogen_policy
        self.infer_charge = infer_charge
        self.device = device
        self.split_by_metals = split_by_metals
        self.max_energy_diff = max_energy_diff
        self._calculator = None  # Loaded in setup()

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map validation inputs to compute() parameters."""
        return {
            "predicted_atom_array_stack": "predicted_atom_array",
            "ground_truth_atom_array_stack": "ground_truth_atom_array",
        }

    def setup(self) -> None:
        """Load MACE calculator for validation."""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("MACE metric requires CUDA but none available — skipping setup")
            return

        from mace.calculators import MACECalculator

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected",
            )
            self._calculator = MACECalculator(
                model_paths=self.model_path,
                device=self.device,
                default_dtype="float32",
            )
        logger.info(f"Loaded MACE calculator from {self.model_path}")

    def teardown(self) -> None:
        """Clean up MACE calculator after validation."""
        # Sync all ranks before cleanup to prevent DDP hangs
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self._calculator is not None:
            del self._calculator
            self._calculator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Unloaded MACE calculator")

    def compute(
        self,
        predicted_atom_array_stack: "AtomArrayStack | AtomArray | list[AtomArray]",
        ground_truth_atom_array_stack: "AtomArrayStack | AtomArray | list[AtomArray]",
    ) -> dict[str, Any]:
        """Compute MACE energy differences for all valid chains.

        Args:
          predicted_atom_array_stack: Predicted structures.
          ground_truth_atom_array_stack: Ground truth structures.

        Returns:
          Dictionary with energy differences. If ``split_by_metals`` is False:
            - ``median_energy_diff``: Median energy diff across all chains (eV)
          If ``split_by_metals`` is True:
            - ``median_energy_diff_with_metals``: Median for chains containing metals
            - ``median_energy_diff_without_metals``: Median for chains without metals
        """
        pred_arrays = normalize_to_atom_array_list(predicted_atom_array_stack)
        gt_arrays = normalize_to_atom_array_list(ground_truth_atom_array_stack)

        # Use cached calculator (loaded in setup())
        if self._calculator is None:
            logger.warning("MACE calculator unavailable — skipping metric computation")
            return {}

        results: dict[str, Any] = {}
        energy_diffs: list[float] = []
        energy_diffs_with_metals: list[float] = []
        energy_diffs_without_metals: list[float] = []

        # Iterate over model pairs
        for model_idx, (pred_array, gt_array) in enumerate(zip(pred_arrays, gt_arrays, strict=False)):
            unique_chains = np.unique(pred_array.chain_id)

            for chain_id in unique_chains:
                pred_chain = pred_array[pred_array.chain_id == chain_id]
                gt_chain = gt_array[gt_array.chain_id == chain_id]

                # Skip if too large
                if len(pred_chain) > self.max_chain_atoms:
                    logger.warning(
                        f"Skipping model {model_idx} chain {chain_id}: "
                        f"{len(pred_chain)} atoms > {self.max_chain_atoms}"
                    )
                    continue

                # Skip if chain doesn't exist in gt
                if len(gt_chain) == 0:
                    continue

                try:
                    pred_energy = compute_mace_energy(
                        pred_chain,
                        self._calculator,
                        self.hydrogen_policy,
                        self.infer_charge,
                    )
                    gt_energy = compute_mace_energy(
                        gt_chain,
                        self._calculator,
                        self.hydrogen_policy,
                        self.infer_charge,
                    )

                    diff = pred_energy - gt_energy

                    # Clip extreme values to max_energy_diff if:
                    # 1. Absolute difference exceeds max_energy_diff, OR
                    # 2. Difference is more than 100x the ground truth energy (likely unphysical)
                    if self.max_energy_diff is not None:
                        is_abs_extreme = abs(diff) > self.max_energy_diff
                        is_rel_extreme = abs(gt_energy) > 0 and abs(diff) > 100 * abs(gt_energy)
                        if is_abs_extreme or is_rel_extreme:
                            diff = self.max_energy_diff

                    energy_diffs.append(diff)

                    # Track by metal presence if requested
                    if self.split_by_metals:
                        chain_elements = {e.upper() for e in gt_chain.element}
                        has_metal = bool(chain_elements & METAL_ELEMENTS)
                        if has_metal:
                            energy_diffs_with_metals.append(diff)
                        else:
                            energy_diffs_without_metals.append(diff)

                except Exception as e:
                    logger.warning(f"Failed to compute energy for model {model_idx} chain {chain_id}: {e}")
                    continue

        # Always return consistent keys with NaN for missing values to avoid DDP sync issues
        if self.split_by_metals:
            results["median_energy_diff_with_metals"] = (
                float(np.median(energy_diffs_with_metals)) if energy_diffs_with_metals else float("nan")
            )
            results["median_energy_diff_without_metals"] = (
                float(np.median(energy_diffs_without_metals)) if energy_diffs_without_metals else float("nan")
            )
        else:
            results["median_energy_diff"] = float(np.median(energy_diffs)) if energy_diffs else float("nan")

        return results
