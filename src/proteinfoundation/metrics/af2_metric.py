"""AlphaFold2 prediction metric using ColabDesign.

This metric is DIFFERENTIABLE and can be used for gradient-based refinement.
"""

import contextlib
import gc
import warnings
from functools import cached_property
from typing import Any

from biotite.structure import AtomArray

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# Suppress warnings
warnings.filterwarnings("ignore")


class AF2Metric(Metric):
    """AlphaFold2-based prediction metric.

    This is a DIFFERENTIABLE metric that uses AF2 to predict structures
    and extract confidence scores. Can be used for gradient-based refinement.

    Args:
        model_name: AF2 model to use. Defaults to ``"model_1_ptm"``.
        num_recycles: Number of recycling iterations. Defaults to ``3``.
        use_templates: Whether to use templates. Defaults to ``False``.
        data_dir: Path to AF2 parameters. If None, uses DATA_PATH env var.
        protocol: "binder" or "hallucination". Defaults to ``"binder"``.

    Example:
        >>> metric = AF2Metric(protocol="binder")
        >>> results = metric.compute(predicted_atom_array=structure)
        >>> # Returns: {"plddt": 85.0, "ptm": 0.9, "pae": 8.5, ...}
    """

    def __init__(
        self,
        model_name: str = "model_1_ptm",
        num_recycles: int = 3,
        use_templates: bool = False,
        data_dir: str | None = None,
        protocol: str = "binder",
        **kwargs: Any,
    ):
        """Initialize AF2 metric.

        Args:
            model_name: AF2 model name
            num_recycles: Number of recycles
            use_templates: Whether to use templates
            data_dir: AF2 parameters directory
            protocol: "binder" or "hallucination"
            **kwargs: Additional arguments for Metric base class
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.num_recycles = num_recycles
        self.use_templates = use_templates
        self.data_dir = data_dir
        self.protocol = protocol

        # Lazy initialization
        self._af_model = None

        logger.info(f"AF2Metric initialized (model={model_name}, protocol={protocol})")

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map kwargs to compute arguments."""
        return {
            "predicted_atom_array": "predicted_atom_array",
        }

    @property
    def af_model(self) -> Any:
        """Lazy initialize AF2 model."""
        if self._af_model is None:
            from proteinfoundation.rewards.alphafold2_reward_utils import get_af2_model_runner

            self._af_model = get_af2_model_runner(
                model_name=self.model_name,
                num_recycles=self.num_recycles,
                use_templates=self.use_templates,
                data_dir=self.data_dir,
            )
        return self._af_model

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Predict with AF2 and return confidence metrics.

        Args:
            predicted_atom_array: Structure to evaluate

        Returns:
            Dictionary with AF2 confidence metrics:
                - plddt: Mean pLDDT score (0-100)
                - ptm: pTM score (0-1 scale)
                - pae: Mean predicted aligned error (Angstroms)
                - max_pae: Max predicted aligned error (Angstroms)
                - ipae: Interface PAE (if multi-chain)
        """
        import jax
        import jax.numpy as jnp

        from proteinfoundation.rewards.alphafold2_reward_utils import (
            atom_array_to_sequence,
            compute_interface_pae,
            extract_af2_confidence_metrics,
        )

        try:
            # Extract sequence
            sequence = atom_array_to_sequence(predicted_atom_array)

            # Get chain information for interface metrics
            chain_ids = []
            seen_res = set()
            for res_id, chain_id in zip(predicted_atom_array.res_id, predicted_atom_array.chain_id, strict=False):
                key = (res_id, chain_id)
                if key not in seen_res:
                    seen_res.add(key)
                    chain_ids.append(ord(chain_id) - ord("A"))  # Convert to 0-indexed

            # Prepare model inputs and run (following public codebase pattern)
            self.af_model.prep_inputs(
                length=len(sequence),
                seed=0,
            )

            # Set sequence (soft mode for prediction without gradients)
            self.af_model.set_seq(sequence)

            # Run prediction (similar to public codebase's single_run)
            self.af_model.run(
                num_recycles=self.num_recycles,
                sample_models=False,
                models=[0],  # Use first model
                backprop=False,  # No gradients for metric evaluation
            )

            # Extract results from model.aux
            result = self.af_model.aux

            # Extract basic metrics
            metrics = extract_af2_confidence_metrics(result)

            # Compute interface PAE if multi-chain
            if "pae" in result and len(set(chain_ids)) > 1:
                pae = result["pae"]
                metrics["ipae"] = compute_interface_pae(pae, chain_ids)

                # Also compute min interface PAE
                chain_ids_arr = jnp.array(chain_ids)
                # Get interface mask (different chains)
                interface_mask = chain_ids_arr[:, None] != chain_ids_arr[None, :]
                if interface_mask.any():
                    metrics["min_ipae"] = float(jnp.min(pae[interface_mask]))

            # Clean up model state
            if hasattr(self.af_model, "_inputs"):
                self.af_model._inputs.clear()
            if hasattr(self.af_model, "aux"):
                self.af_model.aux.clear()

            # Clean up JAX memory
            jax.clear_backends()
            gc.collect()

            return metrics

        except Exception as e:
            logger.error(f"AF2 prediction failed: {e}")
            # Clean up JAX memory on error
            with contextlib.suppress(Exception):
                jax.clear_backends()
            gc.collect()

            return {
                "plddt": 0.0,
                "ptm": 0.0,
                "pae": 100.0,
                "max_pae": 100.0,
                "ipae": 100.0,
                "min_ipae": 100.0,
            }

    def compute_with_gradients(
        self,
        pdb_path: str,
        target_chain: str,
        binder_chain: str,
        binder_sequence_logits: "jnp.ndarray | None" = None,
        soft: float = 1.0,
        hard: float = 0.0,
        temp: float = 1.0,
    ) -> dict[str, Any]:
        """Compute AF2 metrics with sequence gradients for refinement.

        This method supports differentiable sequence optimization by:
        1. Running AF2 with sequence logits (soft) or discrete sequence (hard)
        2. Computing gradients ∂metrics/∂sequence_logits for each metric
        3. Returning both metrics and per-metric gradients

        NOTE: Reward aggregation (weighted sum) is handled by RewardModel, not here.
        This ensures consistency with the configured weights.

        Args:
            pdb_path: Path to PDB file
            target_chain: Target chain ID
            binder_chain: Binder chain ID
            binder_sequence_logits: Sequence logits [n_binder, 20] for differentiable mode
            soft: Weight for soft (logits) optimization (0-1)
            hard: Weight for hard (discrete) optimization (0-1)
            temp: Temperature for soft optimization

        Returns:
            Dictionary containing:
                - 'metrics': Dict of AF2 metrics (plddt, ptm, pae, ipae, etc.)
                - 'grad': Dict with per-metric gradients (all w.r.t. sequence logits)
        """
        import jax.numpy as jnp

        from proteinfoundation.rewards.alphafold2_reward_utils import run_af2_with_gradients

        try:
            # Run AF2 with gradient support
            result = run_af2_with_gradients(
                af_model=self.af_model,
                pdb_path=pdb_path,
                target_chain=target_chain,
                binder_chain=binder_chain,
                binder_sequence_logits=binder_sequence_logits,
                soft=soft,
                hard=hard,
                temp=temp,
                num_recycles=self.num_recycles,
            )

            return result

        except Exception as e:
            logger.error(f"AF2 gradient computation failed: {e}")
            # Return zero metrics and zero gradients
            n_binder = binder_sequence_logits.shape[0] if binder_sequence_logits is not None else 0
            return {
                "metrics": {
                    "af2.plddt": 0.0,
                    "af2.ptm": 0.0,
                    "af2.pae": 100.0,
                    "af2.ipae": 100.0,
                },
                "grad": {"sequence": jnp.zeros((n_binder, 20))} if n_binder > 0 else {},
            }
