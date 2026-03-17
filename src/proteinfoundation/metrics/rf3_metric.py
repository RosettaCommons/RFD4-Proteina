"""RF3 refolding metric for structure prediction evaluation.

This metric is DIFFERENTIABLE — RF3 is a PyTorch-based model whose forward pass
preserves the computation graph, enabling gradient-based optimization.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed
from atomworks.common import exists
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.io.transforms.atom_array import (
    add_id_and_entity_annotations,
    add_chain_iid_annotation,
    add_pn_unit_iid_annotation,
    add_molecule_iid_annotation,
)
from atomworks.ml.utils import nested_dict
from biotite.structure import AtomArray, AtomArrayStack
from foundry.metrics.metric import Metric, MetricManager
from foundry.utils.ddp import RankedLogger
from rf3.inference_engines.rf3 import RF3InferenceEngine
from rf3.utils.inference import InferenceInput, InputOptions

from proteinfoundation.metrics.metrics_utils import normalize_to_atom_array_list
from proteinfoundation.utils.chain_utils import get_chain_of_interest

logger = RankedLogger(__name__, rank_zero_only=True)


# TODO: Probably a job for the configs, but should have options for the MetricManager to do thresholding operations and alignments
class RF3Metric(Metric):
    """RF3-based refolding metric (differentiable).

    Uses the RF3 pythonic API to refold structures and extract confidence metrics.
    Returns multiple confidence scores: plddt, iptm, pae, ipae, min_ipae.

    This metric is differentiable: RF3 is a PyTorch model, so the refolding
    forward pass can propagate gradients back to the input structure for
    gradient-based refinement.

    Args:
        ckpt_path: Path to RF3 checkpoint file.
        num_steps: RF3 diffusion steps. Defaults to ``50``.
        n_recycles: RF3 recycles. Defaults to ``5``.
        diffusion_batch_size: Batch size. Defaults to ``1``.
        template_annotations: Annotation name(s) to use for templating. Multiple
            annotations are OR'd together. Defaults to ``None``.
        out_dir: Directory to save input/refolded structures as CIF files.
            If None, no saving. Defaults to ``None``.

    Example:
        >>> metric = RF3Metric(ckpt_path="/path/to/rf3_latest.pt")
        >>> results = metric.compute(predicted_atom_array=structure)
        >>> # Returns: {"plddt": 85.0, "iptm": 0.9, "ipae": 8.5, ...}

        >>> # Template coordinate-conditioned atoms and save structures
        >>> from atomworks.ml.conditions import C_CRD
        >>> metric = RF3Metric(
        ...     ckpt_path="/path/to/rf3_latest.pt",
        ...     template_annotations=[C_CRD.full_name],
        ...     out_dir="/path/to/output",
        ... )
    """

    # Class-level cache for sharing RF3 engine across metric instances
    _engine_cache: ClassVar[dict[Path, RF3InferenceEngine]] = {}
    _engine_ref_counts: ClassVar[dict[Path, int]] = {}

    @staticmethod
    def _get_current_device() -> int:
        """Get current CUDA device index for this process."""
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return 0

    def __init__(
        self,
        ckpt_path: str | Path,
        num_steps: int = 50,
        n_recycles: int = 5,
        diffusion_batch_size: int = 1,
        metric_manager: MetricManager | None = None,
        template_annotations: str | list[str] | None = None,
        extra_annots_to_symm_resolve: list[str] | None = None,
        pass_criteria: dict[str, dict[str, float]] | None = None,
        out_dir: str | Path | None = None,
        input_kwarg_names: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        """Initialize RF3 metric.

        Args:
            ckpt_path: Path to RF3 checkpoint
            num_steps: RF3 diffusion steps
            n_recycles: RF3 recycles
            diffusion_batch_size: Batch size
            metric_manager: MetricManager for RF3 inference metrics.
            template_annotations: Annotation name(s) to use for templating. Annotations must be
                boolean arrays. Multiple annotations are OR'd together.
            extra_annots_to_symm_resolve: List of extra annotation names to apply symmetry resolution to. Intended for coord-like annotations,
                such as the coordinates of a designed input structure after inserting motif positions.
            pass_criteria: Pass criteria for design evaluation. Should map metric names to min and/or max values.
            out_dir: Directory to save input/refolded structures. If None, no saving.
            input_kwarg_names: Mapping from compute() parameter names to input kwarg names.
                Defaults to ``{"input_atom_array": "predicted_atom_array"}``.
            **kwargs: Additional arguments for Metric base class
        """
        # Set input_kwarg_names before super().__init__() since it accesses kwargs_to_compute_args
        self._input_kwarg_names = input_kwarg_names or {
            "input_atom_array": "predicted_atom_array",
        }
        super().__init__(**kwargs)

        self.ckpt_path = Path(ckpt_path)
        self.num_steps = num_steps
        self.n_recycles = n_recycles
        self.diffusion_batch_size = diffusion_batch_size
        self.metric_manager = metric_manager
        self.template_annotations = (
            [template_annotations] if isinstance(template_annotations, str) else template_annotations
        )
        self.pass_criteria = pass_criteria
        self.out_dir = Path(out_dir) if out_dir else None
        self.extra_annots_to_symm_resolve = extra_annots_to_symm_resolve

        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"RF3Metric will save structures to {self.out_dir}")

        logger.info(f"RF3Metric initialized with checkpoint: {self.ckpt_path}")

    @property
    def engine(self) -> RF3InferenceEngine:
        """Get shared RF3 engine from class-level cache (creates once per checkpoint)."""
        if self.ckpt_path not in RF3Metric._engine_cache:
            current_device = self._get_current_device()
            logger.info(f"Creating RF3 engine for {self.ckpt_path} on device {current_device}")
            RF3Metric._engine_cache[self.ckpt_path] = RF3InferenceEngine(
                ckpt_path=self.ckpt_path,
                n_recycles=self.n_recycles,
                diffusion_batch_size=self.diffusion_batch_size,
                num_steps=self.num_steps,
                seed=None,  # NOTE: We don't fix the seed here, as it may lead to RNG issues when resuming training
                load_weights=False,  # NOTE: We will load weights in setup()
                metrics_cfg=None,
                num_nodes=1,
                devices_per_node=[current_device],
                verbose=True,  # Don't silence warnings
            )
        return RF3Metric._engine_cache[self.ckpt_path]

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map kwargs to compute arguments.

        Keys are compute() parameter names, values specify where to get data from.
        """
        return {
            **self._input_kwarg_names,
            "example_id": ("extra_info", "example_id"),
            "rf3_extra_info": ("extra_info",),
            "out_dir": ("extra_info", "output_base_dir"),
            "chain_of_interest": ("extra_info", "chain_of_interest"),
        }

    def _cleanup_rf3(self) -> None:
        """Clean up RF3 engine and free GPU memory."""
        # Sync all ranks before cleanup to prevent DDP hangs
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Check cache for engine (don't use self.engine which would create one)
        engine = RF3Metric._engine_cache.get(self.ckpt_path)
        if engine is not None and engine._weights_loaded:
            engine.unload_weights()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def setup(self) -> None:
        """Load RF3 engine weights for validation (shared across instances)."""
        # Increment reference count
        RF3Metric._engine_ref_counts[self.ckpt_path] = RF3Metric._engine_ref_counts.get(self.ckpt_path, 0) + 1
        # Load weights only if not already loaded
        if not self.engine._weights_loaded:
            self.engine.load_weights()
            logger.info(f"Loaded RF3 weights from {self.ckpt_path}")

    def teardown(self) -> None:
        """Clean up RF3 engine after validation (only when all instances are done)."""
        # Decrement reference count
        RF3Metric._engine_ref_counts[self.ckpt_path] -= 1
        # Only cleanup when no more references
        if RF3Metric._engine_ref_counts[self.ckpt_path] == 0:
            self._cleanup_rf3()
            logger.info("Unloaded RF3 weights (last reference)")
        else:
            logger.info(
                f"RF3 weights still in use ({RF3Metric._engine_ref_counts[self.ckpt_path]} references remaining)"
            )

    def _save_structures(
        self,
        input_atom_array: AtomArray,
        predicted_atom_array: AtomArray,
        save_path: Path,
        example_id: str,
    ) -> None:
        """Save input and predicted structures to CIF files.

        Args:
            input_atom_array: Original input structure
            predicted_atom_array: RF3-refolded structure
            save_path: Directory path to save structures
        """
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            to_cif_file(
                input_atom_array,
                save_path / f"{example_id}_rf3_input.cif",
            )
            to_cif_file(
                predicted_atom_array,
                save_path / f"{example_id}_rf3_predicted.cif",
            )
            logger.info(f"Saved structures to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save structures to {save_path}: {e}")

    def _prepare_atom_array_for_rf3(self, atom_array: AtomArray) -> AtomArray:
        """Prepare AtomArray for RF3 inference."""
        # ... template based on specified annotations (OR'd together)
        if self.template_annotations:
            template_mask = np.zeros(atom_array.array_length(), dtype=bool)
            for annot_name in self.template_annotations:
                assert (
                    annot_name in atom_array.get_annotation_categories()
                ), f"Annotation '{annot_name}' not found in AtomArray"
                annot = atom_array.get_annotation(annot_name)
                assert annot.dtype == bool, f"Annotation '{annot_name}' must be boolean, got {annot.dtype}"
                template_mask |= annot
            atom_array.set_annotation("is_input_file_templated", template_mask)

        # Remove problematic annotations (which we will re-generate within the RF3 pipeline)
        annotations_to_remove = ["token_id", "atom_id"]
        for annotation in annotations_to_remove:
            if annotation in atom_array.get_annotation_categories():
                atom_array.del_annotation(annotation)
        # TODO: MAY NOT BE NEEDED ANYMORE?
        atom_array.set_annotation("transformation_id", np.full(atom_array.array_length(), "1", dtype=str))
        atom_array = add_id_and_entity_annotations(atom_array)
        atom_array = add_chain_iid_annotation(atom_array)
        atom_array = add_pn_unit_iid_annotation(atom_array)
        atom_array = add_molecule_iid_annotation(atom_array)

        return atom_array

    def is_passing_design(self, metrics: dict[str, float]) -> bool:
        if not exists(self.pass_criteria):
            logger.warning("Pass criteria not defined; this design will be marked as failing.")
            return False

        for metric_name, metric_criteria in self.pass_criteria.items():
            metric_value = metrics.get(metric_name)
            if not exists(metric_value):
                logger.warning(f"Metric {metric_name} not found in metrics; this design will be marked as failing.")
                return False

            if not np.isfinite(metric_value):
                logger.warning(f"Metric {metric_name} is not finite; this design will be marked as failing.")
                return False

            if exists(metric_criteria.get("min")) and metric_value < metric_criteria["min"]:
                return False
            if exists(metric_criteria.get("max")) and metric_value > metric_criteria["max"]:
                return False

        return True

    def compute(
        self,
        input_atom_array: AtomArray | AtomArrayStack | list[AtomArray],
        example_id: str | None = None,
        out_dir: Path | None = None,
        rf3_extra_info: dict | None = None,
        chain_of_interest: str | None = None,
    ) -> dict[str, float]:
        """Refold input structures with RF3 and return confidence metrics.

        Args:
            input_atom_array: Structure(s) to refold with RF3.
            example_id: Identifier for the example (used for saving).
            out_dir: Directory to save input/refolded structures. If None, no saving.
            rf3_extra_info: Extra info dictionary to pass to RF3 inference.
            chain_of_interest: Pre-computed chain of interest. If provided, skip the
                heuristic computation. If ``None``, the chain of interest is inferred
                from the input structure annotations.

        Returns:
            Averaged RF3 confidence metrics across all input structures.
        """
        # Save current device to restore after RF3 inference (RF3 may change it)
        original_device = torch.cuda.current_device()

        # Clear cache and synchronize before RF3
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Set the MetricManager to correspond to the current Metric
        self.engine._metrics_cfg = self.metric_manager

        # Set dataset overrides for this RF3Metric instance, then rebuild the Transform pipeline
        for override in ["diffusion_batch_size", "n_recycles"]:
            self.engine.transform_overrides[override] = getattr(self, override)

        self.engine._construct_pipeline(self.engine.cfg)

        # Update the number of inference timesteps (only needed in shadow model, done in both for future-proofing)
        self.engine.trainer.state["model"].model.inference_sampler.num_timesteps = self.num_steps
        self.engine.trainer.state["model"].shadow.inference_sampler.num_timesteps = self.num_steps

        # Try to fold inputs and collect Metrics
        try:
            atom_arrays = normalize_to_atom_array_list(input_atom_array)

            # ... loop through all inputs, fold them, and collect metrics
            all_metrics = defaultdict(list)
            for i, atom_array in enumerate(atom_arrays):
                current_metrics = nested_dict.flatten(
                    self.fold_single_atom_array(
                        atom_array,
                        example_id=example_id,
                        rf3_extra_info=rf3_extra_info,
                        out_dir=out_dir,
                        structure_idx=i,
                        chain_of_interest=chain_of_interest,
                    )
                )
                current_metrics = {".".join(k): v for k, v in current_metrics.items()}

                for key, value in current_metrics.items():
                    if key == "example_id":
                        continue
                    all_metrics[key].append(value)

                if exists(self.pass_criteria):
                    all_metrics["pass_rate"].append(self.is_passing_design(current_metrics))

            # Average metrics over all structures
            averaged_metrics = {
                key: float(np.mean(values)) for key, values in all_metrics.items() if key != "chain_of_interest"
            }

            # Save chain of interest if possible
            num_unique_coi = len(np.unique(all_metrics["chain_of_interest"]))
            if num_unique_coi == 1:
                averaged_metrics["chain_of_interest"] = all_metrics["chain_of_interest"][0]
            else:
                logger.warning(
                    f"Found {num_unique_coi} unique chains of interest among the input AtomArrays. "
                    "A value of None will be returned for the aggregated chain of interest."
                )
                averaged_metrics["chain_of_interest"] = None

            return averaged_metrics
        finally:
            # Restore original CUDA device (RF3 may have changed it)
            torch.cuda.set_device(original_device)

    # Interface metrics for all interfaces involving the COI.
    # Matrices are upper-triangular: [i][j] filled only when i < j.
    def _coi_interface_vals(self, matrix, coi_idx, n_chains):
        return (
            [matrix[i][coi_idx] for i in range(coi_idx) if matrix[i][coi_idx] is not None]
            + [matrix[coi_idx][j] for j in range(coi_idx + 1, n_chains) if matrix[coi_idx][j] is not None]
        )
    
    def fold_single_atom_array(
        self,
        input_atom_array: AtomArray,
        example_id: str | None = None,
        rf3_extra_info: dict | None = None,
        out_dir: Path | None = None,
        structure_idx: int = 0,
        chain_of_interest: str | None = None,
    ) -> dict[str, float]:
        """Refold structure with RF3 and return confidence metrics.

        Args:
            input_atom_array: Structure to refold
            example_id: Identifier for the example (used for RF3 and saving)
            rf3_extra_info: Extra info dictionary to pass to RF3 inference
            out_dir: Directory to save input/refolded structures. If None, no saving.
            structure_idx: Index of structure (fallback for saving if no example_id)
            chain_of_interest: Pre-computed chain of interest. If ``None``, the
                chain of interest is inferred from the structure annotations.

        Returns:
            Dictionary with RF3 confidence metrics:
                - plddt: Mean pLDDT score (0-100)
                - iptm: Interface pTM score (0-100, scaled from 0-1)
                - pae: Mean predicted aligned error (Angstroms)
                - ipae: Interface PAE (Angstroms)
                - min_ipae: Minimum interface PAE (Angstroms)
        """
        # Use example_id or fallback to structure index
        rf3_example_id = example_id or f"structure_{structure_idx}"

        prepared_array = self._prepare_atom_array_for_rf3(input_atom_array.copy())

        try:
            extra_info = {"extra_annots_to_symm_resolve": self.extra_annots_to_symm_resolve} | (
                rf3_extra_info if exists(rf3_extra_info) else {}
            )
            inference_input = InferenceInput.from_atom_array(
                prepared_array,
                example_id=rf3_example_id,
                extra_info=extra_info,
                input_options=InputOptions(add_missing_atoms=False, long_bond_policy="keep"),
            )

            # Run RF3
            results = self.engine.run(
                inputs=[inference_input],
                out_dir=None,  # Return in-memory
                annotate_b_factor_with_plddt=True,
            )

            if results.get("early_stopped"):
                logger.warning("RF3 early stopped - returning default poor metrics")
                return {
                    "plddt": 0.0,
                    "ptm": 0.0,
                    "iptm": 0.0,
                    "pae": 100.0,
                    "ipae": 100.0,
                    "min_ipae": 100.0,
                }

            # Extract metrics (note we always predict a single structure)
            result = results[rf3_example_id][0]  # RF3Output object

            metrics = result.metrics
            summary_confidences = result.summary_confidences
            chains_in_order = np.unique(result.atom_array.chain_iid)

            if chain_of_interest is not None:
                chain_of_interest_unambiguous = True
                logger.info(f"Using externally-provided chain of interest: {chain_of_interest}")
            else:
                chain_of_interest, chain_of_interest_unambiguous = get_chain_of_interest(prepared_array)
                logger.info(
                    f"Inferred chain of interest: {chain_of_interest}. This will be used to subset interface metrics."
                )
            metrics["chain_of_interest"] = chain_of_interest if chain_of_interest_unambiguous else None

            if summary_confidences:
                # Overall metrics
                if "overall_plddt" in summary_confidences:
                    metrics["overall_plddt"] = summary_confidences["overall_plddt"] * 100
                if summary_confidences.get("ptm") is not None:
                    metrics["overall_ptm"] = summary_confidences["ptm"] * 100.0
                if "overall_pae" in summary_confidences:
                    metrics["overall_pae"] = summary_confidences["overall_pae"]

                # Chain-level and interface metrics for the chain of interest.
                # chains_in_order contains chain_iid integers in sorted order, matching
                # the indexing used in summary_confidences chain lists and chain_pair_* matrices.
                if chain_of_interest is not None and chain_of_interest in chains_in_order:
                    coi_idx = list(chains_in_order).index(chain_of_interest)
                    n_chains = len(chains_in_order)

                    # Per-chain metrics for the COI (all lists indexed by chains_in_order position)
                    if "chain_plddt" in summary_confidences:
                        metrics["plddt_coi"] = summary_confidences["chain_plddt"][coi_idx] * 100
                    if "chain_ptm" in summary_confidences:
                        metrics["ptm_coi"] = summary_confidences["chain_ptm"][coi_idx] * 100.0
                    if "chain_pae" in summary_confidences:
                        metrics["pae_coi"] = summary_confidences["chain_pae"][coi_idx]

                    if "chain_pair_pae" in summary_confidences:
                        vals = self._coi_interface_vals(summary_confidences["chain_pair_pae"], coi_idx, n_chains)
                        if vals:
                            metrics["ipae_coi"] = float(np.mean(vals))
                    if "chain_pair_pae_min" in summary_confidences:
                        vals = self._coi_interface_vals(summary_confidences["chain_pair_pae_min"], coi_idx, n_chains)
                        if vals:
                            metrics["min_ipae_coi"] = float(np.min(vals))
                    if "chain_pair_iptm" in summary_confidences:
                        vals = self._coi_interface_vals(summary_confidences["chain_pair_iptm"], coi_idx, n_chains)
                        if vals:
                            metrics["iptm_coi"] = float(np.mean(vals)) * 100.0

            # Save structures if out_dir is set
            if out_dir:

                # Optionally save an aligned version of the predicted structure
                aligned_aa = []
                for metric_name in metrics:
                    if metric_name.endswith("aligned_predicted_structure"):
                        aligned_aa.append(metrics[metric_name])
                metrics = {k: v for k, v in metrics.items() if not k.endswith("aligned_predicted_structure")}

                if (len(aligned_aa) == 1) and not np.isnan(aligned_aa[0].coord).all():
                    predicted_structure = aligned_aa[0]
                else:
                    predicted_structure = result.atom_array

                if predicted_structure:
                    save_path = out_dir / "rf3_refolding"
                    self._save_structures(input_atom_array, predicted_structure, save_path, rf3_example_id)

            return metrics

        except Exception as e:
            logger.error(f"RF3 inference failed: {e}")
            return {
                "plddt": 0.0,
                "iptm": 0.0,
                "ptm": 0.0,
                "pae": 100.0,
                "ipae": 100.0,
                "min_ipae": 100.0,
            }

