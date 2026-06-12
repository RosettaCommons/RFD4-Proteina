"""RF3 structure prediction oracle.

Folds one designed structure with RF3 and returns confidence scores plus the path
to the saved refolded CIF. Inherits resumability from OracleMetric: the first fold
writes a JSON with all confidence values; subsequent calls load the JSON directly.
"""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed
from atomworks.io.transforms.atom_array import (
    add_chain_iid_annotation,
    add_id_and_entity_annotations,
    add_molecule_iid_annotation,
    add_pn_unit_iid_annotation,
)
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.utils.annotator import ANNOTATOR_REGISTRY, ensure_annotations
from biotite.structure import AtomArray

from foundry.utils.ddp import RankedLogger
from proteinfoundation.metrics.oracle_metric import (
    OracleJobInput,
    OracleMetric,
    extract_confidence_metrics,
)
from proteinfoundation.transforms.utils.conditioning_utils import align_atom_array_to_input_frame
from proteinfoundation.utils.chain_utils import get_chain_of_interest
from rf3.inference_engines.rf3 import RF3InferenceEngine
from rf3.utils.inference import InferenceInput, InputOptions

logger = RankedLogger(__name__, rank_zero_only=True)

_EARLY_STOP_METRICS: dict[str, float] = {
    "plddt": 0.0,
    "ptm": 0.0,
    "iptm": 0.0,
    "pae": 100.0,
    "ipae": 100.0,
    "min_ipae": 100.0,
}


class RF3Oracle(OracleMetric):
    """RF3 structure prediction oracle.

    Folds a single designed structure with RF3, saves the refolded CIF to
    ``{out_dir}/{oracle_name}_refolding/{example_id}_refolded.cif``, and returns
    all RF3 confidence metrics.

    Weights are loaded lazily at validation start and shared across all
    ``RF3Oracle`` instances that use the same checkpoint (ref-counted). A DDP
    barrier is issued during teardown to prevent rank desynchronisation.

    Args:
        ckpt_path: Path to RF3 checkpoint file.
        num_steps: Diffusion steps. Defaults to ``50``.
        n_recycles: Number of recycling iterations. Defaults to ``5``.
        extra_annots_to_symm_resolve: Extra annotation names to apply symmetry
            resolution to (e.g. designed input coordinates).
        oracle_name: Output subdirectory label. Defaults to ``"rf3"``.
        template_annotations: Annotation name(s) to use for templating. Multiple
            names are OR'd together. Defaults to ``None``.

    Example:
        >>> oracle = RF3Oracle(ckpt_path="/path/to/rf3_latest.pt")
        >>> result = oracle.compute(
        ...     input_atom_array=structure,
        ...     example_id="design_001",
        ...     out_dir=Path("/outputs"),
        ... )
        >>> # Returns: {"plddt": 85.0, "iptm": 92.0, ...}
    """

    _engine_cache: ClassVar[dict[Path, RF3InferenceEngine]] = {}
    _engine_ref_counts: ClassVar[dict[Path, int]] = {}

    def __init__(
        self,
        ckpt_path: str | Path,
        num_steps: int = 50,
        n_recycles: int = 5,
        extra_annots_to_symm_resolve: list[str] | None = None,
        oracle_name: str = "rf3",
        template_annotations: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(oracle_name=oracle_name, template_annotations=template_annotations, **kwargs)
        self.ckpt_path = Path(ckpt_path)
        self.num_steps = num_steps
        self.n_recycles = n_recycles
        self.extra_annots_to_symm_resolve = extra_annots_to_symm_resolve

    # ------------------------------------------------------------------
    # Engine management (class-level cache, ref-counted)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_current_device() -> int:
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return 0

    @property
    def engine(self) -> RF3InferenceEngine:
        """Shared RF3 engine (created once per checkpoint path)."""
        if self.ckpt_path not in RF3Oracle._engine_cache:
            current_device = self._get_current_device()
            logger.info(f"Creating RF3 engine for {self.ckpt_path} on device {current_device}")
            RF3Oracle._engine_cache[self.ckpt_path] = RF3InferenceEngine(
                ckpt_path=self.ckpt_path,
                n_recycles=self.n_recycles,
                diffusion_batch_size=1,
                num_steps=self.num_steps,
                seed=None,
                load_weights=False,
                metrics_cfg="default",
                num_nodes=1,
                devices_per_node=[current_device],
                verbose=True,
            )
        return RF3Oracle._engine_cache[self.ckpt_path]

    def setup(self) -> None:
        """Load RF3 weights for validation (shared across instances)."""
        RF3Oracle._engine_ref_counts[self.ckpt_path] = RF3Oracle._engine_ref_counts.get(self.ckpt_path, 0) + 1
        if not self.engine._weights_loaded:
            self.engine.load_weights()
            logger.info(f"Loaded RF3 weights from {self.ckpt_path}")

    def teardown(self) -> None:
        """Unload RF3 weights when the last reference is released."""
        RF3Oracle._engine_ref_counts[self.ckpt_path] -= 1
        if RF3Oracle._engine_ref_counts[self.ckpt_path] == 0:
            self._cleanup_rf3()
            logger.info("Unloaded RF3 weights (last reference)")
        else:
            logger.info(
                f"RF3 weights still in use ({RF3Oracle._engine_ref_counts[self.ckpt_path]} references remaining)"
            )

    def _cleanup_rf3(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        engine = RF3Oracle._engine_cache.get(self.ckpt_path)
        if engine is not None and engine._weights_loaded:
            engine.unload_weights()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Input preparation helpers
    # ------------------------------------------------------------------

    def _prepare_atom_array_for_rf3(self, atom_array: AtomArray) -> AtomArray:
        """Prepare AtomArray for RF3 inference."""
        if self.template_annotations:
            template_mask = np.zeros(atom_array.array_length(), dtype=bool)
            for annot_name in self.template_annotations:
                assert annot_name in atom_array.get_annotation_categories(), (
                    f"Annotation '{annot_name}' not found in AtomArray"
                )
                annot = atom_array.get_annotation(annot_name)
                assert annot.dtype == bool, f"Annotation '{annot_name}' must be boolean, got {annot.dtype}"
                template_mask |= annot
            atom_array.set_annotation("is_input_file_templated", template_mask)

        # Strip annotations the RF3 pipeline regenerates internally (and raises if they exist)
        for annotation in ["token_id", "atom_id", "coord_to_be_noised"]:
            if annotation in atom_array.get_annotation_categories():
                atom_array.del_annotation(annotation)

        atom_array.set_annotation("transformation_id", np.full(atom_array.array_length(), "1", dtype=str))
        atom_array = add_id_and_entity_annotations(atom_array)
        atom_array = add_chain_iid_annotation(atom_array)
        atom_array = add_pn_unit_iid_annotation(atom_array)
        atom_array = add_molecule_iid_annotation(atom_array)
        return atom_array

    # ------------------------------------------------------------------
    # OracleMetric implementation
    # ------------------------------------------------------------------

    def fold_one(self, job: OracleJobInput) -> dict:
        """Fold one design with RF3 in-process.

        Writes the refolded CIF and the confidence JSON to canonical paths and
        returns the metrics dict. RF3 early-stop is a legitimate signal and
        persists :py:data:`_EARLY_STOP_METRICS`; any other inference failure
        raises so the caller sees the real error.
        """
        atom_array = job.atom_array
        example_id = job.example_id
        out_dir = job.out_dir
        chain_of_interest = job.chain_of_interest

        # Add template annotations with annotators if necessary
        if self.template_annotations is not None:
            annotations_to_ensure = [annot for annot in self.template_annotations if annot in ANNOTATOR_REGISTRY]
            if annotations_to_ensure:
                ensure_annotations(atom_array, *annotations_to_ensure)

        original_device = torch.cuda.current_device()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Configure engine for this call (no inner metric manager for the oracle)
        self.engine._metrics_cfg = "default"
        self.engine.transform_overrides["n_recycles"] = self.n_recycles
        self.engine.verbose = False
        try:
            self.engine._construct_pipeline(self.engine.cfg)
        finally:
            self.engine.verbose = True

        # Update the number of inference timesteps (only needed in shadow model, done in both for future-proofing)
        self.engine.trainer.state["model"].model.inference_sampler.num_timesteps = self.num_steps
        self.engine.trainer.state["model"].shadow.inference_sampler.num_timesteps = self.num_steps

        prepared_array = self._prepare_atom_array_for_rf3(atom_array.copy())

        try:
            inference_input = InferenceInput.from_atom_array(
                prepared_array,
                example_id=example_id,
                extra_info={"extra_annots_to_symm_resolve": self.extra_annots_to_symm_resolve},
                input_options=InputOptions(add_missing_atoms=False, long_bond_policy="keep"),
            )

            results = self.engine.run(
                inputs=[inference_input],
                out_dir=None,
                annotate_b_factor_with_plddt=True,
            )

            if results.get("early_stopped"):
                logger.warning(f"RF3 early-stopped for {example_id} — persisting early-stop sentinel metrics")
                metrics = _EARLY_STOP_METRICS.copy()
            else:
                result = results[example_id][0]
                metrics = dict(result.metrics)
                summary_confidences = result.summary_confidences or {}
                chains_in_order = list(np.unique(result.atom_array.chain_iid))

                # Resolve chain of interest
                if chain_of_interest is not None:
                    coi_unambiguous = True
                else:
                    chain_of_interest, coi_unambiguous = get_chain_of_interest(prepared_array)
                metrics["chain_of_interest"] = chain_of_interest if coi_unambiguous else None

                metrics.update(
                    extract_confidence_metrics(
                        summary_confidences=summary_confidences,
                        chain_of_interest=chain_of_interest,
                        chains_in_order=chains_in_order,
                    )
                )

                if out_dir is not None and result.atom_array is not None:
                    cif_path = self.refolded_cif_path(out_dir, example_id)
                    cif_path.parent.mkdir(parents=True, exist_ok=True)
                    to_cif_file(align_atom_array_to_input_frame(result.atom_array), cif_path)
        finally:
            torch.cuda.set_device(original_device)

        self._save_confidence_json(metrics, self.confidence_path(out_dir, example_id))
        return metrics

    def fold_batch(self, jobs: list[OracleJobInput]) -> int:
        n_ok = 0
        for job in jobs:
            if self.already_refolded(job.out_dir, job.example_id):
                continue
            self.fold_one(job)
            n_ok += 1
        return n_ok
