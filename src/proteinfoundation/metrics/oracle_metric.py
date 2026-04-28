"""Base class for oracle refolding metrics (RF3Oracle, AF3Oracle).

An oracle metric folds a designed structure and returns confidence scores plus the
path to the saved refolded CIF. It is intentionally decoupled from post-refolding
comparison metrics (StructureComparisonMetric) — those run separately in the
refolding MetricManager.

Resumability: after each fold, the oracle writes a JSON file containing confidence
scores and the refolded CIF path. On restart, the presence of this file signals that
folding is already complete; the JSON is loaded and returned directly.
"""

import hashlib
import json
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.common import exists
from biotite.structure import AtomArray

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class OracleMetric(Metric):
    """Base class for structure prediction oracles (RF3Oracle, AF3Oracle).

    Subclasses implement :py:meth:`fold_single_atom_array` and inherit the
    resumability / JSON-save logic from this class.

    Args:
        oracle_name: Short label used to name the output subdirectory and JSON files
            (e.g. ``"rf3"`` → ``{out_dir}/rf3_refolding/``).
        template_annotations: Annotation name(s) on the input AtomArray to use for
            templating. Multiple names are OR'd together. Defaults to ``None``.
    """

    def __init__(
        self,
        oracle_name: str,
        template_annotations: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.oracle_name = oracle_name
        self.template_annotations = (
            [template_annotations] if isinstance(template_annotations, str) else template_annotations
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _oracle_dir(self, out_dir: Path) -> Path:
        return out_dir / f"{self.oracle_name}_refolding"

    def _shard_subdir(self, example_id: str) -> str:
        return hashlib.md5(example_id.encode()).hexdigest()[:3]

    def confidence_path(self, out_dir: Path, example_id: str) -> Path:
        """Return the path to the on-the-fly confidence JSON for this example."""
        return self._oracle_dir(out_dir) / self._shard_subdir(example_id) / f"{example_id}_confidences.json"

    def refolded_cif_path(self, out_dir: Path, example_id: str) -> Path:
        """Return the canonical path at which the refolded CIF will be saved."""
        return self._oracle_dir(out_dir) / self._shard_subdir(example_id) / f"{example_id}_refolded.cif"

    def already_refolded(self, out_dir: Path, example_id: str) -> bool:
        """Return True if a confidence JSON for this example already exists."""
        return self.confidence_path(out_dir, example_id).exists()

    def _save_confidence_json(self, confidences: dict, path: Path) -> None:
        """Write confidence dict to JSON (creates parent dirs as needed)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        json_confidences = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in confidences.items()}
        path.write_text(json.dumps(json_confidences, indent=2))

    def _load_confidence_json(self, path: Path) -> dict:
        """Load and return the confidence dict from a JSON file."""
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def fold_single_atom_array(
        self,
        atom_array: AtomArray,
        example_id: str,
        out_dir: Path,
        chain_of_interest: str | None,
    ) -> dict:
        """Fold one structure and return confidence metrics.

        Subclasses must save the refolded CIF at
        ``self.refolded_cif_path(out_dir, example_id)`` before returning, so that
        the caller can parse it for subsequent refolding metrics.

        Args:
            atom_array: The design structure to refold.
            example_id: Identifier used for file naming.
            out_dir: Base output directory (oracle CIFs go under
                ``{out_dir}/{oracle_name}_refolding/{shard}/``).
            chain_of_interest: Pre-computed chain of interest, or ``None``.

        Returns:
            Flat dict with all confidence values.
        """

    # ------------------------------------------------------------------
    # Metric interface
    # ------------------------------------------------------------------

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "input_atom_array": ("predicted_atom_array",),
            "example_id": ("extra_info", "example_id"),
            "out_dir": ("extra_info", "output_base_dir"),
            "chain_of_interest": ("extra_info", "chain_of_interest"),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset(["chain_of_interest"])

    def compute(
        self,
        input_atom_array: AtomArray,
        example_id: str,
        out_dir: Path | None,
        chain_of_interest: str | None = None,
    ) -> dict:
        """Refold the structure, returning confidence metrics and the refolded CIF path.

        If a confidence JSON for ``example_id`` already exists in
        ``{out_dir}/{oracle_name}_refolding/``, loads it and returns immediately
        (skips folding — enables resumability across interrupted runs).

        Args:
            input_atom_array: Design structure to refold.
            example_id: Identifier for file naming and resumability check.
            out_dir: Base output directory. If ``None``, folding runs but nothing is
                saved (not recommended for production use).
            chain_of_interest: Pre-computed chain of interest label, or ``None``.

        Returns:
            Flat dict with confidence values (e.g. ``plddt``, ``iptm``, ``pae``).
        """
        if exists(out_dir):
            out_dir = Path(out_dir)
            if self.already_refolded(out_dir, example_id):
                result = self._load_confidence_json(self.confidence_path(out_dir, example_id))
                result["refolded_structure_path"] = str(self.refolded_cif_path(out_dir, example_id))
                result["confidence_json_path"] = str(self.confidence_path(out_dir, example_id))
                return result
        else:
            logger.warning(
                f"OracleMetric.compute called without out_dir for {example_id}; "
                "refolded structure will not be saved and resumability is disabled."
            )

        result = self.fold_single_atom_array(
            atom_array=input_atom_array,
            example_id=example_id,
            out_dir=out_dir,
            chain_of_interest=chain_of_interest,
        )

        if out_dir is not None:
            self._save_confidence_json(result, self.confidence_path(out_dir, example_id))
            result["refolded_structure_path"] = str(self.refolded_cif_path(out_dir, example_id))
            result["confidence_json_path"] = str(self.confidence_path(out_dir, example_id))

        return result
