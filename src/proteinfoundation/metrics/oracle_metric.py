"""Base class for oracle refolding metrics (RF3Oracle, AF3Oracle).

An oracle metric folds designed structures and returns confidence scores plus
the path to the saved refolded CIF. It is intentionally decoupled from
post-refolding comparison metrics (StructureComparisonMetric) — those run
separately in the refolding MetricManager.

Each subclass implements two abstract methods that share an output contract:

- :py:meth:`OracleMetric.fold_one`   — fold a single design.
- :py:meth:`OracleMetric.fold_batch` — fold many designs (subclasses that can
  amortise setup, like AF3 dispatching one subprocess for the whole batch,
  override this; otherwise it's just a loop over ``fold_one``).

Both write the refolded CIF and a confidence JSON to canonical paths on disk.

The framework-facing :py:meth:`compute` then routes between them: cache hit
→ load JSON; cache miss → call ``fold_one``.
"""

import json
import tempfile
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io.utils.io_utils import apply_sharding_pattern
from atomworks.ml.utils.misc import hash_sequence
from biotite.structure import AtomArray

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass(slots=True)
class OracleJobInput:
    """One design queued for :py:meth:`OracleMetric.fold_one` /
    :py:meth:`OracleMetric.fold_batch`."""

    atom_array: AtomArray
    example_id: str
    out_dir: Path
    chain_of_interest: str | None = None


def _coi_interface_vals(matrix: list, coi_idx: int, n_chains: int) -> list:
    """Bidirectional interface values between the COI and every other chain.

    Used by :py:func:`extract_confidence_metrics` to average forward / reverse
    AF3-style chain-pair matrices.
    """
    vals = []
    for j in range(n_chains):
        if j == coi_idx:
            continue
        v_forward = matrix[coi_idx][j]
        v_reverse = matrix[j][coi_idx]
        if v_forward is not None and v_reverse is not None:
            vals.append((v_forward + v_reverse) / 2.0)
        elif v_forward is not None:
            vals.append(v_forward)
        elif v_reverse is not None:
            vals.append(v_reverse)
    return vals


# AF3 → flat-metrics translation tables, in the order the val pipeline reads them.
# Format: (af3_key, output_key, scale). Scale 100.0 lifts AF3's 0..1 plddt/ptm to 0..100.
_GLOBAL_METRICS: list[tuple[str, str, float]] = [
    ("overall_plddt", "overall_plddt", 100.0),
    ("overall_pae", "overall_pae", 1.0),
    ("ptm", "overall_ptm", 100.0),
    ("iptm", "overall_iptm", 100.0),
    ("ranking_score", "ranking_score", 1.0),
    ("fraction_disordered", "fraction_disordered", 1.0),
    ("has_clash", "has_clash", 1.0),
]

# Per-chain values pulled at chain_X[coi_idx].
_COI_CHAIN_METRICS: list[tuple[str, str, float]] = [
    ("chain_plddt", "plddt_coi", 100.0),
    ("chain_ptm", "ptm_coi", 100.0),
    ("chain_pae", "pae_coi", 1.0),
]

# Interface values reduced over chain_pair_X[coi_idx][j != coi_idx].
# Format: (af3_key, output_key, reduction, scale).
_COI_INTERFACE_METRICS: list[tuple[str, str, Callable, float]] = [
    ("chain_pair_pae", "ipae_coi", np.mean, 1.0),
    ("chain_pair_pae_min", "min_ipae_coi", np.min, 1.0),
    ("chain_pair_iptm", "iptm_coi", np.mean, 100.0),
]


def extract_confidence_metrics(
    summary_confidences: dict,
    chain_of_interest: Any | None,
    chains_in_order: list,
) -> dict[str, float]:
    """Translate an AF3-shaped summary-confidences dict into the flat metric keys
    the val pipeline expects (``overall_plddt``, ``plddt_coi``, ``ptm_coi``, …).

    Both :class:`AF3Oracle` and :class:`RF3Oracle` produce summary-confidences
    in the same shape and call this helper before persisting the result.

    Args:
        summary_confidences: AF3-shaped dict with keys like ``overall_plddt``,
            ``ptm``, ``iptm``, ``chain_plddt``, ``chain_pair_pae``, etc. Values
            in 0..1 are scaled to 0..100 here for the ``plddt`` / ``ptm`` keys.
        chain_of_interest: Chain label for the COI metrics, or ``None``.
        chains_in_order: Ordered chain labels matching the indexing of any
            ``chain_*`` lists / ``chain_pair_*`` matrices in ``summary_confidences``.
    """
    metrics: dict[str, float] = {}

    for src, dst, scale in _GLOBAL_METRICS:
        v = summary_confidences.get(src)
        if v is not None:
            metrics[dst] = float(v) * scale

    if chain_of_interest is None or chain_of_interest not in chains_in_order:
        return metrics

    coi_idx = list(chains_in_order).index(chain_of_interest)
    n_chains = len(chains_in_order)

    for src, dst, scale in _COI_CHAIN_METRICS:
        v = summary_confidences.get(src)
        if v is not None:
            metrics[dst] = float(v[coi_idx]) * scale

    for src, dst, reduce_fn, scale in _COI_INTERFACE_METRICS:
        matrix = summary_confidences.get(src)
        if matrix is None:
            continue
        vals = _coi_interface_vals(matrix, coi_idx, n_chains)
        if vals:
            metrics[dst] = float(reduce_fn(vals)) * scale

    return metrics


def _example_shard_path(out_dir: Path, example_id: str) -> Path:
    """Return a sharded subdirectory under ``out_dir`` for the given example."""
    return out_dir / apply_sharding_pattern(hash_sequence(example_id), "/0:2/0:8/")


def example_output_dir(out_dir: Path, example_id: str) -> Path:
    """Return a sharded per-example output directory under ``out_dir``.

    Args:
        out_dir: Top-level output base directory.
        example_id: Unique example identifier.
    """
    return _example_shard_path(out_dir, example_id)


def example_chain_dir(
    out_dir: Path,
    example_id: str,
    dataset_name: str | None = None,
    task: str | None = None,
) -> Path:
    """Return the foldseek chain CIF directory for the given example.

    Args:
        out_dir: Top-level output base directory.
        example_id: Unique example identifier.
        dataset_name: Dataset name (groups foldseek clustering jobs by dataset).
        task: Task label (e.g. ``"unconditional"``).
    """
    subdir = out_dir / "foldseek_input_chains"
    if dataset_name is not None or task is not None:
        ds_seg = hash_sequence(dataset_name)[:8] if dataset_name is not None else "unknown_dataset"
        task_seg = task if task is not None else "unknown_task"
        subdir = subdir / ds_seg / task_seg
    return _example_shard_path(subdir, example_id)


class OracleMetric(Metric):
    """Base class for structure-prediction oracles (RF3Oracle, AF3Oracle).

    Subclasses implement :py:meth:`fold_one` (single design) and
    :py:meth:`fold_batch` (many designs at once); both write the refolded CIF
    and confidence JSON to canonical paths on disk. :py:meth:`compute` is the
    framework-facing entry point and routes between cache hit (load JSON) and
    cache miss (call ``fold_one`` then load JSON).

    Args:
        oracle_name: Short label included in output filenames
            (e.g. ``"rf3"`` → ``…/{digest8}/{example_id}_rf3_refolded.cif``).
        template_annotations: Annotation name(s) on the input AtomArray to use
            for templating. Multiple names are OR'd together. Defaults to ``None``.
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

    def confidence_path(self, out_dir: Path, example_id: str) -> Path:
        """Return the path to the on-the-fly confidence JSON for this example."""
        return example_output_dir(out_dir, example_id) / f"{example_id}_{self.oracle_name}_confidences.json"

    def refolded_cif_path(self, out_dir: Path, example_id: str) -> Path:
        """Return the canonical path at which the refolded CIF will be saved."""
        return example_output_dir(out_dir, example_id) / f"{example_id}_{self.oracle_name}_refolded.cif"

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
    def fold_one(self, job: OracleJobInput) -> dict:
        """Fold one design and persist its results to disk.

        Implementations must write:

        - the refolded CIF to :py:meth:`refolded_cif_path` ``(out_dir, example_id)``
        - the confidence JSON to :py:meth:`confidence_path` ``(out_dir, example_id)``

        Returns the metrics dict (the same one that was just persisted).
        """

    @abstractmethod
    def fold_batch(self, jobs: list[OracleJobInput]) -> int:
        """Fold every design in ``jobs`` and persist the results to disk."""

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
        """Refold the structure (or load from cache) and return its metrics.

        Returns ``{...confidence keys..., "refolded_structure_path", "confidence_json_path"}``.

        ``out_dir=None`` allocates a tempdir scoped to this call; cache files
        are removed before return, so the returned ``refolded_structure_path``
        will not exist on disk afterwards. Pass an explicit ``out_dir`` to
        keep the artifacts.
        """
        if out_dir is None:
            with tempfile.TemporaryDirectory() as tmp:
                # No caching: fold into a tempdir and read back
                return self._fold_then_load(input_atom_array, example_id, Path(tmp), chain_of_interest)

        out_dir = Path(out_dir)
        if self.already_refolded(out_dir, example_id):
            # Fast path: confidence JSON already exists, so just load and return it
            return self._load_compute_result(out_dir, example_id)

        # Slow path: refold then load
        return self._fold_then_load(input_atom_array, example_id, out_dir, chain_of_interest)

    def _fold_then_load(
        self,
        atom_array: AtomArray,
        example_id: str,
        out_dir: Path,
        chain_of_interest: str | None,
    ) -> dict:
        """Slow path of :py:meth:`compute`: run :py:meth:`fold_one` then read back."""
        self.fold_one(
            OracleJobInput(
                atom_array=atom_array,
                example_id=example_id,
                out_dir=out_dir,
                chain_of_interest=chain_of_interest,
            )
        )
        return self._load_compute_result(out_dir, example_id)

    def _load_compute_result(self, out_dir: Path, example_id: str) -> dict:
        """Load the confidence JSON from disk and stamp on the canonical paths."""
        result = self._load_confidence_json(self.confidence_path(out_dir, example_id))
        result["refolded_structure_path"] = str(self.refolded_cif_path(out_dir, example_id))
        result["confidence_json_path"] = str(self.confidence_path(out_dir, example_id))
        return result
