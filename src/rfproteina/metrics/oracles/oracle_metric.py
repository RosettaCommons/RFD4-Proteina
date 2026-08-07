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
→ load JSON; cache miss → apply :attr:`OracleMetric.pre_oracle_transform` →
call ``fold_one``.
"""

import dataclasses
import json
import tempfile
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from biotite.structure import AtomArray

from rfproteina.metrics.metric import Metric
from rfproteina.metrics.oracles.pre_oracle_transform import PreOracleTransform
from rfproteina.utils.common import design_output_dir, ensure_dir
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass(slots=True)
class OracleJobInput:
    """One design queued for :py:meth:`OracleMetric.fold_one` /
    :py:meth:`OracleMetric.fold_batch`."""

    atom_array: AtomArray
    example_id: str
    out_dir: Path
    save_id: str | None = None
    chain_of_interest: str | None = None
    # Per-chain a3m paths (plain or gzipped), keyed by the input's chain_id; ``None`` folds single-sequence.
    msa_paths_by_chain_id: dict[str, str | Path] | None = None


def _msas_for_present_chains(job: OracleJobInput, transformed: AtomArray, label: str) -> dict[str, str | Path] | None:
    """Drop MSA entries whose chain a :py:class:`PreOracleTransform` removed or renamed."""
    if not job.msa_paths_by_chain_id:
        return job.msa_paths_by_chain_id
    present = set(np.unique(transformed.chain_id).tolist())
    kept = {chain_id: path for chain_id, path in job.msa_paths_by_chain_id.items() if chain_id in present}
    if dropped := sorted(set(job.msa_paths_by_chain_id) - set(kept)):
        logger.warning(
            f"{label}: dropping MSAs for chain(s) {dropped}, which the pre-oracle transform removed or renamed."
        )
    return kept or None


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

# Emitted per chain pair as `<prefix>_<chain_i>_<chain_j>`, forward/reverse averaged.
# Format: (af3_key, output_key_prefix, scale).
_CHAIN_PAIR_METRICS: list[tuple[str, str, float]] = [
    ("chain_pair_pae_min", "min_ipae", 1.0),
    ("chain_pair_iptm", "iptm", 100.0),
]


def extract_confidence_metrics(
    summary_confidences: dict,
    chain_of_interest: Any | None,
    chains_in_order: list,
) -> dict[str, float]:
    """Translate an AF3-shaped summary-confidences dict into the flat metric keys
    the val pipeline expects (``overall_plddt``, ``plddt_coi``, ``ptm_coi``, …).

    Also emits one ``min_ipae_<chain_i>_<chain_j>`` key per chain pair, which ``min_ipae_coi``
    cannot express: it is one ``np.min`` over the COI's pairs, so the tightest hides the rest.

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

    # Emitted for all pairs, not just the COI's, and before the reduction that collapses them.
    for src, dst_prefix, scale in _CHAIN_PAIR_METRICS:
        matrix = summary_confidences.get(src)
        if matrix is None:
            continue
        for i, j in combinations(range(len(chains_in_order)), 2):
            # chains_in_order may carry None where a label failed to map back.
            if chains_in_order[i] is None or chains_in_order[j] is None:
                continue
            vals = [v for v in (matrix[i][j], matrix[j][i]) if v is not None]
            if vals:
                metrics[f"{dst_prefix}_{chains_in_order[i]}_{chains_in_order[j]}"] = float(np.mean(vals)) * scale

    if chain_of_interest is None or chain_of_interest not in chains_in_order:
        return metrics

    coi_idx = list(chains_in_order).index(chain_of_interest)
    n_chains = len(chains_in_order)

    for src, dst, scale in _COI_CHAIN_METRICS:
        v = summary_confidences.get(src)
        if v is not None and v[coi_idx] is not None:
            metrics[dst] = float(v[coi_idx]) * scale

    for src, dst, reduce_fn, scale in _COI_INTERFACE_METRICS:
        matrix = summary_confidences.get(src)
        if matrix is None:
            continue
        vals = _coi_interface_vals(matrix, coi_idx, n_chains)
        if vals:
            metrics[dst] = float(reduce_fn(vals)) * scale

    return metrics


class OracleMetric(Metric):
    """Base class for structure-prediction oracles (RF3Oracle, AF3Oracle).

    Subclasses implement :py:meth:`fold_one` (single design) and
    :py:meth:`fold_batch` (many designs at once); both write the refolded CIF
    and confidence JSON to canonical paths on disk. :py:meth:`compute` is the
    framework-facing entry point and routes between cache hit (load JSON) and
    cache miss (apply :attr:`pre_oracle_transform`, call ``fold_one``, then
    load JSON).

    Args:
        oracle_name: Short label included in output filenames
            (e.g. ``"rf3"`` → ``…/{example_id}/{example_id}_rf3_refolded.cif``).
        pre_oracle_transform: Optional transform applied to the generated
            AtomArray before refolding. May fan one design out into several
            named outputs; see :py:mod:`~rfproteina.metrics.oracles.pre_oracle_transform`
            and :py:meth:`apply_transform`. Defaults to ``None`` (no transform).
        template_query: AtomWorks query string selecting which atoms to pass to the
            oracle as a structural template (e.g.
            ``"mask_coordinate_1_atom & ~is_nucleic_acid"``). ``None`` disables
            templating. Defaults to ``None``.
        frac_under_thresholds: Per-output-key thresholds for which to emit
            ``frac_under_<T>`` rows in :py:meth:`aggregate`. Keys are bare output
            names (e.g. ``"min_ipae_coi"``), values are threshold lists. Uses
            strict ``<``. Defaults to ``None`` (no threshold rows).
    """

    def __init__(
        self,
        oracle_name: str,
        pre_oracle_transform: PreOracleTransform | None = None,
        template_query: str | None = None,
        frac_under_thresholds: dict[str, list[float]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.oracle_name = oracle_name
        self.pre_oracle_transform = pre_oracle_transform
        self.template_query = template_query
        self.frac_under_thresholds = frac_under_thresholds or {}

    def aggregate(self, column: str, values: pd.Series) -> dict[str, float]:
        """Default reductions plus per-column ``frac_under_<T>`` rows.

        A :attr:`pre_oracle_transform` fans this oracle's columns out to
        ``f"{save_id}.{output_key}"``, but thresholds stay keyed by the bare
        output name — one threshold applies to every output of the fan-out — so
        the save_id is stripped before the lookup.
        """
        result = super().aggregate(column, values)
        output_key = column.rpartition(".")[2]
        for thresh in self.frac_under_thresholds.get(output_key, []):
            result[f"frac_under_{thresh}"] = float((values < thresh).mean())
        return result

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def output_stem(self, example_id: str, save_id: str | None = None) -> str:
        """Filename stem shared by one refold's CIF and confidence JSON."""
        return (
            f"{example_id}_{save_id}_{self.oracle_name}" if save_id is not None else f"{example_id}_{self.oracle_name}"
        )

    def confidence_path(self, out_dir: Path, example_id: str, save_id: str | None = None) -> Path:
        """Return the path to the on-the-fly confidence JSON for this example."""
        return design_output_dir(out_dir, example_id) / f"{self.output_stem(example_id, save_id)}_confidences.json"

    def refolded_cif_path(self, out_dir: Path, example_id: str, save_id: str | None = None) -> Path:
        """Return the canonical path at which the refolded CIF will be saved."""
        return design_output_dir(out_dir, example_id) / f"{self.output_stem(example_id, save_id)}_refolded.cif"

    def already_refolded(self, out_dir: Path, example_id: str, save_id: str | None = None) -> bool:
        """Return True if a confidence JSON for this example already exists."""
        return self.confidence_path(out_dir, example_id, save_id).exists()

    def list_save_ids(self) -> list[str] | None:
        """Return the save_ids this oracle fans each design into, or ``None`` if it folds it as-is.

        What gets folded is decided by the currently configured
        :attr:`pre_oracle_transform`, never by what an earlier run happens to
        have left on disk: a save_id this oracle no longer emits is simply not
        looked for, so its stale files are ignored rather than reloaded.
        """
        if self.pre_oracle_transform is None:
            return None
        return self.pre_oracle_transform.expected_save_ids()

    def _save_confidence_json(self, confidences: dict, path: Path) -> None:
        """Write confidence dict to JSON (creates parent dirs as needed)."""
        ensure_dir(path.parent)
        json_confidences = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in confidences.items()}
        path.write_text(json.dumps(json_confidences, indent=2))

    def _load_confidence_json(self, path: Path) -> dict:
        """Load and return the confidence dict from a JSON file."""
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Lifecycle hooks (override in subclasses that load heavy resources)
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Set up oracle resources for validation (called once per epoch)."""
        if self.pre_oracle_transform is not None:
            self.pre_oracle_transform.setup()

    def teardown(self) -> None:
        """Release oracle resources after validation."""
        if self.pre_oracle_transform is not None:
            self.pre_oracle_transform.teardown()

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def fold_one(self, job: OracleJobInput) -> dict:
        """Fold one design and persist its results to disk.

        Implementations must write:

        - the refolded CIF to :py:meth:`refolded_cif_path` ``(out_dir, job.example_id, job.save_id)``
        - the confidence JSON to :py:meth:`confidence_path` ``(out_dir, job.example_id, job.save_id)``

        Returns the metrics dict (the same one that was just persisted).
        """

    @abstractmethod
    def fold_batch(self, jobs: list[OracleJobInput]) -> int:
        """Fold every design in ``jobs`` and persist the results to disk.

        Returns:
          Number of jobs that have a refold on disk once this call returns.
          Implementations that skip already-cached designs must still count
          them: callers compare this against ``len(jobs)`` to assert coverage.
        """

    def apply_transform(self, jobs: list[OracleJobInput]) -> list[OracleJobInput]:
        """Apply :attr:`pre_oracle_transform` to each job's atom_array (if set).

        Each :py:class:`~rfproteina.metrics.oracles.pre_oracle_transform.PreOracleOutput`
        becomes its own :py:class:`OracleJobInput`, carrying the design's
        ``example_id`` and that output's ``save_id`` — so all of one design's
        refolds land in one directory, told apart by
        :py:meth:`output_stem`. An output that names its chain of interest
        overrides the job's, since the incoming one describes the untransformed
        design.

        Anything that goes wrong for one design — ``forward`` raising, or
        returning save_ids other than the declared ones — is logged and that
        design dropped rather than raised on: the caller folds a whole rank's
        designs in one call, and under DDP an exception on the one rank holding
        the offending design strands every other rank in a collective. The
        ``expected_save_ids()`` check below is the exception, and is safe to
        raise from: it reads only this transform's configuration, so every rank
        fails on it identically.

        Raises:
            ValueError: If the transform declares no save_ids, or save_ids that
                are blank, duplicated, or contain a path separator — each of
                which would silently collide or be unrecoverable on disk.
        """
        if self.pre_oracle_transform is None:
            return jobs
        transform_name = type(self.pre_oracle_transform).__name__
        expected = self.pre_oracle_transform.expected_save_ids()
        if not expected or len(set(expected)) != len(expected) or any("/" in s or "\\" in s or not s for s in expected):
            raise ValueError(
                f"{transform_name}.expected_save_ids() returned unusable save_ids {expected!r}: "
                "expected at least one, all distinct, non-empty and free of path separators"
            )
        expanded: list[OracleJobInput] = []
        for job in jobs:
            try:
                outputs = list(self.pre_oracle_transform.forward(job.atom_array))
            except Exception as e:
                logger.warning(f"{transform_name} failed for {job.example_id!r}; skipping that design: {e}")
                continue
            save_ids = [output.save_id for output in outputs]
            if sorted(save_ids) != sorted(expected):
                logger.error(
                    f"{transform_name}.forward returned save_ids {save_ids!r} for {job.example_id!r}, but its "
                    f"expected_save_ids() declares {expected!r}; skipping that design. The two must agree, since "
                    "the fold writes what forward returns and every read looks for what expected_save_ids declares."
                )
                continue
            for output in outputs:
                expanded.append(
                    dataclasses.replace(
                        job,
                        atom_array=output.atom_array,
                        save_id=output.save_id,
                        chain_of_interest=output.chain_of_interest or job.chain_of_interest,
                        msa_paths_by_chain_id=_msas_for_present_chains(
                            job, output.atom_array, f"{job.example_id}[{output.save_id}]"
                        ),
                    )
                )
        return expanded

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
        If :attr:`pre_oracle_transform` is configured, the returned dict
        instead holds one such group per output, each under a
        ``f"{save_id}."`` key prefix (see :py:meth:`apply_transform`) — even
        when the transform only produced one output, so callers never have to
        guess which shape to expect from a given oracle config.

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
        cached = self._load_if_cached(out_dir, example_id)
        if cached is not None:
            # Fast path: every output already on disk, so just load and return them
            return cached

        # Slow path: refold then load
        return self._fold_then_load(input_atom_array, example_id, out_dir, chain_of_interest)

    def _load_if_cached(self, out_dir: Path, example_id: str) -> dict | None:
        """Return this design's result if every output it should have is already on disk, else ``None``.

        The outputs to look for come from
        :py:meth:`~rfproteina.metrics.oracles.pre_oracle_transform.PreOracleTransform.expected_save_ids`,
        so a cache hit never needs to re-run the transform (or
        ``input_atom_array``).
        """
        save_ids = self.list_save_ids() or [None]
        if not all(self.already_refolded(out_dir, example_id, save_id) for save_id in save_ids):
            return None
        return self._load_result(out_dir, example_id)

    def _fold_then_load(
        self,
        atom_array: AtomArray,
        example_id: str,
        out_dir: Path,
        chain_of_interest: str | None,
    ) -> dict:
        """Slow path of :py:meth:`compute`: apply transform, fold, then read back.

        A single job folds via :py:meth:`fold_one` — unchanged from before
        :py:class:`~rfproteina.metrics.oracles.pre_oracle_transform.PreOracleTransform`
        existed. A fanned-out design folds via :py:meth:`fold_batch`, which
        subclasses already implement to skip individually-cached outputs.
        """
        jobs = self.apply_transform(
            [
                OracleJobInput(
                    atom_array=atom_array,
                    example_id=example_id,
                    out_dir=out_dir,
                    chain_of_interest=chain_of_interest,
                )
            ]
        )
        if len(jobs) == 1:
            self.fold_one(jobs[0])
        else:
            self.fold_batch(jobs)
        return self._load_result(out_dir, example_id)

    def _load_compute_result(self, out_dir: Path, example_id: str, save_id: str | None = None) -> dict:
        """Load one refold's confidence JSON from disk and stamp on its canonical paths."""
        result = self._load_confidence_json(self.confidence_path(out_dir, example_id, save_id))
        result["refolded_structure_path"] = str(self.refolded_cif_path(out_dir, example_id, save_id))
        result["confidence_json_path"] = str(self.confidence_path(out_dir, example_id, save_id))
        return result

    def _load_result(self, out_dir: Path, example_id: str) -> dict:
        """Load this design's folded output(s), one ``f"{save_id}."``-prefixed group per output.

        An output whose fold produced nothing is skipped, so one failed member of
        a fan-out costs only its own keys rather than raising and taking every
        sibling's (and every other oracle's) metrics down with it.
        """
        save_ids = self.list_save_ids()
        if save_ids is None:
            return self._load_compute_result(out_dir, example_id)
        result: dict = {}
        for save_id in save_ids:
            if not self.already_refolded(out_dir, example_id, save_id):
                logger.warning(f"{example_id}: no {self.oracle_name} refold for save_id {save_id!r}; skipping it")
                continue
            result.update(
                {
                    f"{save_id}.{key}": value
                    for key, value in self._load_compute_result(out_dir, example_id, save_id).items()
                }
            )
        return result
