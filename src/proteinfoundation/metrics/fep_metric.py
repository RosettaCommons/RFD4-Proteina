"""RBFE (Relative Binding Free Energy) metrics using TMD.

Provides two Foundry Metric implementations:

- ``FEPMetric``: Single-edge RBFE between two ligands in a protein pocket.
  Runs synchronously per sample via the standard ``MetricManager`` flow.
- ``FEPBatchMetric``: Accumulates ligand pairs during validation and runs
  all edges in parallel via MPS at teardown for higher GPU throughput.

Requires TMD (https://github.com/tmd-industries/tmd) with CUDA custom ops.
TMD compiles CUDA kernels via CMake+nvcc during ``uv pip install .``.
On first GPU execution, JAX also JIT-compiles additional kernels, making
the first run significantly slower (~5-10 min extra). Run the smoke test
to warm the cache before production use::

    pytest tests/metrics/test_fep_metric.py::test_fep_smoke -v -s

Installation from source (requires CUDA toolkit with nvcc, CMake >= 3.24.3)::

    git clone https://github.com/tmd-industries/tmd.git
    cd tmd
    uv pip install -r requirements.txt
    uv pip install .

Example (single edge via Metric API):
    >>> metric = FEPMetric(mode="fast")
    >>> result = metric.compute(predicted_atom_array, ground_truth_atom_array)
    >>> print(f"ddG = {result['ddg_kcal_mol']:.2f} kcal/mol")

Example (batch via Metric API):
    >>> batch_metric = FEPBatchMetric(mode="fast", n_mps_workers=16)
    >>> batch_metric.setup()
    >>> for pred, gt in zip(predictions, ground_truths):
    ...     batch_metric.compute(pred, gt)
    >>> batch_metric.teardown()
    >>> results = batch_metric.get_results()

Example (standalone, without Metric framework):
    >>> from proteinfoundation.metrics.fep_metric import compute_rbfe
    >>> result = compute_rbfe(mol_a, mol_b, protein_pdb="protein.pdb", mode="fast")
"""

import contextlib
import logging
import os
from functools import cached_property
from typing import Any, Literal

from biotite.structure import AtomArray
from numpy.typing import NDArray
from rdkit import Chem

from foundry.metrics.metric import Metric
from proteinfoundation.metrics.fep_utils import (
    atom_array_to_pdb_path,
    extract_ligand_as_rdkit,
    extract_protein_atom_array,
    run_rbfe_edge,
)
from proteinfoundation.metrics.fep_utils import (
    compute_rbfe_batch as _compute_rbfe_batch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone convenience functions (usable without the Metric framework)
# ---------------------------------------------------------------------------


def compute_rbfe(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    core: NDArray | None = None,
    protein_pdb: str = "",
    mode: Literal["fast", "standard"] = "fast",
    seed: int = 2025,
    batch_mode: bool = False,
) -> dict[str, float]:
    """Compute relative binding free energy between two ligands.

    Args:
        mol_a: RDKit Mol for ligand A (3D coords, hydrogens).
        mol_b: RDKit Mol for ligand B (3D coords, hydrogens).
        core: Nx2 atom mapping array. If None, auto-computed via MCS (~1s).
        protein_pdb: Path to protein PDB file.
        mode: "fast" (~2 min/edge with MPS) or "standard" (~10 min/edge with MPS).
        seed: Random seed.
        batch_mode: Enable TMD batch mode for lambda window parallelism.
    """
    return run_rbfe_edge(
        mol_a=mol_a,
        mol_b=mol_b,
        core=core,
        protein_pdb=protein_pdb,
        mode=mode,
        seed=seed,
        batch_mode=batch_mode,
    )


def compute_rbfe_batch(
    pairs: list[tuple[Chem.Mol, Chem.Mol, NDArray | None]],
    protein_pdb: str,
    mode: Literal["fast", "standard"] = "fast",
    seed: int = 2025,
    n_mps_workers: int = 16,
) -> list[dict[str, float]]:
    """Compute RBFE for multiple ligand pairs in parallel via MPS.

    Each pair is (mol_a, mol_b, core_or_None). If core is None,
    the atom mapping is auto-computed via MCS.

    Args:
        pairs: List of (mol_a, mol_b, optional_core) tuples.
        protein_pdb: Path to protein PDB (shared across all pairs).
        mode: "fast" or "standard".
        seed: Random seed.
        n_mps_workers: MPS workers per GPU (16 recommended for H200, 12 for RTX 4090).
    """
    return _compute_rbfe_batch(
        pairs=pairs,
        protein_pdb=protein_pdb,
        mode=mode,
        seed=seed,
        n_mps_workers=n_mps_workers,
    )


# ---------------------------------------------------------------------------
# Foundry Metric: single-edge RBFE
# ---------------------------------------------------------------------------


class FEPMetric(Metric):
    """Single-edge RBFE metric using TMD.

    Computes the relative binding free energy between two ligands in a shared
    protein pocket. Extracts protein and ligand components from AtomArrays
    automatically using atomworks annotations.

    The ``predicted_atom_array`` should contain the predicted/designed ligand
    pose in context of the protein. The ``ground_truth_atom_array`` should
    contain the reference ligand in the same protein pocket.
    """

    def __init__(
        self,
        mode: Literal["fast", "standard"] = "fast",
        seed: int = 2025,
        required_tags_any: list[str] | set[str] | None = None,
        required_tags_all: list[str] | set[str] | None = None,
        prohibited_tags: list[str] | set[str] | None = None,
        n_frames_override: int | None = None,
        n_eq_steps_override: int | None = None,
        n_windows_override: int | None = None,
    ):
        if required_tags_any is None:
            required_tags_any = {"fep", "rbfe", "binding_energy"}
        super().__init__(
            required_tags_any=required_tags_any,
            required_tags_all=required_tags_all,
            prohibited_tags=prohibited_tags,
        )
        self.mode = mode
        self.seed = seed
        self.n_frames_override = n_frames_override
        self.n_eq_steps_override = n_eq_steps_override
        self.n_windows_override = n_windows_override

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, str]:
        return {
            "predicted_atom_array": "predicted_atom_array",
            "ground_truth_atom_array": "ground_truth_atom_array",
        }

    def compute(
        self,
        predicted_atom_array: AtomArray,
        ground_truth_atom_array: AtomArray,
    ) -> dict[str, Any]:
        """Compute RBFE for one ligand pair extracted from AtomArrays.

        Args:
            predicted_atom_array: Structure containing the predicted ligand pose
                (protein + ligand).
            ground_truth_atom_array: Structure containing the reference ligand
                (protein + ligand).

        Returns:
            Dict with ddg_kcal_mol, ddg_err_kcal_mol, and per-leg energies.
        """
        protein = extract_protein_atom_array(ground_truth_atom_array)
        mol_a = extract_ligand_as_rdkit(ground_truth_atom_array)
        mol_b = extract_ligand_as_rdkit(predicted_atom_array)

        protein_pdb = atom_array_to_pdb_path(protein)
        try:
            return run_rbfe_edge(
                mol_a=mol_a,
                mol_b=mol_b,
                protein_pdb=protein_pdb,
                mode=self.mode,
                seed=self.seed,
                n_frames_override=self.n_frames_override,
                n_eq_steps_override=self.n_eq_steps_override,
                n_windows_override=self.n_windows_override,
            )
        except Exception as e:
            logger.error(f"FEP computation failed: {e}")
            return {
                "ddg_kcal_mol": float("nan"),
                "ddg_err_kcal_mol": float("nan"),
                "error": str(e),
            }
        finally:
            with contextlib.suppress(OSError):
                os.unlink(protein_pdb)


# ---------------------------------------------------------------------------
# Foundry Metric: batch RBFE with MPS parallelism
# ---------------------------------------------------------------------------


class FEPBatchMetric(Metric):
    """Batch RBFE metric using TMD with MPS parallelism.

    Accumulates ligand pairs during validation (via repeated ``compute()``
    calls from ``MetricManager``) and runs all edges in parallel during
    ``teardown()`` using TMD's CUDA MPS pool for higher GPU throughput.

    Call ``get_results()`` after ``teardown()`` to retrieve the batch results.
    """

    def __init__(
        self,
        mode: Literal["fast", "standard"] = "fast",
        seed: int = 2025,
        n_mps_workers: int = 16,
        required_tags_any: list[str] | set[str] | None = None,
        required_tags_all: list[str] | set[str] | None = None,
        prohibited_tags: list[str] | set[str] | None = None,
    ):
        if required_tags_any is None:
            required_tags_any = {"fep", "rbfe", "binding_energy"}
        super().__init__(
            required_tags_any=required_tags_any,
            required_tags_all=required_tags_all,
            prohibited_tags=prohibited_tags,
        )
        self.mode = mode
        self.seed = seed
        self.n_mps_workers = n_mps_workers
        self._pairs: list[tuple[Chem.Mol, Chem.Mol, None]] = []
        self._protein_pdb: str | None = None
        self._results: list[dict[str, Any]] = []

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, str]:
        return {
            "predicted_atom_array": "predicted_atom_array",
            "ground_truth_atom_array": "ground_truth_atom_array",
        }

    def setup(self) -> None:
        """Reset accumulated pairs for a new validation epoch."""
        self._pairs = []
        self._protein_pdb = None
        self._results = []

    def compute(
        self,
        predicted_atom_array: AtomArray,
        ground_truth_atom_array: AtomArray,
    ) -> dict[str, Any]:
        """Extract and accumulate a ligand pair for batch processing.

        The actual RBFE computation is deferred to ``teardown()``.

        Returns:
            Placeholder dict indicating the result is pending.
        """
        mol_a = extract_ligand_as_rdkit(ground_truth_atom_array)
        mol_b = extract_ligand_as_rdkit(predicted_atom_array)
        self._pairs.append((mol_a, mol_b, None))

        if self._protein_pdb is None:
            protein = extract_protein_atom_array(ground_truth_atom_array)
            self._protein_pdb = atom_array_to_pdb_path(protein)

        return {
            "ddg_kcal_mol": float("nan"),
            "ddg_err_kcal_mol": float("nan"),
            "pending": True,
            "pair_index": len(self._pairs) - 1,
        }

    def teardown(self) -> None:
        """Run batch RBFE on all accumulated pairs and store results."""
        if not self._pairs or self._protein_pdb is None:
            self._results = []
            return

        try:
            self._results = _compute_rbfe_batch(
                pairs=self._pairs,
                protein_pdb=self._protein_pdb,
                mode=self.mode,
                seed=self.seed,
                n_mps_workers=self.n_mps_workers,
            )
        except Exception as e:
            logger.error(f"Batch FEP computation failed: {e}")
            self._results = [
                {"ddg_kcal_mol": float("nan"), "ddg_err_kcal_mol": float("nan"), "error": str(e)} for _ in self._pairs
            ]
        finally:
            if self._protein_pdb is not None:
                with contextlib.suppress(OSError):
                    os.unlink(self._protein_pdb)

    def get_results(self) -> list[dict[str, Any]]:
        """Retrieve batch RBFE results after teardown.

        Returns:
            List of result dicts, one per accumulated pair, in insertion order.
        """
        return self._results
