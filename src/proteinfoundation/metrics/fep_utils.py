"""Utilities for TMD-based Relative Binding Free Energy (RBFE) calculations.

Wraps TMD's alchemical FEP pipeline for use with biotite AtomArrays.
Supports single-edge, batch, and graph-parallel (MPS) modes.

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
"""

import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from atomworks.io.tools.rdkit import atom_array_to_rdkit
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.io_utils import to_pdb_string
from biotite.structure import AtomArray
from numpy.typing import NDArray
from rdkit import Chem

TMD_AVAILABLE = False
try:
    import jax

    jax.config.update("jax_enable_x64", True)

    from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS
    from tmd.fe import atom_mapping
    from tmd.fe.free_energy import HREXParams, LocalMDParams, MDParams, RESTParams, WaterSamplingParams
    from tmd.fe.rbfe import run_complex, run_solvent
    from tmd.fe.utils import get_mol_name, read_sdf_mols_by_name
    from tmd.ff import Forcefield
    from tmd.parallel.utils import get_gpu_count

    TMD_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _check_tmd_available() -> None:
    if not TMD_AVAILABLE:
        raise ImportError(
            "TMD is required for FEP/RBFE calculations but is not installed.\n"
            "Install from source (requires CUDA toolkit with nvcc, CMake >= 3.24.3):\n"
            "  git clone https://github.com/tmd-industries/tmd.git\n"
            "  cd tmd\n"
            "  uv pip install -r requirements.txt\n"
            "  uv pip install .\n"
            "The CUDA custom ops are compiled via CMake during installation.\n"
            "See https://github.com/tmd-industries/tmd for full prerequisites."
        )


# Amber AM1BCC forcefield reads precomputed charges from SDF properties (no OpenEye needed).
RBFE_DEFAULT_FF = "smirnoff_2_0_0_amber_am1bcc.py"

# TMD timestep is 2.5 fs, so steps_per_frame=400 = 1 ps/frame
_STEPS_PER_FRAME = 400

# kJ/mol -> kcal/mol conversion
KJ_TO_KCAL = 1.0 / 4.184


# ---------------------------------------------------------------------------
# AtomArray helpers (no TMD dependency)
# ---------------------------------------------------------------------------


def extract_protein_atom_array(atom_array: AtomArray) -> AtomArray:
    """Extract protein atoms from an AtomArray using chain_type annotation.

    Args:
        atom_array: Input structure containing protein and possibly other entities.

    Returns:
        AtomArray containing only protein atoms.

    Raises:
        ValueError: If no protein atoms are found.
    """
    ensure_annotations(atom_array, "is_protein")
    protein = atom_array[atom_array.is_protein]
    if len(protein) == 0:
        raise ValueError("No protein atoms found in AtomArray")
    return protein


def extract_ligand_as_rdkit(atom_array: AtomArray) -> Chem.Mol:
    """Extract non-polymer ligand atoms and convert to an RDKit Mol.

    Uses atomworks ``is_ligand`` annotation (based on ``chain_type``) to identify
    ligand atoms, then converts via ``atom_array_to_rdkit`` with hydrogens kept
    and sanitization enabled.

    Args:
        atom_array: Input structure containing at least one ligand entity.

    Returns:
        RDKit Mol with 3D coordinates and explicit hydrogens.

    Raises:
        ValueError: If no ligand atoms are found.
    """
    ensure_annotations(atom_array, "is_ligand")
    ligand_atoms = atom_array[atom_array.is_ligand]
    if len(ligand_atoms) == 0:
        raise ValueError("No ligand atoms found in AtomArray")
    return atom_array_to_rdkit(ligand_atoms, hydrogen_policy="keep", sanitize=True)


def compute_atom_mapping(mol_a: Chem.Mol, mol_b: Chem.Mol) -> NDArray:
    """Auto-compute atom mapping between two molecules using MCS.

    Uses TMD's atom_mapping module with default parameters (MCS-based,
    respects chirality, ring topology, etc.). Takes ~1s per pair.

    Returns:
        Nx2 array where each row maps an atom index in mol_a to mol_b.
    """
    _check_tmd_available()
    cores = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)
    if not cores:
        raise ValueError(
            f"No atom mapping found between {get_mol_name(mol_a)} and {get_mol_name(mol_b)}. "
            "The molecules may be too dissimilar for RBFE."
        )
    return cores[0]


@dataclass(frozen=True)
class RBFEPreset:
    """Simulation parameters for a given RBFE mode (fast vs standard)."""

    n_frames: int
    n_eq_steps: int
    n_windows: int
    min_overlap: float

    def build_md_params(self, seed: int = 2025) -> MDParams:
        """Construct a :class:`MDParams` instance with these preset values."""
        local_md = LocalMDParams(local_steps=390, min_radius=1.2, max_radius=3.0)
        water_sampling = WaterSamplingParams(interval=400, n_proposals=1000)
        rest = RESTParams(max_temperature_scale=3.0, temperature_scale_interpolation="exponential")
        return MDParams(
            n_frames=self.n_frames,
            n_eq_steps=self.n_eq_steps,
            steps_per_frame=_STEPS_PER_FRAME,
            seed=seed,
            hrex_params=HREXParams(n_frames_bisection=100, rest_params=rest),
            water_sampling_params=water_sampling,
            local_md_params=local_md,
        )


RBFE_PRESETS: dict[str, RBFEPreset] = {
    "fast": RBFEPreset(n_frames=1000, n_eq_steps=10_000, n_windows=24, min_overlap=0.200),
    "standard": RBFEPreset(n_frames=2000, n_eq_steps=200_000, n_windows=48, min_overlap=0.667),
}


def _get_preset(mode: str) -> RBFEPreset:
    if mode not in RBFE_PRESETS:
        raise ValueError(f"Unknown mode: {mode!r}. Choose from {list(RBFE_PRESETS)}.")
    return RBFE_PRESETS[mode]


def _enable_batch_mode(batch_size: int = 8) -> None:
    """Enable TMD's batch mode for parallelizing lambda windows within bisection."""
    os.environ["TMD_BATCH_MODE"] = "on"
    os.environ["TMD_BISECTION_BATCH_SIZE"] = str(batch_size)


def _disable_batch_mode() -> None:
    os.environ.pop("TMD_BATCH_MODE", None)
    os.environ.pop("TMD_BISECTION_BATCH_SIZE", None)


def _extract_dg(sim_result: "Any") -> tuple[float, float]:
    """Extract dG and dG_err (kJ/mol) from a SimulationResult."""
    dg = float(np.sum(sim_result.final_result.dGs))
    dg_err = float(np.linalg.norm(sim_result.final_result.dG_errs))
    return dg, dg_err


def run_rbfe_edge(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    core: NDArray | None = None,
    protein_pdb: str = "",
    mode: Literal["fast", "standard"] = "fast",
    forcefield_name: str = RBFE_DEFAULT_FF,
    seed: int = 2025,
    batch_mode: bool = False,
    batch_size: int = 8,
    n_frames_override: int | None = None,
    n_eq_steps_override: int | None = None,
    n_windows_override: int | None = None,
) -> dict:
    """Run a single RBFE edge (complex + solvent legs) and return ddG.

    Args:
        mol_a: RDKit Mol for the initial ligand (with 3D coords and hydrogens).
        mol_b: RDKit Mol for the target ligand (with 3D coords and hydrogens).
        core: Nx2 atom mapping array. If None, auto-computed via MCS.
        protein_pdb: Path to the protein PDB file.
        mode: "fast" (~2 min with MPS) or "standard" (~10 min with MPS).
        forcefield_name: TMD forcefield name.
        seed: Random seed.
        batch_mode: Enable TMD batch mode. Default False for single-edge.
        batch_size: Batch size for lambda window parallelism when batch_mode=True.
        n_frames_override: Override production frames (for testing).
        n_eq_steps_override: Override equilibration steps (for testing).
        n_windows_override: Override max lambda windows (for testing).

    Returns:
        dict with ddg_kcal_mol, ddg_err_kcal_mol, complex/solvent dG, elapsed_seconds.
    """
    _check_tmd_available()
    if core is None:
        logger.info("Auto-computing atom mapping via MCS...")
        core = compute_atom_mapping(mol_a, mol_b)
    else:
        core = np.asarray(core)

    preset = _get_preset(mode)
    ff = Forcefield.load_from_file(forcefield_name)
    md_params = preset.build_md_params(seed)
    n_windows = preset.n_windows
    min_overlap = preset.min_overlap

    if n_frames_override is not None:
        md_params = replace(md_params, n_frames=n_frames_override)
    if n_eq_steps_override is not None:
        md_params = replace(md_params, n_eq_steps=n_eq_steps_override)
    if n_windows_override is not None:
        n_windows = n_windows_override

    if batch_mode:
        _enable_batch_mode(batch_size)

    t0 = time.time()
    try:
        logger.info(f"Running complex leg ({mode} mode, batch={batch_mode})...")
        complex_res, _ = run_complex(
            mol_a,
            mol_b,
            core,
            ff,
            protein_pdb,
            md_params=md_params,
            n_windows=n_windows,
            min_overlap=min_overlap,
        )
        complex_dg, complex_dg_err = _extract_dg(complex_res)

        logger.info(f"Running solvent leg ({mode} mode)...")
        solvent_res, _ = run_solvent(
            mol_a,
            mol_b,
            core,
            ff,
            None,
            md_params=md_params,
            n_windows=n_windows,
            min_overlap=min_overlap,
        )
        solvent_dg, solvent_dg_err = _extract_dg(solvent_res)
    finally:
        if batch_mode:
            _disable_batch_mode()

    ddg = (complex_dg - solvent_dg) * KJ_TO_KCAL
    ddg_err = np.sqrt(complex_dg_err**2 + solvent_dg_err**2) * KJ_TO_KCAL
    elapsed = time.time() - t0

    logger.info(f"RBFE edge complete: ddG = {ddg:.2f} ± {ddg_err:.2f} kcal/mol ({elapsed:.1f}s)")

    return {
        "ddg_kcal_mol": ddg,
        "ddg_err_kcal_mol": ddg_err,
        "complex_dg_kcal_mol": complex_dg * KJ_TO_KCAL,
        "complex_dg_err_kcal_mol": complex_dg_err * KJ_TO_KCAL,
        "solvent_dg_kcal_mol": solvent_dg * KJ_TO_KCAL,
        "solvent_dg_err_kcal_mol": solvent_dg_err * KJ_TO_KCAL,
        "elapsed_seconds": elapsed,
        "mol_a": get_mol_name(mol_a),
        "mol_b": get_mol_name(mol_b),
    }


def _run_single_leg(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    core: NDArray,
    leg_name: str,
    ff: Forcefield,
    protein_pdb: str | None,
    md_params: MDParams,
    n_windows: int,
    min_overlap: float,
) -> dict:
    """Run a single RBFE leg (complex or solvent). Used by batch runner via MPS."""
    if leg_name == "complex":
        res, _ = run_complex(
            mol_a,
            mol_b,
            core,
            ff,
            protein_pdb,
            md_params=md_params,
            n_windows=n_windows,
            min_overlap=min_overlap,
        )
    elif leg_name == "solvent":
        res, _ = run_solvent(
            mol_a,
            mol_b,
            core,
            ff,
            None,
            md_params=md_params,
            n_windows=n_windows,
            min_overlap=min_overlap,
        )
    else:
        raise ValueError(f"Unknown leg: {leg_name}")

    dg, dg_err = _extract_dg(res)
    return {
        "leg": leg_name,
        "mol_a": get_mol_name(mol_a),
        "mol_b": get_mol_name(mol_b),
        "dg_kj_mol": dg,
        "dg_err_kj_mol": dg_err,
    }


def compute_rbfe_batch(
    pairs: list[tuple[Chem.Mol, Chem.Mol, NDArray | None]],
    protein_pdb: str,
    mode: Literal["fast", "standard"] = "fast",
    forcefield_name: str = RBFE_DEFAULT_FF,
    seed: int = 2025,
    n_mps_workers: int = 16,
    batch_mode: bool = True,
    batch_size: int = 8,
) -> list[dict]:
    """Run multiple RBFE edges in parallel using MPS.

    Each pair is (mol_a, mol_b, core_or_None). If core is None,
    atom mapping is auto-computed via MCS before submission.

    Args:
        pairs: List of (mol_a, mol_b, core) tuples. core can be None.
        protein_pdb: Path to protein PDB (shared across all pairs).
        mode: "fast" or "standard".
        forcefield_name: TMD forcefield name.
        seed: Random seed.
        n_mps_workers: Number of MPS workers per GPU. 16 recommended for H200.
        batch_mode: Enable TMD batch mode within each worker.
        batch_size: Batch size for lambda window parallelism.

    Returns:
        List of result dicts (same format as run_rbfe_edge), one per pair.
    """
    _check_tmd_available()
    from tmd.parallel.client import CUDAMPSPoolClient, iterate_completed_futures

    preset = _get_preset(mode)
    ff = Forcefield.load_from_file(forcefield_name)
    md_params = preset.build_md_params(seed)
    n_windows = preset.n_windows
    min_overlap = preset.min_overlap

    # Pre-compute atom mappings for pairs that don't have them
    resolved_pairs = []
    for mol_a, mol_b, core in pairs:
        if core is None:
            logger.info(f"Computing atom mapping: {get_mol_name(mol_a)} -> {get_mol_name(mol_b)}")
            core = compute_atom_mapping(mol_a, mol_b)
        else:
            core = np.asarray(core)
        resolved_pairs.append((mol_a, mol_b, core))

    if batch_mode:
        _enable_batch_mode(batch_size)

    t0 = time.time()
    num_gpus = get_gpu_count()
    pool = CUDAMPSPoolClient(num_gpus=num_gpus, workers_per_gpu=n_mps_workers, max_tasks_per_child=1)
    pool.verify()

    futures = []
    future_to_edge = {}

    for mol_a, mol_b, core in resolved_pairs:
        for leg in ["complex", "solvent"]:
            fut = pool.submit(
                _run_single_leg,
                mol_a,
                mol_b,
                core,
                leg,
                ff,
                protein_pdb if leg == "complex" else None,
                md_params,
                n_windows,
                min_overlap,
            )
            future_to_edge[fut.id] = (get_mol_name(mol_a), get_mol_name(mol_b), leg)
            futures.append(fut)

    # Collect results grouped by edge
    leg_results = defaultdict(dict)
    for fut in iterate_completed_futures(futures):
        mol_a_name, mol_b_name, leg = future_to_edge[fut.id]
        edge_key = (mol_a_name, mol_b_name)
        try:
            result = fut.result()
            leg_results[edge_key][leg] = result
            logger.info(
                f"Completed {leg} leg for {mol_a_name}->{mol_b_name}: "
                f"dG={result['dg_kj_mol']*KJ_TO_KCAL:.2f} kcal/mol"
            )
        except Exception as e:
            logger.error(f"Failed {leg} leg for {mol_a_name}->{mol_b_name}: {e}")
            leg_results[edge_key][leg] = {"error": str(e)}

    if batch_mode:
        _disable_batch_mode()

    # Assemble final results
    results = []
    for mol_a, mol_b, _core in resolved_pairs:
        edge_key = (get_mol_name(mol_a), get_mol_name(mol_b))
        complex_leg = leg_results[edge_key].get("complex", {})
        solvent_leg = leg_results[edge_key].get("solvent", {})

        if "error" in complex_leg or "error" in solvent_leg:
            results.append(
                {
                    "ddg_kcal_mol": float("nan"),
                    "ddg_err_kcal_mol": float("nan"),
                    "mol_a": edge_key[0],
                    "mol_b": edge_key[1],
                    "error": complex_leg.get("error", "") + solvent_leg.get("error", ""),
                }
            )
            continue

        complex_dg = complex_leg["dg_kj_mol"]
        complex_dg_err = complex_leg["dg_err_kj_mol"]
        solvent_dg = solvent_leg["dg_kj_mol"]
        solvent_dg_err = solvent_leg["dg_err_kj_mol"]

        ddg = (complex_dg - solvent_dg) * KJ_TO_KCAL
        ddg_err = np.sqrt(complex_dg_err**2 + solvent_dg_err**2) * KJ_TO_KCAL

        results.append(
            {
                "ddg_kcal_mol": ddg,
                "ddg_err_kcal_mol": ddg_err,
                "complex_dg_kcal_mol": complex_dg * KJ_TO_KCAL,
                "complex_dg_err_kcal_mol": complex_dg_err * KJ_TO_KCAL,
                "solvent_dg_kcal_mol": solvent_dg * KJ_TO_KCAL,
                "solvent_dg_err_kcal_mol": solvent_dg_err * KJ_TO_KCAL,
                "mol_a": edge_key[0],
                "mol_b": edge_key[1],
            }
        )

    elapsed = time.time() - t0
    logger.info(f"Batch RBFE complete: {len(results)} edges in {elapsed:.1f}s " f"({elapsed/len(results):.1f}s/edge)")

    return results


def atom_array_to_pdb_path(atom_array: AtomArray) -> str:
    """Write an AtomArray to a temporary PDB file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
        tmp.write(to_pdb_string(atom_array))
    return tmp.name


def load_benchmark_dataset(dataset_path: str) -> dict:
    """Load an RBFE benchmark dataset (PDB + SDF + JSON map).

    Returns:
        dict with keys: protein_pdb, mols (dict name->Mol), edges (list of dicts),
        experimental_dg (dict name->float in kcal/mol)
    """
    _check_tmd_available()
    dataset_path = Path(dataset_path)

    pdb_files = list(dataset_path.glob("*_structure.pdb")) + list(dataset_path.glob("*.pdb"))
    assert len(pdb_files) >= 1, f"No PDB files found in {dataset_path}"
    protein_pdb = str(pdb_files[0])

    sdf_path = dataset_path / "ligands.sdf"
    assert sdf_path.exists(), f"ligands.sdf not found in {dataset_path}"
    mols = read_sdf_mols_by_name(str(sdf_path))

    map_files = list(dataset_path.glob("*_map.json"))
    assert len(map_files) == 1, f"Expected exactly 1 map JSON, found {len(map_files)}"
    with open(map_files[0]) as f:
        edges = json.load(f)

    experimental_dg = {}
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name")
        if mol.HasProp("kcal/mol experimental dG"):
            experimental_dg[name] = float(mol.GetProp("kcal/mol experimental dG"))

    return {
        "protein_pdb": protein_pdb,
        "mols": mols,
        "edges": edges,
        "experimental_dg": experimental_dg,
    }
