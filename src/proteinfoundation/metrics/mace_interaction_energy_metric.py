"""MACE interaction energy metric for protein-ligand complexes.

Uses MACE-POLAR-1 for interaction energy decomposition:
    E_interaction = E(complex) - E(protein) - E(ligand)

Hydrogen addition uses atomworks' ``ensure_hydrogens`` when available, else RDKit infer:
    - Per-residue Dimorphite-DL protonation at target pH
    - RDKit 3D H placement
    - Backbone charge correction for peptide bond context
"""

from __future__ import annotations

import gc
import logging
import os
import traceback
import warnings
from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import ase.io
import numpy as np
import torch
from ase import Atoms
from ase.optimize import BFGS, FIRE, LBFGS
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.conditions.annotator import ensure_annotations
from biotite.structure import AtomArray, AtomArrayStack
from scipy.spatial import cKDTree

from foundry.metrics.metric import Metric
from proteinfoundation.metrics.metrics_utils import normalize_to_atom_array_list

if TYPE_CHECKING:
    from mace.calculators import MACECalculator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_hydrogens(atom_array: AtomArray, *, ph: float) -> AtomArray:
    """Add hydrogens for MACE; prefer atomworks protonation, else RDKit infer."""
    try:
        from atomworks.io.utils.protonation import ensure_hydrogens

        return ensure_hydrogens(atom_array, ph=ph)
    except ModuleNotFoundError:
        from atomworks.io.tools.rdkit import atom_array_from_rdkit, atom_array_to_rdkit

        mol = atom_array_to_rdkit(atom_array, hydrogen_policy="infer", set_coord=True)
        return atom_array_from_rdkit(mol, remove_hydrogens=False, remove_inferred_atoms=False)


def _atom_array_to_ase(array: AtomArray) -> Atoms:
    """Convert a Biotite AtomArray to an ASE Atoms object."""
    symbols = [elem.capitalize() for elem in array.element]
    atoms = Atoms(symbols=symbols, positions=array.coord)
    box = getattr(array, "box", None)
    if box is not None:
        atoms.cell = box
        atoms.pbc = True
    return atoms


def _atom_array_to_ase_with_charge(
    atom_array: AtomArray,
    infer_charge: Literal["atom_array", "zeros", "raise"] = "atom_array",
) -> Atoms:
    """Convert AtomArray to ASE Atoms with charge/spin handling.

    Args:
        atom_array: Structure to convert.
        infer_charge: How to determine total charge.

            - ``"atom_array"``: Sum per-atom ``charge`` annotation.
            - ``"zeros"``: Use charge=0, spin=1.
            - ``"raise"``: Raise if charge is missing.

    Returns:
        ASE Atoms with ``info["charge"]`` and ``info["spin"]`` set.

    Raises:
        ValueError: If charge is missing and ``infer_charge="raise"``.
    """
    ase_atoms = _atom_array_to_ase(atom_array)

    if infer_charge == "zeros":
        defaults = {"charge": 0.0, "spin": 1.0}
    elif infer_charge == "atom_array":
        try:
            charge_sum = float(np.sum(atom_array.get_annotation("charge")))
        except ValueError as e:
            raise ValueError(f"Charge annotation not found in atom_array: {e}") from e
        defaults = {"charge": charge_sum, "spin": 1.0}
    elif infer_charge == "raise":
        defaults = None
    else:
        raise ValueError(f"infer_charge must be 'atom_array', 'zeros', or 'raise', got {infer_charge}")

    for key in ["charge", "spin"]:
        if key in ase_atoms.info and isinstance(ase_atoms.info[key], int | float):
            continue
        if infer_charge == "raise":
            raise ValueError(f"{key.capitalize()} not found in ase_atoms.info and infer_charge='raise'")
        ase_atoms.info[key] = defaults[key]
        logger.debug("Inferred %s=%.1f (policy=%s)", key, ase_atoms.info[key], infer_charge)

    if abs(ase_atoms.info["charge"]) > 10:
        logger.warning("Unusually high charge: %.1f. Verify structure.", ase_atoms.info["charge"])

    return ase_atoms


def _create_subset_ase_atoms(
    source_ase: Atoms,
    mask: np.ndarray,
    calculator: MACECalculator,
    infer_charge: Literal["atom_array", "zeros", "raise"] = "atom_array",
) -> Atoms:
    """Create a subset of ASE Atoms using a boolean mask.

    Args:
        source_ase: Source ASE Atoms (optimized complex).
        mask: Boolean mask selecting atoms to include.
        calculator: MACE calculator to attach.
        infer_charge: Charge inference policy for the subset.

    Returns:
        New ASE Atoms at original positions with calculator attached.
    """
    subset = source_ase[mask]
    subset.calc = calculator

    if infer_charge == "zeros":
        subset.info["charge"] = 0.0
        subset.info["spin"] = 1.0
    elif infer_charge == "atom_array":
        if "charges" in source_ase.arrays:
            subset.info["charge"] = float(np.sum(source_ase.arrays["charges"][mask]))
        else:
            subset.info["charge"] = 0.0
        subset.info["spin"] = 1.0
    elif infer_charge == "raise":
        if "charge" not in subset.info:
            raise ValueError("Charge not found in subset atoms and infer_charge='raise'")

    return subset


# ---------------------------------------------------------------------------
# Clash detection
# ---------------------------------------------------------------------------

CLASH_THRESHOLD = 0.4  # Å — severe steric overlap
CONTACT_THRESHOLD = 1.8  # Å — close contacts (potential covalent bond distance)
POCKET_CUTOFF = 8.0  # Å — distance cutoff for pocket definition


def _get_pocket_coords(
    ligand_coords: np.ndarray,
    protein_coords: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Return protein atoms within *cutoff* of any ligand atom."""
    if len(ligand_coords) == 0 or len(protein_coords) == 0:
        return np.array([]).reshape(0, 3)
    tree = cKDTree(ligand_coords)
    near = tree.query_ball_point(protein_coords, r=cutoff)
    pocket_mask = np.array([len(n) > 0 for n in near])
    return protein_coords[pocket_mask]


def check_clashes(
    ligand_coords: np.ndarray,
    protein_coords: np.ndarray,
    pocket_cutoff: float = POCKET_CUTOFF,
    clash_threshold: float = CLASH_THRESHOLD,
    contact_threshold: float = CONTACT_THRESHOLD,
) -> dict[str, bool | int]:
    """Check for clashes between ligand and nearby protein atoms.

    Args:
        ligand_coords: Ligand atom positions ``(n_lig, 3)``.
        protein_coords: Protein atom positions ``(n_prot, 3)``.
        pocket_cutoff: Distance for pocket definition.
        clash_threshold: Severe steric overlap threshold (Å).
        contact_threshold: Close-contact threshold (Å).

    Returns:
        Dict with ``has_clashes``, ``has_contacts``, ``n_clashes``,
        ``n_contacts``.
    """
    empty = {"has_clashes": False, "has_contacts": False, "n_clashes": 0, "n_contacts": 0}
    if len(ligand_coords) == 0 or len(protein_coords) == 0:
        return empty

    pocket_coords = _get_pocket_coords(ligand_coords, protein_coords, pocket_cutoff)
    if len(pocket_coords) == 0:
        return empty

    tree = cKDTree(pocket_coords)
    dists, _ = tree.query(ligand_coords, k=1)

    n_clashes = int((dists < clash_threshold).sum())
    n_contacts = int((dists < contact_threshold).sum())

    return {
        "has_clashes": n_clashes > 0,
        "has_contacts": n_contacts > 0,
        "n_clashes": n_clashes,
        "n_contacts": n_contacts,
    }


# ---------------------------------------------------------------------------
# Geometry optimization
# ---------------------------------------------------------------------------

_OPTIMIZER_MAP = {"BFGS": BFGS, "LBFGS": LBFGS, "FIRE": FIRE}


def geometry_optimize(
    ase_atoms: Atoms,
    calculator: MACECalculator,
    max_steps: int = 50,
    fmax: float = 0.05,
    optimizer: Literal["BFGS", "LBFGS", "FIRE"] = "LBFGS",
) -> tuple[Atoms, int, bool]:
    """Geometry-optimise an ASE Atoms object using a MACE calculator.

    Falls back to FIRE if the requested optimizer fails.

    Args:
        ase_atoms: Structure to optimise.
        calculator: MACE calculator instance.
        max_steps: Maximum optimisation steps.
        fmax: Force convergence criterion (eV/Å).
        optimizer: Optimizer name (``BFGS``, ``LBFGS``, or ``FIRE``).

    Returns:
        ``(optimised_atoms, n_steps, converged)`` tuple.
    """
    if optimizer not in _OPTIMIZER_MAP:
        raise ValueError(f"optimizer must be one of {list(_OPTIMIZER_MAP.keys())}, got {optimizer}")

    ase_atoms = ase_atoms.copy()
    ase_atoms.calc = calculator

    with torch.enable_grad():
        try:
            opt = _OPTIMIZER_MAP[optimizer](ase_atoms, logfile=None)
            converged = opt.run(fmax=fmax, steps=max_steps)
            n_steps = opt.nsteps
        except Exception as e:
            if optimizer != "FIRE":
                logger.warning("%s failed (%s), falling back to FIRE", optimizer, e)
                try:
                    opt = FIRE(ase_atoms, logfile=None)
                    converged = opt.run(fmax=fmax, steps=max_steps)
                    n_steps = opt.nsteps
                except Exception as e2:
                    logger.warning("FIRE also failed (%s), skipping optimisation", e2)
                    n_steps = 0
                    converged = False
            else:
                raise

    return ase_atoms, n_steps, converged


# ---------------------------------------------------------------------------
# Interaction energy computation
# ---------------------------------------------------------------------------


def compute_interaction_energy(
    atom_array: AtomArray,
    calculator: MACECalculator,
    hydrogen_policy: Literal["infer", "keep"] = "infer",
    infer_charge: Literal["atom_array", "zeros", "raise"] = "atom_array",
    max_opt_steps: int = 50,
    fmax: float = 0.05,
    optimizer: Literal["BFGS", "LBFGS", "FIRE"] = "LBFGS",
    skip_on_clashes: bool = False,
    pH: float = 7.0,
    pocket_cutoff: float = POCKET_CUTOFF,
    clash_threshold: float = CLASH_THRESHOLD,
    contact_threshold: float = CONTACT_THRESHOLD,
    debug: bool = False,
    debug_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Compute MACE interaction energy for a protein-ligand complex.

    E_interaction = E(complex) - E(protein) - E(ligand), all evaluated
    at the same geometry-optimised coordinates.

    Args:
        atom_array: Structure with ``is_ligand`` / ``is_protein`` annotations
            (derived automatically from ``chain_type`` if absent).
        calculator: MACE calculator instance.
        hydrogen_policy: ``"infer"`` strips existing H and re-adds them
            via Dimorphite-DL at *pH*; ``"keep"`` uses existing H.
        infer_charge: Policy for total system charge.
        max_opt_steps: Maximum geometry optimisation steps.
        fmax: Force convergence criterion (eV/Å).
        optimizer: ASE optimiser name.
        skip_on_clashes: Return ``None`` when pre-optimisation clashes are
            detected instead of proceeding.
        pH: Target pH for protonation (used when ``hydrogen_policy="infer"``).
        debug: Save intermediate CIF files and enable DEBUG logging.
        debug_dir: Directory for debug output (created if absent).

    Returns:
        Dict with ``interaction_energy``, ``complex_energy``,
        ``protein_energy``, ``ligand_energy`` (all in eV), plus
        ``n_opt_steps``, ``converged``, ``pre_opt_clashes``,
        ``post_opt_clashes``.  ``None`` if skipped due to clashes.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        if debug_dir is None:
            debug_dir = Path("debug_output")
        debug_dir.mkdir(parents=True, exist_ok=True)

    ensure_annotations(atom_array, "is_ligand", "is_protein")
    n_ligand = int(np.sum(atom_array.is_ligand))
    n_protein = int(np.sum(atom_array.is_protein))

    if n_ligand == 0:
        raise ValueError("No ligand atoms found (is_ligand all False)")
    if n_protein == 0:
        raise ValueError("No protein atoms found (is_protein all False)")

    logger.debug("Structure has %d protein atoms and %d ligand atoms", n_protein, n_ligand)

    # Step 1: Add hydrogens. ``ensure_hydrogens`` (atomworks) preserves Biotite
    # annotations; the RDKit round-trip fallback in ``_infer_hydrogens`` does not,
    # so snapshot ``chain_type`` per chain and restore after protonation — otherwise
    # ``ensure_annotations`` misclassifies ligand chains as protein.
    prepared = atom_array.copy()
    chain_type_by_id: dict[str, int] = {}
    if "chain_type" in prepared.get_annotation_categories():
        for cid in np.unique(prepared.chain_id):
            mask = prepared.chain_id == cid
            chain_type_by_id[str(cid)] = int(prepared.chain_type[mask][0])

    if hydrogen_policy == "infer":
        prepared = _infer_hydrogens(prepared, ph=pH)
        if chain_type_by_id:
            from atomworks.enums import ChainType

            restored = np.empty(len(prepared), dtype=int)
            for cid in np.unique(prepared.chain_id):
                mask = prepared.chain_id == cid
                restored[mask] = chain_type_by_id.get(str(cid), ChainType.POLYPEPTIDE_L)
            prepared.set_annotation("chain_type", restored)

    ensure_annotations(prepared, "is_ligand", "is_protein")
    lig_mask = prepared.is_ligand
    prot_mask = prepared.is_protein

    if debug:
        _log_duplicate_atoms(atom_array)
        to_cif_file(atom_array, debug_dir / "initial_no_hydrogens.cif")
        to_cif_file(prepared, debug_dir / "pre_opt_all.cif")
        to_cif_file(prepared[lig_mask.astype(bool)], debug_dir / "pre_opt_ligand.cif")
        to_cif_file(prepared[prot_mask.astype(bool)], debug_dir / "pre_opt_protein_pocket.cif")
        logger.debug("Saved pre-optimisation debug CIF files to %s", debug_dir)

    # Step 2: Pre-optimisation clash check
    lig_coords = prepared.coord[lig_mask.astype(bool)]
    prot_coords = prepared.coord[prot_mask.astype(bool)]
    pre_opt_clashes = check_clashes(
        lig_coords,
        prot_coords,
        pocket_cutoff=pocket_cutoff,
        clash_threshold=clash_threshold,
        contact_threshold=contact_threshold,
    )

    if pre_opt_clashes["has_clashes"]:
        warnings.warn(
            f"Pre-optimisation clashes: {pre_opt_clashes['n_clashes']} pairs "
            f"< {clash_threshold} Å. Input geometry may be problematic.",
            UserWarning,
            stacklevel=2,
        )
        if skip_on_clashes:
            logger.warning("Skipping structure (skip_on_clashes=True)")
            return None

    # Step 3: Convert to ASE and optimise
    ase_complex = _atom_array_to_ase_with_charge(prepared, infer_charge)
    if "charge" in prepared.get_annotation_categories():
        ase_complex.arrays["charges"] = prepared.get_annotation("charge")

    if debug:
        ase.io.write(debug_dir / "pre_opt_complex.extxyz", ase_complex, format="extxyz")
        logger.debug("Saved pre-optimisation ASE Atoms to %s", debug_dir / "pre_opt_complex.extxyz")

    optimised, n_steps, converged = geometry_optimize(
        ase_complex,
        calculator,
        max_opt_steps,
        fmax,
        optimizer,
    )

    # Step 4: Post-optimisation clash check
    opt_lig = optimised.get_positions()[lig_mask.astype(bool)]
    opt_prot = optimised.get_positions()[prot_mask.astype(bool)]
    post_opt_clashes = check_clashes(
        opt_lig,
        opt_prot,
        pocket_cutoff=pocket_cutoff,
        clash_threshold=clash_threshold,
        contact_threshold=contact_threshold,
    )

    if post_opt_clashes["has_clashes"]:
        warnings.warn(
            f"Post-optimisation clashes: {post_opt_clashes['n_clashes']} pairs < {clash_threshold} Å.",
            UserWarning,
            stacklevel=2,
        )
    if post_opt_clashes["has_contacts"]:
        warnings.warn(
            f"Post-optimisation close contacts: {post_opt_clashes['n_contacts']} pairs "
            f"< {CONTACT_THRESHOLD} Å (possible unphysical bond formation).",
            UserWarning,
            stacklevel=2,
        )

    if debug:
        opt_array = prepared.copy()
        opt_array.coord = optimised.get_positions()
        to_cif_file(opt_array, debug_dir / "post_opt_all.cif")
        to_cif_file(opt_array[lig_mask.astype(bool)], debug_dir / "post_opt_ligand.cif")
        to_cif_file(opt_array[prot_mask.astype(bool)], debug_dir / "post_opt_protein_pocket.cif")
        logger.debug("Saved post-optimisation debug CIF files to %s", debug_dir)

    # Step 5: Energy decomposition at optimised geometry
    with torch.enable_grad():
        optimised.calc = calculator
        e_complex = optimised.get_potential_energy()

    protein_only = _create_subset_ase_atoms(optimised, ~lig_mask, calculator, infer_charge)
    ligand_only = _create_subset_ase_atoms(optimised, lig_mask, calculator, infer_charge)

    with torch.enable_grad():
        e_protein = protein_only.get_potential_energy()
        e_ligand = ligand_only.get_potential_energy()

    e_interaction = e_complex - e_protein - e_ligand

    logger.debug(
        "E_complex=%.4f, E_protein=%.4f, E_ligand=%.4f, E_interaction=%.4f eV",
        e_complex,
        e_protein,
        e_ligand,
        e_interaction,
    )

    return {
        "interaction_energy": float(e_interaction),
        "complex_energy": float(e_complex),
        "protein_energy": float(e_protein),
        "ligand_energy": float(e_ligand),
        "n_opt_steps": n_steps,
        "converged": converged,
        "pre_opt_clashes": pre_opt_clashes,
        "post_opt_clashes": post_opt_clashes,
    }


def _log_duplicate_atoms(atom_array: AtomArray) -> None:
    """Log duplicate atoms (same chain/res/name/ins_code tuple)."""
    ins = (
        atom_array.ins_code.tolist() if "ins_code" in atom_array.get_annotation_categories() else [""] * len(atom_array)
    )
    keys = list(
        zip(
            atom_array.chain_id.tolist(),
            atom_array.res_id.tolist(),
            atom_array.res_name.tolist(),
            atom_array.atom_name.tolist(),
            ins,
            strict=False,
        )
    )
    dupes = [(k, c) for k, c in Counter(keys).items() if c > 1]
    if dupes:
        logger.warning("Duplicate atoms in input: %d groups", len(dupes))
        for k, c in dupes[:10]:
            logger.warning("  %s -> %d", k, c)


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------


class MACEInteractionEnergyMetric(Metric):
    """Compute MACE interaction energies for protein-ligand complexes.

    Workflow per structure:

    1. Add hydrogens (pH-aware via Dimorphite-DL)
    2. Geometry-optimise the full complex
    3. Decompose: E_int = E(complex) - E(protein) - E(ligand)
    """

    def __init__(
        self,
        model_path: str | None = None,
        max_opt_steps: int = 50,
        fmax: float = 0.05,
        optimizer: Literal["BFGS", "LBFGS", "FIRE"] = "LBFGS",
        hydrogen_policy: Literal["infer", "keep"] = "infer",
        infer_charge: Literal["atom_array", "zeros", "raise"] = "atom_array",
        device: str = "cuda",
        skip_if_no_ligand: bool = True,
        skip_on_clashes: bool = False,
        pH: float = 7.0,
        pocket_cutoff: float = POCKET_CUTOFF,
        clash_threshold: float = CLASH_THRESHOLD,
        contact_threshold: float = CONTACT_THRESHOLD,
        max_energy_magnitude: float | None = 1e5,
        debug: bool = False,
        debug_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        if hydrogen_policy not in ("infer", "keep"):
            raise ValueError(f"hydrogen_policy must be 'infer' or 'keep', got {hydrogen_policy}")
        if optimizer not in _OPTIMIZER_MAP:
            raise ValueError(f"optimizer must be one of {list(_OPTIMIZER_MAP.keys())}, got {optimizer}")

        if model_path is None:
            model_path = os.environ.get("MACE_POLAR_MODEL_PATH") or os.environ.get("MACE_MODEL_PATH")
        if model_path is None:
            raise ValueError("MACE model path required via model_path, MACE_POLAR_MODEL_PATH, or MACE_MODEL_PATH.")

        super().__init__(**kwargs)
        self.model_path = model_path
        self.max_opt_steps = max_opt_steps
        self.fmax = fmax
        self.optimizer = optimizer
        self.hydrogen_policy = hydrogen_policy
        self.infer_charge = infer_charge
        self.device = device
        self.skip_if_no_ligand = skip_if_no_ligand
        self.skip_on_clashes = skip_on_clashes
        self.pH = pH
        self.pocket_cutoff = pocket_cutoff
        self.clash_threshold = clash_threshold
        self.contact_threshold = contact_threshold
        self.max_energy_magnitude = max_energy_magnitude
        self.debug = debug
        self.debug_dir = Path(debug_dir) if debug_dir is not None else Path("debug_output")
        self._calculator: MACECalculator | None = None

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map validation inputs to compute() parameters."""
        return {"predicted_atom_array_stack": "predicted_atom_array"}

    def setup(self) -> None:
        """Load MACE calculator.

        Uses ``mace_polar`` factory if the model filename contains
        ``polar``, otherwise falls back to ``MACECalculator``.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected",
            )
            if "polar" in Path(self.model_path).name.lower():
                from mace.calculators import mace_polar

                self._calculator = mace_polar(
                    model=self.model_path,
                    device=self.device,
                    default_dtype="float32",
                )
                logger.info("Loaded MACE-POLAR calculator from %s", self.model_path)
            else:
                from mace.calculators import MACECalculator as _MACECalculator

                self._calculator = _MACECalculator(
                    model_paths=self.model_path,
                    device=self.device,
                    default_dtype="float32",
                )
                logger.info("Loaded MACE calculator from %s", self.model_path)

    def teardown(self) -> None:
        """Unload MACE calculator and free GPU memory."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self._calculator is not None:
            del self._calculator
            self._calculator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Unloaded MACE calculator")

    def _clip_energy(self, energy: float) -> float:
        """Clip energy to ``max_energy_magnitude`` if configured."""
        if self.max_energy_magnitude is not None and abs(energy) > self.max_energy_magnitude:
            return float(np.sign(energy) * self.max_energy_magnitude)
        return energy

    def compute(
        self,
        predicted_atom_array_stack: AtomArrayStack | AtomArray | list[AtomArray],
    ) -> dict[str, Any]:
        """Compute MACE interaction energies for all structures.

        Args:
            predicted_atom_array_stack: One or more protein-ligand complexes.

        Returns:
            Dict with aggregated medians/means and per-structure energy lists.
        """
        atom_arrays = normalize_to_atom_array_list(predicted_atom_array_stack)

        if self._calculator is None:
            raise RuntimeError("Calculator not loaded. Call setup() first.")

        interaction_energies: list[float] = []
        complex_energies: list[float] = []
        protein_energies: list[float] = []
        ligand_energies: list[float] = []
        n_opt_steps_list: list[int] = []
        converged_list: list[bool] = []

        for idx, atom_array in enumerate(atom_arrays):
            try:
                ensure_annotations(atom_array, "is_ligand", "is_protein")

                if np.sum(atom_array.is_ligand) == 0:
                    if self.skip_if_no_ligand:
                        logger.debug("Skipping structure %d: no ligand atoms", idx)
                    else:
                        logger.warning("Structure %d has no ligand atoms", idx)
                    continue

                struct_debug_dir = self.debug_dir / f"structure_{idx}" if self.debug else None
                result = compute_interaction_energy(
                    atom_array,
                    self._calculator,
                    hydrogen_policy=self.hydrogen_policy,
                    infer_charge=self.infer_charge,
                    max_opt_steps=self.max_opt_steps,
                    fmax=self.fmax,
                    optimizer=self.optimizer,
                    skip_on_clashes=self.skip_on_clashes,
                    pH=self.pH,
                    pocket_cutoff=self.pocket_cutoff,
                    clash_threshold=self.clash_threshold,
                    contact_threshold=self.contact_threshold,
                    debug=self.debug,
                    debug_dir=struct_debug_dir,
                )

                if result is None:
                    logger.debug("Skipping structure %d: pre-optimisation clashes", idx)
                    continue

                interaction_energies.append(self._clip_energy(result["interaction_energy"]))
                complex_energies.append(self._clip_energy(result["complex_energy"]))
                protein_energies.append(self._clip_energy(result["protein_energy"]))
                ligand_energies.append(self._clip_energy(result["ligand_energy"]))
                n_opt_steps_list.append(result["n_opt_steps"])
                converged_list.append(result["converged"])

            except Exception as e:
                logger.warning(
                    "Failed to compute interaction energy for structure %d: %s\n%s",
                    idx,
                    e,
                    traceback.format_exc(),
                )
                continue

        n_structures = len(interaction_energies)
        if n_structures == 0:
            return {
                "median_interaction_energy": float("nan"),
                "median_complex_energy": float("nan"),
                "median_protein_energy": float("nan"),
                "median_ligand_energy": float("nan"),
                "mean_n_opt_steps": float("nan"),
                "convergence_rate": float("nan"),
                "n_structures": 0,
                "interaction_energies": [],
                "complex_energies": [],
                "protein_energies": [],
                "ligand_energies": [],
            }

        return {
            "median_interaction_energy": float(np.median(interaction_energies)),
            "median_complex_energy": float(np.median(complex_energies)),
            "median_protein_energy": float(np.median(protein_energies)),
            "median_ligand_energy": float(np.median(ligand_energies)),
            "mean_n_opt_steps": float(np.mean(n_opt_steps_list)),
            "convergence_rate": float(np.mean(converged_list)),
            "n_structures": n_structures,
            "interaction_energies": interaction_energies,
            "complex_energies": complex_energies,
            "protein_energies": protein_energies,
            "ligand_energies": ligand_energies,
        }
