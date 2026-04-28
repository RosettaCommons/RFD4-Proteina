"""In-Pocket Analysis (IPA) interaction energy metric for protein-ligand complexes.

Uses GFN2-xTB via the IPA toolkit to compute pairwise ligand-residue
interaction energies with bias-constrained geometry optimization::

    E_int(residue) = E(dimer) - E(residue) - E(ligand) - E(cap_correction)
    E_total = sum of selected per-residue E_int terms

Installation:
    1. Clone and compile In-Pocket-Analysis::

        git clone <repo-url> /path/to/In-Pocket-Analysis
        cd /path/to/In-Pocket-Analysis
        python install.py          # compiles C++ with g++ -O3

    2. Set the ``IPA_PATH`` environment variable::

        export IPA_PATH=/path/to/In-Pocket-Analysis

    3. Install IPA's Python dependencies (``gemmi``, ``rdkit``)::

        pip install gemmi
        conda install -c conda-forge rdkit   # or pip install rdkit

References:
    - IPA method: https://doi.org/10.1101/2025.09.24.678379
    - ULYSSES (GFN2-xTB engine): https://doi.org/10.1021/acs.jcim.2c00757
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
import tempfile
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
from atomworks.ml.conditions.annotator import ensure_annotations
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import pdbx

from foundry.metrics.metric import Metric
from proteinfoundation.metrics.metrics_utils import normalize_to_atom_array_list

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_all_ligand_residues(atom_array: AtomArray) -> list[dict[str, Any]]:
    """Extract all unique ligand residues from ``is_ligand`` annotation.

    Args:
        atom_array: Structure with ``is_ligand`` annotation.

    Returns:
        List of dicts, each with ``active_residue`` (int), ``chain`` (str),
        and ``active_residue_name`` (str). One entry per unique
        ``(chain_id, res_id)`` ligand residue.

    Raises:
        ValueError: If no ligand atoms found.
    """
    ensure_annotations(atom_array, "is_ligand")
    lig = atom_array[atom_array.is_ligand.astype(bool)]
    if len(lig) == 0:
        raise ValueError("No ligand atoms found (is_ligand all False)")

    seen: set[tuple[str, int]] = set()
    result: list[dict[str, Any]] = []
    for i in range(len(lig)):
        key = (str(lig.chain_id[i]), int(lig.res_id[i]))
        if key not in seen:
            seen.add(key)
            result.append(
                {
                    "active_residue": int(lig.res_id[i]),
                    "chain": str(lig.chain_id[i]),
                    "active_residue_name": str(lig.res_name[i]),
                }
            )
    return result


def _parse_ipa_csv(csv_path: Path) -> list[list[dict[str, Any]]]:
    """Parse an IPA ``interaction_energies.csv`` file.

    The CSV uses semicolon separators with blocks per ligand chemical
    state, each preceded by a SMILES header line.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of per-state blocks, where each block is a list of
        per-residue dicts with keys: ``res``, ``chain``, ``resn``,
        ``charge``, ``flip``, ``eint_kcal``, ``cap_corr_kcal``.
        The first block corresponds to the first (most relevant)
        chemical state.
    """
    all_blocks: list[list[dict[str, Any]]] = []
    if not csv_path.exists():
        logger.warning("IPA CSV not found: %s", csv_path)
        return all_blocks

    current_block: list[dict[str, Any]] = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # SMILES header line starts a new block (contains ":" early)
            if line.endswith(":") and ";" not in line:
                if current_block:
                    all_blocks.append(current_block)
                    current_block = []
                continue
            if line.startswith("res;"):
                continue
            parts = line.split(";")
            if len(parts) < 8:
                continue
            try:
                current_block.append(
                    {
                        "res": parts[0],
                        "chain": parts[1],
                        "resn": int(parts[2]),
                        "charge": parts[3],
                        "flip": parts[5].strip().lower() == "true",
                        "eint_kcal": float(parts[6]),
                        "cap_corr_kcal": float(parts[7]),
                    }
                )
            except (ValueError, IndexError):
                continue

    if current_block:
        all_blocks.append(current_block)
    return all_blocks


def _parse_ipa_stats(stats_path: Path) -> list[dict[str, float]]:
    """Parse an IPA ``stats.txt`` file.

    Each block corresponds to one ligand chemical state (keyed by SMILES).

    Args:
        stats_path: Path to the stats file.

    Returns:
        List of dicts per chemical state with keys:
        ``deformation_energy``, ``ipa_interaction_energy``,
        ``ipa_binding_energy``.
    """
    results: list[dict[str, float]] = []
    if not stats_path.exists():
        logger.warning("IPA stats not found: %s", stats_path)
        return results

    current: dict[str, float] = {}
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    results.append(current)
                    current = {}
                continue
            m = re.match(r"Deformation Energy:\s+([-\d.]+)", line)
            if m:
                current["deformation_energy"] = float(m.group(1))
            m = re.match(r"IPA Interaction Energy:\s+([-\d.]+)", line)
            if m:
                current["ipa_interaction_energy"] = float(m.group(1))
            m = re.match(r"IPA Binding Energy:\s+([-\d.]+)", line)
            if m:
                current["ipa_binding_energy"] = float(m.group(1))

    if current:
        results.append(current)
    return results


# ---------------------------------------------------------------------------
# IPA import helper
# ---------------------------------------------------------------------------


def _import_ipa(
    ipa_path: str | None = None,
) -> tuple[type, type]:
    """Lazily import IPA modules, adding ``IPA_PATH`` to sys.path if needed.

    Args:
        ipa_path: Path to the In-Pocket-Analysis repo root.  Falls back
            to the ``IPA_PATH`` environment variable.

    Returns:
        Tuple of ``(AnalysisConfig, InPocketAnalyzer)`` classes.

    Raises:
        ImportError: If IPA cannot be imported.
    """
    ipa_path = ipa_path or os.environ.get("IPA_PATH")
    if ipa_path is None:
        raise ImportError(
            "IPA_PATH not set.  Set the IPA_PATH environment variable to "
            "the In-Pocket-Analysis repository root (with compiled binaries)."
        )
    resolved = str(Path(ipa_path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

    from in_pocket import AnalysisConfig, InPocketAnalyzer

    return AnalysisConfig, InPocketAnalyzer


# ---------------------------------------------------------------------------
# Standalone compute function
# ---------------------------------------------------------------------------


def compute_ipa_interaction_energy(
    atom_array: AtomArray,
    ipa_path: str | None = None,
    *,
    protonation: Literal["rdkit", "ccd", "vsepr", "obabel"] = "rdkit",
    distance: float = 5.0,
    alpha: float = 1.0,
    kappa: float = 0.025,
    pH: float = 7.0,
    use_ip_structure: bool = True,
    use_ip_residues: bool = False,
    tautomerise: bool = False,
    eval_only: bool = False,
    no_solvation: bool = False,
    ligand_states_selection: Literal["all", "first"] = "all",
    debug: bool = False,
    debug_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Compute IPA interaction energy for a protein-ligand complex.

    Writes the structure to a temporary CIF file, runs the full IPA
    pipeline (protonation, neighbor finding, pairwise QM energies), and
    parses the results.

    Args:
        atom_array: Structure with ``is_ligand`` and ``is_protein``
            annotations.  Should carry author-field numbering
            (``chain_id`` = auth_asym_id, ``res_id`` = auth_seq_id)
            for correct IPA terminus/capping detection.
        ipa_path: Path to IPA repo.  Falls back to ``IPA_PATH`` env var.
        protonation: Protonation algorithm to use.
        distance: Neighbor search radius in Angstrom.
        alpha: IPA bias potential alpha parameter.
        kappa: IPA bias potential kappa parameter.
        pH: Target pH for protonation. Defaults to ``7.0``.
        use_ip_structure: Use IPA-optimized ligand structures for energy
            evaluation.
        use_ip_residues: Use IPA-optimized residue structures.
            Defaults to ``False`` (matching IPA CLI default).
        tautomerise: Enumerate and evaluate tautomers.
        eval_only: Skip geometry optimization (single-point only).
        no_solvation: Disable ALPB implicit solvation.
        ligand_states_selection: Which protonation states to evaluate.
            ``"all"`` evaluates all Dimorphite-generated states,
            ``"first"`` uses only the first (fastest). Defaults to
            ``"all"``.
        debug: Save intermediate files.
        debug_dir: Directory for debug output.

    Returns:
        Dictionary with keys:

        - ``total_interaction_energy`` (float): IPA total interaction
          energy in kcal/mol (sum of selected per-residue terms).
        - ``ipa_binding_energy`` (float): IPA binding energy in kcal/mol
          (interaction + deformation + chemical state).
        - ``deformation_energy`` (float): Deformation energy in kcal/mol.
        - ``per_residue_interactions`` (list[dict]): Per-residue results
          with keys ``res``, ``chain``, ``resn``, ``charge``, ``flip``,
          ``eint_kcal``, ``cap_corr_kcal``.

        Returns ``None`` if computation fails.
    """
    AnalysisConfig, InPocketAnalyzer = _import_ipa(ipa_path)

    ensure_annotations(atom_array, "is_ligand", "is_protein")
    ligand_residues = _extract_all_ligand_residues(atom_array)

    ligand_names = [f"{l['active_residue_name']}:{l['chain']}:{l['active_residue']}" for l in ligand_residues]
    logger.info("Processing %d ligand residue(s): %s", len(ligand_residues), ", ".join(ligand_names))

    from in_pocket import ProtonationMode

    prot_mode_map = {
        "rdkit": ProtonationMode.RDKIT,
        "ccd": ProtonationMode.CCD,
        "vsepr": ProtonationMode.VSEPR,
        "obabel": ProtonationMode.OBABEL,
    }

    with tempfile.TemporaryDirectory(prefix="ipa_metric_") as tmpdir:
        tmpdir = Path(tmpdir)

        cif_path = tmpdir / "structure.cif"
        cif_file = pdbx.CIFFile()
        pdbx.set_structure(cif_file, atom_array, data_block="structure")
        cif_file.write(str(cif_path))

        total_e_int = 0.0
        total_e_bind = 0.0
        total_e_def = 0.0
        all_per_residue: list[dict] = []

        for lig_idx, ligand_info in enumerate(ligand_residues):
            output_dir = tmpdir / f"output_lig{lig_idx}"
            output_dir.mkdir()

            config = AnalysisConfig(
                pdb_file=cif_path,
                active_residue=ligand_info["active_residue"],
                active_residue_chain=ligand_info["chain"],
                active_residue_name=ligand_info["active_residue_name"],
                output_folder=output_dir,
                critical_distance=distance,
                ph=pH,
                use_ip_structure=use_ip_structure,
                use_ip_residues=use_ip_residues,
                protonation_mode=prot_mode_map.get(protonation, ProtonationMode.RDKIT),
                tautomerise=tautomerise,
                evaluate_only=eval_only,
                debug=debug,
            )

            object.__setattr__(config.qm_params, "alpha", alpha)
            object.__setattr__(config.qm_params, "kappa", kappa)
            if no_solvation:
                object.__setattr__(config.qm_params, "use_solvation", False)

            analyzer = InPocketAnalyzer(config)

            original_stdin = sys.stdin
            sys.stdin = io.StringIO("first\n") if ligand_states_selection == "first" else io.StringIO()
            try:
                analyzer.run_analysis()
            finally:
                sys.stdin = original_stdin

            csv_path = output_dir / "interaction_energies.csv"
            stats_path = output_dir / "stats.txt"

            per_residue_blocks = _parse_ipa_csv(csv_path)
            stats_list = _parse_ipa_stats(stats_path)

            if not stats_list:
                logger.warning("No stats found for ligand %s", ligand_names[lig_idx])
                continue

            best_stats = stats_list[0]
            total_e_int += best_stats.get("ipa_interaction_energy", 0.0)
            total_e_bind += best_stats.get("ipa_binding_energy", 0.0)
            total_e_def += best_stats.get("deformation_energy", 0.0)
            if per_residue_blocks:
                all_per_residue.extend(per_residue_blocks[0])

            if debug and debug_dir is not None:
                import shutil

                lig_debug_dir = Path(debug_dir) / f"ligand_{lig_idx}"
                lig_debug_dir.mkdir(parents=True, exist_ok=True)
                for f in output_dir.rglob("*"):
                    if f.is_file():
                        dest = lig_debug_dir / f.relative_to(output_dir)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dest)

        if total_e_int == 0.0 and total_e_bind == 0.0:
            logger.warning("No IPA results for any ligand residue")
            return None

        return {
            "total_interaction_energy": total_e_int,
            "ipa_binding_energy": total_e_bind,
            "deformation_energy": total_e_def,
            "per_residue_interactions": all_per_residue,
        }


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------


class QMInteractionEnergyIPA(Metric):
    """Compute IPA interaction energies for protein-ligand complexes.

    Uses the In-Pocket Analysis toolkit (GFN2-xTB) to compute pairwise
    ligand-residue interaction energies with bias-constrained geometry
    optimization.  Follows the same ``Metric`` pattern as
    :class:`MACEInteractionEnergyMetric`.

    The workflow for each structure:

    1. Write structure to temporary CIF
    2. Run IPA pipeline (protonation, neighbor finding, QM energies)
    3. Parse per-residue interaction energies and summary statistics

    Requires ``IPA_PATH`` environment variable pointing to a compiled
    In-Pocket-Analysis repository (see module docstring for setup).
    """

    def __init__(
        self,
        ipa_path: str | None = None,
        protonation: Literal["rdkit", "ccd", "vsepr", "obabel"] = "rdkit",
        distance: float = 5.0,
        alpha: float = 1.0,
        kappa: float = 0.025,
        pH: float = 7.0,
        use_ip_structure: bool = True,
        use_ip_residues: bool = False,
        tautomerise: bool = False,
        eval_only: bool = False,
        skip_if_no_ligand: bool = True,
        no_solvation: bool = False,
        ligand_states_selection: Literal["all", "first"] = "all",
        debug: bool = False,
        debug_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize IPA interaction energy metric.

        Args:
            ipa_path: Path to IPA repo.  Falls back to ``IPA_PATH`` env var.
            protonation: Protonation algorithm (``"rdkit"``, ``"ccd"``,
                ``"vsepr"``, or ``"obabel"``).
            distance: Neighbor search radius in Angstrom.
            alpha: IPA bias potential alpha parameter.
            kappa: IPA bias potential kappa parameter.
            pH: Target pH for protonation. Defaults to ``7.0``.
            use_ip_structure: Use IPA-optimized ligand structures.
            use_ip_residues: Use IPA-optimized residue structures.
                Defaults to ``False``.
            tautomerise: Enumerate tautomers.
            eval_only: Skip geometry optimization (single-point only).
            skip_if_no_ligand: Skip structures without ligand atoms.
            no_solvation: Disable ALPB implicit solvation.
            ligand_states_selection: Which protonation states to evaluate.
                ``"all"`` evaluates all states (default), ``"first"``
                uses only the first (fastest).
            debug: Save intermediate files to ``debug_dir``.
            debug_dir: Directory for debug output.
        """
        super().__init__(**kwargs)
        self.ipa_path = ipa_path or os.environ.get("IPA_PATH")
        self.protonation = protonation
        self.distance = distance
        self.alpha = alpha
        self.kappa = kappa
        self.pH = pH
        self.use_ip_structure = use_ip_structure
        self.use_ip_residues = use_ip_residues
        self.tautomerise = tautomerise
        self.eval_only = eval_only
        self.skip_if_no_ligand = skip_if_no_ligand
        self.no_solvation = no_solvation
        self.ligand_states_selection = ligand_states_selection
        self.debug = debug
        self.debug_dir = Path(debug_dir) if debug_dir else Path("ipa_debug_output")
        self._ipa_imported = False

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map validation inputs to compute() parameters."""
        return {
            "predicted_atom_array_stack": "predicted_atom_array",
        }

    def setup(self) -> None:
        """Verify IPA is available and import it.

        Initializes the ``IPAExecutable`` (checks binary exists) and
        imports IPA's ``InPocketAnalyzer`` via ``sys.path``.
        """
        from proteinfoundation.metrics.ipa_qm import IPAExecutable

        IPAExecutable.get_or_initialize()
        _import_ipa(self.ipa_path)
        self._ipa_imported = True
        logger.info("IPA metric initialized (IPA_PATH=%s)", self.ipa_path)

    def teardown(self) -> None:
        """Clean up after validation."""
        self._ipa_imported = False

    def compute(
        self,
        predicted_atom_array_stack: AtomArrayStack | AtomArray | list[AtomArray],
    ) -> dict[str, Any]:
        """Compute IPA interaction energies for all structures.

        Args:
            predicted_atom_array_stack: Predicted protein-ligand complexes.
                Structures should have ``is_ligand`` and ``is_protein``
                annotations.

        Returns:
            Dictionary with aggregated and per-structure results:

            - ``median_total_interaction_energy``: Median IPA interaction
              energy across structures (kcal/mol).
            - ``median_ipa_binding_energy``: Median IPA binding energy
              (kcal/mol).
            - ``n_structures``: Number of structures processed.
            - ``total_interaction_energies``: Per-structure E_int list.
            - ``ipa_binding_energies``: Per-structure binding energy list.
            - ``deformation_energies``: Per-structure deformation list.
            - ``per_residue_interactions``: Per-structure list of
              per-residue dicts (see :func:`_parse_ipa_csv`).
        """
        atom_arrays = normalize_to_atom_array_list(predicted_atom_array_stack)

        if not self._ipa_imported:
            raise RuntimeError("IPA not imported. Ensure setup() is called before compute().")

        total_interaction_energies: list[float] = []
        ipa_binding_energies: list[float] = []
        deformation_energies: list[float] = []
        all_per_residue: list[list[dict[str, Any]]] = []

        for idx, atom_array in enumerate(atom_arrays):
            try:
                ensure_annotations(atom_array, "is_ligand", "is_protein")

                n_ligand = np.sum(atom_array.is_ligand)
                if n_ligand == 0:
                    if self.skip_if_no_ligand:
                        logger.debug("Skipping structure %d: no ligand atoms", idx)
                    else:
                        logger.warning("Structure %d has no ligand atoms", idx)
                    continue

                struct_debug_dir = self.debug_dir / f"structure_{idx}" if self.debug else None
                result = compute_ipa_interaction_energy(
                    atom_array,
                    ipa_path=self.ipa_path,
                    protonation=self.protonation,
                    distance=self.distance,
                    alpha=self.alpha,
                    kappa=self.kappa,
                    pH=self.pH,
                    use_ip_structure=self.use_ip_structure,
                    use_ip_residues=self.use_ip_residues,
                    tautomerise=self.tautomerise,
                    eval_only=self.eval_only,
                    no_solvation=self.no_solvation,
                    ligand_states_selection=self.ligand_states_selection,
                    debug=self.debug,
                    debug_dir=struct_debug_dir,
                )

                if result is None:
                    logger.debug("Skipping structure %d: IPA returned no result", idx)
                    continue

                total_interaction_energies.append(result["total_interaction_energy"])
                ipa_binding_energies.append(result["ipa_binding_energy"])
                deformation_energies.append(result["deformation_energy"])
                all_per_residue.append(result["per_residue_interactions"])

            except Exception as e:
                logger.warning("Failed to compute IPA energy for structure %d: %s", idx, e)
                continue

        n_structures = len(total_interaction_energies)

        if n_structures == 0:
            return {
                "median_total_interaction_energy": float("nan"),
                "median_ipa_binding_energy": float("nan"),
                "n_structures": 0,
                "total_interaction_energies": [],
                "ipa_binding_energies": [],
                "deformation_energies": [],
                "per_residue_interactions": [],
            }

        return {
            "median_total_interaction_energy": float(np.median(total_interaction_energies)),
            "median_ipa_binding_energy": float(np.median(ipa_binding_energies)),
            "n_structures": n_structures,
            "total_interaction_energies": total_interaction_energies,
            "ipa_binding_energies": ipa_binding_energies,
            "deformation_energies": deformation_energies,
            "per_residue_interactions": all_per_residue,
        }
