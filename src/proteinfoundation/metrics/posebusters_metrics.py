"""PoseBusters validation metric for ligand quality assessment."""

from typing import Any, Literal

import numpy as np
from atomworks.constants import METAL_ELEMENTS
from atomworks.io.tools.posebusters import DEFAULT_EXCLUDED_CHECKS, validate_molecule
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.ml.utils.timer import timeout
from beartype.typing import Sequence
from biotite.structure import AtomArray, AtomArrayStack

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger
from proteinfoundation.metrics.utils.metrics_utils import normalize_to_atom_array_list

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _ligand_has_metal(ligand_atoms: AtomArray) -> bool:
    """Check if a ligand contains any metal atoms."""
    elements = np.char.upper(np.char.strip(ligand_atoms.element.astype(str)))
    return any(e in METAL_ELEMENTS for e in elements)


def validate_single_ligand(
    ligand_atoms: AtomArray,
    ligand_id: str = "ligand",
    excluded_checks: frozenset[str] | None = None,
    timeout_seconds: float | None = 5.0,
) -> dict[str, Any]:
    """Validate a single ligand molecule with PoseBusters.

    Delegates to :py:func:`atomworks.io.tools.posebusters.validate_molecule`
    for RDKit conversion, PoseBusters execution, and metal fallback logic.

    Args:
      ligand_atoms: AtomArray containing a single ligand molecule.
      ligand_id: Identifier for logging. Defaults to ``"ligand"``.
      excluded_checks: Check names to exclude from pass/fail calculation.
        If ``None``, uses the atomworks default excluded checks.
      timeout_seconds: Timeout in seconds. Defaults to ``5.0``.
        Set to ``None`` to disable.

    Returns:
      Dictionary with check results and metadata.
    """
    if excluded_checks is None:
        excluded_checks = DEFAULT_EXCLUDED_CHECKS

    base_result: dict[str, Any] = {
        "ligand_id": ligand_id,
        "n_atoms": len(ligand_atoms),
    }

    def _run() -> tuple[bool, dict[str, bool]]:
        return validate_molecule(ligand_atoms, excluded_checks=excluded_checks)

    try:
        if timeout_seconds is not None:
            failed, check_results = timeout(timeout_seconds, strategy="signal")(_run)()
        else:
            failed, check_results = _run()
        if not check_results:
            # No checks were run (e.g., metal-only molecule with no organic fragments)
            return {**base_result, "skipped": True, "passes_validation": True}
        return {**base_result, "passes_validation": not failed, **check_results}
    except TimeoutError:
        ranked_logger.warning(f"Validation timed out for ligand {ligand_id} after {timeout_seconds}s")
        return {**base_result, "timed_out": True, "passes_validation": True}


def validate_ligands(
    atom_array: AtomArray,
    excluded_checks: Sequence[str] | None = None,
    timeout_seconds: float | None = 5.0,
) -> list[dict[str, Any]]:
    """Validate all ligands in a structure with PoseBusters.

    Identifies ligand molecules by the ``is_ligand`` annotation and validates
    each unique (chain_id, res_id) combination separately.

    Args:
      atom_array: Structure containing ligands to validate.
      excluded_checks: Check names to exclude from pass/fail calculation.
      timeout_seconds: Timeout in seconds for each ligand validation.
        Defaults to ``5.0``. Set to ``None`` to disable.

    Returns:
      List of per-ligand validation results. Returns empty list if no
      ligands found.
    """
    ensure_annotations(atom_array, "is_ligand")

    ligand_mask = atom_array.is_ligand
    if not np.any(ligand_mask):
        return []

    ligand_atoms = atom_array[ligand_mask]
    excluded_frozen = frozenset(excluded_checks) if excluded_checks else None

    # Get unique ligand molecules using chain_id + res_id combination
    chain_res_tuples = list(zip(ligand_atoms.chain_id, ligand_atoms.res_id, strict=False))
    unique_ligands = list(dict.fromkeys(chain_res_tuples))

    results = []
    for chain_id, res_id in unique_ligands:
        mol_mask = (ligand_atoms.chain_id == chain_id) & (ligand_atoms.res_id == res_id)
        single_ligand = ligand_atoms[mol_mask]

        ligand_id = f"{chain_id}/{single_ligand.res_name[0]}/{res_id}"
        result = validate_single_ligand(
            single_ligand,
            ligand_id=ligand_id,
            excluded_checks=excluded_frozen,
            timeout_seconds=timeout_seconds,
        )
        result["has_metal"] = _ligand_has_metal(single_ligand)
        results.append(result)

    return results


def aggregate_ligand_results(
    ligand_results: list[dict[str, Any]],
    output_level: Literal["aggregate", "per_check", "per_ligand"] = "per_check",
) -> dict[str, Any]:
    """Aggregate per-ligand validation results into summary statistics.

    Args:
      ligand_results: List of per-ligand validation results from
        :py:func:`validate_ligands` or :py:func:`validate_single_ligand`.
      output_level: Controls output verbosity:
        - ``"aggregate"``: Summary statistics only
        - ``"per_check"``: Adds per-check pass rates
        - ``"per_ligand"``: Adds detailed per-ligand results

    Returns:
      Aggregated results dictionary.
    """
    if not ligand_results:
        return {"pass_rate": float("nan"), "n_ligands": 0, "n_passed": 0, "n_timed_out": 0, "n_skipped": 0}

    _base_keys = {"ligand_id", "n_atoms", "passes_validation", "timed_out", "skipped", "has_metal"}

    # Only count ligands that were successfully validated (not crashed, not timed out, not skipped)
    validated_results = [
        r
        for r in ligand_results
        if not r.get("timed_out", False) and not r.get("skipped", False) and set(r.keys()) - _base_keys
    ]
    n_passed = sum(r.get("passes_validation", False) for r in validated_results)
    pass_rate = n_passed / len(validated_results) if validated_results else float("nan")

    output: dict[str, Any] = {
        "pass_rate": float(pass_rate),
        "n_ligands": len(ligand_results),
        "n_passed": n_passed,
        "n_timed_out": sum(r.get("timed_out", False) for r in ligand_results),
        "n_skipped": sum(r.get("skipped", False) for r in ligand_results),
    }

    # Add per-check pass rates if requested
    if output_level in ("per_check", "per_ligand"):
        all_check_names: set[str] = set()
        for r in validated_results:
            all_check_names.update(k for k in r if k not in _base_keys)

        for check_name in sorted(all_check_names):
            check_values = [r.get(check_name, True) for r in validated_results]
            if check_values:
                output[f"{check_name}_pass_rate"] = float(sum(check_values) / len(check_values))

    # Add per-ligand results if requested
    if output_level == "per_ligand":
        for i, r in enumerate(ligand_results):
            for k, v in r.items():
                if k == "ligand_id":
                    output[f"ligand_{i}_id"] = v
                elif isinstance(v, bool):
                    output[f"ligand_{i}_{k}"] = float(v)
                elif isinstance(v, int | float):
                    output[f"ligand_{i}_{k}"] = v

    return output


class PoseBustersMetric(Metric):
    """Validate ligand structures using PoseBusters stereochemical checks.

    Args:
      excluded_checks: Check names to exclude from pass/fail calculation.
      output_level: Controls output verbosity.
      timeout_seconds: Timeout per ligand in seconds. Defaults to ``5.0``.
      input_kwarg_names: Mapping from compute() parameter names to input kwargs.
    """

    def __init__(
        self,
        excluded_checks: Sequence[str] | None = None,
        output_level: Literal["aggregate", "per_check", "per_ligand"] = "per_check",
        timeout_seconds: float | None = 5.0,
        input_kwarg_names: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PoseBusters metric."""
        self._input_kwarg_names = input_kwarg_names or {
            "atom_array_stack": "predicted_atom_array",
        }
        self.excluded_checks = list(excluded_checks) if excluded_checks is not None else None
        self.output_level = output_level
        self.timeout_seconds = timeout_seconds
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map validation inputs to compute() parameters."""
        return self._input_kwarg_names

    def compute(self, atom_array_stack: "AtomArray | AtomArrayStack | list[AtomArray]") -> dict[str, Any]:
        """Compute PoseBusters validation for ligands in the structure(s).

        Args:
          atom_array_stack: Structure(s) containing ligands to validate.

        Returns:
          Dictionary with validation results based on ``output_level``.
        """
        atom_arrays = normalize_to_atom_array_list(atom_array_stack)

        all_ligand_results: list[dict[str, Any]] = []
        for atom_array in atom_arrays:
            ligand_results = validate_ligands(
                atom_array,
                excluded_checks=self.excluded_checks,
                timeout_seconds=self.timeout_seconds,
            )
            all_ligand_results.extend(ligand_results)

        output = aggregate_ligand_results(all_ligand_results, output_level=self.output_level)

        if all_ligand_results:
            metal_results = [r for r in all_ligand_results if r.get("has_metal", False)]
            organic_results = [r for r in all_ligand_results if not r.get("has_metal", False)]
            for prefix, subset in [("metal", metal_results), ("organic", organic_results)]:
                sub = aggregate_ligand_results(subset, output_level=self.output_level)
                for k, v in sub.items():
                    output[f"{prefix}_{k}"] = v

        return output
