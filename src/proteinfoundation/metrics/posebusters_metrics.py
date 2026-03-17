"""PoseBusters validation metric for ligand quality assessment."""

from functools import lru_cache
from typing import Any, Literal

import numpy as np
from atomworks.io.tools.rdkit import atom_array_to_rdkit
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.ml.utils.timer import timeout
from beartype.typing import Sequence
from biotite.structure import AtomArray, AtomArrayStack
from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger
from posebusters import PoseBusters

from proteinfoundation.metrics.metrics_utils import normalize_to_atom_array_list

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _get_mol_config_no_energy() -> dict:
    """Get PoseBusters 'mol' config with energy_ratio modules removed.

    Returns:
      Config dict without energy calculation (faster, no CUDA issues).
    """

    # Get the default mol config
    pb = PoseBusters(config="mol")
    config = pb.config.copy()

    # Remove energy_ratio modules (they use CUDA and are slow)
    config["modules"] = [m for m in config["modules"] if m.get("function") != "energy_ratio"]

    return config


@lru_cache(maxsize=4)
def _get_posebusters(config: str = "mol", skip_energy: bool = True) -> PoseBusters:
    """Get cached PoseBusters instance.

    Args:
      config: PoseBusters configuration preset.
      skip_energy: If True and config is "mol", skip energy_ratio checks
        (faster, avoids CUDA issues in subprocesses). Defaults to ``True``.

    Returns:
      PoseBusters instance.
    """
    if skip_energy and config == "mol":
        return PoseBusters(config=_get_mol_config_no_energy())

    return PoseBusters(config=config)


def validate_single_ligand(
    ligand_atoms: AtomArray,
    ligand_id: str = "ligand",
    config: str = "mol",
    excluded_checks: frozenset[str] | None = None,
    raise_on_error: bool = False,
    timeout_seconds: float | None = 5.0,
) -> dict[str, Any]:
    """Validate a single ligand molecule with PoseBusters.

    Args:
      ligand_atoms: AtomArray containing a single ligand molecule.
      ligand_id: Identifier for error messages. Defaults to ``"ligand"``.
      config: PoseBusters configuration preset. Defaults to ``"mol"``.
      excluded_checks: Check names to exclude from pass/fail calculation.
      raise_on_error: If True, raise exceptions on validation failure.
      timeout_seconds: Timeout in seconds for validation. Defaults to ``30.0``.
        Set to ``None`` to disable timeout.

    Returns:
      Dictionary with check results and metadata:
        - ligand_id: Identifier string
        - n_atoms: Number of atoms in ligand
        - pb_bust_failed: True if PoseBusters failed (unusual chemistry)
        - rdkit_conversion_failed: True if RDKit conversion failed
        - timed_out: True if validation timed out
        - passes_validation: True if all checks passed
        - Additional boolean keys for each PoseBusters check

    Examples:
      >>> from atomworks.ml.utils.testing import cached_parse
      >>> atom_array = cached_parse("6wtf")["atom_array"]
      >>> ligand = atom_array[atom_array.res_name == "SA8"]
      >>> result = validate_single_ligand(ligand)
      >>> result["passes_validation"]
      True
    """

    excluded_checks = excluded_checks or frozenset()

    base_result: dict[str, Any] = {
        "ligand_id": ligand_id,
        "n_atoms": len(ligand_atoms),
        "pb_bust_failed": False,
        "rdkit_conversion_failed": False,
        "timed_out": False,
    }

    # Convert to RDKit molecule
    try:
        mol = atom_array_to_rdkit(
            ligand_atoms,
            set_coord=True,
            hydrogen_policy="keep",
            infer_bonds=False,
            sanitize=True,
            attempt_fixing_corrupted_molecules=True,
        )
    except Exception as e:
        msg = f"RDKit conversion failed for ligand {ligand_id}: {e}"
        if raise_on_error:
            raise ValueError(msg) from e
        ranked_logger.warning(msg)
        return {
            **base_result,
            "rdkit_conversion_failed": True,
            "passes_validation": False,
        }

    # Run PoseBusters validation with optional timeout
    # Use "signal" strategy since PoseBusters uses CUDA internally and "subprocess" (fork)
    # cannot re-initialize CUDA in forked processes
    def _run_posebusters():  # noqa: ANN202
        pb = _get_posebusters(config)
        results_df = pb.bust(mol_pred=mol, full_report=False)
        return results_df.iloc[0].to_dict()

    try:
        if timeout_seconds is not None:
            pb_results = timeout(timeout_seconds, strategy="signal")(_run_posebusters)()
        else:
            pb_results = _run_posebusters()

        # Extract boolean check results
        check_results = {
            k: bool(v) for k, v in pb_results.items() if isinstance(v, bool | int | np.bool_) and k != "file_path"
        }

        # Filter excluded checks for pass/fail determination
        relevant_checks = {k: v for k, v in check_results.items() if k not in excluded_checks}
        passes_validation = all(relevant_checks.values()) if relevant_checks else True

        return {
            **base_result,
            "passes_validation": passes_validation,
            **check_results,
        }

    except TimeoutError:
        msg = f"PoseBusters validation timed out for ligand {ligand_id} after {timeout_seconds}s"
        ranked_logger.warning(msg)
        return {
            **base_result,
            "timed_out": True,
            "passes_validation": True,  # Not a failure, just couldn't check
        }
    except Exception as e:
        msg = f"PoseBusters validation failed for ligand {ligand_id}: {e}"
        if raise_on_error:
            raise ValueError(msg) from e
        ranked_logger.warning(msg)
        return {
            **base_result,
            "pb_bust_failed": True,
            "passes_validation": True,  # Not a failure, just couldn't check
        }


def validate_ligands(
    atom_array: AtomArray,
    config: str = "mol",
    excluded_checks: Sequence[str] | None = None,
    raise_on_error: bool = False,
    timeout_seconds: float | None = 5.0,
) -> list[dict[str, Any]]:
    """Validate all ligands in a structure with PoseBusters.

    Identifies ligand molecules by the ``is_ligand`` annotation and validates
    each unique (chain_id, res_id) combination separately.

    Args:
      atom_array: Structure containing ligands to validate.
      config: PoseBusters configuration preset. Use ``"mol"`` for standalone
        molecules, ``"dock"`` for docked poses, ``"gen"`` for generated.
        Defaults to ``"mol"``.
      excluded_checks: Check names to exclude from pass/fail calculation.
      raise_on_error: If True, raise exceptions on validation failure.
      timeout_seconds: Timeout in seconds for each ligand validation.
        Defaults to ``30.0``. Set to ``None`` to disable.

    Returns:
      List of per-ligand validation results. Each dict contains:
        - ligand_id: Chain/resname/resid identifier
        - n_atoms: Number of atoms
        - pb_bust_failed: True if PoseBusters failed (unusual chemistry)
        - rdkit_conversion_failed: True if RDKit conversion failed
        - passes_validation: True if all checks passed
        - Additional boolean keys for each PoseBusters check

      Returns empty list if no ligands found.

    Examples:
      >>> from atomworks.ml.utils.testing import cached_parse
      >>> atom_array = cached_parse("6wtf")["atom_array"]
      >>> results = validate_ligands(atom_array)
      >>> len(results)
      4
      >>> sum(r["passes_validation"] for r in results)
      4
    """

    ensure_annotations(atom_array, "is_ligand")

    ligand_mask = atom_array.is_ligand
    if not np.any(ligand_mask):
        return []

    ligand_atoms = atom_array[ligand_mask]
    excluded_checks_frozen = frozenset(excluded_checks) if excluded_checks else None

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
            config=config,
            excluded_checks=excluded_checks_frozen,
            raise_on_error=raise_on_error,
            timeout_seconds=timeout_seconds,
        )
        results.append(result)

    return results


def aggregate_ligand_results(
    ligand_results: list[dict[str, Any]],
    output_level: Literal["aggregate", "per_check", "per_ligand"] = "aggregate",
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
      Aggregated results dictionary with:
        - pass_rate: Fraction of validated ligands that passed
        - n_ligands: Total number of ligands
        - n_passed: Number that passed validation
        - n_pb_bust_failed: Number where PoseBusters failed
        - n_rdkit_conversion_failed: Number where RDKit conversion failed
        - Additional keys based on output_level

    Examples:
      >>> results = [{"passes_validation": True, "pb_bust_failed": False, ...}]
      >>> aggregate_ligand_results(results)
      {'pass_rate': 1.0, 'n_ligands': 1, 'n_passed': 1, ...}
    """
    if not ligand_results:
        return {
            "pass_rate": float("nan"),
            "n_ligands": 0,
            "n_passed": 0,
            "n_pb_bust_failed": 0,
            "n_rdkit_conversion_failed": 0,
            "n_timed_out": 0,
        }

    n_ligands = len(ligand_results)
    n_pb_bust_failed = sum(r.get("pb_bust_failed", False) for r in ligand_results)
    n_rdkit_failed = sum(r.get("rdkit_conversion_failed", False) for r in ligand_results)
    n_timed_out = sum(r.get("timed_out", False) for r in ligand_results)

    # Only count ligands that were successfully validated (not failed, not timed out)
    validated_results = [
        r
        for r in ligand_results
        if not r.get("pb_bust_failed", False)
        and not r.get("rdkit_conversion_failed", False)
        and not r.get("timed_out", False)
    ]
    n_passed = sum(r.get("passes_validation", False) for r in validated_results)
    pass_rate = n_passed / len(validated_results) if validated_results else float("nan")

    output: dict[str, Any] = {
        "pass_rate": float(pass_rate),
        "n_ligands": n_ligands,
        "n_passed": n_passed,
        "n_pb_bust_failed": n_pb_bust_failed,
        "n_rdkit_conversion_failed": n_rdkit_failed,
        "n_timed_out": n_timed_out,
    }

    # Add per-check pass rates if requested
    if output_level in ("per_check", "per_ligand"):
        all_check_names: set[str] = set()
        for r in validated_results:
            for k in r:
                if k not in (
                    "ligand_id",
                    "n_atoms",
                    "pb_bust_failed",
                    "rdkit_conversion_failed",
                    "passes_validation",
                ):
                    all_check_names.add(k)

        for check_name in sorted(all_check_names):
            check_values = [r.get(check_name, True) for r in validated_results]
            if check_values:
                check_pass_rate = sum(check_values) / len(check_values)
                output[f"{check_name}_pass_rate"] = float(check_pass_rate)

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

    Iterates over individual ligand molecules in a structure and runs PoseBusters
    validation on each. For direct function access without instantiating a metric,
    use :py:func:`validate_ligands` or :py:func:`validate_single_ligand`.

    Args:
      config: PoseBusters configuration preset. Use ``"mol"`` for standalone
        molecules, ``"dock"`` for docked poses, ``"gen"`` for generated molecules.
        Defaults to ``"mol"``.
      excluded_checks: Check names to exclude from pass/fail calculation.
        Defaults to ``None`` (no exclusions).
      raise_on_error: If True, raise exceptions when validation fails. If False,
        log a warning and mark as ``pb_bust_failed=True``. Defaults to ``False``.
      output_level: Controls output verbosity:
        - ``"aggregate"``: Return only summary statistics (pass_rate, n_ligands, etc.)
        - ``"per_check"``: Include per-check pass rates across all ligands
        - ``"per_ligand"``: Include detailed results for each individual ligand
        Defaults to ``"aggregate"``.
      input_kwarg_names: Mapping from compute() parameter names to input kwarg names.
        Defaults to ``{"atom_array_stack": "predicted_atom_array"}``.

    Examples:
      >>> metric = PoseBustersMetric(output_level="aggregate")
      >>> result = metric.compute(atom_array_stack)
      >>> print(result)
      {'pass_rate': 0.85, 'n_ligands': 10, 'n_passed': 8, 'n_pb_bust_failed': 1}
    """

    def __init__(
        self,
        config: str = "mol",
        excluded_checks: Sequence[str] | None = None,
        raise_on_error: bool = False,
        output_level: Literal["aggregate", "per_check", "per_ligand"] = "aggregate",
        input_kwarg_names: dict[str, str] | None = None,
        timeout_seconds: float | None = 5.0,
        **kwargs: Any,
    ) -> None:
        """Initialize PoseBusters metric."""
        # Default to standard Proteina input names
        self._input_kwarg_names = input_kwarg_names or {
            "atom_array_stack": "predicted_atom_array",
        }

        self.config = config
        self.excluded_checks = list(excluded_checks) if excluded_checks is not None else None
        self.raise_on_error = raise_on_error
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
          Dictionary with validation results. Contents depend on ``output_level``:

          - ``"aggregate"``: ``pass_rate``, ``n_ligands``, ``n_passed``, ``n_pb_bust_failed``
          - ``"per_check"``: Adds ``{check_name}_pass_rate`` for each PoseBusters check
          - ``"per_ligand"``: Adds ``ligand_{i}_{check_name}`` for each ligand and check
        """

        atom_arrays = normalize_to_atom_array_list(atom_array_stack)

        # Process all structures using standalone function
        all_ligand_results: list[dict[str, Any]] = []
        for atom_array in atom_arrays:
            ligand_results = validate_ligands(
                atom_array,
                config=self.config,
                excluded_checks=self.excluded_checks,
                raise_on_error=self.raise_on_error,
                timeout_seconds=self.timeout_seconds,
            )
            all_ligand_results.extend(ligand_results)

        return aggregate_ligand_results(all_ligand_results, output_level=self.output_level)
