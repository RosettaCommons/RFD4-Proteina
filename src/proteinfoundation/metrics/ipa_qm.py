"""GFN2-xTB In-Pocket Analysis QM wrapper.

Provides a Python interface to the IPA GFN2-xTB executable for computing
single-point energies and bias-constrained geometry optimizations of
protein-ligand systems.  The executable is resolved via the ``IPA_PATH``
environment variable using the atomworks :class:`Executable` pattern.

See https://doi.org/10.1101/2025.09.24.678379 for the IPA method.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np
from atomworks.ml.executables import Executable, ExecutableError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Executable wrapper
# ---------------------------------------------------------------------------

_CALCULATION_TYPE_FLAGS: dict[str, str] = {
    "in_pocket": "1",
    "full_opt": "0",
    "single_point": "-1",
}


class IPAExecutable(Executable):
    """Executable wrapper for the In-Pocket Analysis GFN2-xTB binary.

    IPA uses the ULYSSES GFN2-xTB implementation for quantum-based
    protein-ligand interaction energy calculations.  The executable is
    compiled from the In-Pocket-Analysis repository with
    ``python install.py``.

    Examples:
        >>> ipa = IPAExecutable.get_or_initialize()
        >>> bin_path = ipa.get_bin_path()
    """

    name = "gfn2xtb_in_pocket_opt"
    # The IPA executable has no --help/--version flags; it expects
    # positional arguments.  Setting required_verification_text to None
    # skips the subprocess probe (file existence and execute permission
    # are still checked).
    required_verification_text = None
    verification_cmd = "--help"
    version_cmd = "--version"

    @classmethod
    def initialize(
        cls,
        bin_path: PathLike | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> IPAExecutable:
        """Initialize the IPA executable.

        Args:
            bin_path: Path to ``gfn2xtb_in_pocket_opt.exe``.  If ``None``,
                inferred from the ``IPA_PATH`` environment variable.

        Returns:
            Initialized IPAExecutable.

        Raises:
            ExecutableError: If executable not found or invalid.
        """
        if bin_path is None:
            bin_path = cls._infer_bin_path_from_env_var()
        return super().initialize(bin_path, *args, **kwargs)

    @staticmethod
    def _infer_bin_path_from_env_var() -> PathLike:
        """Resolve the GFN2-xTB executable from ``IPA_PATH`` env var."""
        ipa_path = os.environ.get("IPA_PATH")
        if ipa_path is not None:
            bin_path = os.path.join(ipa_path, "gfn2xtb_in_pocket_opt.exe")
            if os.path.isfile(bin_path) and os.access(bin_path, os.X_OK):
                return bin_path

        raise ExecutableError(
            "No `bin_path` provided and `IPA_PATH` environment variable not set "
            "or does not contain a compiled gfn2xtb_in_pocket_opt.exe.\n"
            "Set IPA_PATH to the In-Pocket-Analysis repo root "
            "(after running `python install.py` to compile the C++ binaries), "
            "or provide a `bin_path` to `IPAExecutable.initialize(bin_path=...)`."
        )

    @classmethod
    def _setup(cls, bin_path: PathLike, *args: Any, **kwargs: Any) -> None:
        """No special setup required for IPA."""


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def parse_ipa_output(stdout: bytes) -> dict[str, Any]:
    """Parse GFN2-xTB stdout for energy, charges, and geometry info.

    The output format is defined by the ULYSSES GFN2-xTB implementation
    used in In-Pocket Analysis.

    Args:
        stdout: Raw stdout bytes from the GFN2-xTB executable.

    Returns:
        Dictionary with keys:

        - ``energy_hartree`` (float): Total energy in Hartree.
        - ``rmsd`` (float): RMSD with initial structure in Angstrom.
        - ``total_polarizability`` (float): Total polarizability.
        - ``atom_indices`` (list[int]): Atom indices from output.
        - ``charges`` (np.ndarray): Per-atom Mulliken charges.
        - ``polarizabilities`` (np.ndarray): Per-atom polarizabilities.
        - ``converged`` (bool): Whether the optimization converged.
    """
    lines = stdout.decode("utf-8", errors="replace").split("\n")

    energy = 0.0
    rmsd = 0.0
    total_pol = 0.0
    atom_indices: list[int] = []
    charges: list[float] = []
    polarizabilities: list[float] = []
    converged = True
    fetch_atomic_props = False

    for line in lines:
        if line.startswith(">atom;charge;pol"):
            fetch_atomic_props = True
            continue
        if line.startswith("<atom;charge;pol"):
            fetch_atomic_props = False
            continue

        if line.startswith("Optimized g energy:"):
            energy = float(line.replace("Optimized g energy:", "").replace("Hartree", "").strip())
        elif line.startswith("RMSD with initial structure:"):
            rmsd = float(line.replace("RMSD with initial structure:", "").strip())
        elif "Total Polarizability" in line:
            total_pol = float(line.replace("Total Polarizability", "").strip())
        elif fetch_atomic_props:
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 4:
                atom_indices.append(int(parts[0]))
                charges.append(float(parts[2]))
                polarizabilities.append(float(parts[3]))
        elif "no convergence in" in line:
            converged = False

    return {
        "energy_hartree": energy,
        "rmsd": rmsd,
        "total_polarizability": total_pol,
        "atom_indices": atom_indices,
        "charges": np.array(charges, dtype=np.float64),
        "polarizabilities": np.array(polarizabilities, dtype=np.float64),
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Main calculation runner
# ---------------------------------------------------------------------------


def run_in_pocket_calculation(
    geometry_path: Path | str,
    charge: int,
    *,
    alpha: float = 1.0,
    kappa: float = 0.025,
    include_h_in_bias: bool = False,
    calculation_type: Literal["in_pocket", "full_opt", "single_point"] = "in_pocket",
    electronic_temperature: float = 300.0,
    solvation: bool = True,
    solvent: str = "water",
    output_geometry_path: Path | str | None = None,
    energy_threshold: float = 5e-6,
    gradient_threshold: float = 5e-3,
) -> dict[str, Any]:
    """Run a GFN2-xTB in-pocket calculation via the IPA executable.

    This calls the ``gfn2xtb_in_pocket_opt.exe`` binary with the given
    parameters and parses the resulting output.

    Args:
        geometry_path: Path to input XYZ geometry file.
        charge: Total system charge (integer).
        alpha: Bias potential alpha parameter controlling the strength of
            the restraint toward the reference structure.
        kappa: Bias potential kappa parameter controlling the steepness
            of the restraint.
        include_h_in_bias: If ``True``, include hydrogen atoms in the
            bias potential. If ``False`` (default), H atoms are freely
            optimized.
        calculation_type: Type of calculation:
            ``"in_pocket"`` -- biased optimization toward reference,
            ``"full_opt"`` -- unconstrained optimization,
            ``"single_point"`` -- single-point energy evaluation.
        electronic_temperature: Electronic temperature in Kelvin for
            Fermi smearing.
        solvation: If ``True``, use ALPB implicit solvation.
        solvent: Solvent name for ALPB (e.g. ``"water"``).
        output_geometry_path: Path to write the optimized geometry.
            If ``None``, uses the input path with ``_opt`` suffix.
        energy_threshold: Energy convergence threshold in Hartree.
        gradient_threshold: Gradient convergence threshold in Hartree/Angstrom.

    Returns:
        Dictionary with keys: ``energy_hartree``, ``rmsd``,
        ``total_polarizability``, ``atom_indices``, ``charges``,
        ``polarizabilities``, ``converged``.

    Raises:
        ExecutableError: If the IPA executable is not initialized.
        subprocess.SubprocessError: If the executable fails.
    """
    exe_path = IPAExecutable.get_bin_path()
    geometry_path = str(geometry_path)

    if output_geometry_path is None:
        stem = Path(geometry_path).stem
        output_geometry_path = str(Path(geometry_path).parent / f"{stem}_opt.xyz")
    else:
        output_geometry_path = str(output_geometry_path)

    mode_flag = _CALCULATION_TYPE_FLAGS[calculation_type]
    ign_h_flag = "0" if include_h_in_bias else "1"
    solv_flag = "1" if solvation else "0"

    cmd = [
        str(exe_path),
        geometry_path,
        str(charge),
        str(alpha),
        str(kappa),
        ign_h_flag,
        mode_flag,
        str(electronic_temperature),
        solv_flag,
        solvent,
        output_geometry_path,
        str(energy_threshold),
        str(gradient_threshold),
    ]

    logger.debug("Running IPA: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        logger.warning(
            "IPA executable returned non-zero exit code %d. stderr: %s",
            proc.returncode,
            stderr.decode("utf-8", errors="replace")[:500],
        )

    result = parse_ipa_output(stdout)
    result["output_geometry_path"] = output_geometry_path
    return result
