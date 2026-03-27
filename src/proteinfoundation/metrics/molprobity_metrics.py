"""MolProbity validation metrics for protein structure quality assessment."""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray, AtomArrayStack

from foundry.metrics.metric import Metric
from proteinfoundation.metrics.metrics_utils import unwrap_atom_array

logger = logging.getLogger(__name__)


def compute_molprobity_metrics(atom_array: AtomArray, molprobity_script_path: str) -> dict[str, float]:
    """Compute MolProbity validation metrics for an atom array.

    Args:
      atom_array: AtomArray to validate.
      molprobity_script_path: Path to the molprobity validation script.

    Returns:
      Dictionary containing validation metrics:
        - ``rama_percent_favored``: Ramachandran favored percentage
        - ``rama_percent_outliers``: Ramachandran outlier percentage
        - ``rota_percent_outliers``: Rotamer outlier percentage
        - ``clashscore``: All-atom clash score
        - ``molprobity_score``: Overall MolProbity score
        - ``rms_bonds``: RMS deviation for bond lengths
        - ``bond_outliers``: Number of bond length outliers
        - ``rms_angles``: RMS deviation for bond angles
        - ``angle_outliers``: Number of bond angle outliers
    """
    # Create temporary CIF file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        to_cif_file(atom_array, str(tmp_path))

        # Verify the file was written and is readable
        if not tmp_path.exists():
            raise RuntimeError(f"Failed to write temporary CIF file: {tmp_path}")

        file_size = tmp_path.stat().st_size
        if file_size == 0:
            raise RuntimeError(f"Temporary CIF file is empty: {tmp_path}")
        # Run molprobity script as subprocess
        result = subprocess.run(
            [molprobity_script_path, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            # Include both stdout and stderr for debugging
            error_msg = f"MolProbity script failed with return code {result.returncode}\n"
            error_msg += f"Temp file: {tmp_path} ({file_size} bytes)\n"
            raise RuntimeError(error_msg)

        # Parse output
        return _parse_molprobity_output(result.stdout)

    except subprocess.TimeoutExpired as e:
        raise RuntimeError("MolProbity execution timed out after 120s") from e
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


def _parse_molprobity_output(output: str) -> dict[str, float]:
    """Parse MolProbity script output.

    Expected output format:
      RMS(bonds): 0.012
      Bond outliers: 3
      RMS(angles): 1.234
      Angle outliers: 5
      Rama favored: 95.5
      Rama outliers%: 0.5
      Rota outliers%: 2.3
      Clashscore: 5.2
      Molprobity score: 1.8

    Args:
      output: Stdout text from molprobity script.

    Returns:
      Dictionary of parsed metrics.
    """
    metrics = {}
    patterns = {
        "rama_percent_favored": r"Rama favored:\s*([\d.]+)",
        "rama_percent_outliers": r"Rama outliers%:\s*([\d.]+)",
        "rota_percent_outliers": r"Rota outliers%:\s*([\d.]+)",
        "clashscore": r"Clashscore:\s*([\d.]+)",
        "molprobity_score": r"Molprobity score:\s*([\d.]+)",
        "rms_bonds": r"RMS\(bonds\):\s*([\d.]+|nan)",
        "bond_outliers": r"Bond outliers:\s*(-?\d+)",
        "rms_angles": r"RMS\(angles\):\s*([\d.]+|nan)",
        "angle_outliers": r"Angle outliers:\s*(-?\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            if value == "nan":
                metrics[key] = np.nan
            elif key in ["bond_outliers", "angle_outliers"]:
                # Integer metrics
                int_val = int(value)
                metrics[key] = float(int_val) if int_val >= 0 else np.nan
            else:
                metrics[key] = float(value)
        else:
            logger.warning(f"Could not parse {key} from MolProbity output")
            metrics[key] = np.nan

    return metrics


class MolProbityMetrics(Metric):
    """MolProbity validation metrics for protein structure quality.

    The MolProbity script path can be provided either as an argument or via
    the ``MOLPROBITY_PATH`` environment variable.

    Note: The molprobity script uses the cctbx Python interpreter which has
    built-in access to the chem_data directory (rotamer library).

    Args:
      molprobity_script_path: Path to the molprobity validation script.
        If ``None``, reads from ``MOLPROBITY_PATH`` environment variable.
      skip_on_error: If ``True``, returns NaN for failed structures. Defaults to ``True``.

    Examples:
      >>> # Using environment variable
      >>> metrics = MolProbityMetrics()
      >>> results = metrics.compute(atom_array)
      >>> # Or providing path explicitly
      >>> metrics = MolProbityMetrics(molprobity_script_path="/path/to/script.py")
    """

    def __init__(
        self,
        molprobity_script_path: str | None = None,
        skip_on_error: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Get MolProbity script path from argument or environment variable
        if molprobity_script_path is None:
            molprobity_script_path = os.environ.get("MOLPROBITY_PATH")

        if molprobity_script_path is None:
            raise ValueError(
                "MolProbity script path must be provided either via 'molprobity_script_path' "
                "argument or 'MOLPROBITY_PATH' environment variable"
            )

        self.molprobity_script_path = molprobity_script_path
        self.skip_on_error = skip_on_error

    @property
    def kwargs_to_compute_args(self) -> dict:
        return {
            "atom_array": ("predicted_atom_array",),
        }

    def compute(self, atom_array: AtomArray | AtomArrayStack) -> dict[str, float]:
        """Compute MolProbity metrics for a structure.

        Args:
          atom_array: Structure to validate.

        Returns:
          Dictionary with metrics:
            - ``mean_rama_percent_favored``: Ramachandran favored percentage
            - ``mean_rama_percent_outliers``: Ramachandran outlier percentage
            - ``mean_rota_percent_outliers``: Rotamer outlier percentage
            - ``mean_clashscore``: All-atom clash score
            - ``mean_molprobity_score``: Overall MolProbity score
            - ``mean_rms_bonds``: RMS deviation for bond lengths
            - ``mean_bond_outliers``: Number of bond length outliers
            - ``mean_rms_angles``: RMS deviation for bond angles
            - ``mean_angle_outliers``: Number of bond angle outliers
        """
        atom_array = unwrap_atom_array(atom_array)
        try:
            metrics = compute_molprobity_metrics(atom_array, self.molprobity_script_path)
        except Exception as e:
            logger.warning(f"MolProbity validation failed for structure: {e}")
            if not self.skip_on_error:
                raise
            metrics = {
                key: np.nan
                for key in [
                    "rama_percent_favored",
                    "rama_percent_outliers",
                    "rota_percent_outliers",
                    "clashscore",
                    "molprobity_score",
                    "rms_bonds",
                    "bond_outliers",
                    "rms_angles",
                    "angle_outliers",
                ]
            }

        return {f"mean_{key}": float(value) for key, value in metrics.items()}
