from collections import defaultdict
from typing import Any

import numpy as np
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from biotite.structure import AtomArray, AtomArrayStack

from foundry.metrics.metric import Metric
from proteinfoundation.conditions import C_RAS, RasaEnum


def get_rasa_metrics(
    atom_array: AtomArray,
    probe_radius: float = 1.4,
    atom_radii: str = "ProtOr",
    point_number: int = 100,
) -> dict[str, float]:
    """Compute RASA adherence metrics for a given input atom array.

    Args:
        atom_array (AtomArray): Input AtomArray with predicted structure and rasa annotations.
        probe_radius (float): Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (for water).
        atom_radii (str): Atom radii set to use for calculation. Defaults to "ProtOr".
        point_number (int): Number of points in the Shrake-Rupley algorithm. Defaults to 100.

    Returns:
        dict: Dictionary containing RASA adherence metrics

    NOTE: RASA (Relative Accessible Surface Area) is computed using the Shrake-Rupley algorithm.
    The conditioning bins match those used in training:
        - buried: RASA in [0.0, 0.1)
        - intermediate: RASA in [0.1, 0.2)
        - exposed: RASA >= 0.2
    """

    output_metrics = {}

    # Calculate RASA for generated structure
    generated_rasa = calculate_atomwise_rasa(atom_array, probe_radius, atom_radii, point_number)

    # Define bin edges matching training (from design_task_modifiers.py)
    bin_edges = np.linspace(0, 0.2, 3)  # [0.0, 0.1, 0.2]

    # Bin the generated RASA values
    generated_rasa_bins = np.digitize(generated_rasa, bin_edges, right=False)
    if (generated_rasa_bins == 0).any():
        raise ValueError("Negative RASA value was detected. This indicates an error.")

    # Convert to RasaEnum
    generated_rasa_values = np.full(atom_array.array_length(), RasaEnum.UNSPECIFIED, dtype=int)
    generated_rasa_values[generated_rasa_bins == 1] = RasaEnum.BURIED
    generated_rasa_values[generated_rasa_bins == 2] = RasaEnum.INTERMEDIATE
    generated_rasa_values[generated_rasa_bins == 3] = RasaEnum.EXPOSED
    generated_rasa_values[np.isnan(generated_rasa)] = RasaEnum.UNSPECIFIED

    # Extract RASA conditioning
    rasa_conditioning = C_RAS.annotation(atom_array, default="generate")

    # Compute adherence for each RASA type
    rasa_adherence_dict = {}
    for rasa_annot, rasa_type in zip(
        [RasaEnum.BURIED, RasaEnum.INTERMEDIATE, RasaEnum.EXPOSED],
        ["buried", "intermediate", "exposed"],
        strict=False,
    ):
        metric_name = f"{rasa_type}_conditioning_adherence"
        expected_indices = np.where(rasa_conditioning == rasa_annot)[0]

        if len(expected_indices) > 0:
            rasa_adherence = (generated_rasa_values[expected_indices] == rasa_annot).mean()
            rasa_adherence_dict[metric_name] = rasa_adherence
        else:
            # Would be misleading to give a numerical value if no conditioning of this type was provided
            rasa_adherence_dict[metric_name] = np.nan

    output_metrics.update(rasa_adherence_dict)

    return output_metrics


class RasaAdherenceMetric(Metric):
    """Compute adherence to RASA conditioning."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "atom_array_or_stack": ("predicted_atom_array",),
        }

    def compute(self, atom_array_or_stack: AtomArray | AtomArrayStack | list[AtomArray]) -> dict[str, float]:
        out_dict = defaultdict(list)

        if isinstance(atom_array_or_stack, AtomArray):
            atom_array_list = [atom_array_or_stack]
        elif isinstance(atom_array_or_stack, AtomArrayStack):
            atom_array_list = list(atom_array_or_stack)
        elif isinstance(atom_array_or_stack, list):
            atom_array_list = atom_array_or_stack
        else:
            raise ValueError(
                f"RasaAdherenceMetric expects AtomArray, AtomArrayStack, or list of "
                f"AtomArrays, but got {type(atom_array)}"
            )

        for atom_array in atom_array_list:
            rasa_metrics = get_rasa_metrics(atom_array)
            for k, v in rasa_metrics.items():
                out_dict[k].append(v)

        return {k: np.nanmean(v) for k, v in out_dict.items()}
