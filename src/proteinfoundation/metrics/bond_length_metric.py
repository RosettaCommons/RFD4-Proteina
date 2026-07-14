"""Internal residue bond length metric comparing generated vs CCD ideal."""

from collections import defaultdict
from typing import Any

import numpy as np
from atomworks.constants import METAL_ELEMENTS
from atomworks.io.utils.ccd import atom_array_from_ccd_code
from biotite.structure import AtomArray, AtomArrayStack, get_residue_starts

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger
from proteinfoundation.metrics.utils.metrics_utils import normalize_to_atom_array_list

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _classify_bond(elem1: str, elem2: str) -> str:
    """Classify a bond as organic_organic, metal_organic, or metal_metal."""
    e1_metal = elem1.strip().upper() in METAL_ELEMENTS
    e2_metal = elem2.strip().upper() in METAL_ELEMENTS
    if e1_metal and e2_metal:
        return "metal_metal"
    elif e1_metal or e2_metal:
        return "metal_organic"
    return "organic_organic"


def _get_ccd_bond_lengths(res_name: str) -> dict[tuple[str, str], float] | None:
    """Get ideal bond lengths from CCD for a residue.

    Returns:
      Dict mapping ``(atom_name_1, atom_name_2)`` (sorted) to ideal bond
      length in Angstroms, or None if CCD lookup fails.
    """
    try:
        ccd_aa = atom_array_from_ccd_code(res_name)
    except (KeyError, ValueError, FileNotFoundError, OSError):
        return None

    if ccd_aa is None or ccd_aa.bonds is None:
        return None

    bond_arr = ccd_aa.bonds.as_array()
    if len(bond_arr) == 0:
        return None

    lengths = {}
    for i in range(len(bond_arr)):
        a1, a2 = int(bond_arr[i, 0]), int(bond_arr[i, 1])
        name1 = str(ccd_aa.atom_name[a1]).strip()
        name2 = str(ccd_aa.atom_name[a2]).strip()
        key = tuple(sorted((name1, name2)))
        coord1 = ccd_aa.coord[a1]
        coord2 = ccd_aa.coord[a2]
        if np.any(np.isnan(coord1)) or np.any(np.isnan(coord2)):
            continue
        lengths[key] = float(np.linalg.norm(coord1 - coord2))

    return lengths if lengths else None


def compute_internal_bond_metrics(atom_array: AtomArray) -> dict[str, list[float]]:
    """Compute per-bond relative errors for intra-residue bonds vs CCD ideal.

    Args:
      atom_array: Generated structure with bonds.

    Returns:
      Dict mapping bond type to list of relative errors.
    """
    if atom_array.bonds is None:
        return {}

    bond_arr = atom_array.bonds.as_array()
    if len(bond_arr) == 0:
        return {}

    res_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    starts, stops = res_starts[:-1], res_starts[1:]

    # Build atom_idx → residue_idx mapping
    atom_to_res = np.zeros(atom_array.array_length(), dtype=np.int32)
    for res_idx, (start, stop) in enumerate(zip(starts, stops, strict=False)):
        atom_to_res[start:stop] = res_idx

    # Cache CCD lookups per res_name
    ccd_cache: dict[str, dict[tuple[str, str], float] | None] = {}

    errors: dict[str, list[float]] = defaultdict(list)

    for i in range(len(bond_arr)):
        a1, a2 = int(bond_arr[i, 0]), int(bond_arr[i, 1])

        # Only intra-residue bonds
        if atom_to_res[a1] != atom_to_res[a2]:
            continue

        # Skip bonds with NaN coordinates
        if np.any(np.isnan(atom_array.coord[a1])) or np.any(np.isnan(atom_array.coord[a2])):
            continue

        res_name = str(atom_array.res_name[a1]).strip()

        # Get CCD ideal bond lengths
        if res_name not in ccd_cache:
            ccd_cache[res_name] = _get_ccd_bond_lengths(res_name)

        ccd_lengths = ccd_cache[res_name]
        if ccd_lengths is None:
            continue

        # Match bond by atom names
        name1 = str(atom_array.atom_name[a1]).strip()
        name2 = str(atom_array.atom_name[a2]).strip()
        key = tuple(sorted((name1, name2)))

        ideal = ccd_lengths.get(key)
        if ideal is None or ideal < 1e-6:
            continue

        actual = float(np.linalg.norm(atom_array.coord[a1] - atom_array.coord[a2]))
        rel_error = abs(actual - ideal) / ideal

        elem1 = str(atom_array.element[a1]).strip()
        elem2 = str(atom_array.element[a2]).strip()
        bond_type = _classify_bond(elem1, elem2)

        errors[bond_type].append(rel_error)

    return dict(errors)


class InternalResidueBondMetric(Metric):
    """Compare generated intra-residue bond lengths to CCD ideal geometry.

    Reports mean relative error split by bond type: organic-organic,
    metal-organic, metal-metal.
    """

    def __init__(
        self,
        input_kwarg_names: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._input_kwarg_names = input_kwarg_names or {
            "atom_array_stack": "predicted_atom_array",
        }
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return self._input_kwarg_names

    def compute(self, atom_array_stack: AtomArray | AtomArrayStack | list[AtomArray]) -> dict[str, Any]:
        atom_arrays = normalize_to_atom_array_list(atom_array_stack)

        all_errors: dict[str, list[float]] = defaultdict(list)
        for atom_array in atom_arrays:
            per_type = compute_internal_bond_metrics(atom_array)
            for bond_type, errs in per_type.items():
                all_errors[bond_type].extend(errs)

        output: dict[str, Any] = {}
        for bond_type in ["organic_organic", "metal_organic", "metal_metal"]:
            errs = all_errors.get(bond_type, [])
            output[f"{bond_type}_mean_rel_error"] = float(np.mean(errs)) if errs else float("nan")

        return output
