"""Transform for the internal-coordinate-only conditioning ablation."""

from typing import ClassVar

from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_CRD
from atomworks.ml.transforms.base import Transform

from proteinfoundation.transforms.utils.conditioning_utils import convert_to_internal_coords


class ConvertAbsoluteToInternalCoords(Transform):
    """Convert all absolute coordinate conditioning to a single internal-coord group."""

    # Prohibited prior transforms: this conversion must run before each of these.
    incompatible_previous_transforms: ClassVar[list[str]] = [
        "UnindexFlaggedTokens",
        "CenterOnCoordinateConditionedAtoms",
    ]

    def forward(self, data: dict) -> dict:
        atom_array = as_atom_array_plus(data["atom_array"])

        coord_mask = C_CRD.mask(atom_array, default="generate")
        if coord_mask.any():
            atom_array = convert_to_internal_coords(atom_array, coord_mask)

        # Invariant: no absolute coordinate conditioning survives this transform.
        assert not C_CRD.mask(
            atom_array, default="generate"
        ).any(), "ConvertAbsoluteToInternalCoords: C_CRD mask must be empty after conversion."

        data["atom_array"] = atom_array
        return data
