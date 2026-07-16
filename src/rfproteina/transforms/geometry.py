from collections.abc import Sequence

import numpy as np
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_CRD
from atomworks.ml.transforms.base import Transform
from biotite.structure import AtomArray

from rfproteina.utils.common import true_with_prob


def shift_conditioning_annotations(
    atom_array: AtomArray,
    shift: np.ndarray,
    annotation_names: Sequence[str] = (),
) -> None:
    """Shift coordinate-valued conditioning annotations in place to match a coordinate re-centering.

    Whenever a structure's coordinates are re-centered by subtracting ``shift``, any pre-existing
    coordinate-valued conditioning (e.g. :py:class:`~rfproteina.conditions.ChainCentroid`) must be
    shifted by the same amount, or it will silently refer to the wrong location once the structure
    has moved to its new frame.

    Args:
        atom_array: Atom array to update in place.
        shift: The offset (Å, shape ``(3,)``) being subtracted from ``atom_array.coord``.
        annotation_names: Names of the ``(n_atoms, 3)`` coordinate-valued annotations to shift.
            Each is a no-op if absent from ``atom_array``. Defaults to no annotations.
    """
    categories = atom_array.get_annotation_categories()
    for name in annotation_names:
        if name not in categories:
            continue
        values = atom_array.get_annotation(name)
        finite = np.isfinite(values).all(axis=-1)
        if not finite.any():
            continue
        values = values.copy()
        values[finite] -= shift
        atom_array.set_annotation(name, values)


class RandomRotationTransform(Transform):
    """Apply random SE(3) rotation for data augmentation.

    Rotates all atomic coordinates around center of mass before featurization.
    Coordinates remain centered; re-centering happens later in training based on:
    - Monomers: center on all generated atoms
    - PPI: center on target chains (non-generated)

    Args:
        p_apply: Probability of applying rotation (default: 1.0)
    """

    def __init__(self, p_apply: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.p_apply = p_apply

    def forward(self, data: dict) -> dict:
        """Apply random rotation to coordinates."""
        if not true_with_prob(self.p_apply):
            # Skip rotation with probability 1 - p_apply
            return data

        aa = data["atom_array"]

        # Skip if no atoms (can happen after filtering)
        if aa.array_length() == 0:
            return data

        # Center coordinates at origin before rotation
        # This ensures rotation is around the center of mass
        coords = aa.coord  # [n_atoms, 3]

        # Additional safety check for valid coordinates
        if len(coords) == 0 or np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            # Skip rotation if coordinates are invalid
            return data

        com = coords.mean(axis=0, keepdims=True)  # [1, 3]
        centered_coords = coords - com

        # Generate random rotation matrix using QR decomposition
        # This gives uniform random rotations from SO(3)
        rand_mat = np.random.randn(3, 3)
        q, r = np.linalg.qr(rand_mat)

        # Ensure proper rotation (determinant = 1, not reflection)
        q = q * np.sign(np.linalg.det(q))

        # Apply rotation to centered coordinates
        # coords shape: [n_atoms, 3]
        rotated_coords = centered_coords @ q.T  # Matrix multiplication for rotation

        # Keep coordinates centered; don't add COM back
        # (Later in training, coordinates will be re-centered based on conditioned atoms)
        aa.coord = rotated_coords

        data["atom_array"] = aa
        return data


class CenterOnCoordinateConditionedAtoms(Transform):
    """Center entire structure on the centroid of atoms with coordinate conditioning.

    Centers coordinates by subtracting the centroid of all atoms that are
    coordinate-conditioned (i.e., have known positions). If no atoms are
    coordinate-conditioned, centers on all atoms.

    Args:
        coord_annotation: Annotation name for coordinates to center.
            Defaults to ``"coord_to_be_noised"``.
        conditioning_annotations: Names of coordinate-valued conditioning annotations to re-center
            alongside the coordinates (see :py:func:`shift_conditioning_annotations`). Defaults to
            no annotations; configured via the pipeline (e.g. chain-centroid conditioning).
    """

    def __init__(
        self,
        coord_annotation: str = "coord_to_be_noised",
        conditioning_annotations: Sequence[str] = (),
    ):
        super().__init__()
        self.coord_annotation = coord_annotation
        self.conditioning_annotations = conditioning_annotations

    def forward(self, data: dict) -> dict:
        """Center entire structure on the centroid of the coordinate-conditioned atoms."""
        aa = as_atom_array_plus(data["atom_array"])
        coords = aa.get_annotation(self.coord_annotation)

        # Get coordinate-conditioned mask
        coord_mask = C_CRD.mask(aa, default="generate")

        # If no coordinate-conditioned atoms, use all atoms
        if not coord_mask.any():
            coord_mask = np.ones(aa.array_length(), dtype=bool)

        # Compute centroid of coordinate-conditioned atoms
        # Use nanmean to ignore NaN coordinates from unresolved atoms
        centroid = np.nanmean(coords[coord_mask], axis=0)

        # Center coordinates
        coords = coords - centroid

        # Update annotation
        aa.set_annotation(self.coord_annotation, coords)

        # Any pre-existing coordinate-valued conditioning lives in the same frame as `coords` and
        # must be re-centered by the same amount, or it will point to the wrong location.
        shift_conditioning_annotations(aa, centroid, self.conditioning_annotations)

        data["atom_array"] = aa

        return data
