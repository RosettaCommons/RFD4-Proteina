"""Utilities for handling stereogenic-center geometry"""

from itertools import permutations

import numpy as np
from atomworks.enums import ChiralType
from atomworks.io.utils.annotator import ensure_annotations
from biotite.structure import AtomArray
from scipy.spatial.transform import Rotation

IDEAL_POLYHEDRA: dict[str, np.ndarray] = {
    "tetrahedral": np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float32)
    / np.sqrt(3),
    "square_planar": np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float32),
    "trigonal_bipyramidal": np.array(
        [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [-0.5, -np.sqrt(3) / 2, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float32,
    ),
    "octahedral": np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32
    ),
}


def _enumerate_stereoisomer_templates(template: np.ndarray, tol: float = 1e-5) -> list[np.ndarray]:
    """One representative per proper-rotation orbit of all ``n!`` labelings."""
    reps: list[np.ndarray] = []

    # Iterate through all n! permutations
    for perm in permutations(range(len(template))):
        candidate = template[list(perm)]
        if any(Rotation.align_vectors(candidate, existing)[1] < tol for existing in reps):
            # Skip permutations that are not unique stereocenters
            continue
        reps.append(candidate)

    return reps


STEREOISOMER_TEMPLATES: dict[tuple[str, int], list[np.ndarray]] = {
    (geom, n): _enumerate_stereoisomer_templates(full[:n])
    for geom, full in IDEAL_POLYHEDRA.items()
    for n in range(3, len(full) + 1)
}
"""Pre-computed stereoisomer templates for all common geometries"""


def _template_rmsds(centered: np.ndarray, geometry: str) -> np.ndarray | None:
    """Per-template rmsd between unit-normalized ``centered`` and each template."""
    templates = STEREOISOMER_TEMPLATES.get((geometry, len(centered)))
    if templates is None or not np.isfinite(centered).all():
        return None
    unit = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    rssds = np.array([Rotation.align_vectors(unit, t)[1] for t in templates])
    return rssds / np.sqrt(len(centered))


def classify_stereoisomer(centered: np.ndarray, geometry: str) -> int | None:
    """Index of the :py:data:`STEREOISOMER_TEMPLATES` entry best matching ``centered``.

    Returns:
      Index into ``STEREOISOMER_TEMPLATES[(geometry, len(centered))]``, or
      ``None`` if ``centered`` contains non-finite values or no templates exist
      for ``(geometry, len(centered))``.
    """
    rmsds = _template_rmsds(centered, geometry)
    return None if rmsds is None else int(np.argmin(rmsds))


def near_tied_templates(centered: np.ndarray, geometry: str, tol: float = 0.05) -> set[int] | None:
    """Indices of templates within ``tol`` rmsd of the best match.

    Returns:
      Set of template indices, or ``None`` if no templates exist for
      ``(geometry, len(centered))`` or ``centered`` is non-finite.
    """
    rmsds = _template_rmsds(centered, geometry)
    if rmsds is None:
        return None
    return set(np.where(rmsds <= rmsds.min() + tol)[0].tolist())


def identify_stereogenic_centers(atom_array: AtomArray) -> list[dict]:
    """Stereogenic centers with a supported geometry + neighbor count.

    Gates on :py:meth:`atomworks.enums.ChiralType.is_stereogenic` (unspecified
    tags are excluded) and on having a template for ``(geometry, n_neighbors)``.
    """
    ensure_annotations(atom_array, "chiral_type")

    centers: list[dict] = []
    for idx in np.where(atom_array.chiral_type != 0)[0]:
        ct = ChiralType(int(atom_array.chiral_type[idx]))
        if not ct.is_stereogenic():
            continue
        geometry = ct.to_geometry()
        bonded, _ = atom_array.bonds.get_bonds(int(idx))
        bonded_sorted = sorted(int(b) for b in bonded)
        if (geometry, len(bonded_sorted)) not in STEREOISOMER_TEMPLATES:
            continue
        centers.append(
            {
                "center_idx": int(idx),
                "bonded_idxs": bonded_sorted,
                "geometry": geometry,
                "chiral_type": ct,
            }
        )

    return centers
