"""Binary detection of long helices with stretches lacking protein packing."""

from itertools import pairwise
from typing import Any

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack, annotate_sse, filter_amino_acids
from scipy.spatial.distance import cdist

from rfproteina.metrics import Metric
from rfproteina.metrics.utils.metrics_utils import unwrap_atom_array


def has_protruding_helix(
    atom_array: AtomArray,
    *,
    min_helix_residues: int = 30,
    min_helix_span: float = 35.0,
    min_unpacked_residues: int = 15,
    packing_distance: float = 10.0,
    sequence_neighbor_exclusion: int = 5,
) -> bool:
    """Detect a long P-SEA helix with a consecutive stretch lacking C-alpha contacts.

    All protein chains, including conditioned context, are inspected. Packing
    partners are protein C-alpha atoms outside the helix, excluding same-chain
    neighbors within ``sequence_neighbor_exclusion`` residue numbers. Ligands
    do not count as packing partners. No annotations or coordinates are mutated.

    Args:
      atom_array: One output structure, with coordinates in angstroms.
      min_helix_residues: Minimum consecutive alpha-helical residue count.
      min_helix_span: Minimum first-to-last helix C-alpha distance in angstroms.
      min_unpacked_residues: Minimum consecutive helix residues without packing.
      packing_distance: A packing contact is at or below this distance in angstroms.
      sequence_neighbor_exclusion: Same-chain residue-number distance to exclude.

    Returns:
      True if any helix meets all cutoffs; False includes structures without protein.

    Raises:
      ValueError: A cutoff is invalid or protein C-alpha coordinates are nonfinite.
      TypeError: Input is not a single AtomArray.
    """
    if not isinstance(atom_array, AtomArray):
        raise TypeError("Protruding-helix detection requires one AtomArray per example.")
    for name, value, minimum in (
        ("min_helix_residues", min_helix_residues, 1),
        ("min_unpacked_residues", min_unpacked_residues, 1),
        ("sequence_neighbor_exclusion", sequence_neighbor_exclusion, 0),
    ):
        if not isinstance(value, int | np.integer) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    for name, value in (("min_helix_span", min_helix_span), ("packing_distance", packing_distance)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}.")

    ca = atom_array[filter_amino_acids(atom_array) & (atom_array.atom_name == "CA")]
    if not np.isfinite(ca.coord).all():
        raise ValueError("Protruding-helix detection requires finite protein C-alpha coordinates.")
    if len(ca) < min_helix_residues:
        return False
    chain_key = "chain_iid" if "chain_iid" in ca.get_annotation_categories() else "chain_id"
    chain_ids = ca.get_annotation(chain_key)
    all_indices = np.arange(len(ca))

    for chain_id in np.unique(chain_ids):
        chain_indices = all_indices[chain_ids == chain_id]
        chain = ca[chain_indices]
        helical = annotate_sse(chain) == "a"
        # Residue-number gaps split helices even if both sides are helical.
        breaks = np.flatnonzero((helical[1:] != helical[:-1]) | (np.diff(chain.res_id) != 1)) + 1
        boundaries = np.r_[0, breaks, len(chain)]
        for start, stop in pairwise(boundaries):
            if not helical[start] or stop - start < min_helix_residues:
                continue
            helix_indices = chain_indices[start:stop]
            helix = ca[helix_indices]
            if np.linalg.norm(helix.coord[-1] - helix.coord[0]) < min_helix_span:
                continue
            outside = np.ones(len(ca), dtype=bool)
            outside[helix_indices] = False
            outside_indices = all_indices[outside]
            distances = cdist(helix.coord, ca.coord[outside_indices])
            neighbors = (chain_ids[outside_indices][None, :] == chain_id) & (
                np.abs(helix.res_id[:, None].astype(np.int64) - ca.res_id[outside_indices][None, :])
                <= sequence_neighbor_exclusion
            )
            packed = np.any((distances <= packing_distance) & ~neighbors, axis=1)
            unpacked_boundaries = np.r_[-1, np.flatnonzero(packed), len(helix)]
            if np.any(np.diff(unpacked_boundaries) - 1 >= min_unpacked_residues):
                return True
    return False


class ProtrudingHelixMetric(Metric):
    """Report ``present=1`` if an output contains a protruding helix, otherwise 0.

    Cutoffs have the semantics and units of :func:`has_protruding_helix`.
    The mean across output structures is the fraction with protruding helices.
    """

    def __init__(
        self,
        min_helix_residues: int = 30,
        min_helix_span: float = 35.0,
        min_unpacked_residues: int = 15,
        packing_distance: float = 10.0,
        sequence_neighbor_exclusion: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cutoffs = {
            "min_helix_residues": min_helix_residues,
            "min_helix_span": min_helix_span,
            "min_unpacked_residues": min_unpacked_residues,
            "packing_distance": packing_distance,
            "sequence_neighbor_exclusion": sequence_neighbor_exclusion,
        }

    @property
    def kwargs_to_compute_args(self) -> dict[str, str]:
        """Read the generated structure from the standard inline-metric input."""
        return {"atom_array": "predicted_atom_array"}

    def compute(self, atom_array: AtomArray | AtomArrayStack) -> dict[str, int]:
        """Return one binary flag for one output structure."""
        return {"present": int(has_protruding_helix(unwrap_atom_array(atom_array), **self.cutoffs))}
