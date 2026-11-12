"""3D point-group symmetry package for RFD4-Proteina oligomer design."""

from __future__ import annotations

from typing import Any

from rfproteina.symmetry.operators import (
    BatchSymmetryLike,
    SymmetryLike,
    compute_symmetry_rmsd,
    expand_asu,
    expand_frames_asu,
    expand_invariant_asu,
    has_active_symmetry,
    project_frames_symmetric,
    project_invariant_symmetric,
    project_symmetric,
    reduce_to_asu,
    resolve_batch_symmetry,
    sample_symmetric_ca_noise,
    symmetrize_decoded_atom37,
    zero_global_com,
)
from rfproteina.symmetry.point_groups import (
    VALID_DECODE_MODES,
    DecodeMode,
    GroupFamily,
    SymmetryConfig,
    SymmetrySpec,
    chain_index_to_id,
    get_symmetry_chain_ids,
    parse_symmetry,
)

__all__ = [
    "VALID_DECODE_MODES",
    "BatchSymmetryLike",
    "DecodeMode",
    "ExpandSymmetricASUTransform",
    "GroupFamily",
    "SymmetryConfig",
    "SymmetryLike",
    "SymmetrySpec",
    "chain_index_to_id",
    "compute_symmetry_rmsd",
    "expand_asu",
    "expand_frames_asu",
    "expand_invariant_asu",
    "get_symmetry_chain_ids",
    "has_active_symmetry",
    "parse_symmetry",
    "project_frames_symmetric",
    "project_invariant_symmetric",
    "project_symmetric",
    "reduce_to_asu",
    "resolve_batch_symmetry",
    "sample_symmetric_ca_noise",
    "symmetrize_decoded_atom37",
    "zero_global_com",
]


def __getattr__(name: str) -> Any:
    """Lazily expose pipeline transforms without importing AtomWorks/Biotite on package load."""
    if name == "ExpandSymmetricASUTransform":
        from rfproteina.symmetry.transforms import ExpandSymmetricASUTransform

        return ExpandSymmetricASUTransform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
