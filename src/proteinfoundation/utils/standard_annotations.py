"""Project-specific StandardAnnotations.

Registering these as StandardAnnotations (rather than plain extra_fields) gives:
- Automatic saving to CIF files (no need to list in ``extra_fields``).
- Easy loading from CIF files through ``parse``
- Default generation via :py:meth:`~atomworks.io.utils.standard_annotations.base.StandardAnnotationBase.default_annotation`.
"""

import numpy as np
from atomworks.io.utils.standard_annotations.base import Level, StandardAnnotationBase
from biotite.structure import AtomArray


class SrcCoordinateConditionedRegion(StandardAnnotationBase):
    """Per-atom source identifier for the coordinate-conditioned region.

    Contig-style string identifier naming the source component that provided the coordinate
    conditioning for each atom. ``"."`` means the atom was not coordinate-conditioned.
    """

    name = "src-coordinate-conditioned-region"
    n_body = 1
    level = Level.ATOM
    dtype = np.dtype("<U10")

    @classmethod
    def get_mask_name(cls, n_body=None, level=None) -> str:
        """Return backward-compatible mask key."""
        return "mapped_to_coordinate_conditioned_region"

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        """Infer mask from annotation if present, otherwise all-False."""
        if cls.has_annotation(atom_array):
            return cls.annotation(atom_array) != "."
        return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: ``"."`` (unconditioned) for all atoms."""
        return np.full(atom_array.array_length(), ".", dtype="<U10")


class SrcComponent(StandardAnnotationBase):
    """Per-atom source component identifier.

    Contig-style string identifier naming the source component each atom
    came from. ``"."`` means the component is unassigned.
    """

    name = "src-component"
    n_body = 1
    level = Level.ATOM
    dtype = np.dtype("<U10")

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        """Infer mask from annotation if present, otherwise all-False."""
        if cls.has_annotation(atom_array):
            return cls.annotation(atom_array) != "."
        return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: ``"."`` (unassigned) for all atoms."""
        return np.full(atom_array.array_length(), ".", dtype="<U10")


class CoordsWithInsertion(StandardAnnotationBase):
    """Per-atom coordinate array with aligned coordinate-conditioned regions inserted."""

    name = "coords-with-insertion"
    n_body = 1
    level = Level.ATOM
    dtype = np.float32

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        """Infer mask from annotation if present, otherwise all-False."""
        if cls.has_annotation(atom_array):
            return ~np.isnan(cls.annotation(atom_array)).any(axis=-1)
        return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: NaN coordinates for all atoms."""
        return np.full((atom_array.array_length(), 3), np.nan, dtype=np.float32)


class SrcCoordGroup(StandardAnnotationBase):
    """Per-atom source coordinate group identifier.

    Integer identifying which input ``CoordinateGroup`` this atom belongs to.
    ``-1`` means the atom had no coordinate conditioning (``UNKNOWN_COORDS``).
    """

    name = "src-coord-group"
    n_body = 1
    level = Level.ATOM
    dtype = np.int32

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        """Infer mask from annotation if present, otherwise all-False."""
        if cls.has_annotation(atom_array):
            return cls.annotation(atom_array) != -1
        return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: ``-1`` (not mapped) for all atoms."""
        return np.full(atom_array.array_length(), -1, dtype=np.int32)


class IsUnindexedInsertion(StandardAnnotationBase):
    """Per-atom mask flagging atoms that map to unindexed in-context conditioning regions.

    Set by :py:func:`~proteinfoundation.transforms.unindexing.annotate_unindexed_insertion`.
    ``True`` means the atom was identified as the insertion target for an unindexed token.
    """

    name = "is-unindexed-insertion"
    n_body = 1
    level = Level.ATOM
    dtype = bool
    alias = "is_unindexed_insertion"

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        """Default: all-False (no unindexed insertion)."""
        return np.zeros(atom_array.array_length(), dtype=bool)


# --- Convenience aliases ---
SA_CRDSRC = SrcCoordinateConditionedRegion
SA_SRC = SrcComponent
SA_CWI = CoordsWithInsertion
SA_CRDGRPSRC = SrcCoordGroup
SA_UII = IsUnindexedInsertion
