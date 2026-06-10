from enum import IntEnum

import numpy as np
from atomworks.io.transforms.atom_array import (
    add_id_and_entity_annotations,
    add_molecule_iid_annotation,
)
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import AnnotationList2D
from atomworks.io.utils.scatter import spread_segment_wise
from atomworks.io.utils.standard_annotations.base import Level
from atomworks.ml.conditions import C_CRD, C_IDX, CONDITIONS, ConditionBase
from atomworks.ml.transforms.dssp import SSEnum
from biotite.structure import AtomArray


# TODO: Generalize to give both hotspots and antihotspots!
class Hotspot(ConditionBase):
    """Hotspot annotation for protein design."""

    name = "hotspot"
    n_body = 1
    level = Level.ATOM
    is_mask = True
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        """
        If there exists a hotspot annotation, we derive the mask by setting the mask to true for all atoms in
        any chain that has a hotspot annotation.
        """
        if cls.full_name in atom_array.get_annotation_categories():
            return atom_array.get_annotation(cls.full_name).astype(bool)

        return np.zeros(atom_array.array_length(), dtype=bool)


class AntibodyCDR(ConditionBase):
    name = "cdr"
    n_body = 1
    level = Level.RESIDUE
    is_mask = True
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        return np.zeros(atom_array.array_length(), dtype=bool)


class AntibodyHeavyChain(ConditionBase):
    """
    NOTE: The reason these are residue level and not chain level is because things like scFvs can have the
    heavy and light 'chains' on the same protein polymer.
    """

    name = "ab-heavy"
    n_body = 1
    level = Level.RESIDUE
    is_mask = True
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        return np.zeros(atom_array.array_length(), dtype=bool)


class AntibodyLightChain(ConditionBase):
    name = "ab-light"
    n_body = 1
    level = Level.RESIDUE
    is_mask = True
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        return np.zeros(atom_array.array_length(), dtype=bool)


# NOTE: In RFD3, the "unspecified" bin was the same as fully exposed! That bug is fixed here.
# However, this may cause issues if trying to load the exact weights without any fine-tuning.
# The enum values are chosen to match RFD3 inference-time behavior to make the transition easy for designers.
class RasaEnum(IntEnum):
    UNSPECIFIED = 0
    BURIED = 1
    INTERMEDIATE = 2
    EXPOSED = 3


class Rasa(ConditionBase):
    """Conditioning that provides binned RASA values to the model. Zero denotes no conditioning."""

    name = "rasa-binned"
    n_body = 1
    level = Level.ATOM
    dtype = int

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            binned_rasa_annotation = atom_array.get_annotation(cls.full_name)
            return binned_rasa_annotation != RasaEnum.UNSPECIFIED
        return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        return np.full(atom_array.array_length(), RasaEnum.UNSPECIFIED, dtype=int)


class HBondEnum(IntEnum):
    NONE = 0
    DONOR = 1
    ACCEPTOR = 2


class HydrogenBond(ConditionBase):
    """Conditioning that provides hydrogen bond information (donor/acceptor) to the model."""

    name = "hbond"
    n_body = 1
    level = Level.ATOM
    dtype = int

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            annotation = cls.annotation(atom_array, default="raise")
            return annotation != HBondEnum.NONE
        return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        return np.full(atom_array.array_length(), HBondEnum.NONE, dtype=int)


class SecondaryStructure(ConditionBase):
    """Conditioning that provides secondary structure information to the model."""

    name = "secondary-structure"
    n_body = 1
    level = Level.RESIDUE
    dtype = int

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            annotation = cls.annotation(atom_array, default="raise")
            return annotation != SSEnum.NONE
        return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        return np.full(atom_array.array_length(), SSEnum.NONE, dtype=int)


class IndexGroup(ConditionBase):
    """Indicates atoms within the same equivalence class of known index offsets.
    Does not differentiate between globally indexed vs unindexed groups (that is annotated in C_IDX.mask)
    While the default annotation is monotonically increasing by one, this should not be expected to always hold.

    NOTE: Residues on different chains do NOT have known index offsets, even if both are globally indexed!!
    """

    name = "index-group"
    n_body = 1
    level = Level.TOKEN
    dtype = int

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        return np.ones(atom_array.array_length(), dtype=bool)  # Every atom has an index group

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        # Add molecule annotations if not already present
        if "molecule_iid" not in atom_array.get_annotation_categories():
            ensure_annotations(atom_array, "is_polymer")
            atom_array = add_id_and_entity_annotations(atom_array)
            if "transformation_id" not in atom_array.get_annotation_categories():
                atom_array.set_annotation("transformation_id", np.full(atom_array.array_length(), "1", dtype=str))
            atom_array = add_molecule_iid_annotation(atom_array)

        molecule_annot = atom_array.molecule_iid
        is_globally_indexed = C_IDX.mask(atom_array, default="generate")

        molecule_annot_changes = molecule_annot[1:] != molecule_annot[:-1]
        globally_indexed_changes = is_globally_indexed[1:] != is_globally_indexed[:-1]
        index_group_changes = molecule_annot_changes | globally_indexed_changes

        # Convert mask to indices
        # Add 1, to shift the indices from the end of a residue to the start of a new residue
        index_group_starts = np.where(index_group_changes)[0] + 1
        num_index_groups = len(index_group_starts) + 1  # +1 for the first segment starting at 0

        # The first residue is not included yet -> Insert '[0]'
        index_group_starts_ends = np.concatenate(([0], index_group_starts, [atom_array.array_length()]))
        all_index_groups = spread_segment_wise(index_group_starts_ends, np.arange(num_index_groups))

        return all_index_groups


class CathLabel(ConditionBase):
    """Conditioning that provides cath labels to the model. 'x.x.x.x' is null conditioning."""

    name = "cath-label"
    n_body = 1
    level = Level.RESIDUE  # In practice this is often chain-level
    dtype = str
    NULL_VALUE = "x.x.x.x"

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            annotation = cls.annotation(atom_array, default="raise")
            return annotation != cls.NULL_VALUE
        return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        # Max size of 23 is more than enough (allows for 5-digit ids at each level of the cath hierarchy)
        return np.full(atom_array.array_length(), cls.NULL_VALUE, dtype=np.dtype("U23"))


class InternalCoordinate(ConditionBase):
    """Conditioning that provides internal coordinates which are only meaningful within a given subset of atoms.

    For example, centered and rotated coordinates of the antibody framework and the target, providing their internal
    structures but not the dock between them.
    """

    # NOTE: CamelCase is used since underscores are forbidden and dashes break the query syntax
    name = "InternalCoordinate"
    n_body = 1
    level = Level.ATOM
    dtype = float

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            # ... if annotation exists, derive the mask from it
            return np.isfinite(cls.annotation(atom_array, default="raise")).any(axis=1)
        else:
            # ... otherwise, default to an empty mask
            return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        return np.full((atom_array.array_length(), 3), fill_value=np.nan)


class CoordinateGroup(ConditionBase):
    """Indicates atoms within the same equivalence class of known relative coordinates."""

    # NOTE: CamelCase is used since underscores are forbidden and dashes break the query syntax
    name = "CoordinateGroup"
    n_body = 1
    level = Level.ATOM
    dtype = int
    KNOWN_ABSOLUTE_COORDS = 0  # Standard absolute coordinate conditioning
    UNKNOWN_COORDS = -1  # No form of relative or absolute coordinate conditioning

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        return np.ones(atom_array.array_length(), dtype=bool)  # Every atom has a coordinate group

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        # Three groups by default: no coordinate conditioning, absolute coordinate conditioning,
        # and relative coordinate conditioning
        all_coordinate_groups = np.full(atom_array.array_length(), cls.UNKNOWN_COORDS, dtype=int)
        if C_CRD.get_mask_name() in atom_array.get_annotation_categories():
            has_known_absolute_coords = atom_array.get_annotation(C_CRD.get_mask_name())
            all_coordinate_groups[has_known_absolute_coords] = cls.KNOWN_ABSOLUTE_COORDS
        if InternalCoordinate.get_mask_name() in atom_array.get_annotation_categories():
            has_known_internal_coords = atom_array.get_annotation(InternalCoordinate.get_mask_name())
            all_coordinate_groups[has_known_internal_coords] = 1
        return all_coordinate_groups


class BasePairEnum(IntEnum):
    """
    the Enum to hold int info corresponding to BasePair annotations
    """

    UNSPECIFIED = 0  # mask
    PAIR = 1  # base paired positions
    LOOP = 2  # explicitl-unpaired


class BasePair(ConditionBase):
    """NA secondary-structure pair condition (residue semantics, 2-body sparse storage)."""

    name = "basepair"
    n_body = 2
    level = Level.RESIDUE
    is_symmetric = True
    dtype = int

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> AnnotationList2D:
        # Sparse default: empty/unspecified
        return AnnotationList2D(
            n_atoms=atom_array.array_length(),
            pairs=np.array([], dtype=int),
            values=np.empty((0,), dtype=cls.dtype),
        )

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> AnnotationList2D:
        annotation = cls.annotation(atom_array, default="generate")
        return AnnotationList2D(
            n_atoms=annotation.n_atoms,
            pairs=annotation.pairs,
            values=np.ones(len(annotation.values), dtype=bool),
        )


class NucleicSSState(ConditionBase):
    """Per-residue NA secondary-structure state encoded with ``BasePairEnum``."""

    name = "na-ss-state"
    n_body = 1
    level = Level.RESIDUE
    dtype = int

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        annotation = cls.annotation(atom_array, default="generate")
        return annotation.astype(int) != BasePairEnum.UNSPECIFIED.value

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        return np.full(atom_array.array_length(), BasePairEnum.UNSPECIFIED.value, dtype=cls.dtype)


# Aliases for convenience
C_HOT = CONDITIONS.hotspot
C_CDR = CONDITIONS.cdr
C_ABH = CONDITIONS.ab_heavy
C_ABL = CONDITIONS.ab_light
C_RAS = CONDITIONS.rasa_binned
C_SEC = CONDITIONS.secondary_structure
C_HB = CONDITIONS.hbond
C_IDXGRP = CONDITIONS.index_group
C_CATH = CONDITIONS.cath_label
C_INTCRD = CONDITIONS.InternalCoordinate
C_CRDGRP = CONDITIONS.CoordinateGroup
C_BP = CONDITIONS.basepair
C_NA_SS_STATE = CONDITIONS.na_ss_state
