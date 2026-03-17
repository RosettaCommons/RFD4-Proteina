"""Pipeline for training for La Proteina-style models."""

import torch
import functools
import os
from typing import ClassVar
import logging
import pickle
import rootutils

import numpy as np
from atomworks.constants import (
    AF3_EXCLUDED_LIGANDS,
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_RNA,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_RNA,
)
from atomworks.enums import ChainType
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.conditions import (
    C_CRD,
    C_CTR,
    C_IDX,
    C_NTR,
    C_SEQ,
)
from atomworks.io.utils.annotator import (
    ensure_annotations,
)
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING, TokenEncoding
from atomworks.ml.preprocessing.msa.finding import get_msa_dirs
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_atom_array_has_bonds,
    check_contains_keys,
    check_is_instance,
    check_nonzero_length,
)
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    CopyAnnotation,
)
from atomworks.ml.transforms.atomize import (
    AtomizeByCCDName,
    FlagNonPolymersForAtomization,
)
from atomworks.ml.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    ListBuilder,
    RandomRoute,
    RemoveKeys,
    SubsetToKeys,
    Transform,
)
from atomworks.ml.transforms.covalent_modifications import (
    FlagAndReassignCovalentModifications,
)
from atomworks.ml.transforms.bfactor_conditioned_transforms import SetOccToZeroOnBfactor
from atomworks.ml.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from atomworks.ml.transforms.design_task import SampleDesignTask
from atomworks.ml.transforms.encoding import atom_array_to_encoded_resnames
from atomworks.ml.transforms.featurize_unresolved_residues import (
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
)
from atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveNucleicAcidTerminalOxygen,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedLigandAtomsIfTooMany,
    RemoveUnresolvedPNUnits,
)
from atomworks.ml.transforms.msa.msa import (
    EncodeMSA,
    LoadPolymerMSAs,
    PairAndMergePolymerMSAs,
    FillFullMSAFromEncoded,
    FeaturizeMSALikeAF3,
)
from atomworks.ml.utils.token import (
    apply_and_spread_token_wise,
    apply_segment_wise_2d,
    apply_token_wise,
    get_af3_token_center_masks,
    get_af3_token_representative_masks,
    get_token_starts,
    spread_token_wise,
)
from beartype.typing import Any
from biotite.structure import AtomArray
from einops import rearrange

from proteinfoundation.conditions import (
    C_HB,
    C_HOT,
    C_IDXGRP,
    C_CRDGRP,
    C_INTCRD,
    C_RAS,
    C_SEC,
    C_CATH,
    HBondEnum,
    RasaEnum,
)
from proteinfoundation.transforms.atom_names import (
    AnnotateAtomNameVisibility,
    FeaturizeAtomNamesLikeAF3,
)
from proteinfoundation.transforms.geometry import CenterOnCoordinateConditionedAtoms, RandomRotationTransform
from proteinfoundation.transforms.graph_laplacian import (
    FeaturizeBondsAsLaplacianPositionalEncoding,
)
from proteinfoundation.transforms.random_atomize_residues import (
    RandomAtomizeResidues,
)
from proteinfoundation.transforms.unindexing import UnindexFlaggedTokens
from proteinfoundation.utils.fold_utils import extract_cath_code_by_level

PROTEIN_MSA_DIRS_ENV_VAR = "PROTEIN_MSA_DIRS"
"""Environment variable for protein MSA directories (comma-separated paths)."""

RNA_MSA_DIRS_ENV_VAR = "RNA_MSA_DIRS"
"""Environment variable for RNA MSA directories (comma-separated paths)."""

# ruff: noqa: N806

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Config path
CATH_MAPPING_PATH = os.path.join(os.environ["PROJECT_ROOT"], "projects/proteinfoundation/data/cath_label_mapping.pkl")

logger = logging.getLogger(__name__)


def _resolve_msa_dirs(load_msa: bool) -> tuple[list[dict], list[dict]]:
    """Resolve MSA directories and report whether env vars are set vs. usable."""
    if not load_msa:
        return [], []

    protein_msa_env = os.getenv(PROTEIN_MSA_DIRS_ENV_VAR, "").strip()
    rna_msa_env = os.getenv(RNA_MSA_DIRS_ENV_VAR, "").strip()
    if not protein_msa_env and not rna_msa_env:
        raise ValueError(
            "load_msa=true but PROTEIN_MSA_DIRS/RNA_MSA_DIRS are not set or empty."
        )

    protein_msa_dirs = get_msa_dirs(PROTEIN_MSA_DIRS_ENV_VAR)
    rna_msa_dirs = get_msa_dirs(RNA_MSA_DIRS_ENV_VAR)
    if not (protein_msa_dirs or rna_msa_dirs):
        raise ValueError(
            "load_msa=true but no MSA files were found in the configured directories. "
            f"PROTEIN_MSA_DIRS='{protein_msa_env}', RNA_MSA_DIRS='{rna_msa_env}'."
        )

    return protein_msa_dirs, rna_msa_dirs


class TokenType:
    """Token type constants used in pipeline."""

    PROTEIN = 0
    RNA = 1
    DNA = 2
    ATOMIZED = 3
    UNKNOWN = 4


@functools.lru_cache(maxsize=1)
def _build_atom37_lookups() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sorted atom name arrays for vectorized slot lookups.

    Builds comprehensive lookup arrays by iterating through all standard and
    unknown amino acid and nucleic acid residues. Uses @lru_cache to ensure
    we only build the lookups once per Python session.

    Returns:
        Tuple of (aa_atom_names, aa_atom_slots, na_atom_names, na_atom_slots)
        where names are sorted arrays of atom names and slots are corresponding
        slot indices for the atom37 encoding.

    Note:
        Atom names are stored with trailing spaces stripped to match the format
        used in the UNIFIED_ATOM37_ENCODING. At runtime, we strip atom names from
        AtomArray before lookup.
    """
    # Build AA lookup (all standard + UNK)
    aa_atom_to_slot = {}
    aa_residues = [*STANDARD_AA, UNKNOWN_AA]
    for res_name in aa_residues:
        atoms = UNIFIED_ATOM37_ENCODING.token_atoms[res_name]
        for slot, atom_name in enumerate(atoms):
            if atom_name.strip():  # Skip empty slots
                aa_atom_to_slot[atom_name.strip()] = slot

    # Convert to sorted arrays for searchsorted
    aa_names_sorted = sorted(aa_atom_to_slot.keys())
    aa_atom_names = np.array(aa_names_sorted)
    aa_atom_slots = np.array([aa_atom_to_slot[name] for name in aa_names_sorted], dtype=np.int32)

    # Build NA lookup (all standard RNA + DNA + unknowns)
    na_atom_to_slot = {}
    na_residues = [*STANDARD_RNA, *STANDARD_DNA, UNKNOWN_RNA, UNKNOWN_DNA]
    for res_name in na_residues:
        atoms = UNIFIED_ATOM37_ENCODING.token_atoms[res_name]
        for slot, atom_name in enumerate(atoms):
            if atom_name.strip():
                na_atom_to_slot[atom_name.strip()] = slot

    # Convert to sorted arrays for searchsorted
    na_names_sorted = sorted(na_atom_to_slot.keys())
    na_atom_names = np.array(na_names_sorted)
    na_atom_slots = np.array([na_atom_to_slot[name] for name in na_names_sorted], dtype=np.int32)

    return aa_atom_names, aa_atom_slots, na_atom_names, na_atom_slots


# Module-level cached lookups
_AA_ATOM_NAMES, _AA_ATOM_SLOTS, _NA_ATOM_NAMES, _NA_ATOM_SLOTS = _build_atom37_lookups()


class InferQueryPNUnitIIDsForDimers(Transform):
    """Automatically infer query_pn_unit_iids for dimer structures.

    For simple datasets (TED dimers, AFDB dimers) that don't have query_pn_unit_iids
    in the parquet file, this transform automatically detects protein chains and
    sets query_pn_unit_iids if there are exactly 2 protein chains.

    This enables PPI tasks to work with simple dimer datasets without needing
    metadata columns in the parquet file.
    """

    def forward(self, data: dict) -> dict:
        """Infer query_pn_unit_iids from atom_array if not already present."""
        # Skip if query_pn_unit_iids already exists (from PDB interfaces loader)
        if "query_pn_unit_iids" in data:
            return data

        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_protein", "pn_unit_iid")

        # Get all protein pn_unit_iids
        protein_mask = atom_array.is_protein
        if not protein_mask.any():
            return data  # No proteins, skip

        protein_pn_unit_iids = np.unique(atom_array.pn_unit_iid[protein_mask])

        # Only set query_pn_unit_iids if there are exactly 2 protein chains (dimer)
        if len(protein_pn_unit_iids) == 2:
            data["query_pn_unit_iids"] = protein_pn_unit_iids.tolist()

        return data


class AnnotateAF3TokenCenterAtoms(Transform):
    """Annotate which atoms are AF3 token center atoms.

    Namely:
        - CA for standard amino acids
        - C1' for standard nucleotides
        - For other cases (including atomized residues), take the first and only atom as they are tokenized per-atom.
    """

    requires_previous_transforms: ClassVar[bool] = ["AtomizeByCCDName"]

    def forward(self, data: dict) -> dict:
        """Annotate AF3 token center atoms."""
        atom_array = data["atom_array"]
        is_af3_token_center = get_af3_token_center_masks(atom_array)
        atom_array.set_annotation("is_af3_token_center", is_af3_token_center)
        data["atom_array"] = atom_array
        return data


class ConvertToAtomLevelProteinaFeatures(Transform):
    """Convert inputs to Proteina features and conditioning annotations.

    Output features:
        - coords: [L, 3] coordinates in Angstroms, centered at origin
        - residue_type: [L] residue type encoding from UNIFIED_ATOM37_ENCODING
        - tok_idx: [L] token index for each atom
        - element: [L] atomic number (provided for sequence-conditioned tokens)
        - chain_iid: [L] chain instance ID
        - within_index_group_res_idx: [L] residue index within chain
        - Conditioning masks and values (C_CRD, C_SEQ, C_DIS, C_IDXGRP, etc.) of shape [L]
        - can_be_encoded: [L] bool mask for atoms to be encoded with the autoencoder
        - should_be_lossed: [L] bool mask for atoms to include in loss computation
    """

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(
            data, ["can_be_encoded", "should_be_lossed", "coord_to_be_noised", "is_token_center_atom"]
        )

    def forward(self, data: dict) -> dict:
        """Convert encoded data into atom-level Proteina features."""
        feats = data.get("feats", {})
        aa = as_atom_array_plus(data["atom_array"])
        ensure_annotations(
            aa, "is_standard_or_unknown_aa", "is_standard_or_unknown_dna", "is_standard_or_unknown_rna", "atomize"
        )

        # ... index features
        feats["tok_idx"] = aa.token_id  # [L]
        feats["within_index_group_res_idx"] = aa.within_index_group_res_idx  # [L]

        # Get unique chain IDs and random permutation
        # NOTE: We ensure that the chain_iid is randomly assigned for each chain in the structure,
        # rather than always assigning the first chain as chain_iid=0, second as chain_iid=1, etc.
        unique_chains, inverse_indices = np.unique(aa.chain_iid, return_inverse=True)
        random_chain_indices = np.random.permutation(len(unique_chains))
        feats["chain_iid"] = random_chain_indices[inverse_indices]  # [L]

        # Encode ground-truth residue types using atom_array_to_encoded_resnames (handles proteins, DNA, RNA, atomized)
        # Uses UNIFIED_ATOM37_ENCODING for full support (20 AAs + 4 RNA + 4 DNA + unknowns)
        feats["residue_type"] = atom_array_to_encoded_resnames(aa, UNIFIED_ATOM37_ENCODING, atomize_token="<A>")  # [L]

        # ground truth coordinates
        feats["coords"] = aa.coord_to_be_noised  # [L, 3]

        # Derive token_type from existing boolean annotations
        # 0: protein, 1: RNA, 2: DNA, 3: atomized, 4: unknown/other
        token_type = np.full(aa.array_length(), 4, dtype=np.int16)  # Default to unknown
        token_type[aa.get_annotation("is_standard_or_unknown_dna")] = 2  # DNA
        token_type[aa.get_annotation("is_standard_or_unknown_rna")] = 1  # RNA
        token_type[aa.get_annotation("is_standard_or_unknown_aa")] = 0  # Protein
        token_type[aa.get_annotation("atomize")] = 3
        aa.set_annotation("token_type", token_type)
        feats["token_type"] = token_type  # [L]

        # ... center atoms
        feats["is_center_atom_mask"] = aa.is_token_center_atom  # [L]

        # ... occupancy
        feats["is_resolved_mask"] = aa.occupancy > 0.0  # [L]

        # ... encoding and loss annotations
        # (Will be pooled to the token-level later)
        feats["can_be_encoded"] = aa.can_be_encoded  # [L]
        feats["should_be_lossed"] = aa.should_be_lossed  # [L]
        feats["coord_cond_mask"] = C_CRD.mask(aa, default="generate")  # [L] for diagnostics

        ###########################################################
        # Conditioning information
        ###########################################################

        # +------ Sequence conditioning (residue type conditioned) ------+
        # C_SEQ.annotation returns actual residue names where conditioned, atomworks.constants.MASKED where not
        aa_conditioned = aa.copy()
        aa_conditioned.res_name = C_SEQ.annotation(aa, default="generate")

        feats[C_SEQ.full_name] = atom_array_to_encoded_resnames(
            aa_conditioned, UNIFIED_ATOM37_ENCODING, atomize_token="<A>"
        )  # [L]

        # Elements are provided if they are part of a sequence conditioned token (which should already include
        # ligands and DNA/RNA). If the token is not sequence conditioned, we pass through the auto-encoder without
        # element information, but we'll remove it here to be thorough
        element_mask = C_SEQ.mask(aa, default="raise")
        feats["element"] = np.full(aa.array_length(), -1, dtype=np.int32)
        if any(element_mask):
            feats["element"][element_mask] = aa.atomic_number[element_mask]  # [L]

        # Charges are provided for sequence-conditioned tokens (especially ligands)
        # If charge annotation doesn't exist, initialize with zeros
        if "charge" not in aa.get_annotation_categories():
            aa.set_annotation("charge", np.zeros(aa.array_length(), dtype=np.int16))
        feats["charge"] = aa.charge.astype(np.int16)  # [L]

        # Number of hydrogens for atomized tokens (from RDKit/OpenBabel)
        # Values: 0-4 for actual hydrogen count, -1 for unspecified
        # If nhyd annotation doesn't exist, initialize with -1 (unspecified)
        if "nhyd" not in aa.get_annotation_categories():
            aa.set_annotation("nhyd", np.full(aa.array_length(), -1, dtype=np.int16))
        feats["nhyd"] = aa.nhyd.astype(np.int16)  # [L]

        # +------ CAT domain annotations from CATH ------+
        cath_codes = C_CATH.annotation(aa, default="generate")
        unique_cath_codes = np.unique(cath_codes)  # [num_codes]

        output_cath_codes = np.full((len(cath_codes), 3), -1, dtype=np.int32)
        if os.path.exists(CATH_MAPPING_PATH):
            with open(CATH_MAPPING_PATH, "rb") as f:
                cath_class_mapping = pickle.load(f)

            cath_class_mapping_C = cath_class_mapping["C"]
            cath_class_mapping_A = cath_class_mapping["A"]
            cath_class_mapping_T = cath_class_mapping["T"]

            remapped_unique_cath_codes = np.zeros((len(unique_cath_codes), 3), dtype=np.int32)
            for i, cath_code in enumerate(unique_cath_codes):
                remapped_unique_cath_codes[i, 0] = cath_class_mapping_C.get(
                    extract_cath_code_by_level(cath_code, "C"), -1
                )
                remapped_unique_cath_codes[i, 1] = cath_class_mapping_A.get(
                    extract_cath_code_by_level(cath_code, "A"), -1
                )
                remapped_unique_cath_codes[i, 2] = cath_class_mapping_T.get(
                    extract_cath_code_by_level(cath_code, "T"), -1
                )

            for i, code in enumerate(unique_cath_codes):
                output_cath_codes[cath_codes == code, :] = remapped_unique_cath_codes[i]

        elif len(unique_cath_codes) > 1 or unique_cath_codes[0] != C_CATH.NULL_VALUE:
            raise ValueError(
                f"CATH code mapping file not found at {CATH_MAPPING_PATH}, CATH conditioning must be default"
            )

        feats[C_CATH.full_name] = output_cath_codes

        # +------ Coordinate conditioning (absolute) ------+
        feats[C_CRD.mask_name] = C_CRD.mask(aa, default="generate")  # [L]

        # +------ Coordinate conditioning (internal) ------+
        feats[C_INTCRD.mask_name] = C_INTCRD.mask(aa, default="generate")  # [L]
        feats[C_INTCRD.full_name] = C_INTCRD.annotation(aa, default="generate")  # [L]

        # +------ Other atom-level conditions ------+
        # ... basic
        feats[C_SEQ.mask_name] = C_SEQ.mask(aa, default="generate")  # [L] (which atoms are sequence conditioned)
        feats[C_IDX.mask_name] = C_IDX.mask(aa, default="generate")  # [L] (True = indexed, False = unindexed)
        feats[C_CTR.full_name] = C_CTR.annotation(aa, default="generate")  # [L] (True = C-terminus)
        feats[C_NTR.full_name] = C_NTR.annotation(aa, default="generate")  # [L] (True = N-terminus)

        # ... fancy
        feats[C_HOT.full_name] = C_HOT.annotation(aa, default="generate")
        feats[C_RAS.full_name] = C_RAS.annotation(aa, default="generate")  # NOTE: this needs to be encoded
        feats[C_SEC.full_name] = C_SEC.annotation(aa, default="generate")  # NOTE: this needs to be encoded
        feats[C_HB.full_name] = C_HB.annotation(aa, default="generate")
        feats[C_IDXGRP.full_name] = C_IDXGRP.annotation(aa, default="generate")
        feats[C_CRDGRP.full_name] = C_CRDGRP.annotation(aa, default="generate")

        # +------ Two-body conditions ------+
        # TODO: Not yet supported

        # +------ Gap / missing-atom features ------+
        if "chain_breaks_per_residue" in aa.get_annotation_categories():
            feats["chain_breaks_per_residue"] = aa.chain_breaks_per_residue  # [L]
        if "missing_atom_count" in aa.get_annotation_categories():
            feats["missing_atom_count"] = aa.missing_atom_count  # [L]

        ###########################################################
        # MSA features (if available)
        ###########################################################
        # MSA is encoded earlier in pipeline by FillFullMSAFromEncoded
        # Full MSA: [n_rows, n_tokens] - already token-aligned
        # NOTE: Only `msa` is needed - MSA Pairformer derives all masks internally
        if "encoded" in data and "msa" in data["encoded"]:
            feats["msa"] = data["encoded"]["msa"]  # [n_rows, n_tokens]

        # ... add to data dict
        data["feats"] = feats
        return data


EXISTING_TOKEN_LEVEL_FEATURES = ["graph_laplacian_pe_token_level", "centroid_of_atoms_without_coordinate_conditioning"]
"""Features already at the token level that simply pass through."""

ATOM_LEVEL_FEATURES_TO_CONVERT_TO_TOKEN_LEVEL = [
    "residue_type",
    C_SEQ.full_name,
    "token_type",
    "tok_idx",
    "chain_iid",
    "within_index_group_res_idx",
    "can_be_encoded",
    "should_be_lossed",
    "coord_cond_mask",  # Coordinate conditioning mask (for diagnostics)
    C_SEQ.mask_name,
    C_CTR.full_name,
    C_NTR.full_name,
    C_SEC.full_name,
    C_IDXGRP.full_name,
    C_CATH.full_name,
    "chain_breaks_per_residue",
    "missing_atom_count",
]
"""Features to convert from atom-level to token-level by taking the first atom in each token."""

ATOM37_FEATURES = [
    ("coords", (3,), 0.0),
    ("element", (), -1),
    ("charge", (), 0),  # Atomic charge
    ("nhyd", (), -1),  # Number of hydrogens (0-4 for count, -1 for unspecified)
    ("is_resolved_mask", (), False),
    ("is_center_atom_mask", (), False),
    (C_CRD.mask_name, (), False),
    (C_INTCRD.mask_name, (), False),
    (C_IDX.mask_name, (), True),
    (C_CRDGRP.full_name, (), C_CRDGRP.UNKNOWN_COORDS),
    (C_INTCRD.full_name, (3,), 0.0),
    (C_HOT.full_name, (), False),
    (C_RAS.full_name, (), RasaEnum.UNSPECIFIED),
    (C_HB.full_name, (), HBondEnum.NONE),
]
"""Features to convert from atom-level to token-level atom37 representation.

Schema: (feature_name, extra_dims, pad_value)
"""


class ConvertToAtom37Format(Transform):
    """Convert atom-level features to token-level atom37 representation.

    Atomized residues place their first (and only) atom in slot 1.

    Input (from :py:class:`ConvertToAtomLevelProteinaFeatures`):
        feats with atom-level tensors:
        - coords: [n_atoms, 3]
        - residue_type: [n_atoms]
        - element: [n_atoms]
        - tok_idx: [n_atoms]
        - etc.

    Output:
        feats with token-level tensors:
        - coords: [n_tokens, 37, 3] - Atom37 coordinates
        - element: [n_tokens, 37] - Atomic numbers per slot
        - atom37_mask: [n_tokens, 37] - Valid atom slots
        - Token-level features: residue_type: [n_tokens], etc.
    """

    def forward(self, data: dict) -> dict:
        """Convert atom-level features to token-level atom37."""

        result = {}

        # Extract basic info
        feats = data["feats"]
        atom_array = data["atom_array"]

        token_starts = get_token_starts(atom_array, add_exclusive_stop=False)
        num_tokens = len(token_starts)
        atom_names = np.char.strip(atom_array.atom_name.astype(str))  # [L]

        # +--------- Pass through existing token-level features ---------+
        for feat_name in EXISTING_TOKEN_LEVEL_FEATURES:
            if feat_name in feats:
                result[feat_name] = feats[feat_name]

        # +--------- Convert atom-level features to token-level by using the first atom in each token ---------+

        for feat_name in ATOM_LEVEL_FEATURES_TO_CONVERT_TO_TOKEN_LEVEL:
            if feat_name in feats:
                result[feat_name] = feats[feat_name][token_starts]

        # +--------- Build Atom37 features by scattering atoms into appropriate slots ---------+

        # Initialize atom37 arrays and atom mask
        atom37_arrays = {}
        for feat_name, extra_dims, pad_value in ATOM37_FEATURES:
            if feat_name in feats:
                shape = (num_tokens, 37, *extra_dims)
                atom37_arrays[feat_name] = np.full(shape, pad_value, dtype=feats[feat_name].dtype)  # [L, 37, ...]

        atom37_mask = np.zeros((num_tokens, 37), dtype=bool)  # [L, 37]
        n_atoms = atom_array.array_length()

        # Initialize slot assignments (-1 = invalid)
        atom_slots = np.full(n_atoms, -1, dtype=np.int32)  # [n_atoms]

        # Vectorized slot lookup using np.searchsorted
        token_type = feats["token_type"]  # [n_atoms] with values from TokenType enum

        is_protein = token_type == TokenType.PROTEIN
        if is_protein.any():
            protein_atoms = atom_names[is_protein]
            sorted_positions = np.searchsorted(_AA_ATOM_NAMES, protein_atoms)
            atom_slots[is_protein] = _AA_ATOM_SLOTS[sorted_positions]

        is_na = (token_type == TokenType.RNA) | (token_type == TokenType.DNA)
        if is_na.any():
            na_atoms = atom_names[is_na]
            sorted_positions = np.searchsorted(_NA_ATOM_NAMES, na_atoms)
            atom_slots[is_na] = _NA_ATOM_SLOTS[sorted_positions]

        # Atomized atoms → slot 1
        atom_slots[(token_type == TokenType.ATOMIZED)] = 1

        # Scatter atoms into atom37 arrays (vectorized)
        atom37_mask[atom_array.token_id, atom_slots] = True

        # Scatter all features at once
        for feat_name in atom37_arrays:
            atom37_arrays[feat_name][atom_array.token_id, atom_slots] = feats[feat_name]

        result["atom37_mask"] = atom37_mask
        result.update(atom37_arrays)

        # +--------- Derived atom-level features ---------+

        # Absolute coordinate conditioning (zeros for unconditioned atoms)
        if C_CRD.mask_name in result and "coords" in result:
            coord_cond_mask_expanded = result[C_CRD.mask_name][..., np.newaxis]
            result[C_CRD.full_name] = result["coords"] * coord_cond_mask_expanded

        # Internal coordinate conditioning (zeros for unconditioned atoms)
        if C_INTCRD.mask_name in result and C_INTCRD.full_name in result:
            internal_coord_cond_mask_expanded = result[C_INTCRD.mask_name][..., np.newaxis]
            result[C_INTCRD.full_name] = result[C_INTCRD.full_name] * internal_coord_cond_mask_expanded

        # Override existing atom-level features with atom37 versions
        data["feats"] = result

        return data


def get_bond_order_matrix(atom_array: AtomArray) -> np.ndarray:
    """
    Get the bond order matrix for an atom array where values are as follows:
    - 0 = no bond
    - 1 = single bond
    - 2 = double bond
    - 3 = triple bond
    - 4 = aromatic bond

    All other bond types are mapped to 1 (single bond)
    """
    atom1_idxs, atom2_idxs, bond_orders = atom_array.bonds.as_array().T

    # ============= Map to our bond order encoding ==============
    bond_orders = bond_orders.astype(np.int16)

    # Quadruple bonds and "coordination" bonds are mapped to 1
    bond_orders[bond_orders == 4] = 1
    bond_orders[bond_orders == 8] = 1

    # Various kinds of aromatic bonds are all mapped to 4
    bond_orders[bond_orders == 5] = 4
    bond_orders[bond_orders == 6] = 4
    bond_orders[bond_orders == 7] = 4
    bond_orders[bond_orders == 9] = 4

    # there shouldn't be any other types left
    assert np.all(bond_orders >= 0) and np.all(bond_orders <= 4), "Invalid bond order detected"

    # ============= Build bond order matrix ==============
    bond_order_matrix = np.zeros((atom_array.array_length(), atom_array.array_length()), dtype=np.int16)
    bond_order_matrix[atom1_idxs, atom2_idxs] = bond_orders
    bond_order_matrix[atom2_idxs, atom1_idxs] = bond_orders

    return bond_order_matrix


def get_bond_distances(atom_array: AtomArray) -> np.ndarray:
    """Returns the bond distance (adjacency) list as a 1D array."""
    coords = atom_array.coord
    atom1_idxs, atom2_idxs, _ = atom_array.bonds.as_array().T
    return np.linalg.norm(coords[atom1_idxs] - coords[atom2_idxs], axis=1)


def get_bond_distance_matrix(atom_array: AtomArray) -> np.ndarray:
    """Returns the bond adjacency matrix with bond distances as values."""
    atom1_idxs, atom2_idxs, _ = atom_array.bonds.as_array().T
    bond_distances = get_bond_distances(atom_array)
    bond_distance_matrix = np.full((atom_array.array_length(), atom_array.array_length()), np.inf)
    bond_distance_matrix[atom1_idxs, atom2_idxs] = bond_distances
    bond_distance_matrix[atom2_idxs, atom1_idxs] = bond_distances
    return bond_distance_matrix


class ComputeTokenLevelBondFeaturesLikeAF3(Transform):
    """
    Compute token-level bond features like AF3. Much of the functionality is the same as
    atomworks.ml.transforms.bonds.AddAF3TokenBondFeatures, but we want to get the bond order as well.

    Computes the following token-level bond features:
    - token_bond_mask: [n_tokens, n_tokens] - True if there is a bond between any atom in token i and token j
    - token_bond_order: [n_tokens, n_tokens] - Bond order between token i and token j
    """

    def __init__(self, distance_cutoff: float = np.inf):
        # NOTE: Atomworks parsing already handles filtering bonds by cutoff (actually does something fancier).
        # Thus we default distance cutoff to np.inf.
        self.distance_cutoff = distance_cutoff

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")
        check_atom_array_has_bonds(data)
        check_atom_array_annotation(data, ["is_polymer", "atomize"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        if "feats" not in data:
            data["feats"] = {}

        token_start_end_idxs = get_token_starts(atom_array, add_exclusive_stop=True)
        token_starts = token_start_end_idxs[:-1]
        token_bond_order = apply_segment_wise_2d(get_bond_order_matrix(atom_array), token_start_end_idxs, np.min)

        token_bond_distances = apply_segment_wise_2d(get_bond_distance_matrix(atom_array), token_start_end_idxs, np.min)
        token_bond_mask = token_bond_distances < self.distance_cutoff

        # remove token self-bonds
        np.fill_diagonal(token_bond_mask, False)
        np.fill_diagonal(token_bond_order, 0)

        # remove poly-poly bonds
        is_poly_poly_bond = np.outer(~atom_array.atomize[token_starts], ~atom_array.atomize[token_starts])
        token_bond_mask[is_poly_poly_bond] = False
        token_bond_order[is_poly_poly_bond] = 0

        data["feats"]["bond_mask"] = token_bond_mask
        data["feats"]["bond_order"] = token_bond_order

        return data


def add_can_be_encoded_annotation(
    atom_array: AtomArray,
    residue_names_to_encode: list | set,
    can_be_encoded_annotation: str = "can_be_encoded",
) -> AtomArray:
    """Determine which atoms should be encoded by the autoencoder.

    We encode all non-atomized standard amino acids by default.
    In the future, after re-training the autoencoder, we may expand this to include more residue types.

    Returns:
        atom_array with the `can_be_encoded` annotation set
    """
    # Get a mask of the residue types that the autoencoder can encode
    is_supported_residue = np.isin(atom_array.res_name, residue_names_to_encode)

    # Atomized residues should NOT be encoded (e.g., covalent modifications, atomized unindexed residues, etc.)
    # NOTE: Prior Transforms ensure atomized residues are sequence- and coordinate-conditioned
    if "atomize" in atom_array.get_annotation_categories():
        is_atomized = atom_array.atomize
    else:
        is_atomized = np.zeros(atom_array.array_length(), dtype=bool)

    can_be_encoded = is_supported_residue & ~is_atomized
    if "atomize" in atom_array.get_annotation_categories():
        assert not np.any(atom_array.atomize[can_be_encoded]), "Atomized residues should not be encoded"

    atom_array.set_annotation(can_be_encoded_annotation, can_be_encoded)

    return atom_array


RESIDUE_NAMES_TO_ENCODE = [
    *STANDARD_AA,
    # *STANDARD_RNA,
    # *STANDARD_DNA,
    UNKNOWN_AA,
    # UNKNOWN_RNA,
    # UNKNOWN_DNA,
]
"""Default residue names that can be encoded by the autoencoder."""


class AddCanBeEncodedAnnotation(Transform):
    """
    Annotate which atoms can be encoded by the autoencoder using the `can_be_encoded` annotation.
    For now we can only encode standard amino acids, but in the future we can also encode standard RNA/DNA.

    Chain-level logic for coordinate conditioning (when co_generate_targets=False):
    - **Full chain coordinate-conditioned**: Excluded from encoding (fixed targets)
    - **Partial chain coordinate-conditioned**: Still encoded (motifs within designed chains)

    This matches the loss supervision logic - we encode what we supervise.
    """

    def __init__(
        self,
        residue_names_to_encode: list | set,
    ):
        super().__init__()
        self.residue_names_to_encode = residue_names_to_encode

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        add_can_be_encoded_annotation(atom_array, self.residue_names_to_encode, "can_be_encoded")

        # Respect co_generate_targets flag: exclude fully coordinate-conditioned chains when False
        co_generate = data.get("co_generate_targets", True)  # default True for backward compat
        if not co_generate:
            # Use chain-level logic to distinguish full-chain targets vs partial-chain motifs
            can_be_encoded = atom_array.get_annotation("can_be_encoded")
            coord_conditioned = C_CRD.mask(atom_array)

            # For each chain, check if ALL atoms have absolute coordinate conditioning
            unique_chains = np.unique(atom_array.chain_iid)
            fully_conditioned_chains_mask = np.zeros(atom_array.array_length(), dtype=bool)

            for chain_id in unique_chains:
                chain_mask = atom_array.chain_iid == chain_id
                chain_atoms_conditioned = coord_conditioned[chain_mask]

                # If ALL atoms in this chain have absolute coordinate conditioning → full chain target
                if np.all(chain_atoms_conditioned):
                    fully_conditioned_chains_mask[chain_mask] = True
                # If only SOME atoms are conditioned → motif within chain (still encode)

            # Exclude only fully conditioned chains from encoding
            can_be_encoded = can_be_encoded & ~fully_conditioned_chains_mask
            atom_array.set_annotation("can_be_encoded", can_be_encoded)

        data["atom_array"] = atom_array
        return data


class AddShouldBeLossedAnnotation(Transform):
    """Annotate which atoms should be included in the loss computation.

    We apply loss to atoms that have:
    1. Non-zero occupancy (resolved atoms)
    2. Known index (C_IDX mask = True, i.e., not unindexed)
    3. NOT fully coordinate-conditioned chains (if co_generate_targets=False)

    Unindexed atoms should NOT contribute to the loss.

    Chain-level logic for coordinate conditioning:
    - **Full chain coordinate-conditioned** (all atoms in chain have C_CRD):
      Excluded from loss (e.g., fixed ligands, target proteins)
    - **Partial chain coordinate-conditioned** (only some atoms have C_CRD):
      Included in loss (e.g., motifs within designed protein)

    This allows motif scaffolding to supervise the motif atoms while keeping
    full target chains fixed during generation.

    Examples:
      Case 1 - Ligand binder design:
        Chain A: protein (C_CRD=False) → supervise ✓
        Chain B: ligand (C_CRD=True for all atoms) → don't supervise (fixed)

      Case 2 - Motif scaffolding:
        Chain A: residues 1-50 (C_CRD=False) → supervise ✓
        Chain A: residues 51-60 (C_CRD=True) → supervise ✓ (motif)
        Chain A: residues 61-150 (C_CRD=False) → supervise ✓
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        occupancy_mask = atom_array.occupancy > 0.0
        # Unindexed atoms should NOT be included in loss
        # C_IDX mask: True = indexed (has known position), False = unindexed (position unknown)
        has_known_index = C_IDX.mask(atom_array, default="generate")

        should_be_lossed = occupancy_mask & has_known_index

        # Respect co_generate_targets flag for coordinate-conditioned tokens
        co_generate = data.get("co_generate_targets", True)  # default True for backward compat
        if not co_generate:
            # Use chain-level logic to distinguish full-chain targets vs partial-chain motifs
            coord_conditioned = C_CRD.mask(atom_array)

            # For each chain, check if all atoms have absolute coordinate conditioning
            unique_chains = np.unique(atom_array.chain_iid)
            fully_conditioned_chains_mask = np.zeros(atom_array.array_length(), dtype=bool)

            for chain_id in unique_chains:
                chain_mask = atom_array.chain_iid == chain_id
                chain_atoms_conditioned = coord_conditioned[chain_mask]

                # If all atoms in this chain have absolute coordinate conditioning → full chain target
                if np.all(chain_atoms_conditioned):
                    fully_conditioned_chains_mask[chain_mask] = True
                # If only some atoms are conditioned → motif within chain (still supervise)

            # Exclude only fully conditioned chains from loss
            should_be_lossed = should_be_lossed & ~fully_conditioned_chains_mask

        # NOTE: We apply loss token a token if ALL atoms in the token should be lossed
        atom_array.set_annotation("should_be_lossed", apply_and_spread_token_wise(atom_array, should_be_lossed, np.all))

        data["atom_array"] = atom_array
        return data



class RemoveTokensWithAnyNaNAtom(Transform):
    """Remove tokens if any atom has NaN coordinates.

    There are a few types of situations where NaNs can show up in a token:
    1. A residue has a single missing side-chain atom
        In this case, we cannot generate a latent representation fo the sidechain and therefore cannot create noise to train
    2. A residue is entirely nan coordinates
        Obviously in this case we cannot encode the token and therefore cannot make a noisy version of the token
    3. An atomized token has NaN coordinates (e.g. a ligand is missing an atom coordinate)
        We would need to have a "CA" coordinate to add noise to, so we cannot encode this token either

    NOTE: With the fixing of the PDB, we are hoping to eliminate most if not all occurances of cases (1) and (3).
    As we fix those, this Transform will remove fewer and fewer tokens.
    """

    requires_previous_transforms: ClassVar[list] = [AddWithinPolyResIdxAnnotation, AddWithinChainInstanceResIdx]

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        starts = get_token_starts(atom_array, add_exclusive_stop=True)
        mask_to_keep = np.ones(atom_array.array_length(), dtype=bool)
        for i in range(len(starts) - 1):
            s, e = starts[i], starts[i + 1]
            if np.any(np.isnan(atom_array.coord[s:e])):
                mask_to_keep[s:e] = False
        atom_array = atom_array[mask_to_keep]
        data["atom_array"] = atom_array
        return data


class ValidateNonEmptyStructure(Transform):
    """Validate that structure has atoms after filtering.

    Raises an error if the atom array is empty, which can happen after
    aggressive filtering of hydrogens, unresolved units, and undesired residues.
    """

    def forward(self, data: dict) -> dict:
        aa = data["atom_array"]
        if aa.array_length() == 0:
            example_id = data.get("example_id", "unknown")
            raise ValueError(
                f"Structure has no atoms after filtering (example_id: {example_id}). " "This structure will be skipped."
            )
        return data


class AddChainBreaksPerResidue(Transform):
    """Annotate polymer tokens adjacent to a sequence gap with a binary flag.

    Sets ``chain_breaks_per_residue`` to 1.0 for any polymer token whose left
    or right neighbor in the same chain has a gap (``within_index_group_res_idx``
    difference > 1), and 0.0 otherwise.  Non-polymer tokens are always 0.
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        starts = get_token_starts(atom_array, add_exclusive_stop=False)
        n_tokens = len(starts)
        chain_breaks = np.zeros(n_tokens, dtype=np.float32)
        res_idx = atom_array.within_chain_res_idx[starts]
        chain_iids = atom_array.chain_iid[starts]
        is_polymer = atom_array.is_polymer[starts]
        is_indexed = C_IDX.mask(atom_array)[starts]

        for i in range(n_tokens):
            if not is_polymer[i] or not is_indexed[i]:
                continue
            if i < n_tokens - 1 and chain_iids[i] == chain_iids[i + 1] and (res_idx[i + 1] - res_idx[i]) > 1 and is_indexed[i + 1]:
                chain_breaks[i] = 1.0
            if i > 0 and chain_iids[i] == chain_iids[i - 1] and (res_idx[i] - res_idx[i - 1]) > 1 and is_indexed[i - 1]:
                chain_breaks[i] = 1.0

        atom_array.set_annotation(
            "chain_breaks_per_residue",
            spread_token_wise(atom_array, chain_breaks),
        )
        data["atom_array"] = atom_array
        return data


class AddMissingAtomCount(Transform):
    """Annotate every atom with the number of unresolved atoms in its token.

    For polymer residues the count reflects how many heavy atoms in that
    residue have ``occupancy == 0``.  For ligand PN-units the count is the
    number of unresolved atoms across the whole unit, broadcast to every atom.
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        is_unresolved = atom_array.occupancy == 0

        missing_count = np.zeros(len(atom_array), dtype=np.float32)

        # Polymer residues: count unresolved atoms per residue
        starts = get_residue_starts(atom_array, add_exclusive_stop=True)
        for i in range(len(starts) - 1):
            s, e = starts[i], starts[i + 1]
            if atom_array.is_polymer[s]:
                missing_count[s:e] = float(np.sum(is_unresolved[s:e]))

        # Ligand PN-units: count unresolved atoms per unit
        is_ligand = ~atom_array.is_polymer
        for pn_unit in np.unique(atom_array.pn_unit_iid[is_ligand]):
            unit_mask = (atom_array.pn_unit_iid == pn_unit) & is_ligand
            missing_count[unit_mask] = float(np.sum(is_unresolved[unit_mask]))

        atom_array.set_annotation("missing_atom_count", missing_count)
        data["atom_array"] = atom_array
        return data


class EnsureAtomizedResiduesAreSequenceConditioned(Transform):
    """
    Ensure that all atomized residues are sequence conditioned.
    Temporary annotation that be removed for e.g. unfixed ligands.
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        if "atomize" in atom_array.get_annotation_categories():
            seq_cond_mask = C_SEQ.mask(atom_array, default="generate")
            C_SEQ.set_mask(atom_array, seq_cond_mask | atom_array.atomize)
        data["atom_array"] = atom_array
        return data


class EnsureRepresentativeAndCenterAtomAnnotations(Transform):
    """
    Ensure that the 'is_token_representative_atom' and 'is_token_center_atom' annotations are present in the AtomArray.

    If not present, it computes and adds these annotations using the aa_design version of
    get_af3_token_representative_masks and the AtomWorks version of get_af3_token_center_masks.
    """

    incompatible_previous_transforms: ClassVar[list[str]] = ["UnindexFlaggedTokens"]

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["is_inference"])

    def forward(self, data: dict) -> None:
        atom_array = data["atom_array"]
        is_inference = data["is_inference"]
        token_starts = get_token_starts(atom_array)
        has_index = C_IDX.mask(atom_array, default="generate")

        # At train time, these might not be identical because the DesignTask is allowed to mark only part of a
        # ground-truth residue for unindexing (e.g. tip-atom cases). At inference time, they should be identical, since
        # when tip-atom(s) are added they get a unique res_id and that residue should be fully unindexed
        fully_indexed_token_mask = apply_token_wise(atom_array, has_index, np.all)
        partially_indexed_token_mask = apply_token_wise(atom_array, has_index, np.any)
        if is_inference and not np.array_equal(fully_indexed_token_mask, partially_indexed_token_mask):
            raise ValueError("At inference time, all residues should be either fully indexed or fully unindexed. ")

        if "is_token_representative_atom" not in atom_array.get_annotation_categories():
            rep_mask = get_af3_token_representative_masks(atom_array, enforce_one_per_token=False)
            rep_counts_per_token = apply_token_wise(atom_array, rep_mask, np.sum)
            rep_counts_zero_mask = rep_counts_per_token == 0

            # At inference time, we may have unindexed residues without representative atoms (e.g. tip-atoms)
            # This occurs because when constructing the original AtomArray at inference we also add the in-context
            # conditioning (typically from an external file).
            if not is_inference and np.any(rep_counts_zero_mask):
                raise ValueError("All tokens should have a representative atom during training.")
            elif is_inference and np.any(rep_counts_zero_mask & fully_indexed_token_mask):
                raise ValueError("All fully indexed tokens should have a representative atom.")

            # Set the token start atom as the representative atom for tokens without one
            token_starts_without_rep_atoms = token_starts[rep_counts_zero_mask]
            rep_mask[token_starts_without_rep_atoms] = True

            if np.any(rep_counts_per_token > 1):
                raise ValueError("One token should never have multiple representative atoms.")

            atom_array.set_annotation("is_token_representative_atom", rep_mask)

        if "is_token_center_atom" not in atom_array.get_annotation_categories():
            center_mask = get_af3_token_center_masks(atom_array, enforce_one_per_token=False)
            center_counts_per_token = apply_token_wise(atom_array, center_mask, np.sum)
            center_counts_zero_mask = center_counts_per_token == 0

            if not is_inference and np.any(center_counts_zero_mask):
                raise ValueError("All tokens should have a center atom during training.")
            elif is_inference and np.any(center_counts_zero_mask & fully_indexed_token_mask):
                raise ValueError("All fully indexed tokens should have a center atom.")

            # Set the token start atom as the central atom for tokens without one
            token_starts_without_center_atoms = token_starts[center_counts_zero_mask]
            center_mask[token_starts_without_center_atoms] = True

            if np.any(center_counts_per_token > 1):
                raise ValueError("One token should never have multiple center atoms.")

            atom_array.set_annotation("is_token_center_atom", center_mask)
        return data


class FeaturizeCentroidOfAtomsWithoutCoordinateConditioning(Transform):
    """Add centroid of non-coordinate-conditioned atoms as a feature.

    Computes the centroid (center of mass) of all atoms that are NOT
    coordinate-conditioned and adds it as the ``centroid_of_atoms_without_coordinate_conditioning`` feature.
    This represents the approximate center of the region to be generated.

    Args:
        coord_annotation: Annotation name for coordinates to use.
            Defaults to ``"coord_to_be_noised"``.
        feature_name: Feature name for output centroid.
            Defaults to ``"centroid_of_atoms_without_coordinate_conditioning"``.
    """

    def __init__(
        self,
        coord_annotation: str = "coord_to_be_noised",
        feature_name: str = "centroid_of_atoms_without_coordinate_conditioning",
    ):
        super().__init__()
        self.coord_annotation = coord_annotation
        self.feature_name = feature_name

    def forward(self, data: dict) -> dict:
        """Compute and add centroid of non-coordinate-conditioned atoms."""
        aa = as_atom_array_plus(data["atom_array"])
        coords = aa.get_annotation(self.coord_annotation)

        # Get mask for atoms without absolute coordinate conditioning
        not_coord_mask = ~C_CRD.mask(aa, default="generate")

        # Compute centroid of non-coordinate-conditioned atoms
        # Use nanmean to ignore NaN coordinates from unresolved atoms
        centroid = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        if not_coord_mask.any():
            centroid = np.nanmean(coords[not_coord_mask], axis=0)

        # Add to features dict
        feats = data.get("feats", {})
        feats[self.feature_name] = centroid
        data["feats"] = feats

        return data


class DeleteNucleicAcids(Transform):
    """
    Delete nucleic acids from the atom array.

    NOTE: This is temporary until we actually want to train with DNA/RNA
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        atom_array = atom_array[~np.isin(atom_array.chain_type, ChainType.get_nucleic_acids())]
        data["atom_array"] = atom_array
        return data


class EncodeQuerySequenceForMSA(Transform):
    """Encode query sequence for MSA using specified TokenEncoding."""

    def __init__(self, encoding: TokenEncoding, atomize_token: str = "<A>"):
        self.encoding = encoding
        self.atomize_token = atomize_token

    def forward(self, data: dict) -> dict:
        aa = data["atom_array"]
        # Encode at atom-level first, then convert to token-level
        atom_level_encoded = atom_array_to_encoded_resnames(
            aa, self.encoding, atomize_token=self.atomize_token
        )
        # Convert to token-level by taking first atom of each token
        token_starts = get_token_starts(aa)
        data["msa_query_seq"] = atom_level_encoded[token_starts]
        return data

class ConcatenateMSAFeatures(Transform):
    """
    Construct final MSA tensors by taking a dictionary of static and recycle dynamic MSA features
    """
    def check_input(self, data: dict) -> None:
        check_contains_keys(
            data,
            [
                "msa_features",
                "feats",
            ],
        )
        assert len(data["msa_features"]["msa_features_per_recycle_dict"]["msa"]) == 1, (
            "Expected only one recycle for MSA feature concatenation."
        )

    def forward(self, data: dict) -> dict:
        # Aggregate and stack MSA features
        msa_feats = data["msa_features"]

        msa_stacked_by_recycle = torch.stack(
            msa_feats["msa_features_per_recycle_dict"]["msa"]
        ).float()  # [n_recycles, n_sequences, n_tokens_across_chains, n_types_of_tokens]
        has_deletion_stacked_by_recycle = torch.stack(
            msa_feats["msa_features_per_recycle_dict"]["has_insertion"]
        )  # [n_recycles, n_sequences, n_tokens_across_chains]
        deletion_value_stacked_by_recycle = torch.stack(
            msa_feats["msa_features_per_recycle_dict"]["insertion_value"]
        )  # [n_recycles, n_sequences, n_tokens_across_chains]

        msa_stack = torch.concatenate(
            [
                msa_stacked_by_recycle,
                rearrange(has_deletion_stacked_by_recycle, "... -> ... 1"),
                rearrange(deletion_value_stacked_by_recycle, "... -> ... 1"),
            ],
            dim=-1,
        )  # [n_recycles, n_msa, n_tokens_across_chains, n_types_of_tokens + 2] (float)

        # Add pairing information if present
        if "residue_is_paired" in msa_feats["msa_features_per_recycle_dict"]:
            residue_is_paired_stacked_by_recycle = torch.stack(
                msa_feats["msa_features_per_recycle_dict"]["residue_is_paired"]
            )
            msa_stack = torch.concatenate(
                [
                    msa_stack,
                    rearrange(residue_is_paired_stacked_by_recycle, "... -> ... 1"),
                ],
                dim=-1,
            )  # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains, n_types_of_tokens + 3] (float)

        profile = msa_feats["msa_static_features_dict"]["profile"]
        deletion_mean = msa_feats["msa_static_features_dict"]["insertion_mean"]
        # collate all features into a single tensor
        msa_stack = msa_stack.squeeze(0) # remove recycle dim for final msa input

        num_msa_sequences = msa_stack.shape[0]
        data["feats"]["msa_final"] = torch.concatenate(
            [
                msa_stack,
                rearrange(profile, " ... -> 1 ...").expand(num_msa_sequences, -1, -1),
                rearrange(deletion_mean, " ... -> 1 ... 1").expand(num_msa_sequences, -1, -1),
            ],
            dim=-1,
        )  # [n_msa, n_tokens_across_chains, n_types_of_tokens + 3 + n_profile + n_deletion_mean] (float)

        return data

######################################################################################
# Pipelines
######################################################################################


def _get_design_task_name(data: dict) -> str | None:
    """Get the design task name from the data dict."""
    return (data.get("task") or {}).get("name")


def _build_design_train_pipeline(
    crop_size: int | None = None,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    crop_center_cutoff_distance: float = 15.0,
    max_atoms_in_crop: int | None = None,
    is_inference: bool = False,
    p_atomize_residues: float = 0.0,
    design_tasks: dict[str, dict[str, Any]] = {},
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    b_factor_occupancy_min: float | None = None,  # For filtering out unresolved atoms from PDB or distillation sets
    p_random_rotation: float = 1.0,  # Probability of applying random rotation
    # MSA parameters
    load_msa: bool = False,
    max_msa_sequences: int = 10_000,
    **kwargs,  # to dump remaining args for now
) -> list[Transform]:
    # T = transform shorthand for readability
    T = ListBuilder()

    # (We may want to run the train pipe with is_inference=True; e.g., for simple validations of our training objective)
    T += AddData({"is_inference": is_inference})
    ###########################################################
    ### Step 0: Data Augmentation                           ###
    ###########################################################
    # Apply random rotation for data augmentation (only during training)
    if not is_inference and p_random_rotation > 0.0:
        T += RandomRotationTransform(p_apply=p_random_rotation)

    ###########################################################
    ### Step 1: Filter unwanted information                 ###
    ###########################################################
    # ... cleanup
    T += RemoveKeys(["atom_array_stack"], require_keys_exist=False)
    T += RemoveHydrogens()
    T += FilterToSpecifiedPNUnits(extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing")
    T += RemoveTerminalOxygen()
    T += RemoveNucleicAcidTerminalOxygen()
    T += RemoveUnresolvedPNUnits()
    T += HandleUndesiredResTokens(undesired_res_tokens=undesired_res_names)  # e.g., non-standard residues
    T += RemovePolymersWithTooFewResolvedResidues(min_residues=1)
    T += SetOccToZeroOnBfactor(bmin=b_factor_occupancy_min)
    T += RemoveUnresolvedLigandAtomsIfTooMany(unresolved_ligand_atom_limit=5)

    # Validate structure has atoms after filtering
    T += ValidateNonEmptyStructure()

    # ... add basic annotations
    ###########################################################
    ### 2. Add basic annotations                            ###
    ###########################################################
    T += AddGlobalAtomIdAnnotation(allow_overwrite=True)
    T += AddWithinChainInstanceResIdx()
    T += AddWithinPolyResIdxAnnotation()
    T += FlagAndReassignCovalentModifications()
    T += FlagNonPolymersForAtomization()
    T += RandomAtomizeResidues(p_atomize=p_atomize_residues)
    T += AtomizeByCCDName(
        atomize_by_default=True,
        res_names_to_ignore=RESIDUE_NAMES_TO_ENCODE,
        move_atomized_part_to_end=False,
        validate_atomize=False,
    )
    T += AddGlobalTokenIdAnnotation()

    ###########################################################
    ### Step 3: Crop And Remove Unusable tokens             ###
    ###########################################################
    T += RemoveTokensWithAnyNaNAtom()  # Remove tokens with any NaN coordinates (e.g., from unresolved atoms). We cannot generate noisy versions of these
    
    if crop_size and (crop_contiguous_probability > 0 or crop_spatial_probability > 0):
        assert crop_size > 0, "Crop size must be greater than 0"
        assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"

        T += RandomRoute.from_list(
            [
                (
                    crop_contiguous_probability,  # DISCUSS: Does this even make sense for design??
                    CropContiguousLikeAF3(
                        crop_size=crop_size,
                        keep_uncropped_atom_array=True,
                        max_atoms_in_crop=max_atoms_in_crop,
                        annotate_crop_boundary=True,
                    ),
                ),
                (
                    crop_spatial_probability,
                    CropSpatialLikeAF3(
                        crop_size=crop_size,
                        crop_center_cutoff_distance=crop_center_cutoff_distance,
                        keep_uncropped_atom_array=True,
                        max_atoms_in_crop=max_atoms_in_crop,
                        raise_if_missing_query=False,
                        annotate_crop_boundary=True,
                    ),
                ),
            ]
        )

    ###########################################################
    ### Step 3.5: Infer query_pn_unit_iids for dimers      ###
    ###########################################################
    # For simple datasets (TED dimers, AFDB dimers) that don't have query_pn_unit_iids
    # from the parquet loader, automatically detect and set them for PPI tasks
    T += InferQueryPNUnitIIDsForDimers()

    ###########################################################
    ### Step 3.6: Load MSAs                                 ###
    ###########################################################
    # Loading only - encoding and filling done within the featurization pipeline
    protein_msa_dirs, rna_msa_dirs = _resolve_msa_dirs(load_msa)

    if protein_msa_dirs or rna_msa_dirs:
        T += LoadPolymerMSAs(
            protein_msa_dirs=protein_msa_dirs,
            rna_msa_dirs=rna_msa_dirs,
            max_msa_sequences=max_msa_sequences,
            use_paths_in_chain_info=True,
        )
        T += PairAndMergePolymerMSAs(dense=True)

    ###########################################################
    ### Step 4: Build design task                           ###
    ###########################################################
    # Design task sampling: Sample and apply design tasks
    #
    # This section implements a flexible design task sampling system where different protein design
    # objectives can be conditionally applied based on the input data. The system works as follows:
    # 1. SampleDesignTask evaluates all available design tasks and samples one based on their
    #    frequencies among eligible tasks (those that can be applied to the current data)
    # 2. Each design task implements can_apply() to determine applicability and
    #    annotate_for_task_selection() to add required annotations
    # 3. The sampled task is then applied via ConditionalRoute to generate the appropriate
    #    design problem configuration
    # 4. Possible `DesignTaskModifier` functions can then be applied afterwards to modify the task
    #    These exist to reduce the overhead of writing full-fledged new design tasks by simply allowing
    #    to change the condition annotation values (e.g. `rasa`, `sequence`, `distance`) for a given,
    #    existing task.
    #
    # If no design tasks are provided (or empty dict), skip design task routing entirely.
    if design_tasks:
        T += SampleDesignTask(design_tasks=design_tasks)

        # ... create the design task
        T += ConditionalRoute(
            condition_func=_get_design_task_name,
            transform_map={name: task["transform"] for name, task in design_tasks.items()},
        )

        # ... (optionally) modify the design task post-hoc
        # TODO: Template this

    return T.tolist()




def _build_design_featurize_pipeline(
    residue_names_to_encode: list | set,
    metrics_tags: list[str] | set[str] | None = None,
    graph_laplacian_use_bond_order: bool = True,
    load_msa: bool = False,
    max_msa_sequences: int = 10_000,
    msa_featurize_n_recycles: int = 1,
    msa_featurize_n_msa: int = 1024,
) -> list[Transform]:
    """Build shared featurization pipeline across both training and validation.

    Args:
        residue_names_to_encode: Residue names to encode.
        metrics_tags: Tags to use for determining which Metrics apply to the outputs of this pipeline.
        graph_laplacian_use_bond_order: If True, weight graph edges by bond order for Laplacian PE.
            If False, use binary adjacency (all bonds weighted equally).

    Returns:
        List of transforms for featurization pipeline.
    """
    T = ListBuilder()

    # Add metrics tags if provided
    if metrics_tags is not None:
        T += AddData({"metrics_tags": metrics_tags})

    ###########################################################
    ### Step 1: Annotation fixes                            ###  # TODO: Should these be in validate / train portions?
    ###########################################################
    # TODO: Should these "Ensure" transforms be in the train pipeline instead of featurize?
    T += EnsureAtomizedResiduesAreSequenceConditioned()

    # ... Annotate finalized token ids after we are sure we are not motifying `atomize` feature anymore
    T += AddGlobalTokenIdAnnotation(allow_overwrite=True)

    # Create in-context conditioning corresponding to unindexed regions
    T += EnsureRepresentativeAndCenterAtomAnnotations()
    T += UnindexFlaggedTokens()
    T += AddChainBreaksPerResidue()
    T += AddMissingAtomCount()

    ###########################################################
    ### MSA Encoding (if MSAs were loaded)                  ###
    ###########################################################
    # MSA encoding uses UNIFIED_ATOM37_ENCODING (same as feats["residue_type"])
    # Remapping to encoding for MSA Pairformer happens in the model (until we re-train a new MSA encoder)
    protein_msa_dirs, rna_msa_dirs = _resolve_msa_dirs(load_msa)

    if protein_msa_dirs or rna_msa_dirs:
        # First, we encode the query sequence (which may be cropped)...
        T += EncodeQuerySequenceForMSA(encoding=UNIFIED_ATOM37_ENCODING)
        # ... then we encode the (paired) MSA sequences (full, un-cropped)
        T += EncodeMSA(
            encoding=UNIFIED_ATOM37_ENCODING,
            token_to_use_for_gap=UNIFIED_ATOM37_ENCODING.token_to_idx["<G>"],
        )
        # ... then finally, copy over the MSA where applicable (handling atomized and cropped regions appropriately)
        T += FillFullMSAFromEncoded(
            pad_token=UNIFIED_ATOM37_ENCODING.token_to_idx["<G>"],  # We use gap token for padding
            encoded_seq_path=("msa_query_seq",),
        )
        T += PairAndMergePolymerMSAs(dense=True)  # Re-merge after encoding/filling

    ###########################################################
    ### Step 2: Featurize all conditions                    ###
    ###########################################################
    # Prepare coordinates for noising (without modifying the ground truth)
    # ... add placeholder coordinates for noising
    T += CopyAnnotation(annotation_to_copy="coord", new_annotation="coord_to_be_noised")

    # Center coordinates and add centroid of the atoms without coordinate conditioning as a feature
    T += CenterOnCoordinateConditionedAtoms(coord_annotation="coord_to_be_noised")
    # NOTE: Previously, we called this feature the "ori_token", which is a misnomer since it's no longer the origin
    T += FeaturizeCentroidOfAtomsWithoutCoordinateConditioning(
        coord_annotation="coord_to_be_noised",
        feature_name="centroid_of_atoms_without_coordinate_conditioning",
    )

    # Annotate and encode atom names using AF3 encoding
    T += AnnotateAtomNameVisibility()
    T += FeaturizeAtomNamesLikeAF3()

    # Graph Laplacian positional encodings for atomized molecules (ligands, etc.)
    T += FeaturizeBondsAsLaplacianPositionalEncoding(
        encoding_dim=32,
        use_bond_order=graph_laplacian_use_bond_order,
    )

    # Add annotation for which atoms should be encoded / lossed in the model
    T += AddShouldBeLossedAnnotation()

    # Encode standard amino acids, RNA, and DNA nucleotides (atomised things are filtered out already)
    T += AddCanBeEncodedAnnotation(residue_names_to_encode=residue_names_to_encode)

    T += ConvertToAtomLevelProteinaFeatures()
    T += ConvertToAtom37Format()  # Convert atom-level to token-level atom37 representation
    T += ComputeTokenLevelBondFeaturesLikeAF3()  # Compute bond graphs (mask and order)
    # ConvertToTorch - only include MSA keys when MSA is enabled and configured
    convert_keys = ["feats"]
    if protein_msa_dirs or rna_msa_dirs:
        convert_keys.extend(["encoded", "full_msa_details"])
    T += ConvertToTorch(keys=convert_keys)
    
    # MSA featurization - only if MSA is enabled and configured
    if protein_msa_dirs or rna_msa_dirs:
        T += FeaturizeMSALikeAF3(
            encoding=UNIFIED_ATOM37_ENCODING,
            n_recycles=msa_featurize_n_recycles,
            n_msa=msa_featurize_n_msa,
        )
        T += ConcatenateMSAFeatures()
    # Subset to necessary keys only
    # NOTE: MSA features are now included in feats dict (msa, msa_is_padded_mask, etc.)
    keys_to_keep = [
        "example_id",
        "feats",
        "atom_array",
        "extra_info",
        "task",
        "metrics_tags",
        "num_tokens",
        "specification",
    ]

    T += SubsetToKeys(keys_to_keep)

    return T.tolist()


def _build_design_validation_pipeline(
    # MSA parameters
    load_msa: bool = False,
    max_msa_sequences: int = 10_000,
    **kwargs,  # to dump remaining args for now
) -> list[Transform]:
    # T = transform shorthand for readability
    # ... initialize
    T = ListBuilder()
    T += AddData({"is_inference": True})

    # ... cleanup
    T += RemoveTerminalOxygen()
    T += RemoveNucleicAcidTerminalOxygen()
    T += RemoveHydrogens()

    # ... add basic annotations
    T += AddGlobalAtomIdAnnotation(allow_overwrite=True)
    T += AddWithinChainInstanceResIdx()
    T += AddWithinPolyResIdxAnnotation()
    T += AtomizeByCCDName(
        atomize_by_default=True,
        res_names_to_ignore=RESIDUE_NAMES_TO_ENCODE,
    )

    # ... load MSAs (encoding/filling moved to featurization pipeline)
    protein_msa_dirs, rna_msa_dirs = _resolve_msa_dirs(load_msa)

    if protein_msa_dirs or rna_msa_dirs:
        T += LoadPolymerMSAs(
            protein_msa_dirs=protein_msa_dirs,
            rna_msa_dirs=rna_msa_dirs,
            max_msa_sequences=max_msa_sequences,
            use_paths_in_chain_info=True,
        )
        T += PairAndMergePolymerMSAs(dense=True)

    return T.tolist()


def build_design_train_pipeline(
    crop_size: int | None = None,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    crop_center_cutoff_distance: float = 15,
    b_factor_occupancy_min: float | None = None,
    p_atomize_residues: float = 0.0,
    p_random_rotation: float = 1.0,
    design_tasks: dict[str, dict] | None = None,
    is_inference: bool = False,
    metrics_tags: list[str] | set[str] | None = None,
    residue_names_to_encode: list | set = RESIDUE_NAMES_TO_ENCODE,
    graph_laplacian_use_bond_order: bool = True,
    # MSA parameters
    load_msa: bool = False,
    max_msa_sequences: int = 10_000,
    msa_featurize_n_recycles: int = 1,
    msa_featurize_n_msa: int = 1024,
) -> Transform:
    """Build training pipeline for protein design tasks.

    Supports both monomer and multi-chain (PPI) training.
    Pipeline: filter → augment → crop → design task → featurize

    MSA directories are loaded from environment variables:
        - ``PROTEIN_MSA_DIRS``: Comma-separated paths for protein MSAs
        - ``RNA_MSA_DIRS``: Comma-separated paths for RNA MSAs

    Args:
        crop_size: Max residues after cropping
        crop_contiguous_probability: Probability of contiguous cropping
        crop_spatial_probability: Probability of spatial cropping
        crop_center_cutoff_distance: Distance cutoff (Å) for spatial crop
        p_atomize_residues: Probability of atomizing standard residues
        p_random_rotation: Probability of random rotation augmentation
        design_tasks: Dictionary of design tasks where each task is a dict with
                     'frequency' and 'transform' keys. The 'transform' key holds
                     the instantiated DesignTask object.
                     If None, no design tasks are applied.
                     Example:
                         {
                             "unconditional": {
                                 "frequency": 2.0,
                                 "transform": UnconditionalTask(...)
                             }
                         }
        is_inference: If True, skips augmentation
        metrics_tags: Tags to use for determining which Metrics apply to the outputs of this pipeline.
        graph_laplacian_use_bond_order: If True, weight graph edges by bond order for Laplacian PE.
            If False, use binary adjacency (all bonds weighted equally).
        max_msa_sequences: Maximum number of MSA sequences to load per chain.

    Returns:
        Compose of transforms for training pipeline
    """

    transforms = []
    transforms += _build_design_train_pipeline(
        crop_size=crop_size,
        crop_contiguous_probability=crop_contiguous_probability,
        crop_spatial_probability=crop_spatial_probability,
        crop_center_cutoff_distance=crop_center_cutoff_distance,
        b_factor_occupancy_min=b_factor_occupancy_min,
        design_tasks=design_tasks,
        is_inference=is_inference,
        p_atomize_residues=p_atomize_residues,
        p_random_rotation=p_random_rotation,
        load_msa=load_msa,
        max_msa_sequences=max_msa_sequences,
    )
    transforms += _build_design_featurize_pipeline(
        metrics_tags=metrics_tags,
        residue_names_to_encode=residue_names_to_encode,
        graph_laplacian_use_bond_order=graph_laplacian_use_bond_order,
        load_msa=load_msa,
        max_msa_sequences=max_msa_sequences,
        msa_featurize_n_recycles=msa_featurize_n_recycles,
        msa_featurize_n_msa=msa_featurize_n_msa,
    )

    return Compose(transforms)


def build_design_validation_pipeline(
    metrics_tags: list[str] | set[str] | None = None,
    residue_names_to_encode: list | set = RESIDUE_NAMES_TO_ENCODE,
    graph_laplacian_use_bond_order: bool = True,
    # MSA parameters
    load_msa: bool = False,
    max_msa_sequences: int = 10_000,
    msa_featurize_n_recycles: int = 1,
    msa_featurize_n_msa: int = 1024,
) -> Transform:
    """Build the RFD3 validation pipeline.

    MSA directories are loaded from environment variables:
        - ``PROTEIN_MSA_DIRS``: Comma-separated paths for protein MSAs
        - ``RNA_MSA_DIRS``: Comma-separated paths for RNA MSAs

    Args:
        metrics_tags: Tags to use for determining which Metrics apply to the outputs of this pipeline.
        residue_names_to_encode: Residue names to encode.
        graph_laplacian_use_bond_order: If True, weight graph edges by bond order for Laplacian PE.
            If False, use binary adjacency (all bonds weighted equally).
        max_msa_sequences: Maximum number of MSA sequences to load per chain.

    Returns:
    """
    transforms = []
    transforms += _build_design_validation_pipeline(
        load_msa=load_msa,
        max_msa_sequences=max_msa_sequences,
    )
    transforms += _build_design_featurize_pipeline(
        metrics_tags=metrics_tags,
        residue_names_to_encode=residue_names_to_encode,
        graph_laplacian_use_bond_order=graph_laplacian_use_bond_order,
        load_msa=load_msa,
        max_msa_sequences=max_msa_sequences,
        msa_featurize_n_recycles=msa_featurize_n_recycles,
        msa_featurize_n_msa=msa_featurize_n_msa,
    )
    return Compose(transforms)
