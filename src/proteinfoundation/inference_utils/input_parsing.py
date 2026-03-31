import copy
import itertools
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
from atomworks.constants import STANDARD_AA
from atomworks.enums import ChainType
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from atomworks.ml.utils.token import (
    get_token_starts,
)
from biotite import structure as struc
from biotite.structure import AtomArray, BondList, get_residue_starts, spread_residue_wise
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from foundry.common import exists
from foundry.utils.ddp import RankedLogger
from proteinfoundation.conditions import (
    C_CATH,
    C_CRDGRP,
    C_HB,
    C_HOT,
    C_IDXGRP,
    C_INTCRD,
    C_RAS,
    C_SEC,
    HBondEnum,
    RasaEnum,
    SSEnum,
)
from proteinfoundation.inference_utils.components import (
    ComponentValidationError,
    fetch_mask_from_idx,
    get_design_pattern_with_constraints,
    get_motif_components_and_breaks,
)
from proteinfoundation.inference_utils.inference_utils import (
    REQUIRED_CONDITIONING_ANNOTATION_VALUES,
    REQUIRED_INFERENCE_ANNOTATIONS,
    check_has_required_conditioning_annotations,
    convert_existing_annotations_to_bool,
    create_cb_atoms,
    create_o_atoms,
    extract_ligand_array,
    get_motif_features,
    inference_load_,
    set_com,
    set_common_annotations,
    set_default_conditioning_annotations,
    set_indices,
)
from proteinfoundation.inference_utils.parsing import InputSelection
from proteinfoundation.inference_utils.symmetry.symmetry_utils import (
    SymmetryConfig,
    center_symmetric_src_atom_array,
    make_symmetric_atom_array,
)
from proteinfoundation.transforms.utils.conditioning_utils import convert_to_internal_coords

logging.basicConfig(level=logging.DEBUG)

logger = RankedLogger(__name__, rank_zero_only=True)


#################################################################################
# Custom infer_ori functions
#################################################################################


@contextmanager
def validator_context(validator_name: str, data: dict | None = None) -> Iterator[None]:
    """Context manager for validator execution with logging."""
    logger.debug(f"Starting validator: {validator_name}")
    try:
        yield
        logger.debug(f"✓ Completed validator: {validator_name}")
    except Exception as e:
        logger.error(
            f"✗ Failed in validator: {validator_name}\n" f"  Error: {e!s}\n" f"  Error type: {type(e).__name__}"
        )
        raise e


def create_diffused_residues(n: int, additional_annotations: dict | None = None, polymer_type: str = "P") -> AtomArray:
    from proteinfoundation.inference_utils.inference_utils import (
        BACKBONE_ATOM_NAMES_DNA,
        BACKBONE_ATOM_NAMES_RNA,
        NA_BACKBONE_ATOM_NAME_TO_ELEMENT,
    )

    if n <= 0:
        raise ValueError(f"Negative/null residue count ({n}) not allowed.")

    if polymer_type == "P":
        res_name = "ALA"
        bb_len = 5
        bb_atom_names = ["N", "CA", "C", "O", "CB"]
        bb_elements = ["N", "C", "C", "O", "C"]
        chain_type = ChainType.POLYPEPTIDE_L
    elif polymer_type == "R":
        res_name = "A"
        bb_len = len(BACKBONE_ATOM_NAMES_RNA)
        bb_atom_names = BACKBONE_ATOM_NAMES_RNA
        chain_type = ChainType.RNA
        bb_elements = [NA_BACKBONE_ATOM_NAME_TO_ELEMENT[item] for item in bb_atom_names]
    elif polymer_type == "D":
        res_name = "DA"
        bb_len = len(BACKBONE_ATOM_NAMES_DNA)
        bb_atom_names = BACKBONE_ATOM_NAMES_DNA
        chain_type = ChainType.DNA
        bb_elements = [NA_BACKBONE_ATOM_NAME_TO_ELEMENT[item] for item in bb_atom_names]
    else:
        raise ValueError(f"invalid polymer type detected: {polymer_type}, check contig!")

    atoms = []
    [
        atoms.extend(
            [
                struc.Atom(
                    np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    res_name=res_name,
                    res_id=idx,
                )
                for _ in range(bb_len)
            ]
        )
        for idx in range(1, n + 1)
    ]
    array = struc.array(atoms)
    array.set_annotation("element", np.array(bb_elements * n, dtype="<U3"))
    array.set_annotation("atom_name", np.array(bb_atom_names * n, dtype="<U3"))
    array.set_annotation("chain_type", np.array([chain_type] * array.array_length(), dtype=np.int8))
    ensure_annotations(array, "is_protein", "is_rna", "is_dna", "is_ligand")
    array = set_default_conditioning_annotations(array, motif=False, additional=additional_annotations)
    array = set_common_annotations(array)

    return array


def create_motif_residue(
    token: AtomArray,
    strip_sidechains_by_default: bool,
) -> AtomArray:
    if (
        strip_sidechains_by_default
        and token.res_name[0] in STANDARD_AA
        and not token.is_motif_atom_unindexed.astype(bool)[0]
    ):
        n_atoms = token.shape[0]
        diffuse_oxygen = False
        if n_atoms < 3:
            raise ValueError(f"Mising backbone atoms in input motif residue: {token}")
        if n_atoms == 3:
            # Handle cases with N, CA, C only;
            token = token + create_o_atoms(token.copy())
            diffuse_oxygen = True  # flag oxygen for generation

        # Subset to the first 4 atoms (N, CA, C, O) only
        names = token.atom_name
        token = token[np.isin(names, ["N", "CA", "C", "O"])]
        assert token.shape[0] == 4, f"Expected 4 atoms after subsetting, original names: {names}"

        # exactly N, CA, C, O but no CB. Place CB onto idealized position and conver to ALA
        # Sequence name ALA ensures the padded atoms to be diffused from the fixed backbone
        # are placed on the CB so as to not leak the identity of the residue.
        token = token + create_cb_atoms(token.copy())

        # Sequence name must be set to ALA such that the central atom is correctly CB
        token.res_name = np.full_like(token.res_name, "ALA", dtype=token.res_name.dtype)
        token.set_annotation(
            "is_motif_atom_with_fixed_coord",
            np.where(
                np.arange(token.shape[0], dtype=int) < (4 - int(diffuse_oxygen)),
                token.is_motif_atom_with_fixed_coord,
                0,
            ),
        )

    check_has_required_conditioning_annotations(token)
    token = set_common_annotations(token)
    token.set_annotation("res_id", np.full(token.shape[0], 1))  # Reset to 1

    return token


def _check_has_backbone_connections_to_nonstandard_residues(
    atom_array_accum: AtomArray | None,
    src_atom_array: AtomArray | None,
) -> bool:
    """
    Check if the source atom array has backbone C-N bonds involving non-standard residues.
    This is used to determine if we need to restore bonds in accumulate_components.
    This function ONLY returns True for backbone C-N bonds connecting:
    - Standard amino acids to non-standard residues (PTMs, modified AAs, etc.)
    - Non-standard residues to standard amino acids
    - Non-standard residues to other non-standard residues
    Sidechain bonds (e.g., ligand attached to CB, disulfide bridges) are IGNORED.
    Only backbone peptide bonds (C-N) are considered.
    Returns:
        True if there are backbone C-N bonds involving at least one non-standard residue
    """
    if atom_array_accum is None or src_atom_array is None:
        return False
    if (
        not hasattr(atom_array_accum, "bonds")
        or atom_array_accum.bonds is None
        or not hasattr(src_atom_array, "bonds")
        or src_atom_array.bonds is None
    ):
        return False

    bonds = src_atom_array.bonds.as_array()
    if len(bonds) == 0:
        return False

    # Early exit: Check if there are ANY non-standard residues first
    unique_res_names = np.unique(atom_array_accum.res_name)
    has_nonstandard = any(res_name not in STANDARD_AA for res_name in unique_res_names)

    if not has_nonstandard:
        # All residues are standard - no need to check bonds
        return False

    # Now check only backbone C-N bonds involving non-standard residues
    for bond in bonds:
        atom_i, atom_j, _ = bond
        atom_i_obj = src_atom_array[int(atom_i)]
        atom_j_obj = src_atom_array[int(atom_j)]

        # ONLY consider backbone C-N bonds
        is_backbone_bond = (atom_i_obj.atom_name == "C" and atom_j_obj.atom_name == "N") or (
            atom_i_obj.atom_name == "N" and atom_j_obj.atom_name == "C"
        )

        if not is_backbone_bond:
            # Skip non-backbone bonds (sidechain, disulfides, ligand attachments, etc.)
            continue

        # Check if at least one residue is non-standard
        res_i_is_standard = atom_i_obj.res_name in STANDARD_AA
        res_j_is_standard = atom_j_obj.res_name in STANDARD_AA

        # If at least one is non-standard, we have a backbone connection that needs restoration
        if not (res_i_is_standard and res_j_is_standard):
            logger.debug(
                f"Found backbone C-N bond involving non-standard residue: "
                f"{atom_i_obj.res_name}{atom_i_obj.res_id}:{atom_i_obj.atom_name} - "
                f"{atom_j_obj.res_name}{atom_j_obj.res_id}:{atom_j_obj.atom_name}"
            )
            return True

    return False


def _restore_bonds_for_nonstandard_residues(
    atom_array_accum: struc.AtomArray,
    src_atom_array: struc.AtomArray | None,
    source_to_accum_idx: dict[int, int],
) -> struc.AtomArray:
    """
    Restores and creates bonds for non-standard residues (PTMs, modified AAs, etc.)
    from source structure and between consecutive residues.
    This function:
    1. Preserves inter-residue bonds from the source structure (if available)
    2. Adds backbone C-N bonds between consecutive residues where at least one is non-standard
    Args:
        atom_array_accum: The accumulated atom array to add bonds to
        src_atom_array: The source atom array containing original bond information
        source_to_accum_idx: Mapping from source atom indices to accumulated array indices
    Returns:
        atom_array_accum with bonds added
    """
    # Initialize bonds if needed
    if atom_array_accum.bonds is None:
        atom_array_accum.bonds = struc.BondList(atom_array_accum.array_length())

    # Step 1: Restore inter-residue bonds from the source atom array (only for non-standard residues)
    if src_atom_array is not None and hasattr(src_atom_array, "bonds") and src_atom_array.bonds is not None:
        original_bonds = src_atom_array.bonds.as_array()
        if len(original_bonds) > 0:
            # Extract bonds where both atoms are in the accumulated array
            bonds_to_add = []
            for bond in original_bonds:
                atom_i, atom_j, bond_type = bond
                # Check if both atoms are in our mapping
                if int(atom_i) in source_to_accum_idx and int(atom_j) in source_to_accum_idx:
                    # Check if at least one atom is from a non-standard residue
                    src_res_i = src_atom_array[int(atom_i)].res_name
                    src_res_j = src_atom_array[int(atom_j)].res_name

                    # Only preserve if at least one residue is non-standard
                    if src_res_i not in STANDARD_AA or src_res_j not in STANDARD_AA:
                        new_i = source_to_accum_idx[int(atom_i)]
                        new_j = source_to_accum_idx[int(atom_j)]
                        bonds_to_add.append([new_i, new_j, int(bond_type)])

            if bonds_to_add:
                # Add the preserved bonds
                new_bonds = struc.BondList(
                    atom_array_accum.array_length(),
                    np.array(bonds_to_add, dtype=np.int64),
                )
                atom_array_accum.bonds = atom_array_accum.bonds.merge(new_bonds)
                logger.info(
                    f"Preserved {len(bonds_to_add)} inter-residue bonds involving"
                    " non-standard residues from source structure"
                )

    # Step 2: Add backbone bonds between consecutive residues where at least one is non-standard
    # This handles: PTM-to-diffused, diffused-to-PTM, PTM-to-PTM, ligand-to-protein
    bonds_to_add = []

    # Group by residue
    token_starts = get_token_starts(atom_array_accum, add_exclusive_stop=True)

    for i in range(len(token_starts) - 2):  # -2 because we need pairs and token_starts has exclusive stop
        curr_start, curr_end = token_starts[i], token_starts[i + 1]
        next_start, next_end = token_starts[i + 1], token_starts[i + 2]

        curr_residue = atom_array_accum[curr_start:curr_end]
        next_residue = atom_array_accum[next_start:next_end]

        # Check if at least one residue is non-standard (PTMs, modified AAs, etc.)
        curr_is_nonstandard = curr_residue.res_name[0] not in STANDARD_AA
        next_is_nonstandard = next_residue.res_name[0] not in STANDARD_AA

        # Only add bonds if at least one residue is non-standard
        if not (curr_is_nonstandard or next_is_nonstandard):
            continue

        # Check if consecutive in same chain
        if curr_residue.chain_id[0] != next_residue.chain_id[0]:
            continue
        if next_residue.res_id[0] - curr_residue.res_id[0] != 1:
            continue

        # Find C atom in current residue (C-terminus connection point)
        c_mask = curr_residue.atom_name == "C"
        if not np.any(c_mask):
            # If a non-standard residue doesn't have a C atom, it can't connect to next residue
            # This is expected for some atomized residues or ligands at chain termini
            if curr_is_nonstandard and next_is_nonstandard:
                # Both are non-standard but no C in current - might be an atomized region without proper termini
                logger.debug(
                    f"Non-standard residue {curr_residue.res_name[0]} (res_id {curr_residue.res_id[0]}) "
                    f"has no C atom - cannot form backbone bond to next residue"
                )
            continue
        c_idx = curr_start + np.where(c_mask)[0][0]

        # Find N atom in next residue (N-terminus connection point)
        n_mask = next_residue.atom_name == "N"
        if not np.any(n_mask):
            # If a non-standard residue doesn't have an N atom, it can't connect to previous residue
            # This is expected for some atomized residues or ligands at chain termini
            if curr_is_nonstandard and next_is_nonstandard:
                # Both are non-standard but no N in next - might be an atomized region without proper termini
                logger.debug(
                    f"Non-standard residue {next_residue.res_name[0]} (res_id {next_residue.res_id[0]}) "
                    f"has no N atom - cannot form backbone bond from previous residue"
                )
            continue
        n_idx = next_start + np.where(n_mask)[0][0]

        # Check if this bond already exists (from source preservation)
        existing_bonds = atom_array_accum.bonds.as_array()
        bond_exists = False
        if len(existing_bonds) > 0:
            for existing_bond in existing_bonds:
                if (existing_bond[0] == c_idx and existing_bond[1] == n_idx) or (
                    existing_bond[0] == n_idx and existing_bond[1] == c_idx
                ):
                    bond_exists = True
                    break

        if not bond_exists:
            bonds_to_add.append([c_idx, n_idx, struc.BondType.SINGLE])

    if bonds_to_add:
        new_bonds = struc.BondList(atom_array_accum.array_length(), np.array(bonds_to_add, dtype=np.int64))
        atom_array_accum.bonds = atom_array_accum.bonds.merge(new_bonds)
        logger.info(f"Added {len(bonds_to_add)} backbone bonds involving non-standard residues")

    return atom_array_accum


def accumulate_components(
    components_to_accumulate: list[str | int],
    *,
    # Tokens from input
    indexed_tokens: dict[str, AtomArray],
    unindexed_tokens: dict[str, AtomArray],
    # Additional parameters
    atom_array_accum: list = [],
    start_chain: str = "A",
    start_resid: int = 1,
    unindexed_breaks: list[bool] | None = [],
    src_atom_array: AtomArray | None = None,
    strip_sidechains_by_default: bool = False,
    **kwargs: Any,
) -> AtomArray:
    # ... Create list of components
    assert (x := (set(list(indexed_tokens.keys()) + list(unindexed_tokens.keys())))).issubset(
        y := set(components_to_accumulate)
    ), f"Unindexed and indexed set {x} is not subset of components to accumulate {y}"
    all_tokens = indexed_tokens | unindexed_tokens
    all_annots = []
    [all_annots.extend(list(tok.get_annotation_categories())) for tok in all_tokens.values()]
    if src_atom_array is not None:
        all_annots.extend(list(src_atom_array.get_annotation_categories()))
    all_annots = set(all_annots)
    atom_array_accum = [] if atom_array_accum is None else atom_array_accum
    unindexed_breaks = [None] * len(components_to_accumulate) if unindexed_breaks is None else unindexed_breaks

    # ... For-loop accum variables
    unindexed_components_started = False  # once one unindexed component is added, stop adding diffused residues
    chain = start_chain
    res_id = start_resid
    molecule_id = 0
    source_to_accum_idx: dict[int, int] = {}
    current_accum_idx = sum(len(arr) for arr in atom_array_accum)

    # ... Insert contig information one- by one-
    assert len(components_to_accumulate) == len(
        unindexed_breaks
    ), "Mismatch in number of components to accumulate and breaks"
    for component, is_break in zip(components_to_accumulate, unindexed_breaks, strict=False):
        src_indices = None
        if exists(is_break) and is_break and not unindexed_components_started:
            chain = start_chain
            unindexed_components_started = True

        if component == "/0":
            # Reset iterators on next chain
            chain = chr(ord(chain) + 1)
            molecule_id += 1
            res_id = 1
            continue

        # ... Create array to insert
        if str(component)[0].isalpha():  # motif (e.g. "A22")
            n = 1

            # ... Fetch the motif residue
            token = all_tokens[component]
            if src_atom_array is not None:
                try:
                    src_mask = fetch_mask_from_idx(component, atom_array=src_atom_array)
                    src_indices = np.where(src_mask)[0]
                except ComponentValidationError:
                    src_indices = None

            # ... Ensure motif residues are set properly
            token = create_motif_residue(token, strip_sidechains_by_default=strip_sidechains_by_default)

            # ... Insert breakpoint when break clause is met
            if exists(is_break) and is_break:
                token.set_annotation(
                    "is_motif_atom_unindexed_motif_breakpoint",
                    np.ones(token.shape[0], dtype=int),
                )
            else:
                token.set_annotation(
                    "is_motif_atom_unindexed_motif_breakpoint",
                    np.zeros(token.shape[0], dtype=int),
                )
        else:
            if component[-1] in ["P", "R", "D"]:  # if polymer type specified
                polymer_type = component[-1]  # can be 'P'rotein, 'R'NA, 'D'NA
                n = int(component[:-1])
            else:
                polymer_type = "P"
                n = int(component)

            # ... Skip if none or unindexed
            if n == 0 or unindexed_components_started:
                res_id += n
                continue

            # ... Create diffused residues
            token = create_diffused_residues(n, all_annots, polymer_type)

        # ... Set index of insertion
        token = set_indices(
            array=token,
            chain=chain,
            res_id_start=res_id,
            molecule_id=molecule_id,
            component=component,
        )

        assert (
            len(get_token_starts(token)) == n
        ), f"Mismatch in number of residues: expected {n}, got {len(get_token_starts(token))} in \n{token}"

        if (
            src_atom_array is not None
            and str(component)[0].isalpha()
            and src_indices is not None
            and len(src_indices) == len(token)
        ):
            for i, src_idx in enumerate(src_indices):
                source_to_accum_idx[int(src_idx)] = current_accum_idx + i

        # ... Insert & Increment residue ID
        atom_array_accum.append(token)
        res_id += n
        current_accum_idx += len(token)

    # ... Concatenate all components
    atom_array_accum = struc.concatenate(atom_array_accum)
    atom_array_accum.set_annotation("pn_unit_iid", atom_array_accum.chain_id)

    should_restore_bonds = (
        src_atom_array is not None
        and bool(source_to_accum_idx)
        and _check_has_backbone_connections_to_nonstandard_residues(atom_array_accum, src_atom_array)
    )
    if should_restore_bonds:
        assert not unindexed_tokens, (
            "PTM backbone bond restoration is not compatible with unindexed components. "
            "PTMs must be specified as indexed components (using 'contig' parameter, not 'unindex'). "
            f"Found unindexed components: {list(unindexed_tokens.keys())}"
        )
        atom_array_accum = _restore_bonds_for_nonstandard_residues(
            atom_array_accum, src_atom_array, source_to_accum_idx
        )

    # Reset res_id for unindexed residues to avoid duplicates
    if np.any(atom_array_accum.is_motif_atom_unindexed.astype(bool)) and not np.all(
        atom_array_accum.is_motif_atom_unindexed.astype(bool)
    ):
        max_id = np.max(atom_array_accum[~atom_array_accum.is_motif_atom_unindexed.astype(bool)].res_id)
        atom_array_accum.res_id[atom_array_accum.is_motif_atom_unindexed.astype(bool)] += max_id + 1

    # ... Bonds
    if atom_array_accum.bonds is None:
        atom_array_accum.bonds = BondList(atom_array_accum.array_length())
    return atom_array_accum


# ========================================================================
# Input specification
# ========================================================================


class InputSpecification(BaseModel):
    """Validated and parsed input specification before resolution."""

    model_config = ConfigDict(
        hide_input_in_errors=False,
        arbitrary_types_allowed=True,
        validate_assignment=False,
        str_strip_whitespace=True,
        str_min_length=1,
        extra="forbid",
    )
    # fmt: off
    # ========================================================================
    # Data inputs, motif generation & selection
    # ========================================================================
    # Data inputs
    atom_array_input: AtomArray | None = Field(None, description="Loaded atom array", exclude=True)
    input: str | None =  Field(None, description="Path to input PDB/CIF file")
    # Motif selection from input file
    contig:  InputSelection | None = Field(None, description="Contig specification string (e.g. 'A1-10,B1-5')")
    unindex: InputSelection | None = Field(None,
        description="Unindexed components string (components must not overlap with contig). "\
        "E.g. 'A15-20,B6-10' or dict. We recommend specifying")
    internal_coord_groups: list[InputSelection] | None = Field(
        None,
        description=(
            "List of internal coordinate groups to create. Must be a subset of the union of `contig` and `unindex`."
        ),
    )
    # Extra args:
    length:  str | None = Field(
        None, description="Length range as 'min-max' or int. Constrains length of contig if provided"
    )
    ligand:  str | None = Field(None, description="Ligand name or index to include in design.")
    cif_parser_args: dict[str, Any] | None = Field(None, description="CIF parser arguments")
    extra: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Extra metadata to include in output (useful for logging additional info in metadata)",
    )
    dialect: int = Field(2, description="RFdiffusion3 input dialect. 1: legacy, 2: release.")

    # ========================================================================
    # Conditioning
    # ========================================================================
    # Sequence and coordinate conditioning
    select_fixed_atoms: InputSelection | None = Field(None,
        description='''Atoms to fix coordinates for. Examples:
        - True (default when inputs provided): All atoms pulled from the input are fixed in 3d space
        - False: All atoms pulled from the input are unfixed in 3d space
        - ContigStr: Components to fix in 3d space, e.g. "A1-10,B1-3" fixes residues 1-10 in chain A
          and residues 1-3 in chain B.
        - {"A1": "N,CA,C,O,CB,CG", "A2-10": "BKBN"} fixes backbone and CB for residues 1 and 2,
          and all atoms for residues 3-10 in chain A.
    '''.replace('\t\t', '\t')
    )
    select_unfixed_sequence: InputSelection | None = Field(None, description='''Components to unfix sequence for.
        - True (default when inputs provided): All atoms from the input have fixed sequences by default.
        - False: All atoms pulled from the input have diffused sequences by default.
        - ContigStr: Components to unfix sequence for, e.g. "A5-10,B1-3" unfixes sequence for residues
          5-10 in chain A and residues 1-3 in chain B.
        - Dictionary: Allowed but not recommended.
        NOTE: Excludes ligands (ligands / DNA always has fixed sequence).
    '''.replace('\t\t', '\t')
    )
    # Assignments of conditioning annotations
    # RASA accessibilty
    select_buried: InputSelection | None = Field(None, description="Selection of RASA buried conditioning")
    select_partially_buried: InputSelection | None = Field(
        None, description="Selection of RASA partially buried conditioning"
    )
    select_exposed: InputSelection | None = Field(None, description="Selection of RASA exposed conditioning")
    # Hotspots & Hbonds
    select_hbond_acceptor: InputSelection | None = Field(None, description="Atom-wise hydrogen bond acceptor")
    select_hbond_donor: InputSelection | None = Field(None, description="Atom-wise hydrogen bond donor")
    select_hotspots: InputSelection | None = Field(None, description="Atom-level or token-level hotspots for PPI")
    redesign_motif_sidechains: bool | str = Field(
        False,
        description=(
            "Perform fixed-backbone sequence design on when 'contig' is provided. "
            "Changes the default behaviour when not using `select_fixed_atoms`."
        ),
    )

    # ========================================================================
    # Global conditioning & symmetry
    # ========================================================================
    # Symmetry
    symmetry: SymmetryConfig | None = Field(None, description="Symmetry specification, see docs/symmetry.md")
    infer_ori_strategy: str | None = Field(None, description="Strategy for inferring origin; `com` or `hotspots`")
    allow_non_motif_centered: bool = Field(
        False,
        description="Allow centering strategies other than centering on the coordinate-conditioned regions.",
    )
    # Additional global conditioning
    plddt_enhanced: bool | None = Field(True, description="Enable pLDDT enhancement")
    is_non_loopy: bool | None = Field(True, description="Non-loopy conditioning")
    # Partial diffusion
    partial_t_backbone: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Backbone time step for partial diffusion. 0 is noise, 1 is data.",
    )
    partial_t_latent: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Sequence/rotamer time step for partial diffusion. 0 is noise, 1 is data.",
    )
    add_cath_code: dict[str, bool | str | dict[str, str | list[str]]] | None = Field(
        None, description="CATH-style fold type conditioning. Dict of the form {'1.2.x': 'A1-100'}"
    )
    # fmt: on

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def is_partial_diffusion(self) -> bool:
        """Whether partial diffusion is enabled."""
        return exists(self.partial_t_backbone) or exists(self.partial_t_latent)

    # ========================================================================
    # Loading / saving
    # ========================================================================

    @classmethod
    def from_json(cls, path: str) -> "InputSpecification":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_rfd3_out(cls, path: str) -> "InputSpecification":
        """Load from path to rfd3 outputs, either .cif, .cif.gz, .json or denoised / noisy trajectory files"""
        path = path.replace(".cif.gz", ".cif").replace(".cif", ".json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Output file not found at {path}")
        with open(path) as f:
            data = json.load(f)
        if "input_specification" in data:
            spec_args = data["input_specification"]
            return cls(**spec_args)
        else:
            raise ValueError(f"No input specification found in json output: {path}")

    def get_dict_to_save(self, exclude_extra: bool = False) -> dict:
        # Returns dictionary for saving (reproducible) outputs to json
        return self.model_dump(
            exclude_defaults=True,
            exclude={"atom_array_input"} | set({"extra"} if exclude_extra else {}),
        )

    # ========================================================================
    # Pre-Validation / canonicalization
    # ========================================================================

    @model_validator(mode="before")
    @classmethod
    def validate_input_schema(cls, data: dict) -> dict:
        if not (exists(data.get("input")) or exists(data.get("contig")) or exists(data.get("length"))):
            raise ValueError("Either 'input' or 'contig' or 'length' must be provided.")

        # unused input check
        is_partial_diffusion = exists(data.get("partial_t_backbone")) or exists(data.get("partial_t_latent"))
        if exists(data.get("input")) and not (
            exists(data.get("contig"))
            or exists(data.get("unindex"))
            or is_partial_diffusion
            or exists(data.get("ligand"))
            or exists(data.get("internal_coord_groups"))
        ):
            raise ValueError("Input provided but unused in composition specification.")

        if not is_partial_diffusion:
            # non-partial diffusion checks
            if exists(data.get("unindex")) and not (exists(data.get("contig")) or exists(data.get("length"))):
                raise ValueError("Unindex provided but neither a length nor contig was specified.")
        else:
            # partial diffusion checks
            if exists(data.get("length")):
                raise ValueError("Length argument must not be provided during partial diffusion.")
            if not (exists(data.get("input")) or exists(data.get("atom_array_input"))):
                raise ValueError("Partial diffusion requires input file or input atom array.")

        return data

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, data: dict) -> dict:
        # Canonicalize length argument
        data["length"] = str(data["length"]) if exists(data.get("length")) else None

        # Normalize input to str
        data["input"] = str(data["input"]) if exists(data.get("input")) else None
        return data

    @model_validator(mode="before")
    @classmethod
    def load_input(cls, data: dict) -> dict:
        """Loads selection data from the input AtomArray.

        Note that this AtomArray will NOT include any newly-added diffused residues which are not present in the input.
        In the case of partial diffusion, the diffused regions will already be present in the input AtomArray.
        """
        with validator_context("load_input"):
            # ... Find provided selections
            selections = [
                # Motif
                "contig",
                "unindex",
                # Aux
                "select_fixed_atoms",
                "select_unfixed_sequence",
                # Conditioning
                "select_buried",
                "select_partially_buried",
                "select_exposed",
                "select_hbond_acceptor",
                "select_hbond_donor",
                "select_hotspots",
            ]
            selections = [s for s in selections if s in data]

            if exists(data.get("contig")) and not isinstance(data["contig"], str):
                raise ValueError(f"Contig selection must be a string or None, got {type(data['contig'])} instead.")

            # ... Early return if no input file provided / atom array input
            if not exists(data.get("input")) and not exists(data.get("atom_array_input")):
                if not selections:
                    return data
                # Allow for entirely-diffused contigs without input structures
                elif selections == ["contig"] and exists(data["contig"]):
                    contig_components = data["contig"].split(",")
                    if any(c[0].isalpha() for c in contig_components):
                        raise ValueError(
                            "Contig selection without input structure must contain only diffused components"
                        )

                    # The length has not yet been selected so we only store the raw contig string
                    data["contig"] = InputSelection(data={}, raw=data["contig"], mask=None, tokens=None)

                    return data
                else:
                    raise ValueError(f"Atom array input must be provided before parsing selections: {selections}")

            # ... Load atom array from input file if provided
            if exists(data["input"]):
                if exists(data.get("atom_array_input")):
                    raise ValueError("Both 'input' and 'atom_array_input' provided; please provide only one.")
                atom_array = inference_load_(data["input"], cif_parser_args=data.get("cif_parser_args"))["atom_array"]

                # Center for symmetric design
                if exists(data.get("symmetry")) and data["symmetry"].get("id"):
                    atom_array = center_symmetric_src_atom_array(atom_array)

                if "atom_id" in atom_array.get_annotation_categories():
                    atom_array.del_annotation("atom_id")

                data["atom_array_input"] = atom_array

            atom_array = data["atom_array_input"]

            # ... Set defaults if not provided
            if not exists(data.get("select_fixed_atoms")):
                raise ValueError(
                    "`select_fixed_atoms` must be specified. "
                    "The simplest way to do this is to set it to `true` or `false`."
                )
            if not exists(data.get("select_unfixed_sequence")):
                data["select_unfixed_sequence"] = InputSelection.from_any(False, atom_array=atom_array)

            # Coerce selections
            for sele in selections:
                if not isinstance(data.get(sele), InputSelection):
                    data[sele] = InputSelection.from_any(data[sele], atom_array=atom_array)

            # Specify internal coordinate groups (handled separately)
            if exists(data.get("internal_coord_groups")):
                internal_coord_groups = data["internal_coord_groups"]
                if not isinstance(internal_coord_groups, list):
                    raise ValueError(
                        f"internal_coord_groups must be a list of selections, got {type(internal_coord_groups)}."
                    )
                data["internal_coord_groups"] = [
                    InputSelection.from_any(coord_group, atom_array=atom_array) for coord_group in internal_coord_groups
                ]

        return data

    # ========================================================================
    # Post-Validation
    # ========================================================================

    @model_validator(mode="after")
    def assert_exclusivity(self) -> "InputSpecification":
        with validator_context("assert_exclusivity"):
            # ... Assert and indexed do not overlap
            if exists(self.contig) and exists(self.unindex):
                indexed_set = set(self.contig.keys())
                unindexed_set = set(self.unindex.keys())
                overlap = indexed_set & unindexed_set
                if overlap:
                    raise ValueError(f"Indexed and unindexed components must not overlap, got: {overlap}")

            # ... Assert mutual exclusivity of rasa binning
            exclusive_sets = [
                ("Motifs", ("contig", "unindex")),
                (
                    "RASA",
                    ("select_buried", "select_partially_buried", "select_exposed"),
                ),
            ]

            names_and_exclusive_mask_lists = []
            for name, excl_set in exclusive_sets:
                masks = [getattr(self, field, None) for field in excl_set]
                masks = [m.get_mask() for m in masks if m is not None]
                if not masks:
                    continue
                names_and_exclusive_mask_lists.append((name, excl_set, masks))

            # Handle coord conditioning, with an unknown number of selections
            coord_mask_items = [getattr(self, "select_fixed_atoms", None)]
            internal_coord_groups = getattr(self, "internal_coord_groups", None)
            if internal_coord_groups is not None:
                coord_mask_items.extend(internal_coord_groups)
            coord_masks = [m.get_mask() for m in coord_mask_items if m is not None]
            if coord_masks:
                names_and_exclusive_mask_lists.append(
                    ("Coordinate Conditioning", ("select_fixed_atoms", "internal_coord_groups"), coord_masks)
                )

            for name, excl_set, masks in names_and_exclusive_mask_lists:
                mask_sum = np.zeros_like(masks[0], dtype=int)
                for m in masks:
                    if m is not None:
                        mask_sum += m.astype(int)
                if np.any(mask_sum > 1):
                    raise ValueError(
                        f"Selections for `{name}` must be mutually exclusive, "
                        f"got overlapping selections: {excl_set}. Mask sum: {mask_sum}"
                    )

        return self

    @model_validator(mode="after")
    def attempt_expansion(self) -> "InputSpecification":
        if self.is_partial_diffusion and exists(self.contig):
            contig = self.contig
            length = self.length
            try:
                get_design_pattern_with_constraints(contig.raw, length=length, atom_array=self.atom_array_input)
            except Exception as e:
                raise ValueError(f"Failed to expand contig ({contig.raw}): {e}") from e
        return self

    @model_validator(mode="after")
    def validate_contig(self) -> "InputSpecification":
        if exists(self.contig):
            components = self.contig.raw.split(",")
            current_chain_type = None
            for c in components:
                if c[0].isalpha():
                    continue
                if c == "/0":
                    current_chain_type = None
                    continue

                chain_type = "P"
                if c[-1].upper() == "D":
                    chain_type = "D"
                elif c[-1].upper() == "R":
                    chain_type = "R"
                elif c[-1].isalpha() and c[-1].upper() != "P":
                    raise ValueError(
                        f"Invalid component in contig: {c}. Final character must be numeric or one of 'D', 'R', 'P'."
                    )

                if current_chain_type is None:
                    current_chain_type = chain_type
                elif current_chain_type != chain_type:
                    raise ComponentValidationError(
                        f"Chain types must be consistent within a single chain, "
                        f"got: {current_chain_type} and {chain_type}"
                    )
        return self

    @model_validator(mode="after")
    def _assign_types_to_input(self) -> "InputSpecification":
        """Assign conditioning annotations to the input atom array before adding new diffused residues."""
        aa = self.atom_array_input
        if not exists(aa):
            return self

        # ... Selections and their annotation values
        selection_fields = {
            # field name:         (annotation name, assigned value, non-selected value)
            "select_fixed_atoms": ("is_motif_atom_with_fixed_coord", True, False),
            "select_unfixed_sequence": ("is_motif_atom_with_fixed_seq", False, True),
            "unindex": ("is_motif_atom_unindexed", True, False),
            "select_hotspots": ("is_atom_level_hotspot", True, False),
            "select_hbond_acceptor": ("active_acceptor", True, False),
            "select_hbond_donor": ("active_donor", True, False),
            "select_buried": ("rasa_bin", 0, 3),
            "select_partially_buried": ("rasa_bin", 1, 3),
            "select_exposed": ("rasa_bin", 2, 3),
        }
        selection_fields = {k: v for k, v in selection_fields.items() if exists(getattr(self, k, None))}

        # ... Init global
        [
            aa.set_annotation(name, np.full(aa.array_length(), val, dtype=int))
            for name, val in REQUIRED_CONDITIONING_ANNOTATION_VALUES.items()
        ]

        # Application of selections to each token fn;
        def apply_selections(start: int, end: int) -> None:
            chain_id = aa.chain_id[start]
            res_id = aa.res_id[start]

            # Assign all select fields to atom array annotations.
            for selection_name, (
                annotation_name,
                set_value,
                default_value,
            ) in selection_fields.items():
                # ... Get input values
                selection = getattr(self, selection_name)

                # Important line: selects from data dictionary based on src chain & res_id (Not name!)
                atom_names_sele = selection.get(f"{chain_id}{res_id}")

                if atom_names_sele is None:
                    continue
                mask = np.isin(aa.atom_name[start:end], atom_names_sele)
                if annotation_name in aa.get_annotation_categories():
                    # ... Set only mask overridden features if exists in atom array
                    aa.get_annotation(annotation_name)[start:end] = np.where(mask, set_value, default_value).astype(
                        np.int_
                    )
                else:
                    # ... Otherwise, set the entire annotation and use defaults for unselected
                    mask_aa = np.zeros(aa.array_length(), dtype=bool)
                    mask_aa[start:end] = mask
                    annotation_values = np.where(
                        mask_aa,
                        set_value,
                        default_value,
                    ).astype(np.int_)
                    aa.set_annotation(annotation_name, annotation_values)

        # ... Set default assignments per-token based on whether redesigning
        starts = get_residue_starts(aa, add_exclusive_stop=True)
        for start, end in itertools.pairwise(starts):
            # ... Relax sequence and sidechains
            if aa.res_name[start] in STANDARD_AA and self.redesign_motif_sidechains:
                is_bkbn = np.isin(aa.atom_name[start:end], ["N", "CA", "C", "O"])
                aa.is_motif_atom_with_fixed_coord[start:end] = is_bkbn.astype(int)
                aa.is_motif_atom_with_fixed_seq[start:end] = np.full_like(is_bkbn, False, dtype=int)

            # ... Apply selections on top
            apply_selections(start, end)

        # HACK: Necessary while still using the annotation adaptor, since C_CRDGRP default relies on C_CRD
        C_CRD.set_mask(aa, aa.is_motif_atom_with_fixed_coord.astype(bool))

        if exists(self.internal_coord_groups):
            for coord_group_selection in self.internal_coord_groups:
                coord_group_mask = coord_group_selection.get_mask()
                aa = convert_to_internal_coords(aa, coord_group_mask)

                # HACK: Necessary while we're still using the annotation adaptor
                # to ensure that the C_CRD mask is correct
                aa.is_motif_atom_with_fixed_coord[coord_group_mask] = False
        else:
            C_INTCRD.set_mask(aa, C_INTCRD.default_mask(aa))
            C_INTCRD.set_annotation(aa, C_INTCRD.default_annotation(aa))
            C_CRDGRP.set_annotation(aa, C_CRDGRP.default_annotation(aa))

        # Clean up temporary C_CRD mask
        aa.del_annotation(C_CRD.mask_name)

        return self

    # ========================================================================
    # Building
    # ========================================================================

    def build(self, return_metadata: bool = False) -> "AtomArray | tuple[AtomArray, dict]":
        """Main build pipeline."""
        atom_array_input_annotated = copy.deepcopy(self.atom_array_input)
        atom_array = self._build_init(atom_array_input_annotated)

        # Apply post-processing
        atom_array = self._append_ligand(atom_array, atom_array_input_annotated)
        atom_array = self._add_diffused_features(atom_array)
        atom_array = self._apply_symmetry(atom_array, atom_array_input_annotated)

        # Apply globals to all tokens (including diffused)
        atom_array = self._set_origin(atom_array)
        atom_array = self._apply_globals(atom_array)

        # Final validation and cleanup
        check_has_required_conditioning_annotations(atom_array, required=REQUIRED_INFERENCE_ANNOTATIONS)
        convert_existing_annotations_to_bool(atom_array)

        # ... Route return type
        if not return_metadata:
            return copy.deepcopy(atom_array)
        else:
            metadata = self.get_dict_to_save()
            metadata["extra"] = metadata.get("extra", {}) | {
                "num_tokens_in": len(get_token_starts(atom_array)),
                "num_residues_in": len(get_residue_starts(atom_array)),
                "num_chains": len(np.unique(atom_array.chain_id)),
                "num_atoms": len(atom_array),
                "num_residues": len(np.unique(list(zip(atom_array.chain_id, atom_array.res_id, strict=False)))),
            }
            return copy.deepcopy(atom_array), metadata

    # ============================================================================
    # Building functions
    # ============================================================================

    def _build_init(self, atom_array_input_annotated: AtomArray | None) -> AtomArray:
        # ... Fetch tokens
        indexed_tokens = {}
        unindexed_tokens = {}
        if exists(atom_array_input_annotated):
            if exists(self.contig):
                indexed_tokens = self.contig.get_tokens(atom_array_input_annotated)
            if exists(self.unindex):
                unindexed_tokens = self.unindex.get_tokens(atom_array_input_annotated)

        # Subset to only fixed coordindate atoms
        unindexed_tokens = {
            k: tok[tok.is_motif_atom_with_fixed_coord.astype(bool) | C_INTCRD.mask(tok)]
            for k, tok in unindexed_tokens.items()
        }
        unindexed_components, unindexed_breaks = self.break_unindexed(self.unindex)

        if not self.is_partial_diffusion:
            # ... Sample the contig string
            components_to_accumulate = get_design_pattern_with_constraints(
                self.contig.raw if exists(self.contig) else self.length,
                length=self.length,
                atom_array=atom_array_input_annotated,
            )
            self.extra["sampled_contig"] = ",".join([str(x) for x in components_to_accumulate])

            # ... Include unindexed components in accumulation
            unindexed_breaks = [None] * len(components_to_accumulate) + unindexed_breaks
            components_to_accumulate += unindexed_components

            # ... Accumulate from scratch
            atom_array = accumulate_components(
                components_to_accumulate,
                indexed_tokens=indexed_tokens,
                unindexed_tokens=unindexed_tokens,
                atom_array_accum=[],
                unindexed_breaks=unindexed_breaks,
                strip_sidechains_by_default=self.redesign_motif_sidechains,
                src_atom_array=atom_array_input_annotated,
            )
        else:
            # ... Set common annotations
            atom_array_in = copy.deepcopy(atom_array_input_annotated)
            ensure_annotations(atom_array_in, "is_standard_aa")
            atom_array_in = atom_array_in[atom_array_in.is_standard_aa]
            atom_array_in = set_common_annotations(atom_array_in, set_src_component_to_res_name=False)

            # ... Override motif annotations from pipeline
            zeros = np.zeros(atom_array_in.array_length(), dtype=int)
            atom_array_in.is_motif_atom_unindexed = zeros  # reset unindexed annotation since those are copied already.
            atom_array_in.is_motif_atom_with_fixed_coord = (
                self.select_fixed_atoms.get_mask().astype(int) if exists(self.select_fixed_atoms) else zeros
            )
            atom_array_in.is_motif_atom_with_fixed_seq = (
                ~self.select_unfixed_sequence.get_mask() if exists(self.select_unfixed_sequence) else zeros
            ).astype(int)

            # ... Subset to residues only
            ensure_annotations(atom_array_in, "is_protein")
            atom_array_in = atom_array_in[atom_array_in.is_protein]

            # ... Set chain ID for unindexed residues as whatever the input has
            start_resid = np.max(atom_array_in.res_id) + 1
            start_chain = atom_array_in.chain_id[0]

            # ... Accumulate from input
            components_to_accumulate = unindexed_components
            atom_array = accumulate_components(
                # No accumulation of components
                components_to_accumulate=components_to_accumulate,
                indexed_tokens={},
                # Append all inputs to unindexed tokens
                unindexed_tokens=unindexed_tokens,
                atom_array_accum=[atom_array_in],
                start_chain=start_chain,
                start_resid=start_resid,
                unindexed_breaks=unindexed_breaks,
            )

        return atom_array

    # ============================================================================
    # Auxiliary functions
    # ============================================================================

    @staticmethod
    def break_unindexed(unindex: InputSelection | None) -> tuple[list, list]:
        if not exists(unindex):
            return [], []

        # ... If original type was string, use that
        if isinstance(unindex.raw, str):
            unindexed_string = unindex.raw
        elif isinstance(unindex.raw, dict):
            unindexed_string = ",".join(unindex.raw.keys())
        else:
            logger.info(
                "`Unindex` provided as non-string, separate keys in dictionary"
                " will be considered separate contiguous components"
            )
            unindexed_string = ",".join(unindex.keys())

        # ... Break expected unindexed contig string
        unindexed_components, breaks = get_motif_components_and_breaks(unindexed_string)

        return unindexed_components, breaks

    # ============================================================================
    # Setter functions
    # ============================================================================

    def _append_ligand(self, atom_array: AtomArray, atom_array_input_annotated: AtomArray | None) -> AtomArray:
        """Append ligand if specified."""
        if exists(self.ligand):
            ligand_array = extract_ligand_array(
                atom_array_input_annotated,
                self.ligand,
                fixed_atoms={},
                set_defaults=False,
                additional_annotations=set(
                    list(atom_array.get_annotation_categories())
                    + list(atom_array_input_annotated.get_annotation_categories())
                ),
            )

            # It is forbidden to have polymers and non-polymers with the same chain ID
            non_ligand_chain_ids = np.unique(atom_array.chain_id)
            duplicate_chain_ids = np.unique(ligand_array.chain_id[np.isin(ligand_array.chain_id, non_ligand_chain_ids)])
            max_existing_chain_id = (
                max([ord(c) for c in non_ligand_chain_ids]) if len(non_ligand_chain_ids) > 0 else ord("A") - 1
            )
            for i, duplicate_chain_id in enumerate(duplicate_chain_ids):
                new_chain_id = chr(max_existing_chain_id + i + 1)
                ligand_array.chain_id[ligand_array.chain_id == duplicate_chain_id] = new_chain_id
                logger.warning(
                    f"Chain ID {duplicate_chain_id} contains both polymer and non-polymer residues. "
                    f"Assigning non-polymers to new chain {new_chain_id}"
                )

            atom_array = atom_array + ligand_array
        return atom_array

    def _apply_symmetry(self, atom_array: AtomArray, atom_array_input_annotated: AtomArray | None) -> AtomArray:
        """Apply symmetry transformation if specified."""
        if exists(self.symmetry) and self.symmetry.id:
            atom_array = make_symmetric_atom_array(
                atom_array,
                self.symmetry,
                sm=self.ligand,
                src_atom_array=atom_array_input_annotated,
            )
        return atom_array

    def _set_origin(self, atom_array: AtomArray) -> AtomArray:
        """Set origin token and initialize coordinates."""
        if self.is_partial_diffusion:
            # Partial diffusion: use COM, keep all coordinates
            if exists(self.symmetry) and self.symmetry.id:
                # For symmetric structures, avoid COM centering that would collapse chains
                logger.info("Partial diffusion with symmetry: skipping COM centering to preserve chain spacing")
            else:
                atom_array = set_com(atom_array)
        else:
            # Standard: set ori token, zero out diffused atoms
            atom_array = set_com(
                atom_array,
                infer_ori_strategy=self.infer_ori_strategy,
                allow_non_motif_centered=self.allow_non_motif_centered,
            )
            # Diffused atoms are always initialized at origin during regular diffusion (all information removed)
            atom_array.coord[~atom_array.is_motif_atom_with_fixed_coord.astype(bool)] = 0.0
        return atom_array

    def _add_diffused_features(self, atom_array: AtomArray) -> AtomArray:
        """Add non-global features that can apply to both diffused or motif regions.

        The term `add` is used instead of `select` to clarify which features apply to the fully-assembled AtomArray.

        TODO: This could be made fully general by allowing the user to specify which diffused region (1st, 2nd, etc)
        their contig ranges apply to. Right now, the stochastic length of the diffused region makes precise control
        difficult without simply resorting to chain-level specifications.
        """

        # Process CATH code input
        cath_code_annot = np.full(atom_array.array_length(), C_CATH.NULL_VALUE, dtype=np.dtype("U23"))
        if exists(self.add_cath_code):
            for cath_code, selection in self.add_cath_code.items():
                # ... Get mask for contig
                contig_selection = InputSelection.from_any(selection, atom_array=atom_array)
                contig_mask = contig_selection.get_mask()

                # ... Process CATH code
                cath_code_split = cath_code.split(".")
                cath_code_split += ["x"] * (4 - len(cath_code_split))
                full_cath_code = ".".join(cath_code_split[:4])

                # ... Update annotation
                cath_code_annot[contig_mask] = full_cath_code

        atom_array.set_annotation(C_CATH.full_name, cath_code_annot)

        return atom_array

    def _apply_globals(self, atom_array: AtomArray) -> AtomArray:
        # Temperature conditioning
        if exists(self.is_non_loopy):
            is_non_loopy_annot = np.zeros(atom_array.array_length(), dtype=int)
            is_motif_token = get_motif_features(atom_array)["is_motif_token"]
            diffused_region_mask = ~(is_motif_token.astype(bool))
            if exists(self.is_non_loopy):
                is_non_loopy_annot[diffused_region_mask] = 1 if self.is_non_loopy else -1
            atom_array.set_annotation("is_non_loopy", is_non_loopy_annot)
            atom_array.set_annotation("is_non_loopy_atom_level", is_non_loopy_annot)
        else:
            zeros = np.zeros(atom_array.array_length(), dtype=int)
            atom_array.set_annotation("is_non_loopy", zeros)
            atom_array.set_annotation("is_non_loopy_atom_level", zeros)

        if self.plddt_enhanced:
            atom_array.set_annotation("ref_plddt", np.full((atom_array.array_length(),), True, dtype=int))

        # Partial diffusion time annotation
        if exists(self.partial_t_backbone):
            atom_array.set_annotation(
                "partial_t_backbone", np.full(atom_array.shape[0], self.partial_t_backbone, dtype=float)
            )
        if exists(self.partial_t_latent):
            atom_array.set_annotation(
                "partial_t_latent", np.full(atom_array.shape[0], self.partial_t_latent, dtype=float)
            )
        return atom_array

    @classmethod
    def safe_init(cls, **spec_kwargs) -> "InputSpecification":
        if spec_kwargs.get("dialect", 2) < 2:
            raise ValueError(
                "Dialect 1 has been deprecated and is no longer supported. Please switch to using dialect 2. "
                "See detailed comparison of the two syntaxes in the RFD3 documentation."
            )
        else:
            return cls(**spec_kwargs)


# ============================================================================
# Public API
# ============================================================================


def create_atom_array_from_design_specification(
    **spec_kwargs,
) -> tuple[AtomArray, dict]:
    if int(spec_kwargs.get("dialect", 2)) < 2:
        raise ValueError(
            "Dialect 1 has been deprecated and is no longer supported. Please switch to using dialect 2. "
            "See detailed comparison of the two syntaxes in the RFD3 documentation."
        )

    # Create input specfication and build
    spec = InputSpecification(**spec_kwargs)
    atom_array, metadata = spec.build(return_metadata=True)
    return atom_array, metadata


# HACK: Converting RFD3 AtomArray annotations to the Condition format
# The input parsing code itself should be refactored such that this isn't necessary.
def adapt_legacy_atom_array_for_condition_pipeline(atom_array: AtomArray) -> AtomArray:
    """
    Adapts an AtomArray created by create_atom_array_from_design_specification
    to be compatible with the new Condition-based validation pipeline (condition_2d_pipe).

    This function converts old-style annotation names to the new Condition API format.
    The new pipeline will handle creating motif annotations (is_motif_atom, is_motif_token, etc.)
    from the Condition masks via HackInRequiredAnnotations and AddIsX transforms.

    Args:
        atom_array: AtomArray with legacy annotations from create_atom_array_from_design_specification

    Returns:
        atom_array: AtomArray with Condition-compatible annotations
    """
    n_atoms = atom_array.array_length()

    # Verify that the input has the expected legacy annotations
    check_has_required_conditioning_annotations(atom_array, required=REQUIRED_INFERENCE_ANNOTATIONS)

    # 1. Convert motif-related annotations to Condition API
    # The legacy system uses is_motif_atom_with_fixed_coord, is_motif_atom_with_fixed_seq, etc.
    # The new system derives these from C_CRD, C_SEQ, C_IDX masks

    # C_CRD: Coordinate conditioning (atoms with fixed coordinates)
    # This corresponds to is_motif_atom_with_fixed_coord in the legacy system
    has_fixed_coord = atom_array.get_annotation("is_motif_atom_with_fixed_coord").astype(bool)
    C_CRD.set_mask(atom_array, has_fixed_coord)

    # C_SEQ: Sequence conditioning (atoms with fixed sequence)
    # This corresponds to is_motif_atom_with_fixed_seq in the legacy system
    has_fixed_seq = atom_array.get_annotation("is_motif_atom_with_fixed_seq").astype(bool)
    C_SEQ.set_mask(atom_array, has_fixed_seq)

    # C_IDX: Index conditioning (indexed atoms, i.e., not unindexed)
    # This is the opposite of is_motif_atom_unindexed
    is_indexed = ~atom_array.get_annotation("is_motif_atom_unindexed").astype(bool)
    C_IDX.set_mask(atom_array, is_indexed)

    # C_IDXGRP: Index group annotation (denotes equivalence classes of residues with known index offsets)
    # Note that the input parsing code will have already added novel res_ids for partially-unindexed
    # (e.g. tip-atom) residues, so it is safe to use get_residue_starts on the full AtomArray --
    # it will catch unindexed residues or residue fragments.
    residue_starts_array = atom_array[get_residue_starts(atom_array)]
    index_group_breaks = residue_starts_array.is_motif_atom_unindexed_motif_breakpoint

    # Group id is cumulative sum of prior breakpoints.
    index_group_id_residue_level = np.cumsum(index_group_breaks, dtype=int)

    # HACK: We have to manually reset indexed group ids due to the cumsum. Will be cleaned up post-refactor
    is_indexed_residue = C_IDX.mask(residue_starts_array, default="raise")
    default_index_groups = C_IDXGRP.default_annotation(atom_array)

    # Note that this may create gaps in the group ids, which is fine since the absolute values should never matter
    if is_indexed_residue.any():
        max_absolute_indexed_group_id = default_index_groups[is_indexed].max()
        index_group_id_residue_level[~is_indexed_residue] += max_absolute_indexed_group_id + 1

    # Set the new Condition annotation
    index_group_id_atom_level = spread_residue_wise(atom_array, index_group_id_residue_level)
    index_group_id_atom_level[is_indexed] = default_index_groups[is_indexed]
    C_IDXGRP.set_annotation(atom_array, index_group_id_atom_level)

    # 2. Convert RASA annotation if present
    # Old: rasa_bin with values 0/1/2 for buried/intermediate/exposed
    # New: C_RAS with RasaEnum values
    if "rasa_bin" in atom_array.get_annotation_categories():
        input_rasa_values = atom_array.get_annotation("rasa_bin").astype(int)
        assert set(input_rasa_values).issubset({0, 1, 2, 3}), "Unexpected input rasa value"
        enum_rasa_values = C_RAS.default_annotation(atom_array)

        # NOTE: The input rasa values do not necessarily match the internal enum values!
        enum_rasa_values[input_rasa_values == 0] = RasaEnum.BURIED
        enum_rasa_values[input_rasa_values == 1] = RasaEnum.INTERMEDIATE
        enum_rasa_values[input_rasa_values == 2] = RasaEnum.EXPOSED

        C_RAS.set_annotation(atom_array, enum_rasa_values)

    # 3. Convert hotspot annotation if present
    # Old: is_atom_level_hotspot (boolean)
    # New: C_HOT (boolean)
    if "is_atom_level_hotspot" in atom_array.get_annotation_categories():
        hotspot_values = atom_array.get_annotation("is_atom_level_hotspot").astype(bool)
        C_HOT.set_annotation(atom_array, hotspot_values)

    # 4. Convert H-bond donor/acceptor annotations if present
    # Old: active_donor, active_acceptor (boolean)
    # New: C_HB (int)
    hbond_annot = C_HB.default_annotation(atom_array)
    if "active_donor" in atom_array.get_annotation_categories():
        donor_values = atom_array.get_annotation("active_donor").astype(bool)
        hbond_annot[donor_values] = HBondEnum.DONOR

    if "active_acceptor" in atom_array.get_annotation_categories():
        acceptor_values = atom_array.get_annotation("active_acceptor").astype(bool)
        hbond_annot[acceptor_values] = HBondEnum.ACCEPTOR

    C_HB.set_annotation(atom_array, hbond_annot)

    # 5. Convert secondary structure annotations if present
    # Old: is_helix, is_sheet, is_loop (separate boolean annotations)
    # New: C_SEC with SSEnum values (NONE=0, HELIX=1, SHEET=2, LOOP=3)
    has_ss_annotations = (
        "is_helix" in atom_array.get_annotation_categories()
        or "is_sheet" in atom_array.get_annotation_categories()
        or "is_loop" in atom_array.get_annotation_categories()
    )

    if has_ss_annotations:
        # Initialize with NONE
        ss_values = np.full(n_atoms, SSEnum.NONE, dtype=int)

        # Set values based on the individual boolean annotations
        # Note: These should be mutually exclusive, but we prioritize in order: helix, sheet, loop
        # TODO: This is hacky, should be fixed!
        if "is_helix" in atom_array.get_annotation_categories():
            is_helix = atom_array.get_annotation("is_helix").astype(bool)
            ss_values[is_helix] = SSEnum.HELIX

        if "is_sheet" in atom_array.get_annotation_categories():
            is_sheet = atom_array.get_annotation("is_sheet").astype(bool)
            ss_values[is_sheet] = SSEnum.SHEET

        if "is_loop" in atom_array.get_annotation_categories():
            is_loop = atom_array.get_annotation("is_loop").astype(bool)
            ss_values[is_loop] = SSEnum.LOOP

        C_SEC.set_annotation(atom_array, ss_values)

    # Convert to AtomArrayPlus
    atom_array = as_atom_array_plus(atom_array)

    return atom_array
