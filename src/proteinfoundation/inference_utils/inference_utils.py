"""
Utilities for inference input preparation
"""

import ast
import copy
import logging
import os
from os import PathLike

import biotite.structure as struc
import numpy as np
from atomworks import parse
from atomworks.constants import STANDARD_DNA
from atomworks.io.parser import (
    STANDARD_PARSER_ARGS,
)
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.preprocessing.utils.structure_utils import (
    get_atom_mask_from_cell_list,
)
from atomworks.ml.conditions import CONDITIONS
from atomworks.ml.utils.token import (
    apply_and_spread_token_wise,
)
from biotite.structure import AtomArray
from foundry.common import exists
from foundry.utils.ddp import RankedLogger
from foundry.utils.standard_annotations import SA_SRC

from proteinfoundation.conditions import C_HOT, C_INTCRD
from proteinfoundation.inference_utils.components import (
    fetch_mask_from_component,
    get_name_mask,
    unravel_components,
)

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)

sequence_encoding = AF3SequenceEncoding()
_aa_like_res_names = sequence_encoding.all_res_names[sequence_encoding.is_aa_like]



#########################################
######### CONSTANTS #####################
#########################################
BACKBONE_ATOM_NAMES_DNA = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
]

BACKBONE_ATOM_NAMES_RNA = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
]
"""RNA backbone atom names"""

BACKBONE_ATOM_ELEMENTS_RNA = np.array(
    ["P", "O", "O", "O", "C", "C", "O", "C", "O", "C", "O", "C"]
)
"""RNA backbone atom elements"""

NA_BACKBONE_ATOM_NAME_TO_ELEMENT = dict(zip(BACKBONE_ATOM_NAMES_RNA, BACKBONE_ATOM_ELEMENTS_RNA))
"""Atom name to element mapping for NA backbone atoms"""

#########################################
######### HACKS TO BE REMOVED ###########
#########################################
# TODO: We should remove our reliance on the old annotations entirely, and then
# also remove the need for the Condition adapter function. For now, kept this
# way to quickly get signal.

REQUIRED_CONDITIONING_ANNOTATION_VALUES = {
    "is_motif_atom_with_fixed_seq": False,
    "is_motif_atom_with_fixed_coord": False,
    "is_motif_atom_unindexed": False,
    "is_motif_atom_unindexed_motif_breakpoint": False,
}
REQUIRED_CONDITIONING_ANNOTATIONS = list(REQUIRED_CONDITIONING_ANNOTATION_VALUES.keys())
REQUIRED_INFERENCE_ANNOTATIONS = REQUIRED_CONDITIONING_ANNOTATIONS + [SA_SRC.full_name]
"""Annotations assigned to every valid atom array"""

OPTIONAL_CONDITIONING_VALUES = {
    "is_atom_level_hotspot": 0,
    "is_helix_conditioning": 0,
    "is_sheet_conditioning": 0,
    "is_loop_conditioning": 0,
    "active_donor": 0,
    "active_acceptor": 0,
    "rasa_bin": 3,  # NOTE: We are completely insulated against this choice in the adapter
    "ref_plddt": 0,
    "is_non_loopy": 0,
    "partial_t_backbone": np.nan,
    "partial_t_latent": np.nan,
    "has_2d_condition": 0,
}
"""Optional conditioning annotations and their default values if not provided."""

CONDITIONING_VALUES = REQUIRED_CONDITIONING_ANNOTATION_VALUES | OPTIONAL_CONDITIONING_VALUES
"""Annotations that must be present in the AtomArray at inference time."""

INFERENCE_ANNOTATIONS = REQUIRED_INFERENCE_ANNOTATIONS + list(OPTIONAL_CONDITIONING_VALUES.keys())
"""All annotations that might be desired at inference time. Determines what AtomArray annotations will be preserved."""


#################################################################################
# Annotation management
#################################################################################


def check_has_required_conditioning_annotations(
    atom_array: AtomArray, required: list[str] = REQUIRED_CONDITIONING_ANNOTATIONS
) -> bool:
    """
    Checks if the atom array has the correct conditioning annotations
    """
    received = atom_array.get_annotation_categories()
    for required_annotation in required:
        if required_annotation not in received:
            raise ValueError(f"Missing annotation category in atom_array: {required_annotation}")
    return True


def convert_existing_annotations_to_bool(
    atom_array: AtomArray, annotations: list[str] = REQUIRED_CONDITIONING_ANNOTATIONS
) -> AtomArray:
    # When loading from cif, annotations are loaded as strings when they should be boolean
    for annotation in annotations:
        if annotation not in atom_array.get_annotation_categories():
            continue
        tmp = atom_array.get_annotation(annotation).copy()
        atom_array.get_annotation(annotation).dtype = bool
        if isinstance(tmp[0], (str, np.str_, np.dtypes.StrDType)):
            tmp = np.array([ast.literal_eval(x) for x in tmp], dtype=bool)
        else:
            tmp = np.asarray(tmp, dtype=bool)
        atom_array.set_annotation(annotation, tmp)
    return atom_array


def set_default_conditioning_annotations(
    atom_array: AtomArray,
    motif: bool = False,
    unindexed: bool = False,
    mask: np.ndarray | None = None,
    dtype: type = bool,
    additional: set | list | None = None,
) -> AtomArray:
    """
    Adds default annotations to the atom array

    Args:
        motif: True if default for a fully fixed motif, False if default for a fully diffused motif
        unindexed: True if the tokens in the atom array should be motif
        mask: boolean mask for array of which atoms to apply the assignments to.
    NB: In both cases, the defaults for unindexed are False
    """

    # All annotations set to true for motif
    fill = True if motif else False
    if mask is not None:
        # TODO: support defaulting to nulls
        check_has_required_conditioning_annotations(atom_array)
        trues = np.full(mask.sum(), True, dtype=dtype)
        falses = np.full(mask.sum(), False, dtype=dtype)

        atom_array.is_motif_atom_unindexed[mask] = trues if unindexed else falses
        atom_array.is_motif_atom_unindexed_motif_breakpoint[mask] = falses

        # Others:
        for annotation in REQUIRED_CONDITIONING_ANNOTATIONS:
            if annotation in [
                "is_motif_atom_unindexed",
                "is_motif_atom_unindexed_motif_breakpoint",
            ]:
                continue

            vals = copy.deepcopy(atom_array.get_annotation(annotation))
            vals[mask] = trues if fill else falses
            atom_array.set_annotation(annotation, vals)
    else:
        for annotation in REQUIRED_CONDITIONING_ANNOTATIONS:
            if annotation in [
                "is_motif_atom_unindexed",
            ]:
                atom_array.set_annotation(
                    annotation,
                    np.full(atom_array.array_length(), unindexed, dtype=dtype),
                )
            elif annotation in [
                "is_motif_atom_unindexed_motif_breakpoint",
            ]:
                atom_array.set_annotation(annotation, np.full(atom_array.array_length(), False, dtype=dtype))
            else:
                atom_array.set_annotation(annotation, np.full(atom_array.array_length(), fill, dtype=dtype))

    if additional is not None:
        for annot, val in OPTIONAL_CONDITIONING_VALUES.items():
            if annot in additional and annot not in atom_array.get_annotation_categories():
                atom_array.set_annotation(annot, np.full(atom_array.array_length(), val))
        for condition in CONDITIONS:
            if condition.mask_name in additional and condition.mask_name not in atom_array.get_annotation_categories():
                condition.set_mask(atom_array, condition.default_mask(atom_array))
            if condition.full_name in additional and condition.full_name not in atom_array.get_annotation_categories():
                condition.set_annotation(atom_array, condition.default_annotation(atom_array))

    return atom_array


#################################################################################
# Setter functions for annotations
#################################################################################


def set_common_annotations(array: AtomArray, set_src_component_to_res_name: bool = True) -> AtomArray:
    annots = array.get_annotation_categories()
    if "occupancy" not in annots:
        array.set_annotation("occupancy", np.ones(array.shape[0], dtype=float))
    if "b_factor" not in annots:
        array.set_annotation("b_factor", np.zeros(array.shape[0], dtype=float))
    if "charge" not in annots:
        array.set_annotation("charge", np.zeros(array.shape[0], dtype=int))
    if not SA_SRC.has_annotation(array):
        if set_src_component_to_res_name:
            SA_SRC.set_annotation(array, np.full(array.shape[0], array.res_name.copy(), dtype="<U10"))
        else:
            SA_SRC.set_annotation(array, SA_SRC.default_annotation(array))
    return array


def set_indices(array: AtomArray, chain: str, res_id_start: int, molecule_id: int, component: str | int) -> AtomArray:
    n = array.shape[0]
    array.chain_id = np.full(n, chain, dtype=array.chain_id.dtype)
    array.res_id = np.full(n, res_id_start + array.res_id - 1, dtype=array.res_id.dtype)
    array.molecule_id = np.full(n, molecule_id, dtype=np.int32)
    SA_SRC.set_annotation(array, np.full(n, component, dtype="<U10"))
    return array


#################################################################################
# Getters
#################################################################################


def extract_ligand_array(
    atom_array_input: AtomArray,
    ligand: str,
    fixed_atoms: dict = {},
    set_defaults: bool = True,
    additional_annotations: set | list | None = None,
) -> AtomArray:
    if not exists(atom_array_input):
        raise ValueError("No input file/atom array provided. Cannot add requested ligand.")

    ligand_arrays = []
    for lig in ligand.split(","):
        for name in unravel_components(
            lig, atom_array=atom_array_input, allow_multiple_matches=True
        ):  # additional nesting to allow multiple indices per ligand
            mask = fetch_mask_from_component(name, atom_array=atom_array_input)
            ligand_array = atom_array_input[mask].copy()

            # ... Set as fully fixed motif
            if set_defaults:
                ligand_array = set_default_conditioning_annotations(
                    ligand_array, motif=True, additional=additional_annotations
                )  # should be pre-set!
            ligand_array = set_common_annotations(ligand_array)

            # ... Unfix all names not specified if specified in motif_atoms
            if lig in fixed_atoms or name in fixed_atoms:
                if (lig in fixed_atoms and name in fixed_atoms) and name != lig:
                    raise ValueError(
                        f"Got both ligand name and its pdb indices in fixed_atoms dictionary: {lig} and {name}. Please only provide one."
                    )
                fixed = fixed_atoms.get(lig, fixed_atoms.get(name))
                if fixed:
                    fixed_mask = get_name_mask(ligand_array.atom_name, fixed)
                    ligand_array.is_motif_atom_with_fixed_coord[~fixed_mask] = np.zeros(np.sum(~fixed_mask), dtype=int)
                else:
                    ligand_array.is_motif_atom_with_fixed_coord = np.zeros(ligand_array.shape[0], dtype=int)
            ligand_arrays.append(ligand_array)

    ligand_arrays = struc.concatenate(ligand_arrays)
    return ligand_arrays


def extract_na_array(atom_array_input: AtomArray) -> AtomArray:
    # TODO : do it more nicely, take into account modifications to NA reses e.g. 5IU
    if (na_mask := np.isin(atom_array_input.res_name, list(STANDARD_DNA))).any():
        na_array = atom_array_input[na_mask]
        # ...replace chain_id A with literally anything else available
        Achain_mask = na_array.chain_id == "A"

        all_nonAchains = np.unique((atom_array_input + na_array).chain_id).tolist()
        all_nonAchains.remove("A")

        if len(all_nonAchains) > 1:
            new_chain = "".join(all_nonAchains)  # join_them_all !! so definitely unique
        elif len(all_nonAchains) == 1:
            new_chain = all_nonAchains[0] + all_nonAchains[0]
        else:
            new_chain = "B"

        na_array.chain_id[Achain_mask] = new_chain
        na_array = set_default_conditioning_annotations(na_array, motif=True)
        return na_array
    else:
        raise ValueError("Could not find any NA tokens in input file, but requested to add all NA")


#################################################################################
# File IO utilities
#################################################################################


def inference_load_(file: PathLike, *, assembly_id: str = "1", cif_parser_args: dict | None = None) -> dict:
    # Default cif_parser_args to an empty dictionary if not provided
    if cif_parser_args is None:
        cif_parser_args = {}

    # Convenience utilities to default to loading from and saving to cache if a cache_dir is provided, unless explicitly overridden
    if cif_parser_args.get("cache_dir"):
        cif_parser_args.setdefault("load_from_cache", True)
        cif_parser_args.setdefault("save_to_cache", True)

    merged_cif_parser_args = {
        **STANDARD_PARSER_ARGS,
        **{
            "fix_arginines": False,
            "add_missing_atoms": False,
            "fix_formal_charges": False,
            "remove_ccds": [],
        },
        **cif_parser_args,
    }
    merged_cif_parser_args["hydrogen_policy"] = "remove"

    # Ensure the required annotations can be loaded
    merged_cif_parser_args["extra_fields"] = list(
        set(merged_cif_parser_args.get("extra_fields", []) + REQUIRED_CONDITIONING_ANNOTATIONS)
    )

    # Use the parse function with the merged CIF parser arguments
    result_dict = parse(
        filename=file,
        build_assembly=(assembly_id,),  # Convert list to tuple (make hashable)
        **merged_cif_parser_args,
    )

    atom_array = result_dict["assemblies"][assembly_id][0]
    atom_array = convert_existing_annotations_to_bool(atom_array)

    data = {
        "atom_array": atom_array,  # First model
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }

    return data


def ensure_input_is_abspath(args: dict, path: PathLike | None) -> dict:
    """
    Ensures the input source is an absolute path if exists, if not it will convert

    args:
        spec: Inference specification for atom array
        path: None or file to which the input is relative to.
    """
    if isinstance(args, str):
        raise ValueError(
            f"Expected args to be a dictionary, got a string: {args}. If you are using an input JSON ensure it contains dictionaries of arguments"
        )
    if "input" not in args or not exists(args["input"]):
        return args
    input = args["input"]
    if not os.path.isabs(input):
        input = os.path.abspath(os.path.join(os.path.dirname(path), input))
        ranked_logger.info(f"Input source path is relative, converted to absolute path: {input}")
        args["input"] = input
    return args


#################################################################################
# Custom infer_ori functions
#################################################################################


def infer_ori_from_hotspots(atom_array: AtomArray) -> np.ndarray:
    if "is_atom_level_hotspot" in atom_array.get_annotation_categories():
        hotspot_atom_array = atom_array[atom_array.is_atom_level_hotspot.astype(bool)]
    elif C_HOT.full_name in atom_array.get_annotation_categories():
        hotspot_atom_array = atom_array[C_HOT.mask(atom_array, default="raise")]
    else:
        raise ValueError(
            f"Atom array must contain 'is_atom_level_hotspot' or '{C_HOT.full_name}' annotation to infer ori from hotspots."
        )

    hotspot_com = hotspot_atom_array.coord.mean(axis=0)

    # We can only perform distance computations on atoms with non-NaN coordinates
    nan_coords_mask = np.any(np.isnan(atom_array.coord), axis=1)
    non_nan_atom_array = atom_array[~nan_coords_mask]

    # Perform the distance computation
    # RFD2 used 10 Angstroms instead of 12, but was for residue-level hotspots
    DISTANCE_CUTOFF = 12.0
    cell_list = struc.CellList(non_nan_atom_array, cell_size=DISTANCE_CUTOFF)
    nearby_atoms_mask = get_atom_mask_from_cell_list(
        hotspot_atom_array.coord,
        cell_list,
        len(non_nan_atom_array),
        cutoff=DISTANCE_CUTOFF,
    )  # (n_query, n_cell_list)

    nearby_atoms_mask = np.any(nearby_atoms_mask, axis=0)  # (n_cell_list,)
    nearby_atoms_com = non_nan_atom_array.coord[nearby_atoms_mask].mean(axis=0)

    vector_from_core_to_hotspot = hotspot_com - nearby_atoms_com
    vector_from_core_to_hotspot = vector_from_core_to_hotspot / np.linalg.norm(vector_from_core_to_hotspot)

    # This is following RFD2. Both this and the distance cutoff should definitely be configs with defaults
    DISTANCE_ABOVE_HOTSPOTS = 10.0
    ori_token = hotspot_com + DISTANCE_ABOVE_HOTSPOTS * vector_from_core_to_hotspot

    return ori_token


def infer_ori_from_com(atom_array: AtomArray) -> np.ndarray:
    xyz = atom_array.coord
    mask = np.isfinite(xyz).all(axis=-1)  # Ensure no NaN coordinates
    com = np.mean(xyz[..., mask, :], axis=0)
    return com


# This can't go in constants.py because that leads to a circular dependency
INFER_ORI_STRATEGIES_TO_FUNCTIONS = {
    "com": infer_ori_from_com,
    "hotspots": infer_ori_from_hotspots,
    "preserve": lambda atom_array: np.array([0.0, 0.0, 0.0], dtype=atom_array.coord.dtype),
}
"""
Constant mapping from infer_ori_strategy keys to the corresponding functions. These functions should take an AtomArray
as input and return a three-element list or numpy array of floats.
"""


def set_com(
    atom_array: AtomArray,
    ori_token: list[float] | None = None,
    infer_ori_strategy: str | None = None,
    allow_non_motif_centered=False,
) -> AtomArray:
    """General function for handling centering of input structures prior to running inference

    In the proteinfoundation model, we always center on the coordinate-conditioned (motif) regions.
    However, in RFD3 the origin is set to the centroid of the diffused region during training, so at inference time
    the user can choose from various strategies.

    Args:
        atom_array (AtomArray): Input atom array to have its centroid adjusted.
        ori_token (list[float] | None): Optional explicit origin to center the structure on.
        infer_ori_strategy (str | None): Optional strategy to infer the origin for centering. Options include 'com',
            which centers on the centroid of all atoms, and 'hotspots', which places the origin radially out from the hotspots.
        allow_non_motif_centered (bool): Whether to allow centering on non-motif regions. Should only be enabled
            for evaluating external models, never for running the proteinfoundation model.
    """
    if not allow_non_motif_centered and (exists(ori_token) or exists(infer_ori_strategy)):
        raise ValueError("Only motif centering is allowed, but another form of COM information was provided.")
    if exists(ori_token):
        if exists(infer_ori_strategy):
            raise ValueError("If providing ori_token, cannot provide infer_ori_strategy.")
        center = np.array([float(x) for x in ori_token], dtype=atom_array.coord.dtype)
        atom_array.coord = atom_array.coord - center
        ranked_logger.info(f"Received ori_token argument. Setting origin as {center}.")
    elif exists(infer_ori_strategy):
        if infer_ori_strategy in INFER_ORI_STRATEGIES_TO_FUNCTIONS:
            center = INFER_ORI_STRATEGIES_TO_FUNCTIONS[infer_ori_strategy](atom_array)
            atom_array.coord = atom_array.coord - center
            ranked_logger.info(f"Inferred origin using strategy '{infer_ori_strategy}'. Setting origin as {center}.")
        else:
            raise ValueError(f"Unrecognized infer_ori_strategy: {infer_ori_strategy}")
    else:
        # No offset
        if np.any(atom_array.is_motif_atom_with_fixed_coord.astype(bool)):
            center = np.nan_to_num(
                np.mean(
                    atom_array.coord[atom_array.is_motif_atom_with_fixed_coord.astype(bool)],
                    axis=0,
                )
            )
            atom_array.coord -= center
            ranked_logger.info(f"No infer_ori_strategy or ori_token provided. Centering on motif COM at {center}.")
        else:
            atom_array.coord = np.zeros_like(atom_array.coord, dtype=atom_array.coord.dtype)
    return atom_array


#################################################################################
# Patching of bad inputs
#################################################################################


def generate_idealized_cb_position(N: np.array, Ca: np.array, C: np.array) -> np.array:
    """
    Generate Cb coordiantes given (N, CA, C) as if the given coordinates were from an idealized Alanine.

    Args:
        - N (np.array): coordinates of (pseudo) N atoms [..., L, 3]
        - Ca (np.array): coordinates of (pseudo) Ca atoms [..., L, 3]
        - C (np.array): coordinates of (pseudo) C atoms [..., L, 3]

    Returns:
        Cb (torch.Tensor): coordinates of (pseudo) Cb atoms [..., L, 3]
            These will be placed at the idealized Cb distance (based on ALA) from Ca, assuming a frame of the following form:
                - x-axis: along the Ca-C bond
                - z-axis: perpendicular to the Ca-N-C plane, right-handed wrt to (Ca-C) & (Ca-N) vectors.
                - y-axis: in the plane of the Ca-N-C bonds, such that the overall frame is right-handed.
    Reference:
        - https://github.com/google-deepmind/alphafold/blob/d95a92aae161240b645fc10e9d030443011d913e/alphafold/common/residue_constants.py#L126-L335
            ALA:
                ['N',  0, (-0.525,  1.363,  0.000)],  # ca-n bond dist: 1.4606142543
                ['CA', 0, ( 0.000,  0.000,  0.000)],
                ['C',  0, ( 1.526,  0.000,  0.000)],  # ca-c bond dist: 1.526
                ['CB', 0, (-0.529, -0.774, -1.205)],  # cb-ca bond dist: 1.5267422834
    """
    if np.linalg.norm(N) == 0 and np.linalg.norm(C) == 0 and np.linalg.norm(Ca) == 0:
        return np.zeros_like(N)

    def _safe_normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=float)
        norms = np.linalg.norm(vec, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vec / norms

    normalize = _safe_normalize

    # ... get local frame x-axis
    to_C = C - Ca
    frame_x = normalize(to_C)

    # ... get local frame z-axis
    to_N = N - Ca
    to_out_of_plane = np.cross(frame_x, normalize(to_N), axis=-1)
    frame_z = normalize(to_out_of_plane)

    # ... get local frame y-axis
    frame_y = normalize(np.cross(frame_z, frame_x, axis=-1))

    # ... place virtual Cb at the desired location
    Cb = Ca + (-0.529 * frame_x - 0.774 * frame_y - 1.205 * frame_z)
    return Cb


def create_cb_atoms(array: AtomArray) -> AtomArray:
    # array of length 4 with N, CA, C, O
    # Returns array with CB placed ideally
    if array.atom_name.tolist() != ["N", "CA", "C", "O"]:
        raise ValueError(f"Input array must contain exactly 4 atoms: N, CA, C, O. Got : {array.atom_name.tolist()}")
    cb_atoms = array[array.atom_name == "CA"].copy()
    cb_atoms.atom_name = np.array(["CB"], dtype=cb_atoms.atom_name.dtype)
    cb_pos = generate_idealized_cb_position(
        array.coord[array.atom_name == "N"].squeeze(),
        array.coord[array.atom_name == "CA"].squeeze(),
        array.coord[array.atom_name == "C"].squeeze(),
    )
    cb_atoms.coord = cb_pos[None]
    return cb_atoms


def create_o_atoms(array: AtomArray) -> AtomArray:
    if array.atom_name.tolist() != ["N", "CA", "C"]:
        raise ValueError(f"Input array must contain exactly 4 atoms: N, CA, C, O. Got : {array.atom_name.tolist()}")

    ca_atoms = array[array.atom_name == "CA"].copy()
    ca_atoms.atom_name = np.array(["O"], dtype=ca_atoms.atom_name.dtype)
    ca_atoms.element = np.array(["O"], dtype=ca_atoms.element.dtype)
    ca_atoms.coord = array.coord[array.atom_name == "C"].squeeze()[None]

    return ca_atoms


def get_motif_features(atom_array: AtomArray) -> dict[str, np.ndarray]:
    is_fixed = atom_array.is_motif_atom_with_fixed_coord.astype(bool)
    is_sequence_fixed = atom_array.is_motif_atom_with_fixed_seq.astype(bool)
    is_unindexed = atom_array.is_motif_atom_unindexed.astype(bool)

    # Motif atom if has any conditioning
    is_motif_atom = is_fixed | is_sequence_fixed | is_unindexed | C_INTCRD.mask(atom_array)
    is_motif_token = apply_and_spread_token_wise(
        atom_array, is_motif_atom, function=lambda x: np.any(x)
    )  # Has any atoms with conditioning

    return {"is_motif_atom": is_motif_atom, "is_motif_token": is_motif_token}
