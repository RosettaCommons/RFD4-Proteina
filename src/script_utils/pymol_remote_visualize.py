import logging
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
import torch
from atomworks.io.utils.standard_annotations.base import Level
from atomworks.io.utils.visualize import get_pymol_session, view_pymol
from atomworks.ml.conditions import Condition

from rfproteina.proteina import Proteina

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def get_atom_array_style_cmd(
    atom_array: struc.AtomArray | struc.AtomArrayStack,
    obj: str,
    label: bool = False,
) -> str:
    """Generate PyMOL commands to style an atom array visualization.

    Creates a series of PyMOL commands that style different parts of a molecular structure, including:
    - Applying a color spectrum to polymer chains
    - Showing backbone atoms as sticks and CA atoms as spheres
    - Styling non-polymer atoms with different colors and representation
    - Highlighting specially annotated atoms with different colors and visualizations

    Args:
        atom_array: The biotite AtomArray or AtomArrayStack to be styled
        obj: PyMOL object name to apply the styling to
        label: Whether to label all atoms with their 0-indexed atom_id

    Returns:
        str: A PyMOL command string that styles the atom array
    """
    commands = [f"hide everything, {obj}"]
    annotations = atom_array.get_annotation_categories()

    offset = 1  # ... default offset since pymol 1-indexes atom id's
    atom_ids = np.arange(1, atom_array.array_length() + 1)
    if "atom_id" in annotations:
        offset = 0
        atom_ids = atom_array.get_annotation("atom_id")

    # Convert MAS residues to ALA for compatibility
    atom_array = atom_array.copy()
    # atom_array.res_name[atom_array.res_name == "MAS"] = "ALA"

    # Style the backbone for each polymer chain with a color spectrum
    for chain_id in struc.get_chains(atom_array):
        if (~atom_array.hetero[atom_array.chain_id == chain_id]).any():
            commands.append(
                f"spectrum resi, RFd_darkblue RFd_blue RFd_lightblue RFd_purple RFd_pink RFd_melon RFd_navaho, "
                f"{obj} and chain {chain_id} and elem C"
            )

    # Add basic styling commands for protein backbone and non-polymer components
    commands.extend(
        [
            f"show sticks, model {obj} and name n+c+ca+cb",
            f"show spheres, model {obj} and name ca",
            f"set sphere_scale, 0.23, model {obj} and name ca",
            f"set sphere_transparency, 0, model {obj} and name ca",
            f"color grey60, model {obj} and not polymer and elem C",
            f"show nb_spheres, model {obj} and not polymer",
            f"show sticks, model {obj} and not polymer",
        ]
    )
    if label:
        # label 0-indexed for correspondence with biotite
        commands.append("label all, ID") if offset == 0 else commands.append(f"label all, ID-{offset}")

    # Style atoms marked for "atomize" if present
    if "atomize" in annotations and atom_array.atomize.any():
        atomize_ids = np.where(atom_array.atomize)[0]
        atomize_ids = atom_ids[atomize_ids]
        commands.extend(
            [
                f"select {obj}_atomize, model {obj} and id {'+'.join(str(id) for id in atomize_ids)}",
                f"show sticks, {obj}_atomize and byres {obj}_atomize",
            ]
        )

    # Handle 1-D conditions (only display masks)
    for cond in Condition:
        if cond.n_body == 1 and cond.mask(atom_array, default="generate").any():
            _atom_ids = atom_ids[np.where(cond.mask(atom_array, default="generate"))[0]]
            selection_name = f"mask_{cond.name}_{cond.n_body}_{cond.level}_{obj}"
            if cond.level == Level.ATOM:
                _selection = f"model {obj} and id {'+'.join(str(id) for id in _atom_ids)}"
                commands.extend([f"select {selection_name}, {_selection}"])
            elif cond.level == Level.RESIDUE or cond.level == Level.TOKEN:
                _selection = f"model {obj} and byres (id {'+'.join(str(id) for id in _atom_ids)})"
                commands.extend([f"select {selection_name}, {_selection}"])

    return "\n".join(commands)


def viz(
    atom_array: struc.AtomArray | struc.AtomArrayStack,
    id: str = "obj",
    clear: bool = False,
    label: bool = False,
    view_ori_token: bool = False,
    stylize: bool = False,
    connect_mode: Literal[0, 1, 2, 3, 4] = 4,
) -> None:
    """Quickly visualize a molecular structure in PyMOL with predefined styling.

    This function creates a PyMOL session, loads the atom array structure, and applies
    a set of styling commands to make the visualization informative and aesthetically pleasing.
    The styling highlights different structural components and annotated features.

    Args:
        atom_array: The biotite AtomArray or AtomArrayStack to visualize
        id: PyMOL object identifier (default: "obj")
        clear: Whether to clear existing PyMOL objects before visualization (default: True)
        label: Whether to label all atoms with their 0-indexed atom_id (default: True)
        view_ori_token: Whether to view the ori token (default: False)
        stylize: Whether to apply styling as defined in `get_atom_array_style_cmd` (default: False)
        connect_mode: How to determine bond rendering. See https://pymolwiki.org/Connect_mode
            0 = distance-based (excluding HETATM to HETATM) and CONECT records (default)
            1 = CONECT records
            2 = distance-based, ignores CONECT records
            3 = distance-based (including HETATM to HETATM) and CONECT records
            4 = for loading mmCIF: use the chemical components dictionary to look up bonds

    Example:
        >>> from biotite.structure import AtomArray
        >>> # Create or load an atom array
        >>> structure = AtomArray(...)
        >>> # Visualize the structure in PyMOL
        >>> viz(structure)
    """
    atom_array = atom_array.copy()
    # pymol only considers chain_id annotation, which can make weird looking artifacts if we have two different
    # chains with the same chain_id. We always disambiguate by using the chain_iid annotation, so we need to
    # have pymol use that to do the same
    if "chain_iid" in atom_array.get_annotation_categories():
        atom_array.chain_id = atom_array.get_annotation("chain_iid")
        atom_array.chain_id = atom_array.chain_id.astype(str)

    pymol = get_pymol_session()
    pymol.do(f"set valence, 1; set connect_mode, {connect_mode};")
    if clear:
        pymol.do("delete d*")
        pymol.delete("all")
    slot = np.random.randint(0, 10_000)
    obj_name = view_pymol(atom_array, id=id, grid_slot=slot)
    if stylize:
        cmd = get_atom_array_style_cmd(atom_array, obj_name, label=label)
        pymol.do(cmd)

    if view_ori_token:
        pymol.do(f"pseudoatom ori_{obj_name}, pos=[0,0,0]")
        pymol.do(
            [
                f"set grid_slot, {slot}, ori_{obj_name}",
                f"show spheres, ori_{obj_name}",
                f"set sphere_color, white, ori_{obj_name}",
                f"set sphere_scale, 0.5, ori_{obj_name}",
                f"set sphere_transparency, 0.5, ori_{obj_name}",
            ]
        )


def viz_x_decoded(batch: dict[str, Any], x: dict[str, torch.Tensor], model: Proteina, **kwargs) -> None:
    """Decode a partially-latent sample and visualize it using pymol-remote.

    Args:
        batch: input batch containing the sample to visualize, input AtomArray, padding mask, and encoding mask
        x: the latent sample to decode and visualize
        model: A Proteina instance to be used for decoding/formatting
        **kwargs: additional keyword arguments to pass to the viz function

    Returns:
        None, visualizes the decoded sample in PyMOL
    """

    sample_prots = model.sample_formatting(
        x=x,
        ret_mode="coors37_n_aatype",
        batch=batch,
    )

    formatted_results = model.format_sample_prots_to_atom_arrays(sample_prots, batch, batch_idx=0)

    viz(formatted_results[0]["generated_atom_array"], **kwargs)
