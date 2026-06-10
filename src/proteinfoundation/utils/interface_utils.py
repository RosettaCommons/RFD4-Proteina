"""Utilities for detecting and analyzing protein-protein interfaces."""

import functools
import operator

import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import cKDTree

from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def get_interface_residues(
    structure: AtomArray,
    binder_chain: str,
    cutoff: float = 8.0,
) -> list[int]:
    """Get interface residues on the binder chain within cutoff distance of target.

    This function identifies interface residues using a KD-tree based approach for efficient
    distance calculations. Returns 0-indexed residue indices.

    Args:
        structure: Biotite AtomArray containing both target and binder chains
        binder_chain: Chain ID of the binder protein (e.g., "B")
        cutoff: Distance cutoff in Angstroms for interface definition. Defaults to ``8.0``.

    Returns:
        List of 0-indexed residue indices on the binder chain that are at the interface.
        Sorted in ascending order.

    Example:
        >>> from atomworks.io.utils.io_utils import load_any
        >>> structure = load_any("complex.cif")[0]
        >>> interface_residues = get_interface_residues(structure, "B", cutoff=8.0)
        >>> # Use for mpnn_fixed: fix these residues during sequence design

    Note:
        For mpnn_fixed in designability metrics, these interface residues are fixed
        during ProteinMPNN redesign to preserve binding interactions.
    """
    # Get all chains in the structure
    _all_chains = np.unique(structure.chain_id).tolist()

    # Filter to CA atoms only for distance calculation
    struct_ca = structure[structure.atom_name == "CA"]

    # Separate binder and target chains
    binder = struct_ca[struct_ca.chain_id == binder_chain]
    target = struct_ca[struct_ca.chain_id != binder_chain]

    if len(binder) == 0:
        raise ValueError(f"No CA atoms found for binder chain {binder_chain}")

    if len(target) == 0:
        raise ValueError(f"No CA atoms found for target chains (non-{binder_chain})")

    # Create KD-trees for efficient distance calculations
    binder_tree = cKDTree(binder.coord)
    target_tree = cKDTree(target.coord)

    # Find all pairs of atoms within cutoff distance
    # query_ball_tree returns indices of binder atoms close to each target atom
    pairs = target_tree.query_ball_tree(binder_tree, cutoff)

    # Get unique binder atoms that are at the interface
    binder_interface_atoms = np.array(sorted(set(functools.reduce(operator.iadd, pairs, []))))

    if len(binder_interface_atoms) == 0:
        logger.warning(f"No interface residues found with cutoff {cutoff}Å")
        return []

    # Get residue IDs for interface atoms
    binder_interface = np.unique(binder[binder_interface_atoms].res_id)

    # Convert to 0-indexed residue indices
    # Calculate offset to convert from PDB residue numbering to 0-indexed
    offset = int(binder.res_id.min())

    interface_residues = []
    for res in binder_interface:
        # Convert PDB residue number to 0-indexed
        residue_idx = int(res) - offset
        interface_residues.append(residue_idx)

    logger.info(f"Found {len(interface_residues)} interface residues (cutoff={cutoff}Å) on chain {binder_chain}")

    return sorted(interface_residues)


def get_interface_residues_all_atom(
    structure: AtomArray,
    binder_chain: str,
    cutoff: float = 6.0,
) -> list[int]:
    """Get interface residues using all atoms (not just CA).

    More sensitive than CA-only detection. Useful for ligand-protein interfaces
    or detailed interface analysis.

    Args:
        structure: Biotite AtomArray containing both target and binder
        binder_chain: Chain ID of the binder
        cutoff: Distance cutoff in Angstroms. Defaults to ``6.0`` (tighter than CA).

    Returns:
        List of 0-indexed residue indices on the binder chain that are at the interface.

    Note:
        Uses all heavy atoms, providing higher resolution interface detection than CA-only.
    """
    # Separate binder and target (all atoms)
    binder = structure[structure.chain_id == binder_chain]
    target = structure[structure.chain_id != binder_chain]

    if len(binder) == 0:
        raise ValueError(f"No atoms found for binder chain {binder_chain}")

    if len(target) == 0:
        raise ValueError(f"No atoms found for target chains (non-{binder_chain})")

    # Create KD-trees
    binder_tree = cKDTree(binder.coord)
    target_tree = cKDTree(target.coord)

    # Find interface atoms
    pairs = target_tree.query_ball_tree(binder_tree, cutoff)
    binder_interface_atoms = np.array(sorted(set(functools.reduce(operator.iadd, pairs, []))))

    if len(binder_interface_atoms) == 0:
        logger.warning(f"No interface atoms found with cutoff {cutoff}Å")
        return []

    # Get unique residue IDs
    binder_interface = np.unique(binder[binder_interface_atoms].res_id)

    # Convert to 0-indexed
    # For CA atoms in the binder chain, get min res_id as offset
    binder_ca = binder[binder.atom_name == "CA"]
    offset = int(binder_ca.res_id.min())

    interface_residues = [int(res) - offset for res in binder_interface]

    logger.info(
        f"Found {len(interface_residues)} interface residues (all-atom, cutoff={cutoff}Å) on chain {binder_chain}"
    )

    return sorted(interface_residues)


def format_interface_residues_for_mpnn(
    interface_residues: list[int],
    chain_id: str,
) -> list[str]:
    """Format interface residues for ProteinMPNN fixed_positions.

    Converts 0-indexed residue indices to ProteinMPNN's 1-indexed format.

    Args:
        interface_residues: List of 0-indexed residue indices
        chain_id: Chain ID to prepend (e.g., "B")

    Returns:
        List of formatted position strings (e.g., ["B1", "B5", "B10"])

    Example:
        >>> interface_res = [0, 4, 9]  # 0-indexed
        >>> fix_pos = format_interface_residues_for_mpnn(interface_res, "B")
        >>> # Returns: ["B1", "B5", "B10"] (1-indexed)
    """
    return [f"{chain_id}{res + 1}" for res in interface_residues]
