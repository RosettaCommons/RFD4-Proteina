import numpy as np

from proteinfoundation.inference_utils.components import fetch_mask_from_idx
from proteinfoundation.utils.standard_annotations import SA_SRC


def expand_contig_to_resid_from_string(contig_string: str) -> list[str]:
    """
    Expand a contig string to a list of residue indices.
    Arguments:
        contig_string: string of the form "X1-5", e.g.
    Returns:
        list of residue indices
    """
    chain = contig_string[0]
    res_range = contig_string[1:].split("-")
    res_start = int(res_range[0])
    res_end = int(res_range[1])
    return [f"{chain}{i}" for i in range(res_start, res_end + 1)]


def expand_contig_unsym_motif(unsym_motif_names: list[str]) -> list[str]:
    """
    Expand a list of unsym motif names to a list of residue indices.
    Arguments:
        unsym_motif_names: list of unsym motif names
    Returns:
        list of residue indices
    """
    expanded_contigs = [expand_contig_to_resid_from_string(n) for n in unsym_motif_names if "-" in n]
    # now remove any unexpanded contigs
    unsym_motif_names = [n for n in unsym_motif_names if "-" not in n]
    if len(expanded_contigs) != 0:
        for c in expanded_contigs:
            unsym_motif_names.extend(c)
    return unsym_motif_names


def get_unsym_motif_mask(atom_array: np.ndarray, unsym_motif_names: list[str]) -> np.ndarray:
    """
    Get a mask of the unsym motif atoms.
    Arguments:
        atom_array: atom array
        unsym_motif_names: list of unsym motif names
    Returns:
        mask of the unsym motif atoms
    """

    is_unsym_motif = np.zeros(len(atom_array), dtype=bool)
    for n in unsym_motif_names:
        is_unsym_motif = np.logical_or(is_unsym_motif, atom_array.res_name == n)
        if SA_SRC.has_annotation(atom_array) and n in SA_SRC.annotation(atom_array):
            is_unsym_motif = np.logical_or(is_unsym_motif, SA_SRC.annotation(atom_array) == n)
        elif n[0].isalpha() and n[1:].isdigit():
            residue_mask = fetch_mask_from_idx(n, atom_array=atom_array)
            is_unsym_motif = np.logical_or(is_unsym_motif, residue_mask)
    return is_unsym_motif
