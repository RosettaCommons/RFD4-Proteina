"""Utilities for identifying the chain of interest in a multi-chain structure."""

import numpy as np
from atomworks.ml.conditions import C_CRD, C_DIS
from atomworks.ml.conditions.annotator import ensure_annotations
from biotite.structure import AtomArray
from foundry.utils.ddp import RankedLogger

from proteinfoundation.conditions import C_INTCRD

logger = RankedLogger(__name__, rank_zero_only=True)


def get_chain_of_interest(atom_array: AtomArray) -> tuple[str, bool]:
    """Get the chain ID or IID of the chain of interest in the AtomArray.

    The chain of interest corresponds to the primary designed chain, and is determined by the following heuristic:
    1. If there is only one unique chain, return that chain.
    2. Subset to only consider protein chains.
    3. Subset to chains with FEWEST coordinate conditioned atoms (absolute or internal).
    4. Subset to chains with FEWEST internal coordinate conditioned atoms.
    5. Subset to chains with FEWEST distance conditioned atoms.
    6. If this is still ambiguous, return the first chain in sorted order and give a warning.

    TODO: This heuristic will need to be adjusted for more complex use-cases.

    Args:
        atom_array: Input AtomArray.

    Returns:
        Tuple of (chain_id_or_iid, is_unambiguous).
    """
    if "chain_iid" in atom_array.get_annotation_categories():
        chain_identifiers = atom_array.get_annotation("chain_iid")
    else:
        chain_identifiers = atom_array.get_annotation("chain_id")
    unique_chain_identifiers = np.unique(chain_identifiers)

    if len(unique_chain_identifiers) < 2:
        return unique_chain_identifiers[0], True

    # Only consider protein chains
    ensure_annotations(atom_array, "is_protein")
    protein_chain_identifiers = [
        cid for cid in unique_chain_identifiers if atom_array[chain_identifiers == cid].is_protein.all()
    ]
    if len(protein_chain_identifiers) == 1:
        return protein_chain_identifiers[0], True

    # Subset by all coordinate conditioning (absolute or internal)
    coord_conditioned_fraction_by_chain = {
        cid: C_CRD.mask(atom_array[chain_identifiers == cid]).mean()
        + C_INTCRD.mask(atom_array[chain_identifiers == cid]).mean()
        for cid in protein_chain_identifiers
    }
    least_coord_conditioned_fraction = min(coord_conditioned_fraction_by_chain.values())
    least_coord_conditioned_chain_identifiers = [
        cid for cid, frac in coord_conditioned_fraction_by_chain.items() if frac == least_coord_conditioned_fraction
    ]
    if len(least_coord_conditioned_chain_identifiers) == 1:
        return least_coord_conditioned_chain_identifiers[0], True

    # Further subset by internal coordinate conditioning
    internal_coord_conditioned_fraction_by_chain = {
        cid: C_INTCRD.mask(atom_array[chain_identifiers == cid]).mean() for cid in protein_chain_identifiers
    }
    least_internal_coord_conditioned_fraction = min(internal_coord_conditioned_fraction_by_chain.values())
    least_internal_coord_conditioned_chain_identifiers = [
        cid
        for cid, frac in internal_coord_conditioned_fraction_by_chain.items()
        if frac == least_internal_coord_conditioned_fraction
    ]
    if len(least_internal_coord_conditioned_chain_identifiers) == 1:
        return least_internal_coord_conditioned_chain_identifiers[0], True

    # Further subset by distance conditioning
    has_distance_conditioning = C_DIS.mask(atom_array).as_dense_array().any(axis=-1)
    distance_conditioned_fraction_by_chain = {
        cid: has_distance_conditioning[chain_identifiers == cid].mean()
        for cid in least_coord_conditioned_chain_identifiers
    }
    least_distance_conditioned_fraction = min(distance_conditioned_fraction_by_chain.values())
    least_distance_conditioned_chain_identifiers = [
        cid
        for cid, frac in distance_conditioned_fraction_by_chain.items()
        if frac == least_distance_conditioned_fraction
    ]
    if len(least_distance_conditioned_chain_identifiers) == 1:
        return least_distance_conditioned_chain_identifiers[0], True

    # Ambiguous - return first chain with warning
    logger.warning("Could not unambiguously determine chain of interest; defaulting to first chain in sorted order.")
    return sorted(least_distance_conditioned_chain_identifiers)[0], False
