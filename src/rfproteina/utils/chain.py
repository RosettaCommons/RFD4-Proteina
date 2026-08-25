"""Utilities for identifying the chain of interest in a multi-chain structure."""

import numpy as np
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.ml.conditions import C_CRD, C_SEQ
from biotite.structure import AtomArray

from rfproteina.conditions import C_INTCRD
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def get_chain_of_interest(atom_array: AtomArray) -> tuple[str | None, bool]:
    """Get the chain ID or IID of the chain of interest in the AtomArray.

    The chain of interest corresponds to the primary designed chain, and is determined by the following heuristic:
    1. If there is only one unique chain, return that chain.
    2. Subset to only consider protein chains. If none remain, return the first
       chain in sorted order with a warning.
    3. Subset to chains with the LEAST sequence conditioning relative to chain length
       (the designed chain has its sequence generated, so it is least sequence-conditioned).
    4. Tie-break: subset to chains with the LEAST coordinate conditioning relative to chain
       length (absolute or internal).
    5. If this is still ambiguous, return the first chain in sorted order and give a warning.

    TODO: This heuristic will need to be adjusted for more complex use-cases.

    Args:
        atom_array: Input AtomArray.

    Returns:
        Tuple of (chain_id_or_iid, is_unambiguous). ``chain_id_or_iid`` is
        ``None`` when the input contains no protein chains.
    """
    if "chain_iid" in atom_array.get_annotation_categories():
        chain_identifiers = atom_array.get_annotation("chain_iid")
    else:
        chain_identifiers = atom_array.get_annotation("chain_id")
    unique_chain_identifiers = np.unique(chain_identifiers)

    if len(unique_chain_identifiers) < 2:
        return unique_chain_identifiers[0], True

    # Prefer protein chains when present, but fall back to all chains for
    # all-RNA/all-DNA/all-nucleic multichain cases.
    ensure_annotations(atom_array, "is_protein")
    protein_chain_identifiers = [
        cid for cid in unique_chain_identifiers if atom_array[chain_identifiers == cid].is_protein.all()
    ]
    if len(protein_chain_identifiers) == 1:
        return protein_chain_identifiers[0], True

    if len(protein_chain_identifiers) == 0:
        logger.warning("No pure-protein chains found; defaulting to first chain in sorted order.")
        return sorted(unique_chain_identifiers)[0], False

    chain_subset_by_id = {cid: atom_array[chain_identifiers == cid] for cid in protein_chain_identifiers}

    # Subset by sequence conditioning relative to chain length. The designed chain has its
    # sequence generated rather than given, so it is the least sequence-conditioned chain.
    sequence_conditioned_fraction_by_chain = {
        cid: C_SEQ.mask(chain_subset_by_id[cid]).mean() for cid in protein_chain_identifiers
    }
    least_sequence_conditioned_fraction = min(sequence_conditioned_fraction_by_chain.values())
    least_sequence_conditioned_chain_identifiers = [
        cid
        for cid, frac in sequence_conditioned_fraction_by_chain.items()
        if frac == least_sequence_conditioned_fraction
    ]
    if len(least_sequence_conditioned_chain_identifiers) == 1:
        return least_sequence_conditioned_chain_identifiers[0], True

    # Tie-break by coordinate conditioning relative to chain length (absolute or internal).
    coord_conditioned_fraction_by_chain = {
        cid: C_CRD.mask(chain_subset_by_id[cid]).mean() + C_INTCRD.mask(chain_subset_by_id[cid]).mean()
        for cid in least_sequence_conditioned_chain_identifiers
    }
    least_coord_conditioned_fraction = min(coord_conditioned_fraction_by_chain.values())
    least_coord_conditioned_chain_identifiers = [
        cid for cid, frac in coord_conditioned_fraction_by_chain.items() if frac == least_coord_conditioned_fraction
    ]
    if len(least_coord_conditioned_chain_identifiers) == 1:
        return least_coord_conditioned_chain_identifiers[0], True

    # Ambiguous - return first chain with warning
    logger.warning("Could not unambiguously determine chain of interest; defaulting to first chain in sorted order.")
    return sorted(least_coord_conditioned_chain_identifiers)[0], False
