"""Read the insertion head's predicted token mapping via per-chain Hungarian assignment."""

import numpy as np
import torch
from atomworks.ml.conditions import C_CRD, C_IDX
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray
from scipy.optimize import linear_sum_assignment

from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def _any_per_token(per_atom_mask: np.ndarray, token_starts: np.ndarray) -> np.ndarray:
    """Reduce a per-atom boolean mask to one value per token (True if any atom is set)."""
    if len(token_starts) == 0:
        return np.zeros(0, dtype=bool)
    return np.maximum.reduceat(per_atom_mask.astype(np.int8), token_starts).astype(bool)


def build_predicted_insertion_mapping(
    insertion_logits: torch.Tensor,
    full_array_with_unindexed: AtomArray,
) -> dict[int, int]:
    """Match each unindexed token to a distinct, eligible generated token on the same chain.

    A target is eligible when it carries no coordinate conditioning of its own, which rules out
    the indexed-motif tokens whose transfer `annotate_unindexed_insertion` could not apply.
    Identity is not part of eligibility (see the comment in the body). An unindexed token with no
    eligible target is omitted from the mapping and reported as unmapped.

    Args:
      insertion_logits: ``[n_valid, n_valid]`` logits, ``n_valid = n_gen + n_unindexed``.
      full_array_with_unindexed: Atom array of generated + unindexed in-context tokens.

    Returns:
      ``{unindexed_token_id: target_generated_token_id}``. Empty when there are no
      unindexed tokens or the logits shape is inconsistent with the token counts.
    """
    token_starts = get_token_starts(full_array_with_unindexed)
    all_tokens = full_array_with_unindexed[token_starts]
    is_indexed_token = C_IDX.mask(all_tokens, default="generate")
    generated_tokens = all_tokens[is_indexed_token]
    unindexed_tokens = all_tokens[~is_indexed_token]
    n_gen, n_unindexed = len(generated_tokens), len(unindexed_tokens)
    if n_unindexed == 0:
        return {}

    if n_gen + n_unindexed != insertion_logits.shape[0]:
        logger.warning(
            f"Insertion logits shape {tuple(insertion_logits.shape)} inconsistent with token "
            f"counts (gen={n_gen}, unindexed={n_unindexed}); no insertion mapping produced."
        )
        return {}

    logits = insertion_logits.float().cpu().numpy()  # [n_valid, n_valid]
    gen_chain = np.asarray(generated_tokens.chain_id)  # [n_gen]
    unindexed_chain = np.asarray(unindexed_tokens.chain_id)  # [n_unindexed]
    gen_token_id = np.asarray(generated_tokens.token_id)  # [n_gen]
    unindexed_token_id = np.asarray(unindexed_tokens.token_id)  # [n_unindexed]

    # A generated token that already carries coordinate conditioning belongs to the indexed motif.
    # Transferring a second, different coordinate onto it is contradictory, and
    # `annotate_unindexed_insertion` raises ConditionConflictError for it by design -- a hard
    # failure that aborts the run. Such targets are therefore never proposed.
    #
    # Identity is deliberately NOT constrained here: the insertion head's logits carry where the
    # motif actually went, and forcing a same-identity target elsewhere in the chain would
    # manufacture a geometrically wrong mapping in place of an honest failure. A wrong-identity
    # target still fails downstream in atom matching, which is the correct outcome.
    #
    # A row with no eligible target is left unmapped on purpose: a missing mapping entry is counted
    # by UnindexedInsertionMappingMetric as an unmapped token, a clean and measurable outcome.
    is_coord_conditioned_token = _any_per_token(
        C_CRD.mask(full_array_with_unindexed, default="generate"), token_starts
    )
    gen_is_conditioned = is_coord_conditioned_token[is_indexed_token]  # [n_gen]

    # Per-chain Hungarian assignment: each unindexed token -> a DISTINCT generated token on the same chain
    mapping: dict[int, int] = {}
    for chain in np.unique(unindexed_chain):
        u_idx = np.flatnonzero(unindexed_chain == chain)  # rows, in [0, n_unindexed)
        # Only unconditioned generated tokens are candidate targets.
        g_idx = np.flatnonzero((gen_chain == chain) & ~gen_is_conditioned)  # cols, in [0, n_gen)

        if len(g_idx) == 0:
            continue

        # Solve linear assignment problem - maximize the sum of logits for the assigned pairs.
        # A rectangular matrix with fewer columns than rows assigns min(rows, cols) pairs, so a
        # shortage of eligible targets leaves the remaining tokens unmapped rather than raising.
        score = logits[np.ix_(n_gen + u_idx, g_idx)]  # [n_u_chain, n_g_chain]
        rows, cols = linear_sum_assignment(score, maximize=True)
        for r, c in zip(rows, cols, strict=True):
            mapping[int(unindexed_token_id[u_idx[r]])] = int(gen_token_id[g_idx[c]])

    n_dropped = len(unindexed_token_id) - len(mapping)
    if n_dropped:
        logger.info(
            f"Insertion mapping: {n_dropped}/{len(unindexed_token_id)} unindexed token(s) had no "
            f"eligible (unconditioned) target and stay unmapped."
        )

    return mapping
