"""Batch-level invariant checking for the rfproteina pipeline."""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.scatter import apply_segment_wise
from atomworks.io.utils.selection import get_residue_starts, get_segments
from atomworks.io.utils.standard_annotations import S_ATM, STANDARD_ANNOTATIONS
from atomworks.io.utils.standard_annotations.base import Level
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
from atomworks.ml.transforms.atom_array import add_global_token_id_annotation
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray

from rfproteina.conditions import C_CENTROID, C_HB, C_IDXGRP, C_NCYS, HBondEnum

logger = logging.getLogger(__name__)


@dataclass
class InvariantViolation:
    """A single invariant violation detected in a pipeline batch."""

    check: str
    message: str
    count: int = 1


def detect_batch_invariant_violations(data: dict, is_inference: bool) -> list[InvariantViolation]:
    """Detect invariant violations in a pipeline output batch.

    Args:
        data: Pipeline output dict containing at least ``"atom_array"`` and ``"feats"``.
        is_inference: Whether the pipeline is in inference mode.

    Returns:
        List of :py:class:`InvariantViolation` instances; empty if all checks pass.
    """
    violations: list[InvariantViolation] = []
    aa = data["atom_array"]
    feats = data["feats"]

    # --- Structural / atom-array checks ---
    violations += check_molecules_on_separate_chains(aa)
    violations += check_standard_annotation_level_violations(aa)
    violations += check_global_token_id_consistent(aa)
    violations += check_centroid_conditioning_all_or_nothing_violations(aa, is_inference=is_inference)

    # --- Feats / token-level checks ---
    violations += check_conditioning_violations(aa, feats)
    violations += check_atomized_indexed_violations(feats)
    violations += check_coordinates_not_nan(feats, is_inference=is_inference)
    violations += check_element_encoding_violations(feats)
    violations += check_unresolved_tokens_not_lossed_violations(aa, feats)
    violations += check_sequence_conditioning_violations(feats, is_inference)
    violations += check_hbond_no_both_violations(feats)
    violations += check_not_cysteine_sequence_conditioning_violations(aa)

    # --- Cross-cutting checks (require both aa and feats) ---
    violations += check_coordinate_centering_violations(aa, feats)
    violations += check_can_be_encoded_violations(aa, feats)
    violations += check_token_features_consistent_violations(aa)
    violations += check_no_oxt_on_encodable_tokens(aa, feats)

    return violations


# ---------------------------------------------------------------------------
# Atom-array checks
# ---------------------------------------------------------------------------


def check_molecules_on_separate_chains(aa: AtomArray) -> list[InvariantViolation]:
    """Return violations where a chain contains multiple molecules"""
    categories = set(aa.get_annotation_categories())
    if "chain_iid" in categories:
        chain_identifier = aa.chain_iid
        molecule_identifier = aa.molecule_iid if "molecule_iid" in categories else aa.molecule_id
    else:
        chain_identifier = aa.chain_id
        molecule_identifier = aa.molecule_id

    violations = []
    for chain_id in np.unique(chain_identifier):
        molecule_identifiers = molecule_identifier[chain_identifier == chain_id]
        n_molecules = len(np.unique(molecule_identifiers))
        if n_molecules > 1:
            violations.append(
                InvariantViolation(
                    check="multiple_molecules_on_chain",
                    message=(f"Chain {chain_id} contains {n_molecules} different molecules"),
                    count=1,
                )
            )
    return violations


def check_centroid_conditioning_all_or_nothing_violations(
    aa: AtomArray, is_inference: bool
) -> list[InvariantViolation]:
    """Return violations where centroid conditioning covers some but not all of an example.

    Centroid conditioning (``C_CENTROID``) is a per-chain value, and must be all-or-nothing
    per example: either every chain has a finite centroid on every one of its atoms, or none
    do, matching :py:class:`~rfproteina.datapipes.pipeline.FeaturizeChainCentroid`. Two ways
    to break that are reported separately: conditioning only *some* chains, and conditioning
    only part of a chain (which leaves the rest of it with a NaN centroid the model reads as
    "not conditioned").

    At train time, a chain with no resolved coordinates (fully unresolved) is exempt from this,
    as its ground-truth centroid would be undefined.
    """
    categories = set(aa.get_annotation_categories())
    if C_CENTROID.full_name not in categories:
        return []

    chain_identifier = aa.chain_iid if "chain_iid" in categories else aa.chain_id
    centroid_mask = C_CENTROID.mask(aa, default="generate")

    aap = as_atom_array_plus(aa)
    coords = aap.coord_to_be_noised if "coord_to_be_noised" in aap.get_annotation_categories() else None

    n_chains = 0
    n_conditioned = 0
    partially_conditioned: list[str] = []
    for chain_id in np.unique(chain_identifier):
        chain_mask = chain_identifier == chain_id
        if not is_inference and coords is not None and np.isnan(coords[chain_mask]).all():
            continue
        n_chains += 1
        if centroid_mask[chain_mask].any():
            n_conditioned += 1
            if not centroid_mask[chain_mask].all():
                partially_conditioned.append(str(chain_id))

    violations = []
    if partially_conditioned:
        violations.append(
            InvariantViolation(
                check="centroid_conditioning_partial_chain",
                message=(
                    f"chain(s) {partially_conditioned} have a centroid on only some of their atoms; "
                    "the centroid is a per-chain value and must cover every atom of the chain"
                ),
                count=len(partially_conditioned),
            )
        )
    if n_chains and 0 < n_conditioned < n_chains:
        violations.append(
            InvariantViolation(
                check="centroid_conditioning_not_all_or_nothing",
                message=(
                    f"{n_conditioned}/{n_chains} chain(s) have centroid conditioning; "
                    "must be applied to all chains or none"
                ),
                count=n_chains,
            )
        )
    return violations


def _segment_has_one_value(segment: np.ndarray) -> bool:
    """Whether every entry of one segment of a StandardAnnotation carries the same value."""
    if segment.ndim == 1:
        return np.unique(segment).size <= 1
    first = segment[:1]
    equal = segment == first
    if np.issubdtype(segment.dtype, np.floating):
        equal |= np.isnan(segment) & np.isnan(first)
    return bool(equal.all())


def check_standard_annotation_level_violations(aa: AtomArray) -> list[InvariantViolation]:
    """Return violations where a StandardAnnotation has inconsistent values within its segment.

    For each StandardAnnotation present on the atom array, values must be uniform
    within every segment defined by the annotation's declared
    :py:class:`~atomworks.io.utils.standard_annotations.base.Level`.

    ``ATOM``-level and multi-body (``n_body > 1``) annotations are skipped.
    """
    categories = set(aa.get_annotation_categories())
    violations = []

    for sa in STANDARD_ANNOTATIONS:
        if sa.n_body != 1 or sa.level == Level.ATOM or sa.full_name not in categories:
            continue

        values = aa.get_annotation(sa.full_name)

        if sa.level == Level.RESIDUE:
            starts = get_residue_starts(aa, add_exclusive_stop=True)
        elif sa.level == Level.TOKEN:
            starts = get_token_starts(aa, add_exclusive_stop=True)
        elif sa.level == Level.CHAIN:
            chain_identifier = aa.chain_iid if "chain_iid" in categories else aa.chain_id
            starts = get_segments(chain_identifier, add_exclusive_stop=True)
        elif sa.level == Level.MOLECULE:
            molecule_identifier = aa.molecule_id if "molecule_id" in categories else aa.chain_id
            starts = get_segments(molecule_identifier, add_exclusive_stop=True)
        else:
            continue

        uniform_per_seg = np.asarray(apply_segment_wise(starts, values, _segment_has_one_value), dtype=bool)
        n_bad = int((~uniform_per_seg).sum())
        if n_bad:
            violations.append(
                InvariantViolation(
                    check="standard_annotation_level_violated",
                    message=(
                        f"'{sa.full_name}' (level={sa.level.value}) has inconsistent values in {n_bad} segment(s)"
                    ),
                    count=n_bad,
                )
            )

    return violations


def check_global_token_id_consistent(aa: AtomArray) -> list[InvariantViolation]:
    """Return violations where the token_id annotation does not match the annotation implied by the AtomArray."""
    old_token_id = aa.get_annotation("token_id").copy()
    aa_copy = aa.copy()
    aa_copy.del_annotation("token_id")
    aa_copy = add_global_token_id_annotation(aa_copy)
    difference_mask = aa_copy.token_id != old_token_id
    n_bad = int(difference_mask.sum())
    if n_bad:
        return [
            InvariantViolation(
                check="global_token_id_inconsistent",
                message=f"'token_id' annotation is inconsistent with recomputed values in {n_bad} atom(s)",
                count=n_bad,
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Feats / token-level checks
# ---------------------------------------------------------------------------


def check_conditioning_violations(aa: AtomArray, feats: dict) -> list[InvariantViolation]:
    """Return violations where atomized/coord-conditioned tokens lack sequence conditioning."""
    # Lazy imports to avoid circular dependency with pipeline.py
    from rfproteina.datapipes.pipeline import TokenType, build_atom37_slots

    token_type: torch.Tensor = feats["token_type"]  # [n_tokens]
    seq_mask: torch.Tensor = feats["mask_sequence_1_residue"]  # [n_tokens]
    atom37_mask: torch.Tensor = feats["atom37_mask"]  # [n_tokens, 37]
    crd_mask: torch.Tensor = feats["mask_coordinate_1_atom"] | feats["mask_InternalCoordinate_1_atom"]  # [n_tokens, 37]

    violations = []

    is_atomized = token_type == TokenType.ATOMIZED
    atomized_not_seq = is_atomized & ~seq_mask
    if atomized_not_seq.any():
        count = int(atomized_not_seq.sum())
        violations.append(
            InvariantViolation(
                check="atomized_must_be_seq_conditioned",
                message=f"{count} atomized token(s) lack sequence conditioning",
                count=count,
            )
        )

    # Build [n_tokens, 37] backbone mask by scattering lazy-annotator results to atom37.
    # Tokens with coord conditioning only on backbone atoms don't need seq conditioning.
    aap = as_atom_array_plus(aa)
    ensure_annotations(aap, "is_protein_backbone", "is_rna_backbone", "is_dna_backbone")
    is_backbone = aap.is_protein_backbone | aap.is_rna_backbone | aap.is_dna_backbone  # [n_atoms]

    atom_slots = build_atom37_slots(atom_array=aap)

    n_tokens = int(atom37_mask.shape[0])
    bb_atom37 = np.zeros((n_tokens, 37), dtype=bool)
    valid = atom_slots >= 0
    bb_atom37[aap.token_id[valid], atom_slots[valid]] = is_backbone[valid]
    bb_mask = torch.from_numpy(bb_atom37).to(crd_mask.device)

    has_non_bb_crd = (crd_mask & atom37_mask & ~bb_mask).any(dim=1)  # [n_tokens]
    non_bb_crd_not_seq = has_non_bb_crd & ~seq_mask
    if non_bb_crd_not_seq.any():
        count = int(non_bb_crd_not_seq.sum())
        violations.append(
            InvariantViolation(
                check="coord_conditioned_must_be_seq_conditioned",
                message=f"{count} token(s) with non-backbone coordinate conditioning lack sequence conditioning",
                count=count,
            )
        )

    return violations


def check_atomized_indexed_violations(feats: dict) -> list[InvariantViolation]:
    """Return violations where atomized tokens are unindexed."""
    # Lazy import to avoid circular dependency with pipeline.py
    from rfproteina.datapipes.pipeline import TokenType

    token_type: torch.Tensor = feats["token_type"]  # [n_tokens]
    idx_mask_raw: torch.Tensor | None = feats.get(C_IDX.mask_name)  # [n_tokens, 37]

    if idx_mask_raw is None:
        return []

    is_atomized = token_type == TokenType.ATOMIZED
    if not is_atomized.any():
        return []

    is_indexed = idx_mask_raw.any(dim=1)
    atomized_unindexed = is_atomized & ~is_indexed
    if atomized_unindexed.any():
        count = int(atomized_unindexed.sum())
        return [
            InvariantViolation(
                check="atomized_must_be_indexed",
                message=f"{count} atomized token(s) are unindexed",
                count=count,
            )
        ]
    return []


def check_coordinates_not_nan(feats: dict, is_inference: bool = False) -> list[InvariantViolation]:
    """Return violations where coordinates feeding the model contain NaN/inf, or padding is non-zero.

    At training time, every valid atom slot (per ``atom37_mask``) must have finite coordinates.
    At inference time, the binder chain has not been generated yet so its ``coords`` may be NaN
    legitimately. Instead, only the atoms that are explicitly coordinate-conditioned —
    ``condition_coordinate_1_atom`` masked by ``mask_coordinate_1_atom`` — OR atomized (because
    chirality features need to be calculated from coordinate information) are checked, since
    those are the only coordinates that actually reach the model.
    The padding non-zero check is applied in both modes.
    """
    # lazy import to avoid circular dependency with pipeline.py
    from rfproteina.datapipes.pipeline import TokenType

    if "coords" not in feats or "atom37_mask" not in feats:
        return []

    coords: torch.Tensor = feats["coords"]  # [n_tokens, 37, 3]
    atom37_mask: torch.Tensor = feats["atom37_mask"]  # [n_tokens, 37]

    atomized_mask = feats["token_type"] == TokenType.ATOMIZED  # [n_tokens,]
    atomized_mask_atom37 = torch.zeros_like(atom37_mask, dtype=torch.bool)
    ATOMIZED_SLOT_IDX = next(i for i, a in enumerate(UNIFIED_ATOM37_ENCODING.token_atoms["<A>"]) if a.strip())
    atomized_mask_atom37[:, ATOMIZED_SLOT_IDX] = atomized_mask

    violations = []

    if is_inference:
        mask_to_check = (feats.get("mask_coordinate_1_atom") | atomized_mask_atom37) & atom37_mask
    else:
        mask_to_check = atom37_mask

    coords_to_check = coords[mask_to_check]  # [n_valid_atoms, 3]
    if torch.isnan(coords_to_check).any():
        violations.append(
            InvariantViolation(
                check="nan_in_valid_coords",
                message="NaN found in atom coordinates",
                count=1,
            )
        )
    if torch.isinf(coords_to_check).any():
        violations.append(
            InvariantViolation(
                check="inf_in_valid_coords",
                message="Inf found in atom coordinates",
                count=1,
            )
        )

    invalid_mask = ~atom37_mask
    if invalid_mask.any():
        n_nonzero = int((coords[invalid_mask] != 0.0).any(dim=-1).sum())
        if n_nonzero:
            violations.append(
                InvariantViolation(
                    check="nonzero_padding_coords",
                    message=f"{n_nonzero} invalid atom slot(s) have non-zero coordinates",
                    count=n_nonzero,
                )
            )

    return violations


def check_element_encoding_violations(feats: dict) -> list[InvariantViolation]:
    """Return violations where element encoding leaks info to non-seq-conditioned tokens."""
    if "element" not in feats or "atom37_mask" not in feats or "mask_sequence_1_residue" not in feats:
        return []

    element: torch.Tensor = feats["element"]  # [n_tokens, 37]
    atom37_mask: torch.Tensor = feats["atom37_mask"]  # [n_tokens, 37]
    seq_mask: torch.Tensor = feats["mask_sequence_1_residue"]  # [n_tokens]

    violations = []
    non_seq = ~seq_mask

    if non_seq.any():
        n_leaked = int((element[non_seq] != -1).sum())
        if n_leaked:
            violations.append(
                InvariantViolation(
                    check="element_leakage",
                    message=f"{n_leaked} atom slot(s) in non-seq-conditioned tokens expose element info",
                    count=n_leaked,
                )
            )

    if seq_mask.any():
        valid_in_seq = atom37_mask[seq_mask]
        element_in_seq = element[seq_mask]
        if valid_in_seq.any():
            n_missing = int((element_in_seq[valid_in_seq] < 0).sum())
            if n_missing:
                violations.append(
                    InvariantViolation(
                        check="missing_element_in_seq_conditioned",
                        message=f"{n_missing} valid atom(s) in seq-conditioned tokens lack element info",
                        count=n_missing,
                    )
                )

    return violations


def check_unresolved_tokens_not_lossed_violations(aa: AtomArray, feats: dict) -> list[InvariantViolation]:
    """Return violations where tokens with unresolved atoms have should_be_lossed=True."""
    if "should_be_lossed" not in feats:
        return []
    categories = set(aa.get_annotation_categories())
    if "token_id" not in categories or not hasattr(aa, "occupancy"):
        return []

    should_be_lossed = feats["should_be_lossed"].cpu().numpy()  # [n_tokens]
    token_id_per_atom = aa.token_id  # [n_atoms]
    is_resolved = aa.occupancy > 0.0  # [n_atoms]

    unique_tokens, token_indices = np.unique(token_id_per_atom, return_inverse=True)
    token_has_unresolved = np.zeros(len(unique_tokens), dtype=bool)
    np.logical_or.at(token_has_unresolved, token_indices, ~is_resolved)

    n_violated = int((token_has_unresolved & should_be_lossed).sum())
    if n_violated:
        return [
            InvariantViolation(
                check="unresolved_token_lossed",
                message=f"{n_violated} token(s) with unresolved atoms have should_be_lossed=True",
                count=n_violated,
            )
        ]
    return []


def check_hbond_no_both_violations(feats: dict) -> list[InvariantViolation]:
    """Return violations where a valid atom's hydrogen-bond annotation is ``HBondEnum.BOTH``.

    TEMPORARY: the model does not yet support the ``BOTH`` class.

    TODO: Remove this check once the training workflow is patched.
    """
    if C_HB.full_name not in feats or "atom37_mask" not in feats:
        return []

    hb: torch.Tensor = feats[C_HB.full_name]  # [n_tokens, 37]
    atom37_mask: torch.Tensor = feats["atom37_mask"]  # [n_tokens, 37]

    n_bad = int(((hb == HBondEnum.BOTH) & atom37_mask).sum())
    if n_bad:
        return [
            InvariantViolation(
                check="hbond_both_not_supported",
                message=(
                    f"{n_bad} valid atom(s) have HBondEnum.BOTH, which the model does not yet "
                    "support (TEMPORARY check -- will be removed once the training workflow is patched)"
                ),
                count=n_bad,
            )
        ]
    return []


def check_not_cysteine_sequence_conditioning_violations(aa: AtomArray) -> list[InvariantViolation]:
    """Return violations where not-cysteine and sequence conditioning co-occur on a residue."""
    aap = as_atom_array_plus(aa)
    ncys = C_NCYS.annotation(aap, default="generate")  # [n_atoms] bool
    if not ncys.any():
        return []

    both = ncys & C_SEQ.mask(aap, default="generate")  # [n_atoms] bool
    n_bad = int(both[get_residue_starts(aap)].sum())  # residues with both conditions
    if n_bad:
        return [
            InvariantViolation(
                check="not_cysteine_and_sequence_conditioned",
                message=f"{n_bad} residue(s) have both not-cysteine and sequence conditioning",
                count=n_bad,
            )
        ]
    return []


def check_idxgrp_violations(aa: AtomArray, feats: dict) -> list[InvariantViolation]:
    """
    Return violations where C_IDXGRP groups have duplicate or non-monotonically increasing within_index_group_res_idx.

    This test is mainly to ensure all index groups were split correctly. If what should be two separate groups are
    incorrectly classified as the same group, you would see either non-monotonically increasing res_ids or possible
    duplicates (see example below).

    Good grouping:
    idxgrp:                     0 0 0 0 0 0 0 0 1 1 1 1 1
    within_index_group_res_idx: 0 1 2 3 4 5 6 7 0 1 2 3 4

    Bad grouping:
    Error:                                      * * * * *
    idx_group:                  0 0 0 0 0 0 0 0 0 0 0 0 0
    within_index_group_res_idx: 0 1 2 3 4 5 6 7 0 1 2 3 4
    """
    categories = set(aa.get_annotation_categories())
    if C_IDXGRP.full_name not in feats or "within_index_group_res_idx" not in categories:
        return []

    token_starts = get_token_starts(aa)
    non_atomized_token_starts = token_starts[~S_ATM.mask(aa, default="generate")[token_starts]]
    idxgrp = C_IDXGRP.annotation(aa, default="generate")[non_atomized_token_starts]  # [n_tokens]
    within_res_idx = aa.within_index_group_res_idx[non_atomized_token_starts]  # [n_tokens]

    violations = []
    for gid in np.unique(idxgrp):
        group_mask = idxgrp == gid
        group_res_idxs = within_res_idx[group_mask]

        n_tokens = int(group_mask.sum())
        n_unique = int(np.unique(group_res_idxs).size)
        if n_unique != n_tokens:
            violations.append(
                InvariantViolation(
                    check="idxgrp_res_idx_duplicates",
                    message=f"Group {int(gid)} within_index_group_res_idx has {n_tokens - n_unique} \
                        duplicate(s) among {n_tokens} token(s)",
                    count=n_tokens - n_unique,
                )
            )

        if n_tokens > 1 and not np.all(np.diff(group_res_idxs) > 0):
            n_bad = int((np.diff(group_res_idxs) <= 0).sum())
            violations.append(
                InvariantViolation(
                    check="idxgrp_res_idx_not_monotonic",
                    message=f"Group {int(gid)} within_index_group_res_idx is not strictly \
                        increasing ({n_bad} violation(s))",
                    count=n_bad,
                )
            )

    return violations


def find_index_group_res_id_gaps(aa: AtomArray) -> list[str]:
    """Describe each internal res_id gap within a C_IDXGRP index group (atomized residues skipped).

    Returns one ``"index group <g>: res_id jumps from <a> to <b>"`` per gap; gaps on index-group
    boundaries are expected and not reported.
    """
    starts = get_residue_starts(aa)
    starts = starts[~S_ATM.mask(aa, default="generate")[starts]]
    index_groups = C_IDXGRP.annotation(aa, default="generate")
    messages: list[str] = []
    prev_group: int | None = None
    prev_res_id: int | None = None
    for start in starts:
        group = int(index_groups[start])
        res_id = int(aa.res_id[start])
        if group == prev_group and prev_res_id is not None and (res_id - prev_res_id) not in (0, 1):
            messages.append(f"index group {group}: res_id jumps from {prev_res_id} to {res_id}")
        prev_group, prev_res_id = group, res_id
    return messages


# ---------------------------------------------------------------------------
# Cross-cutting checks (require both aa and feats)
# ---------------------------------------------------------------------------


def check_coordinate_centering_violations(aa: AtomArray, feats: dict) -> list[InvariantViolation]:
    """Return violations where coordinates are not centered on coord-conditioned atoms."""
    aap = as_atom_array_plus(aa)
    if "coord_to_be_noised" not in aap.get_annotation_categories():
        return []

    crd_mask = C_CRD.mask(aap, default="generate")
    if not crd_mask.any():
        return []

    centroid_norm = float(np.linalg.norm(np.mean(aap.coord_to_be_noised[crd_mask], axis=0)))
    violations = []
    if centroid_norm >= 1e-3:
        violations.append(
            InvariantViolation(
                check="coordinates_not_centered",
                message=f"Atom-array centroid norm {centroid_norm:.2e} (threshold 1e-3)",
                count=1,
            )
        )

    if "coords" not in feats or "mask_coordinate_1_atom" not in feats or "atom37_mask" not in feats:
        return violations

    coords_token: torch.Tensor = feats["coords"]  # [n_tokens, 37, 3]
    crd_mask_token: torch.Tensor = feats["mask_coordinate_1_atom"]  # [n_tokens, 37]
    atom37_mask: torch.Tensor = feats["atom37_mask"]  # [n_tokens, 37]
    valid_and_conditioned = atom37_mask & crd_mask_token
    if valid_and_conditioned.any():
        feats_centroid_norm = float(coords_token[valid_and_conditioned].mean(dim=0).norm().item())
        if feats_centroid_norm >= 1e-3:
            violations.append(
                InvariantViolation(
                    check="feats_coordinates_not_centered",
                    message=f"Feats centroid norm {feats_centroid_norm:.2e} (threshold 1e-3)",
                    count=1,
                )
            )

    return violations


def check_sequence_conditioning_violations(feats: dict, is_inference: bool) -> list[InvariantViolation]:
    """Return violations where residue_type encoding is inconsistent with conditioning masks."""
    required = {"mask_sequence_1_residue", "residue_type", "atom37_mask"}
    if not required.issubset(feats):
        return []

    # Lazy import to avoid circular dependency with pipeline.py
    from rfproteina.datapipes.pipeline import TokenType

    seq_mask: torch.Tensor = feats["mask_sequence_1_residue"]  # [n_tokens]
    res_type_gt: torch.Tensor = feats["residue_type"]  # [n_tokens]
    token_type: torch.Tensor = feats["token_type"]  # [n_tokens]

    mask_token_idx = UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]
    atomized_token_idx = UNIFIED_ATOM37_ENCODING.token_to_idx["<A>"]
    max_valid_idx = UNIFIED_ATOM37_ENCODING.n_tokens

    violations = []

    if (res_type_gt < 0).any() or (res_type_gt >= max_valid_idx).any():
        violations.append(
            InvariantViolation(
                check="residue_type_out_of_range",
                message="Ground-truth residue_type contains out-of-range indices",
                count=1,
            )
        )

    # This check only applies in training, since residue_type will be passed through the encoder
    can_be_encoded = feats["can_be_encoded"]
    if not is_inference and (can_be_encoded & (res_type_gt == mask_token_idx)).any():
        violations.append(
            InvariantViolation(
                check="residue_type_contains_mask_token",
                message="Ground-truth residue_type contains <M> (mask) token",
                count=int((res_type_gt == mask_token_idx).sum()),
            )
        )

    cond_seq_key = "condition_sequence_1_residue"
    if cond_seq_key in feats:
        res_type_cond: torch.Tensor = feats[cond_seq_key]

        if seq_mask.any():
            mismatch = (res_type_gt[seq_mask] != res_type_cond[seq_mask]).sum()
            if mismatch:
                violations.append(
                    InvariantViolation(
                        check="seq_conditioned_residue_type_mismatch",
                        message=f"{int(mismatch)} seq-conditioned token(s): condition_sequence != residue_type",
                        count=int(mismatch),
                    )
                )

        not_seq = ~seq_mask
        if not_seq.any():
            n_not_mask = int((res_type_cond[not_seq] != mask_token_idx).sum())
            if n_not_mask:
                violations.append(
                    InvariantViolation(
                        check="non_seq_conditioned_not_masked",
                        message=f"{n_not_mask} non-seq-conditioned token(s) have condition_sequence != <M>",
                        count=n_not_mask,
                    )
                )

        is_atomized = token_type == TokenType.ATOMIZED
        if is_atomized.any():
            n_not_a = int((res_type_cond[is_atomized] != atomized_token_idx).sum())
            if n_not_a:
                violations.append(
                    InvariantViolation(
                        check="atomized_condition_sequence_not_A",
                        message=f"{n_not_a} atomized token(s) have condition_sequence != <A>",
                        count=n_not_a,
                    )
                )

    return violations


def check_can_be_encoded_violations(aa: AtomArray, feats: dict) -> list[InvariantViolation]:
    """Return violations where can_be_encoded doesn't match the set of encodable residues."""
    if "can_be_encoded" not in feats:
        return []

    # Lazy import to avoid circular dependency with pipeline.py
    from rfproteina.datapipes.pipeline import RESIDUE_NAMES_TO_ENCODE

    token_starts = get_token_starts(aa)
    res_names = aa.res_name[token_starts]
    categories = set(aa.get_annotation_categories())

    is_atomized = aa.atomize[token_starts] if "atomize" in categories else np.zeros(len(token_starts), dtype=bool)
    is_encodable = np.array([n in RESIDUE_NAMES_TO_ENCODE for n in res_names]) & ~is_atomized

    can_be_encoded = feats["can_be_encoded"].cpu().numpy()
    violations = []

    n_invalid = int((can_be_encoded & ~is_encodable).sum())
    if n_invalid:
        violations.append(
            InvariantViolation(
                check="can_be_encoded_ineligible",
                message=f"{n_invalid} token(s) have can_be_encoded=True but are not eligible",
                count=n_invalid,
            )
        )

    n_missing = int((~can_be_encoded & is_encodable).sum())
    if n_missing:
        violations.append(
            InvariantViolation(
                check="can_be_encoded_missing",
                message=f"{n_missing} eligible token(s) have can_be_encoded=False",
                count=n_missing,
            )
        )

    return violations


def check_no_oxt_on_encodable_tokens(aa: AtomArray, feats: dict) -> list[InvariantViolation]:
    """Return violations where OXT atoms appear on ``can_be_encoded`` non-atomized tokens.

    The autoencoder's restype mask excludes OXT, so any OXT atom on an encodable
    token creates an asymmetry between the reconstruction-loss target and the
    decoder's advertised output mask. Such atoms must be stripped upstream.
    """

    categories = set(aa.get_annotation_categories())

    can_be_encoded = feats["can_be_encoded"].cpu().numpy()  # [n_tokens]
    token_id_per_atom = aa.token_id  # [n_atoms]
    is_atomized = aa.atomize if "atomize" in categories else np.zeros(len(aa), dtype=bool)
    is_oxt = aa.atom_name == "OXT"  # [n_atoms]

    encodable_per_atom = can_be_encoded[token_id_per_atom]  # [n_atoms]
    bad = is_oxt & encodable_per_atom & ~is_atomized
    n_bad = int(bad.sum())
    if n_bad:
        return [
            InvariantViolation(
                check="oxt_on_encodable_token",
                message=f"{n_bad} OXT atom(s) found on can_be_encoded non-atomized token(s)",
                count=n_bad,
            )
        ]
    return []


def check_token_features_consistent_violations(aa: AtomArray) -> list[InvariantViolation]:
    """Return violations where token-level features have inconsistent values within a token."""
    # Lazy import to avoid circular dependency with pipeline.py
    from rfproteina.datapipes.pipeline import ATOM_LEVEL_FEATURES_TO_CONVERT_TO_TOKEN_LEVEL

    categories = set(aa.get_annotation_categories())
    if "token_id" not in categories:
        return []

    token_starts = get_token_starts(aa, add_exclusive_stop=False)  # [n_tokens]
    n_atoms = aa.array_length()
    atom_token_idx = aa.token_id  # [n_atoms], values 0..n_tokens-1

    violations = []
    for feat_name in ATOM_LEVEL_FEATURES_TO_CONVERT_TO_TOKEN_LEVEL:
        if feat_name not in categories:
            continue
        feat_values = aa.get_annotation(feat_name)

        first = feat_values[token_starts][atom_token_idx]  # token-first value per atom
        differs = feat_values != first
        if np.issubdtype(feat_values.dtype, np.floating):
            # NaN == NaN (all-NaN token is consistent); NaN != any real value.
            differs &= ~(np.isnan(feat_values) & np.isnan(first))
        atom_differs = differs.reshape(n_atoms, -1).any(axis=1)  # [n_atoms]

        # A token is bad if any of its atoms differ from the token's first atom.
        n_bad = int((np.add.reduceat(atom_differs.astype(np.int64), token_starts) > 0).sum())
        if n_bad:
            violations.append(
                InvariantViolation(
                    check="token_feature_inconsistent",
                    message=f"'{feat_name}' has inconsistent values in {n_bad} token(s)",
                    count=n_bad,
                )
            )

    return violations


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class ValidateBatchInvariants(Transform):
    """Validate batch-level invariants at the end of the pipeline.

    Calls :py:func:`detect_batch_invariant_violations` on every example and
    handles violations according to ``on_violation``.

    Args:
        on_violation: Action taken when violations are found.
            ``"raise"`` (default) - raise :py:exc:`ValueError`.
            ``"warn"`` - emit a warning via :mod:`logging` and continue.
            ``"ignore"`` - silently continue (useful when used as a probe).
    """

    def __init__(self, is_inference: bool, on_violation: str = "raise"):
        super().__init__()
        if on_violation not in ("raise", "warn", "ignore"):
            raise ValueError(f"on_violation must be 'raise', 'warn', or 'ignore', got {on_violation!r}")
        self.on_violation = on_violation
        self.is_inference = is_inference

    def forward(self, data: dict) -> dict:
        violations = detect_batch_invariant_violations(data, is_inference=self.is_inference)
        if not violations or self.on_violation == "ignore":
            return data

        example_id = data.get("example_id", "unknown")
        lines = "\n".join(f"  [{v.check}] {v.message}" for v in violations)
        message = f"Batch invariant violation(s) for example '{example_id}':\n{lines}"

        if self.on_violation == "raise":
            raise ValueError(message)
        else:
            logger.warning(message)

        return data
