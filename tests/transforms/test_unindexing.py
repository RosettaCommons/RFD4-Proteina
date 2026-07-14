"""Tests for UnindexFlaggedTokens transform."""

import copy

import numpy as np
import pytest
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.testing import assert_same_atom_array_or_stack
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from atomworks.ml.transforms.base import Compose
from atomworks.ml.utils.testing import cached_parse
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray

from rfproteina.conditions import C_IDXGRP
from rfproteina.datapipes.pipeline import _build_design_clean_and_sample_pipeline
from rfproteina.transforms.unindexing import UnindexFlaggedTokens
from rfproteina.transforms.utils.conditioning_utils import sample_unindexed_groups

_N_UNINDEXED_TOKENS = 6

# Conditions compared by partition equivalence (same grouping) rather than identical integer values.
# Their per-value binary annotation categories are excluded from assert_same_atom_array_or_stack
# and checked separately via the pair-mask comparison below.
_PAIR_MASK_CONDITIONS = [C_IDXGRP]

# ID annotations always excluded from the standard check (globally-unique counters that shift
# between training and inference). Checked relative to their minimum when check_ids=True.
_ID_ANNOTATIONS = ["res_id", "atom_id", "token_id"]

# Annotations that UnindexFlaggedTokens *adds* to its output (see unindexing.py). They are
# absent from the pre-transform snapshot by construction, so they are never part of a
# snapshot-vs-output comparison.
_UNINDEXING_ADDED_ANNOTATIONS = {"insertion_source_idx", "is_unindexed_source"}


def assert_same_atom_array_with_group_equivalence(
    aa1: AtomArray, aa2: AtomArray, check_ids: bool = False, exclude_annots: set[str] = None
) -> None:
    """Assert aa1 == aa2, comparing group conditions by partition equivalence rather than raw values.

    Annotations for conditions in _PAIR_MASK_CONDITIONS are excluded from the standard
    per-value comparison and instead checked via pair masks (i.e. only the partition —
    which atoms share a group — must match, not the exact integer labels).

    ID annotations in _ID_ANNOTATIONS are always excluded from the standard check.
    When check_ids=True they are additionally verified to have the same relative values
    (each array shifted by its own minimum).
    """
    all_cats = set(aa1.get_annotation_categories()) | set(aa2.get_annotation_categories())

    excluded = set(exclude_annots or set())
    # Annotations added by the transform are never in the pre-transform snapshot.
    excluded.update(_UNINDEXING_ADDED_ANNOTATIONS)
    if not check_ids:
        excluded.update(_ID_ANNOTATIONS)
        excluded.update(["within_chain_source_res_idx", "within_index_group_res_idx"])
    for cond in _PAIR_MASK_CONDITIONS:
        prefix = f"condition_{cond.name}_"
        excluded.update(cat for cat in all_cats if cat.startswith(prefix))

    assert_same_atom_array_or_stack(aa1, aa2, annotations_to_compare=sorted(all_cats - excluded), compare_bonds=False)

    for cond in _PAIR_MASK_CONDITIONS:
        ann1 = cond.annotation(aa1, default="generate")
        ann2 = cond.annotation(aa2, default="generate")
        pair_mask_1 = ann1[:, None] == ann1[None, :]
        pair_mask_2 = ann2[:, None] == ann2[None, :]
        assert np.array_equal(pair_mask_1, pair_mask_2), f"Condition {cond.name} groupings differ between arrays"

    if check_ids:
        present = all_cats & set(_ID_ANNOTATIONS)
        for name in sorted(present):
            ann1 = aa1.get_annotation(name)
            ann2 = aa2.get_annotation(name)
            rel1 = ann1 - ann1.min()
            rel2 = ann2 - ann2.min()
            if not np.array_equal(rel1, rel2):
                mismatch_idxs = np.where(rel1 != rel2)[0]
                msg = f"ID annotation '{name}' relative values differ at {len(mismatch_idxs)} atoms. First few:\n"
                for idx in mismatch_idxs[:5]:
                    msg += f"\t{idx}: {ann1[idx]} vs {ann2[idx]}\n"
                raise AssertionError(msg)


_cif_parser_args = STANDARD_PARSER_ARGS | {"hydrogen_policy": "remove"}


def _make_unindexed_data(pdb_id: str, seed: int = 42, p_split_group: float = 0.0):
    """Build a data dict with 6 manually-unindexed protein tokens and dense random conditions.

    Returns:
        Tuple of (snapshot_aa, data_train, data_infer) where snapshot_aa is the atom array
        immediately before UnindexFlaggedTokens is applied, data_train and data_infer are
        data dicts ready to be passed to UnindexFlaggedTokens.
    """
    raw = cached_parse(pdb_id, **_cif_parser_args)
    pipeline = Compose(
        _build_design_clean_and_sample_pipeline(is_inference=True, design_tasks={}, p_random_rotation=0.0)
    )
    data = pipeline(raw)
    aa = data["atom_array"]

    # Find the first _N_UNINDEXED_TOKENS protein tokens to unindex
    ensure_annotations(aa, "is_protein", "within_chain_source_res_idx")
    token_starts = get_token_starts(aa, add_exclusive_stop=True)
    is_protein_token = aa.is_protein[token_starts[:-1]]
    protein_token_positions = np.where(is_protein_token)[0]

    # Build atom-level mask for the atoms belonging to those tokens
    atoms_to_unindex = np.zeros(aa.array_length(), dtype=bool)
    for t_pos in protein_token_positions[:_N_UNINDEXED_TOKENS]:
        atoms_to_unindex[token_starts[t_pos] : token_starts[t_pos + 1]] = True

    # Set C_IDX = False for the chosen atoms
    idx_mask = C_IDX.mask(aa, default="generate").copy()
    idx_mask[atoms_to_unindex] = False
    C_IDX.set_mask(aa, idx_mask)

    # Set C_IDXGRP via sample_unindexed_groups (handles indexed + single contiguous unindexed block)
    C_IDXGRP.set_annotation(aa, sample_unindexed_groups(aa, p_split_group=p_split_group))

    # Apply dense random conditioning masks so we test condition preservation thoroughly
    rng = np.random.default_rng(seed)
    n_atoms = aa.array_length()
    C_CRD.set_mask(aa, rng.random(n_atoms) < 0.7)
    C_SEQ.set_mask(aa, rng.random(n_atoms) < 0.7)

    aa.set_annotation("within_index_group_res_idx", aa.within_chain_source_res_idx.copy())

    # Reset indices to zero to avoid any bias.
    for index_group in np.unique(C_IDXGRP.annotation(aa, default="generate")):
        group_mask = C_IDXGRP.annotation(aa, default="generate") == index_group

        # We leave the residue indices unchanged for globally indexed groups
        is_indexed = C_IDX.mask(aa[group_mask], default="raise")
        if (is_indexed != is_indexed[0]).any():
            raise ValueError(f"Index group {index_group} contains a mix of indexed and unindexed atoms.")
        if is_indexed[0]:
            continue

        aa.within_index_group_res_idx[group_mask] -= aa.within_index_group_res_idx[group_mask].min()

    snapshot = copy.deepcopy(aa)

    data_train = {**data, "atom_array": copy.deepcopy(aa), "is_inference": False}
    data_infer = {**data, "atom_array": copy.deepcopy(aa), "is_inference": True}

    return snapshot, data_train, data_infer


@pytest.mark.parametrize("p_split_group", [0.0, 1.0])
@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm", "7rxs"])
def test_indexed_region_identical_across_modes(pdb_id: str, p_split_group: float) -> None:
    """
    Truly indexed atoms are unchanged in both training and inference modes.
    NOTE: In training mode we expect the original atoms left behind to have stripped annotations,
    however in this test we do not apply any DesignTask so this behavior should not happen.
    Therefore the atoms should be 100% identical to before unindexing.
    """
    snapshot, data_train, data_infer = _make_unindexed_data(pdb_id, p_split_group=p_split_group)
    transform = UnindexFlaggedTokens()
    out_train = transform(data_train)["atom_array"]
    out_infer = transform(data_infer)["atom_array"]

    n_orig = snapshot.array_length()
    idx_mask = C_IDX.mask(snapshot, default="generate")  # [n_orig]
    orig_indexed = snapshot[idx_mask]

    # In train mode the original atoms are kept in place; take first n_orig and select indexed
    train_indexed = out_train[:n_orig][idx_mask]
    assert_same_atom_array_with_group_equivalence(orig_indexed, train_indexed, check_ids=True)

    # In infer mode unindexed atoms are removed; indexed atoms occupy the front of the array
    n_indexed = int(idx_mask.sum())
    infer_indexed = out_infer[:n_indexed]
    assert_same_atom_array_with_group_equivalence(orig_indexed, infer_indexed, check_ids=False)


@pytest.mark.parametrize("p_split_group", [0.0, 1.0])
@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm", "7rxs"])
def test_appended_tokens_match_original_unindexed_atoms(pdb_id: str, p_split_group: float) -> None:
    """Copies appended to the end of the array preserve coordinates and conditions from the original."""
    snapshot, data_train, data_infer = _make_unindexed_data(pdb_id, p_split_group=p_split_group)
    transform = UnindexFlaggedTokens()
    out_train = transform(data_train)["atom_array"]
    out_infer = transform(data_infer)["atom_array"]

    idx_mask = C_IDX.mask(snapshot, default="generate")
    orig_unindexed = snapshot[~idx_mask]
    n_unindexed = orig_unindexed.array_length()

    for out in [out_train, out_infer]:
        # Appended copies are the last n_unindexed atoms
        appended = out[-n_unindexed:]
        assert_same_atom_array_with_group_equivalence(orig_unindexed, appended)


@pytest.mark.parametrize("p_split_group", [0.0, 1.0])
@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm", "7rxs"])
def test_within_index_group_res_idx_per_unindexed_group(pdb_id: str, p_split_group: float) -> None:
    """Each unindexed C_IDXGRP group has within_index_group_res_idx starting at 0 with no duplicates."""
    _, data_train, _ = _make_unindexed_data(pdb_id, p_split_group=p_split_group)
    transform = UnindexFlaggedTokens()
    out = transform(data_train)["atom_array"]

    token_starts = get_token_starts(out)
    idxgrp = C_IDXGRP.annotation(out, default="generate")[token_starts]  # token-level
    within_res_idx = out.within_index_group_res_idx[token_starts]  # token-level
    is_indexed = C_IDX.mask(out, default="generate")[token_starts]  # token-level

    for gid in np.unique(idxgrp):
        group_mask = idxgrp == gid
        if is_indexed[group_mask].all():
            continue  # indexed groups are not required to start at 0

        group_res_idxs = within_res_idx[group_mask]
        assert (
            group_res_idxs.min() == 0
        ), f"Group {gid} within_index_group_res_idx min={group_res_idxs.min()}, expected 0"
        assert len(np.unique(group_res_idxs)) == len(
            group_res_idxs
        ), f"Group {gid} within_index_group_res_idx has duplicates: {group_res_idxs}"
