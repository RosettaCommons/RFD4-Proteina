"""Tests for UnindexFlaggedTokens transform."""

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zstandard
from atomworks.io.config import ParseConfig
from atomworks.io.parser import STANDARD_PARSER_ARGS, parse
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.query import QueryExpression
from atomworks.io.utils.testing import assert_same_atom_array_or_stack
from atomworks.ml.conditions import C_CRD, C_CTR, C_IDX, C_NTR, C_SEQ
from atomworks.ml.conditions.base import ConditionConflictError
from atomworks.ml.transforms.atom_array import add_global_token_id_annotation
from atomworks.ml.transforms.base import Compose
from atomworks.ml.utils.testing import cached_parse
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray

from rfproteina.conditions import C_CRDGRP, C_IDXGRP, C_INTCRD
from rfproteina.datapipes.pipeline import _build_design_clean_and_sample_pipeline
from rfproteina.transforms.internal_coords import MaterializeInternalCoords
from rfproteina.transforms.unindexing import (
    UnindexFlaggedTokens,
    annotate_unindexed_insertion,
)
from rfproteina.transforms.utils.conditioning_utils import sample_unindexed_groups
from rfproteina.utils.standard_annotations import SA_CWI, SA_IDXGRPSRC, SA_UII

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
_UNINDEXING_ADDED_ANNOTATIONS = {
    "insertion_source_idx",
    "is_unindexed_source",
}


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
        excluded.update(["within_chain_source_res_idx"])
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


@pytest.mark.parametrize(
    "internal",
    [[True, False, False], [True, True, False], [False, True, False]],
    ids=["indexed", "mixed", "unindexed"],
)
@pytest.mark.parametrize("storage", ["canonical", "alias", "alias_and_canonical"])
def test_training_unindexing_preserves_internal_targets(internal: list[bool], storage: str) -> None:
    aa = as_atom_array_plus(AtomArray(3))
    aa.chain_id[:] = "A"
    aa.res_id = np.arange(1, 4)
    aa.res_name[:] = "ALA"
    aa.atom_name[:] = "CA"
    aa.element[:] = "C"
    aa.coord = np.arange(9, dtype=np.float32).reshape(3, 3)
    aa.set_annotation("chain_iid", np.full(3, "A_1"))
    aa.set_annotation("molecule_iid", np.zeros(3, dtype=int))
    aa.set_annotation("is_polymer", np.ones(3, dtype=bool))
    aa.set_annotation("atomize", np.zeros(3, dtype=bool))
    aa.set_annotation("within_chain_source_res_idx", np.arange(3))
    add_global_token_id_annotation(aa)
    C_IDX.set_annotation(aa, np.array([True, False, True]))
    C_IDXGRP.set_annotation(aa, C_IDXGRP.default_annotation(aa))
    internal = np.asarray(internal)
    C_CRD.set_annotation_from_ground_truth(aa, ~internal)
    groups = np.where(internal, 1, C_CRDGRP.KNOWN_ABSOLUTE_COORDS)
    C_CRDGRP.set_annotation(aa, groups)
    targets = np.where(internal[:, None], aa.coord + 17, np.nan)
    C_INTCRD.set_annotation(aa, targets)
    conditions = (C_IDX, C_IDXGRP, C_CRD, C_CRDGRP, C_INTCRD)
    if storage != "canonical":
        for condition in conditions:
            aa.set_annotation(condition.aliases[0], condition.annotation(aa).copy())
            if storage == "alias":
                aa.del_annotation(condition.full_name)

    expected_groups = groups[[0, 1, 2, 1]].copy()
    expected_groups[1] = C_CRDGRP.UNKNOWN_COORDS
    expected_targets = targets[[0, 1, 2, 1]].copy()
    expected_targets[1] = np.nan

    pipeline = Compose([MaterializeInternalCoords(), UnindexFlaggedTokens()])
    out = pipeline({"atom_array": aa, "is_inference": False})["atom_array"]

    np.testing.assert_array_equal(C_CRDGRP.annotation(out), expected_groups)
    np.testing.assert_array_equal(C_INTCRD.annotation(out), expected_targets)
    np.testing.assert_array_equal(C_IDX.annotation(out), [True, True, True, False])
    for condition in conditions:
        for alias in condition.aliases:
            assert alias not in out.get_annotation_categories()
            np.testing.assert_array_equal(QueryExpression(f"sa_mask('{alias}')").mask(out), condition.mask(out))


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
    C_IDX.set_annotation(aa, idx_mask)

    # Set C_IDXGRP via sample_unindexed_groups (handles indexed + single contiguous unindexed block)
    C_IDXGRP.set_annotation(aa, sample_unindexed_groups(aa, p_split_group=p_split_group))

    # Apply dense random conditioning masks so we test condition preservation thoroughly
    rng = np.random.default_rng(seed)
    n_atoms = aa.array_length()
    coord_mask = rng.random(n_atoms) < 0.7
    C_CRD.set_annotation(
        aa,
        np.where(coord_mask[:, None], C_CRD.ground_truth_value(aa), C_CRD.default_annotation(aa)),
    )
    sequence_mask = rng.random(n_atoms) < 0.7
    C_SEQ.set_annotation(
        aa,
        np.where(sequence_mask, C_SEQ.ground_truth_value(aa), C_SEQ.default_annotation(aa)),
    )

    C_NTR.set_annotation(aa, C_NTR.annotation(aa, default="generate"))
    C_CTR.set_annotation(aa, C_CTR.annotation(aa, default="generate"))

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
    assert np.array_equal(
        C_NTR.mask(train_indexed, default="raise"), C_NTR.mask(snapshot.copy(), default="generate")[idx_mask]
    )
    assert np.array_equal(
        C_CTR.mask(train_indexed, default="raise"), C_CTR.mask(snapshot.copy(), default="generate")[idx_mask]
    )

    # In infer mode unindexed atoms are removed; indexed atoms occupy the front of the array
    n_indexed = int(idx_mask.sum())
    infer_indexed = out_infer[:n_indexed]
    assert_same_atom_array_with_group_equivalence(orig_indexed, infer_indexed, check_ids=False)
    assert np.array_equal(
        C_NTR.mask(infer_indexed, default="raise"), C_NTR.mask(snapshot.copy(), default="generate")[idx_mask]
    )
    assert np.array_equal(
        C_CTR.mask(infer_indexed, default="raise"), C_CTR.mask(snapshot.copy(), default="generate")[idx_mask]
    )


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
        assert_same_atom_array_with_group_equivalence(
            orig_unindexed,
            appended,
            exclude_annots={C_NTR.full_name, C_CTR.full_name},
        )
        assert not C_NTR.mask(appended, default="generate").any()
        assert not C_CTR.mask(appended, default="generate").any()


# Captured array_to_annotate (tests/data/annotate_unindexed_input.cif.zst): generated tokens plus two
# appended unindexed motifs whose LIVE coord/res_name are corrupted, so only their frozen annotations are true.
_ANNOTATE_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "annotate_unindexed_input.cif.zst"
_ANNOTATE_MAPPING = {53: 25, 54: 26, 55: 27, 56: 28, 57: 8, 58: 5, 59: 4}
# (target_token_id, expected live coord group, expected source index group) per mapped motif token.
_ABS_TARGETS = [(t, C_CRDGRP.KNOWN_ABSOLUTE_COORDS, 5) for t in (25, 26, 27, 28)]
_INT_TARGETS = [(t, 3, 6) for t in (8, 5, 4)]


def _load_annotate_fixture() -> AtomArray:
    """Load the captured array_to_annotate fixture with a deterministic per-token ``token_id``."""
    with tempfile.TemporaryDirectory() as td:
        cif = Path(td) / "f.cif"
        cif.write_bytes(zstandard.ZstdDecompressor().decompress(_ANNOTATE_FIXTURE.read_bytes()))
        aa = parse(
            str(cif),
            config=ParseConfig.from_preset(
                "rcsb",
                load_standard_annotations=True,
                return_atom_array_plus=True,
                add_missing_atoms=False,
                convert_mse_to_met=False,
            ),
        )["asym_unit"][0]
    aa = as_atom_array_plus(aa)
    # token_id is not stored in the CIF; assign it exactly as the pipeline does (per-token arange).
    return add_global_token_id_annotation(aa)


def test_annotate_unindexed_insertion_reapplies_conditioning() -> None:
    """Each mapped target carries the motif's FROZEN conditioning (not its corrupted live coord/res_name):
    the coord group is copied live, the source index group recorded, and the token flips to indexed with
    its motif reference coords staged."""
    aa = _load_annotate_fixture()
    out = annotate_unindexed_insertion(aa, token_mapping=_ANNOTATE_MAPPING)

    idx = C_IDX.mask(out, default="generate")
    uii = SA_UII.mask(out)
    crdgrp = C_CRDGRP.annotation(out, default="generate")
    idxgrpsrc = SA_IDXGRPSRC.annotation(out)
    seq = C_SEQ.annotation(out, default="generate")
    cwi = SA_CWI.annotation(out)

    for tid, want_group, want_src_index_group in _ABS_TARGETS + _INT_TARGETS:
        m = out.token_id == tid
        assert idx[m].all(), f"target {tid} not flipped to indexed"
        assert uii[m].all(), f"target {tid} not flagged as an unindexed insertion"
        assert (crdgrp[m] == want_group).all(), f"target {tid} live coord group {crdgrp[m]} != {want_group}"
        assert (idxgrpsrc[m] == want_src_index_group).all(), f"target {tid} src index group != {want_src_index_group}"
        assert (seq[m] != "GLY").all(), f"target {tid} took the live (corrupted) res_name, not the frozen seq"
        assert np.isfinite(cwi[m]).all(), f"target {tid} has no staged motif reference coords"

    # Sequence comes from the frozen annotation: each target gets its motif residue, not the corrupted GLY.
    for tid, motif_res in ((25, "THR"), (26, "GLN"), (27, "ALA"), (8, "SER"), (5, "ARG"), (4, "TRP")):
        assert seq[out.token_id == tid][0] == motif_res

    for source_tid, target_tid in _ANNOTATE_MAPPING.items():
        source = aa[aa.token_id == source_tid]
        if C_CRDGRP.is_internal(source).any():
            target = out[out.token_id == target_tid]
            np.testing.assert_array_equal(C_INTCRD.annotation(target), C_INTCRD.annotation(source))


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_unindexing_preserves_source_condition_annotations(pdb_id: str) -> None:
    """Unindexed sources retain the populated C_CRD/C_SEQ annotations used during insertion transfer."""
    _snapshot, _train, data_infer = _make_unindexed_data(pdb_id)
    aa = UnindexFlaggedTokens()(data_infer)["atom_array"]

    unindexed = ~C_IDX.mask(aa, default="generate")
    crd_masked = C_CRD.mask(aa, default="generate") & unindexed
    seq_masked = C_SEQ.mask(aa, default="generate") & unindexed
    assert crd_masked.any() and seq_masked.any(), "expected coord/seq-conditioned unindexed atoms"
    # Equality proves the annotation remains populated rather than being replaced by its empty default.
    assert np.allclose(C_CRD.annotation(aa, default="generate")[crd_masked], aa.coord[crd_masked], equal_nan=True)
    assert (C_SEQ.annotation(aa, default="generate")[seq_masked] == aa.res_name[seq_masked]).all()


def test_annotate_unindexed_insertion_raises_on_conditioning_conflict() -> None:
    """When a mapped target is already conditioned with a different value, the transfer raises loudly."""
    aa = _load_annotate_fixture()
    # Pre-condition target token 25 (the THR motif token 53 maps here) with a different sequence identity,
    # so transferring the motif's frozen THR onto an already-ALA-conditioned target contradicts.
    target = aa.token_id == 25
    seq_ann = C_SEQ.annotation(aa, default="generate").copy()
    seq_ann[target] = "ALA"
    C_SEQ.set_annotation(aa, seq_ann)
    with pytest.raises(ConditionConflictError):
        annotate_unindexed_insertion(aa, token_mapping=_ANNOTATE_MAPPING)
