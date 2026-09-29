"""Golden-value tests for token bond features across ``res_id`` gaps at inference.

Training recovers the hop count across a discontinuity by running the BFS on the pre-crop
bond graph. Inference has no pre-crop graph, so ``compute_token_bond_features`` builds one by
bonding shut the ``res_id`` gaps of each IndexGroup (``fill_res_id_gaps=True``). These tests
pin the resulting values for both cases the fill has to cover: a ``res_id`` gap inside an
indexed chain, and an unindexed motif whose residue copies
:py:class:`~rfproteina.transforms.unindexing.UnindexFlaggedTokens` slices apart.
"""

import rootutils

from rfproteina.utils.env import load_dotenv

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv(override=True)

import logging

import numpy as np
import pytest
from atomworks.io.tools.inference import sequence_to_annotated_atom_array
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.conditions import C_IDX
from atomworks.ml.utils.annotator import ensure_annotations
from atomworks.ml.utils.expandable_segment import insert_expandable_segment, realize_expandable_segments
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray

from rfproteina.conditions import C_IDXGRP
from rfproteina.datapipes.bonds import compute_token_bond_features
from rfproteina.transforms.invariants import check_index_group_molecule_connectivity_violations
from rfproteina.transforms.unindexing import UnindexFlaggedTokens


def build_poly_gly_chain(res_ids: list[int]) -> AtomArray:
    """Build a poly-GLY chain from the CCD, keeping only ``res_ids``.

    Residues absent from ``res_ids`` are dropped along with the peptide bonds that joined
    them, so a non-consecutive pair leaves both a numeric ``res_id`` gap and a real break in
    the bond graph -- what an inference input with a gap actually looks like.
    """
    aa = sequence_to_annotated_atom_array(["GLY"] * max(res_ids), chain_id="A", chain_type="polypeptide(L)")
    aa = as_atom_array_plus(aa[np.isin(aa.res_id, res_ids)])

    # CCD templates come back unresolved (NaN coords), and the bond-distance filter drops any
    # bond it cannot measure. Only connectivity is under test here, so any finite value will do.
    aa.coord = np.zeros_like(aa.coord)

    # Pipeline annotations that no lazy generator provides.
    n_atoms = aa.array_length()
    aa.set_annotation("chain_iid", np.full(n_atoms, "A_1"))
    aa.set_annotation("atomize", np.zeros(n_atoms, dtype=bool))
    aa.set_annotation("insertion_source_idx", np.full(n_atoms, -1, dtype=np.int32))
    # atomworks assigns molecule ids by connecting whole pn_units, so a chain is one molecule
    # however its res_ids run and whatever breaks its bond graph has.
    aa.set_annotation("molecule_iid", np.full(n_atoms, "A_0"))

    ensure_annotations(aa, "token_id", "within_chain_source_res_idx")
    return aa


def test_indexed_res_id_gap_reads_as_the_gap_size():
    """A ``res_id`` gap inside one chain must read as its own size, not as "unrelated"."""
    res_ids = [1, 2, 3, 7, 8, 9]  # a three-residue hole between res 3 and res 7
    aa = build_poly_gly_chain(res_ids)
    C_IDXGRP.set_annotation(aa, np.zeros(aa.array_length(), dtype=np.int32))

    # Unfilled the two segments are simply disconnected: 0, the same value used for
    # cross-IndexGroup pairs and the self-diagonal.
    unfilled = compute_token_bond_features(aa, fill_res_id_gaps=False)
    assert unfilled["token_bond_path_length"][2, 3] == 0

    feats = compute_token_bond_features(aa, fill_res_id_gaps=True)

    # Every pair now reads as its res_id separation, within and across the gap.
    expected = np.abs(np.subtract.outer(res_ids, res_ids)).astype(np.int16)
    np.testing.assert_array_equal(feats["token_bond_path_length"], expected)

    # The fill only bonds tokens the input implies are adjacent: a gap of four is bridged through
    # filler residues, so residues 3 and 7 stay unbonded and their canonical ordering stays 0.
    # fmt: off
    np.testing.assert_array_equal(
        feats["canonical_polymer_ordering"],
        np.array([
            [0, -1,  0,  0,  0,  0],
            [1,  0, -1,  0,  0,  0],
            [0,  1,  0,  0,  0,  0],
            [0,  0,  0,  0, -1,  0],
            [0,  0,  0,  1,  0, -1],
            [0,  0,  0,  0,  1,  0],
        ], dtype=np.int8),
    )
    # fmt: on
    assert not feats["bond_mask"].any()  # poly-poly bonds are suppressed, inferred or not


@pytest.mark.parametrize(
    "design_res_ids, unindexed_res_ids, expected_path_lengths",
    [
        # Contiguous motif
        ([1, 2, 3, 4, 5, 6], [4, 5, 6], [[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
        # Motif carrying a two-residue hole
        ([1, 2, 3, 4, 5, 8, 9], [4, 5, 8], [[0, 1, 4], [1, 0, 3], [4, 3, 0]]),
        # Interior motif in an unbroken chain, with indexed residues flanking it
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6], [[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
    ],
)
def test_unindexed_motif_spacing_matches_training(design_res_ids, unindexed_res_ids, expected_path_lengths):
    """Both what the excision takes and what it leaves must read the same as they do in training.

    Both arms run on the same cropped chain. In training each copy points back at an indexed
    source token and inherits its path lengths, and those are read off the pre-crop bond graph the
    crop kept a handle on, so a hole is still crossed through the residues the crop removed. At
    inference there is no pre-crop graph and no source to point at, so the copies' ``res_id``
    spacing is the only signal left -- and it must reproduce the training block exactly.

    Since training copies the unindexed source tokens and inference excises them, we do not compare these,
    but we do compare features within the unindexed block and the rest of the indexed block.
    """

    def unindexed_block(is_inference: bool) -> dict[str, dict[str, np.ndarray]]:
        aa = build_poly_gly_chain(design_res_ids)
        is_unindexed = np.isin(aa.res_id, unindexed_res_ids)
        C_IDX.set_annotation(aa, ~is_unindexed)
        C_IDXGRP.set_annotation(aa, np.where(is_unindexed, 1, 0).astype(np.int32))

        source_atom_array = source_token_idxs = None
        if not is_inference:
            source_atom_array = build_poly_gly_chain(list(range(min(design_res_ids), max(design_res_ids) + 1)))
            source_token_idxs = np.array(design_res_ids) - min(design_res_ids)

        expanded = UnindexFlaggedTokens()({"atom_array": aa, "is_inference": is_inference})["atom_array"]
        feats = compute_token_bond_features(
            expanded,
            path_length_cutoff=64,
            source_atom_array=source_atom_array,
            source_token_idxs=source_token_idxs,
            fill_res_id_gaps=is_inference,
        )
        assert not check_index_group_molecule_connectivity_violations(expanded, feats, is_inference=is_inference)

        # Copies are appended last, and are the only tokens with a source in training.
        token_starts = get_token_starts(expanded)
        copies = np.arange(len(token_starts) - len(unindexed_res_ids), len(token_starts))
        assert is_inference or (expanded.insertion_source_idx[token_starts][copies] != -1).all()

        # Regions that were never unindexed to begin with
        scaffold = np.where(C_IDX.mask(expanded, default="generate")[token_starts])[0]
        if not is_inference:
            scaffold = scaffold[~expanded.is_unindexed_source[token_starts][scaffold]]

        return {
            name: {k: v[np.ix_(block, block)] for k, v in feats.items()}
            for name, block in (("copies", copies), ("scaffold", scaffold))
        }

    inference = unindexed_block(is_inference=True)
    np.testing.assert_array_equal(
        inference["copies"]["token_bond_path_length"], np.array(expected_path_lengths, np.int16)
    )

    training = unindexed_block(is_inference=False)
    for block in ("copies", "scaffold"):
        for key in ("token_bond_path_length", "canonical_polymer_ordering"):
            np.testing.assert_array_equal(
                inference[block][key], training[block][key], err_msg=f"train/inference mismatch in {block} {key}"
            )


def test_no_path_and_too_far_are_told_apart():
    """Connected paths beyond the cutoff get overflow; disconnected pairs remain zero."""
    # Connected but beyond the search depth: the sentinel, and nothing to complain about.
    far = build_poly_gly_chain(list(range(1, 11)))
    C_IDXGRP.set_annotation(far, np.zeros(far.array_length(), dtype=np.int32))
    feats = compute_token_bond_features(far, path_length_cutoff=3, fill_res_id_gaps=True)
    assert feats["token_bond_path_length"][0, 9] == 4  # path_length_cutoff + 1
    assert not check_index_group_molecule_connectivity_violations(far, feats, is_inference=True)

    # A severed chain retains its molecule label and has no numbering gap to fill.
    broken = build_poly_gly_chain([1, 2, 3, 4])
    C_IDXGRP.set_annotation(broken, np.zeros(broken.array_length(), dtype=np.int32))
    starts = get_residue_starts(broken)
    ((peptide_bond, *_),) = [
        [(a, b)] for a, b, _ in broken.bonds.as_array() if a < starts[2] <= b and broken.atom_name[b] == "N"
    ]
    broken.bonds.remove_bond(*peptide_bond)

    feats = compute_token_bond_features(broken, path_length_cutoff=64, fill_res_id_gaps=True)
    assert feats["token_bond_path_length"][1, 2] == 0

    (violation,) = check_index_group_molecule_connectivity_violations(broken, feats, is_inference=True)
    assert violation.check == "index_group_molecule_must_be_connected"
    assert "A/2-A/3" in violation.message


def test_unindexing_preserves_bonds_within_a_motif():
    """Unindexed copies must keep the bonds that joined them in the source structure.

    :py:meth:`UnindexFlaggedTokens.expand_unindexed_motifs` copies each IndexGroup out in one
    slice, so the bonds inside it come along and those leaving it do not. Copied residue by residue
    instead, a motif would read as unrelated single residues -- no chain ordering, no hop counts --
    though the source plainly had them bonded. Bonds reaching outside the group must still be
    dropped: a group's placement relative to anything else is exactly what unindexing withholds.
    """
    aa = build_poly_gly_chain([1, 2, 3, 4, 5, 6])
    # Residues 3-4 are one unindexed motif and 5-6 another, so res 4 and res 5 are bonded in the
    # source but belong to different groups.
    group = np.select([np.isin(aa.res_id, [3, 4]), np.isin(aa.res_id, [5, 6])], [1, 2], default=0)
    C_IDX.set_annotation(aa, group == 0)
    C_IDXGRP.set_annotation(aa, group.astype(np.int32))

    expanded = UnindexFlaggedTokens()({"atom_array": aa, "is_inference": True})["atom_array"]

    starts = get_residue_starts(expanded)
    residue_of = np.searchsorted(starts, np.arange(expanded.array_length()), side="right") - 1
    bonded_residues = {
        frozenset((int(residue_of[a]), int(residue_of[b])))
        for a, b, _ in expanded.bonds.as_array()
        if residue_of[a] != residue_of[b]
    }

    # The two indexed residues come first, then the four copies in res_id order.
    indexed_1, indexed_2, copy_3, copy_4, copy_5, copy_6 = range(len(starts))
    assert frozenset((copy_3, copy_4)) in bonded_residues  # one motif, adjacent: bond preserved
    assert frozenset((copy_5, copy_6)) in bonded_residues
    assert frozenset((copy_4, copy_5)) not in bonded_residues  # different motifs: no bond
    assert frozenset((indexed_1, indexed_2)) in bonded_residues  # the indexed region is untouched
    assert frozenset((indexed_2, copy_3)) not in bonded_residues


def test_excision_bonds_an_indexed_flank_to_a_designed_one():
    """Closing the gap must leave a real bond behind, not one the ``res_id`` fill invents.

    The bonds that held the motif's flanks went with the motif, so an excision the designed flank
    absorbs has to re-join them itself -- through the polymerisation atoms, since the two sides are
    of different kinds: one residue taken from the input, one designed stub.
    """
    aa = build_poly_gly_chain([1, 2, 3, 4, 5, 6])
    after_res_3 = int(get_residue_starts(aa, add_exclusive_stop=True)[3]) - 1
    aa = realize_expandable_segments(
        insert_expandable_segment(aa, seg_min=3, seg_max=3, atom_idx=after_res_3, insert_before=False)
    )

    # Residues 1-3 come from the input and 4-6 are the designed stubs, so excising res 3 leaves an
    # indexed residue on one side of the gap and the expandable segment on the other.
    is_unindexed = aa.res_id == 3
    C_IDX.set_annotation(aa, ~is_unindexed)
    C_IDXGRP.set_annotation(aa, is_unindexed.astype(np.int32))

    expanded = UnindexFlaggedTokens()({"atom_array": aa, "is_inference": True})["atom_array"]

    # The scaffold closes up: eight residues numbered without a hole, then the copy of the motif.
    starts = get_residue_starts(expanded)
    np.testing.assert_array_equal(expanded.res_id[starts][:8], np.arange(1, 9))

    # And res 2 (from the input) and res 3 (the first stub) are bonded in the array itself.
    left, right = 1, 2
    residue_of = np.searchsorted(starts, np.arange(expanded.array_length()), side="right") - 1
    ((exiting_atom, _),) = [
        (a, b) for a, b, _ in expanded.bonds.as_array() if {residue_of[a], residue_of[b]} == {left, right}
    ]
    assert expanded.atom_name[exiting_atom] == "C"  # the peptide linkage, not some arbitrary pair


@pytest.mark.parametrize("flanked_by_expandable_segment", [True, False], ids=["expandable", "from_the_input"])
def test_excision_only_closes_the_gap_a_designed_flank_can_absorb(
    flanked_by_expandable_segment: bool, caplog: pytest.LogCaptureFixture
):
    """What excising an interior motif does to the numbering must follow what flanks it.

    An expandable segment stands for residues the model designs, so its length is ours to choose
    and the chain simply closes around the removed motif -- which means bonding the two residues
    the excision left adjacent, since the bonds that held them went with the motif. Two regions
    taken verbatim from the input have the indices the user gave them, so the gap is kept and read
    at its own length. Either way the scaffold must stay one molecule.
    """
    if flanked_by_expandable_segment:
        # Three designed residues in the middle of the chain: res 4-6 become MASKED stubs, and what
        # was res 4-6 becomes res 7-9.
        aa = build_poly_gly_chain([1, 2, 3, 4, 5, 6])
        after_res_3 = int(get_residue_starts(aa, add_exclusive_stop=True)[3]) - 1
        aa = realize_expandable_segments(
            insert_expandable_segment(aa, seg_min=3, seg_max=3, atom_idx=after_res_3, insert_before=False)
        )
    else:
        aa = build_poly_gly_chain(list(range(1, 10)))

    is_unindexed = np.isin(aa.res_id, [7, 8])  # an interior motif, in both arms
    C_IDX.set_annotation(aa, ~is_unindexed)
    C_IDXGRP.set_annotation(aa, is_unindexed.astype(np.int32))

    with caplog.at_level(logging.WARNING):
        expanded = UnindexFlaggedTokens()({"atom_array": aa, "is_inference": True})["atom_array"]

    # The scaffold is what survives the excision -- six residues, then the last one; the two copies
    # of the motif follow it.
    left, right = 5, 6
    scaffold_res_ids = expanded.res_id[get_residue_starts(expanded)][:7]
    feats = compute_token_bond_features(expanded, path_length_cutoff=64, fill_res_id_gaps=True)
    assert not check_index_group_molecule_connectivity_violations(expanded, feats, is_inference=True)

    if flanked_by_expandable_segment:
        np.testing.assert_array_equal(scaffold_res_ids, [1, 2, 3, 4, 5, 6, 7])
        assert feats["token_bond_path_length"][left, right] == 1
        assert "lie between two indexed regions" not in caplog.text
    else:
        # The residues the motif separated keep the indices they came in with, and read as the
        # three residues apart that those indices say they are.
        np.testing.assert_array_equal(scaffold_res_ids, [1, 2, 3, 4, 5, 6, 9])
        assert feats["token_bond_path_length"][left, right] == 3
        assert "A_1/7-8 lie between two indexed regions" in caplog.text
