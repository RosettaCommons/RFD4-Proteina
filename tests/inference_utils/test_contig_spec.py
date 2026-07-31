"""Tests for the contig-string subset + expandable-segment-insertion assembly.

Focuses on the tricky, easily-broken part of ``docs/condition_spec_yaml.md``'s
`contig` grammar: chain-run-first (not left-to-right) chain-id/type assignment,
the pre-insertion `original_res_id` snapshot surviving renumbering shifts,
bootstrapping the very first sentinel when nothing has been assembled yet, and
the validation contig assembly relies on to reject genuinely ambiguous input
(mixed-chain-run, duplicate/`MASKED` names, overlapping/duplicate-chain-id
preserved segments) rather than silently mis-assembling or crashing deep inside
`insert_expandable_segment`.
"""

import numpy as np
import pytest
from atomworks.enums import ChainType

from rfproteina.inference_utils.contig_spec import EXPANDABLE_SEGMENT_RES_ID_SENTINEL, apply_contig


def test_chain_run_first_assignment_not_left_to_right(two_chain_complex):
    """The worked example from design-doc feedback: `A1-10,/0,E1:1-20,B1-10` -> E1 on chain B.

    A single left-to-right walk over the tokens could not know this: `E1` precedes
    `B1-10` in the string, but both are in the *same* chain-run (only one `/0`
    separates them from `A1-10`), so the run's chain id is only knowable once the
    whole run (including the later `B1-10` token) has been scanned.
    """
    assembled, original_res_id = apply_contig(two_chain_complex, "A1-10, /0, E1:1-20, B1-10")

    e1_mask = assembled.res_name == "E1"
    assert e1_mask.sum() == 1
    assert assembled.chain_id[e1_mask][0] == "B"
    # The expandable-segment sentinel has no original res_id -- flagged with the sentinel.
    assert original_res_id[e1_mask][0] == EXPANDABLE_SEGMENT_RES_ID_SENTINEL

    # Every output chain is renumbered from 1, sentinels included. `E1` leads chain B, so it
    # takes res_id 1 and the preserved B backbone follows at 2-11 (not a leading res_id 0).
    a_res_id = assembled.res_id[assembled.chain_id == "A"]
    assert sorted(set(a_res_id)) == list(range(1, 11))
    assert assembled.res_id[e1_mask][0] == 1
    b_only_mask = (assembled.chain_id == "B") & ~e1_mask
    assert sorted(set(assembled.res_id[b_only_mask])) == list(range(2, 12))


def test_original_res_id_snapshot_survives_expandable_segment_shift(lysozyme):
    """A preserved segment after an inserted expandable segment keeps its *original* res_id.

    `contig` renumbers `res_id` to a fresh contiguous per-run counter, so the second
    preserved block's real (assembled) `res_id` no longer matches its res_id in the
    input file once `X1` is inserted before it. `original_res_id` must still reflect
    the pre-contig numbering (10-15, then 50-55), which `conditions:` selections rely
    on to reference "residue 50 of my input file" regardless of anything inserted
    earlier in the same chain.
    """
    assembled, original_res_id = apply_contig(lysozyme, "A10-15, X1:5-8, A50-55")

    x1_mask = assembled.res_name == "X1"
    assert x1_mask.sum() == 1
    assert original_res_id[x1_mask][0] == EXPANDABLE_SEGMENT_RES_ID_SENTINEL

    preserved_mask = ~x1_mask
    # Original numbering is preserved exactly as in the input file, regardless of the
    # insertion in between.
    first_block = original_res_id[preserved_mask][assembled.res_id[preserved_mask] <= 6]
    second_block = original_res_id[preserved_mask][assembled.res_id[preserved_mask] > 7]
    assert sorted(set(first_block)) == list(range(10, 16))
    assert sorted(set(second_block)) == list(range(50, 56))

    # Assembled res_id is a single fresh contiguous per-chain-run counter (1-13), unaffected by
    # the original file's numbering gap (16-49) or the insertion, and monotonic in array order.
    assert sorted(set(assembled.res_id)) == list(range(1, 14))
    assert list(assembled.res_id) == sorted(assembled.res_id)


def test_wholly_expandable_run_gets_fresh_chain_id_and_grows_contiguously(two_chain_complex):
    """A chain-run with no preserved anchor at all gets a fresh chain id, and multiple
    expandable segments within that same run all land on that one new chain,
    contiguously -- not just the first one."""
    assembled, _ = apply_contig(two_chain_complex, "A1-5, /0, X1:5-8, X2:10")

    x1_chain = assembled.chain_id[assembled.res_name == "X1"][0]
    x2_chain = assembled.chain_id[assembled.res_name == "X2"][0]
    assert x1_chain == x2_chain
    assert x1_chain not in ("A",)  # never reuses the run's own preserved-anchor chain id

    new_chain_res_id = assembled.res_id[assembled.chain_id == x1_chain]
    assert list(new_chain_res_id) == [1, 2]


def test_wholly_expandable_leading_runs_deferred_until_a_later_run_has_preserved_atoms(two_chain_complex):
    """A contig whose *leading* run(s) have no preserved anchor at all (nothing built yet to
    derive a new chain's sentinel schema from) still works -- rather than bootstrapping from
    a throwaway copy of the original input, they're deferred and only realized once a later
    run supplies real preserved atoms to build onto. Two leading wholly-expandable runs
    exercise more than one deferred entry at once.
    """
    assembled, original_res_id = apply_contig(two_chain_complex, "E1:5-8, /0, E2:3, /0, A1-5")

    e1_mask = assembled.res_name == "E1"
    e2_mask = assembled.res_name == "E2"
    assert e1_mask.sum() == 1
    assert e2_mask.sum() == 1
    assert original_res_id[e1_mask][0] == EXPANDABLE_SEGMENT_RES_ID_SENTINEL
    assert original_res_id[e2_mask][0] == EXPANDABLE_SEGMENT_RES_ID_SENTINEL
    # Each deferred run's sentinel gets its own fresh chain id, none of them the later run's
    # preserved chain "A", and no two deferred runs share a chain either.
    e1_chain, e2_chain = assembled.chain_id[e1_mask][0], assembled.chain_id[e2_mask][0]
    assert len({e1_chain, e2_chain, "A"}) == 3
    # Despite being processed after "A" (the run that unblocks them), the deferred runs end
    # up positioned before it in the assembled array, matching the contig's declared order.
    first_idx = {c: int(np.argmax(assembled.chain_id == c)) for c in (e1_chain, e2_chain, "A")}
    assert first_idx[e1_chain] < first_idx[e2_chain] < first_idx["A"]
    assert sorted(set(assembled.res_id[assembled.chain_id == "A"])) == list(range(1, 6))


def test_unsuffixed_expandable_segment_defaults_to_anchor_chain_type_not_polypeptide(trna):
    """An unsuffixed expandable segment in an RNA-anchored run defaults to RNA, not P.

    This is the subtle case the design explicitly called out: defaulting to
    polypeptide unconditionally (the old behavior) would be silently wrong for any
    non-protein anchor.
    """
    assembled, _ = apply_contig(trna, "A1-10, X1:5-8")
    x1_mask = assembled.res_name == "X1"
    assert ChainType(int(assembled.chain_type[x1_mask][0])) == ChainType.RNA


def test_anchorless_run_takes_its_expandable_segments_suffix(lysozyme):
    """A run with no preserved segment has no anchor, so its single chain type comes from
    whatever `P`/`R`/`D` suffix its segments carry -- an unsuffixed sibling inherits it
    rather than dragging the run back to the polypeptide default."""
    assembled, _ = apply_contig(lysozyme, "A1-10, /0, X1:5R, X2:5")
    for name in ("X1", "X2"):
        segment_mask = assembled.res_name == name
        assert ChainType(int(assembled.chain_type[segment_mask][0])) == ChainType.RNA


def test_anchor_chain_with_mixed_chain_types_raises(lysozyme):
    """`_anchor_chain_type` assumes one ChainType per original chain id -- a chain that
    (unusually) mixes ChainTypes can't unambiguously default an unsuffixed expandable
    segment's type, so this must be rejected rather than silently picking whichever
    ChainType happens to appear first."""
    lysozyme.chain_type[lysozyme.res_id > 60] = int(ChainType.RNA)
    with pytest.raises(ValueError, match="multiple ChainTypes"):
        apply_contig(lysozyme, "A10-70")


def test_contig_with_no_preserved_segments_raises(lysozyme):
    """A contig referencing no preserved segment at all has no reason to use an `input`
    file -- that's just unconditional generation, out of scope for condition_spec."""
    with pytest.raises(ValueError, match="no preserved segments"):
        apply_contig(lysozyme, "X1:5-8")


@pytest.mark.parametrize(
    "contig,match",
    [
        ("A1-10, B1-10", "mixes preserved segments"),  # two original chains, no `/0` between them
        ("A1-10, X1:5-8, A20-25, X1:5-8", "Duplicate expandable-segment name"),
        ("A1-10, <M>:5-8", "reserved internally"),  # `<M>` is reserved for realized-segment sentinels
        ("A1-10, A5-15", "overlap"),  # silently duplicating shared residues
        ("A52-48", "end < start"),  # a reversed single-segment range is a typo, not a valid ordering
        ("A1-10, /0, A20-30", "both named 'A'"),  # `/0` can't split one original chain in two
        ("Z1-10", "matched no atoms"),  # typo'd chain id -> a clear error, not a raw IndexError
        ("A1-10, X1:5-8R", "disagrees with its run's anchor chain type"),  # R suffix vs. protein anchor
        # One run becomes one chain, so its segments can't request two chain types -- the
        # second request would otherwise be silently dropped onto the first's chain.
        ("A1-10, /0, X1:5P, X2:5R", "conflicting chain types"),
    ],
)
def test_invalid_contigs_raise_clear_errors(lysozyme, contig, match):
    with pytest.raises(ValueError, match=match):
        apply_contig(lysozyme, contig)
