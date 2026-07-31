"""End-to-end tests for the condition-spec pipeline (load -> contig -> conditions -> finalize).

Covers ``docs/condition_spec_yaml.md``'s full order of operations against real
structures, exercising the trickiest interactions: a `conditions:` selection that
targets a preserved region by its *original* (pre-contig) `res_id`, one that targets
an expandable segment by its custom name, mask-only vs. enum vs. fixed-vector
(`ChainCentroid`) vs. group-requiring (`C_IDX`/`C_INTCRD`) condition kinds, a real CIF
write+reload round trip, and confirming an untouched condition reads back as its lazy
default (no explicit back-fill step exists in this design) -- plus the input-loading
correctness issues a red-team pass found: symmetry-copy duplication and cross-file
molecule/index-group aliasing when combining more than one input file.
"""

import numpy as np
import pytest
from atomworks.io.parser import parse
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.conditions import C_SEQ
from atomworks.ml.utils.testing import cached_parse

from rfproteina.conditions import C_CDR, C_CENTROID, C_CRDGRP, C_HOT, C_IDX, C_IDXGRP, C_INTCRD, C_RAS, RasaEnum
from rfproteina.inference_utils.condition_spec import (
    build_annotated_atom_array_from_condition_spec,
    load_condition_spec_inputs,
    split_global_args,
)
from rfproteina.utils.parse_presets import ANNOTATED_CIF


def _build_spec_atom_array(lysozyme_cif_path, tmp_path, conditions, contig="A10-20, A60-70, MOTIF:5-8"):
    spec = {
        "input": str(lysozyme_cif_path),
        "input_as_annotated_cif": True,
        # Both preserved blocks share chain A (one chain-run); MOTIF trails both, so
        # neither preserved block's renumbering is perturbed by the insertion -- makes
        # the expected post-contig res_id values easy to state and check below.
        "contig": contig,
        "conditions": conditions,
    }
    data = build_annotated_atom_array_from_condition_spec(spec, spec_dir=tmp_path, example_id="e2e")
    return data["atom_array"]


def test_end_to_end_condition_spec_survives_cif_round_trip(lysozyme_cif_path, tmp_path):
    conditions = {
        "C_HOT": {True: [{"select": "res_id == 15"}]},  # preserved region, by ORIGINAL res_id
        "C_RAS": {"BURIED": [{"select": "res_name == 'MOTIF'"}]},  # expandable segment, by custom name
        # Whole chain, not a single residue: the centroid is a per-chain value, and
        # `check_centroid_conditioning_all_or_nothing_violations` rejects partial coverage.
        "C_CENTROID": {"1.0,2.0,3.0": [{"select": "chain_id == 'A'"}]},
    }
    arr = _build_spec_atom_array(lysozyme_cif_path, tmp_path, conditions)

    out_path = tmp_path / "e2e.cif"
    to_cif_file(arr, out_path, save_standard_annotations=True, chain_disambiguation="chain_iid")
    reloaded = parse(out_path, config=ANNOTATED_CIF)["assemblies"]["1"][0]

    for built in (arr, reloaded):
        # C_HOT: original res_id 15 in "A10-20" -> renumbered res_id 15 - 10 + 1 = 6.
        hot_mask = C_HOT.mask(built, default="generate")
        assert hot_mask[built.res_id == 6].all()
        assert not hot_mask[built.res_id != 6].any()

        # C_RAS: expandable segment referenced by its custom res_name, not res_id.
        motif_mask = built.res_name == "MOTIF"
        assert motif_mask.sum() == 1
        ras_mask = C_RAS.mask(built, default="generate")
        ras_annot = C_RAS.annotation(built, default="generate")
        assert ras_mask[motif_mask].all()
        assert ras_annot[motif_mask][0] == RasaEnum.BURIED.value

        # C_CENTROID covers the whole (single) chain, and survives the round trip with its
        # literal xyz intact -- spot-checked at renumbered res_id 17 (original 65 in "A60-70":
        # 65 - 60 + 12, the 2nd preserved block continuing the 1st's per-run counter at 12).
        centroid_mask = C_CENTROID.mask(built, default="generate")
        centroid_annot = C_CENTROID.annotation(built, default="generate")
        assert centroid_mask.all()
        np.testing.assert_allclose(centroid_annot[built.res_id == 17][0], [1.0, 2.0, 3.0], atol=1e-4)

        # No explicit back-fill step: an untouched condition still reads back as its
        # lazy default, not as some stale/missing state.
        assert not C_CDR.mask(built, default="generate").any()


def test_group_field_merges_disjoint_selections_and_assigns_fresh_ids(lysozyme_cif_path, tmp_path):
    """`C_IDX: false` / `C_INTCRD: true` need a `group:` -- two disjoint `select:`s sharing
    one `group:` label must merge into a single group id, and different labels must never
    collide with each other or with the reserved CoordinateGroup values (0/-1).
    """
    conditions = {
        "C_IDX": {
            False: [
                {"select": "chain_id == 'A' and res_id >= 10 and res_id <= 20", "group": "g1"},
                {"select": "res_name == 'MOTIF'", "group": "g1"},  # disjoint, same group
                {"select": "chain_id == 'A' and res_id >= 60 and res_id <= 70", "group": "g2"},
            ]
        },
        "C_INTCRD": {True: [{"select": "res_id == 65", "group": "motif_intcrd"}]},
    }
    arr = _build_spec_atom_array(lysozyme_cif_path, tmp_path, conditions, contig="A10-20, A60-70, A90-100, MOTIF:5-8")

    idx_groups = C_IDXGRP.annotation(arr, default="generate")
    g1_mask = (arr.res_id <= 11) | (arr.res_name == "MOTIF")  # block 1 (res_id 1-11) + MOTIF (res_id 34)
    g2_mask = (arr.res_id >= 12) & (arr.res_id <= 22)  # block 2
    block3_mask = (arr.res_id >= 23) & (arr.res_id <= 33)  # left fully indexed
    assert len(set(idx_groups[g1_mask])) == 1
    assert len(set(idx_groups[g2_mask])) == 1
    assert set(idx_groups[g1_mask]) != set(idx_groups[g2_mask])
    assert not C_IDX.mask(arr, default="generate")[g1_mask | g2_mask].any()
    assert C_IDX.mask(arr, default="generate")[block3_mask].all()

    intcrd_mask = arr.res_id == 17  # orig res_id 65 -> assembled 17
    assert C_INTCRD.mask(arr, default="generate")[intcrd_mask].all()
    assert not np.isnan(C_INTCRD.annotation(arr, default="generate")[intcrd_mask]).any()
    crdgrp = C_CRDGRP.annotation(arr, default="generate")
    assert crdgrp[intcrd_mask][0] not in (C_CRDGRP.KNOWN_ABSOLUTE_COORDS, C_CRDGRP.UNKNOWN_COORDS)


@pytest.mark.parametrize(
    "values,expected",
    [
        ({False: [{"select": "res_id == 15", "group": "g"}], True: [{"select": "res_id == 15"}]}, True),
        ({True: [{"select": "res_id == 15"}], False: [{"select": "res_id == 15", "group": "g"}]}, False),
    ],
)
def test_grouped_and_plain_condition_entries_are_last_write_wins(lysozyme_cif_path, tmp_path, values, expected):
    """A grouped value must not jump the queue: deferring grouped entries to the end made
    `C_IDX: false` (grouped) always beat a later `C_IDX: true`, whichever came last."""
    arr = _build_spec_atom_array(lysozyme_cif_path, tmp_path, {"C_IDX": values})
    selected = arr.res_id == 6  # Original res_id 15 after contig assembly.
    assert (C_IDX.mask(arr, default="generate")[selected] == expected).all()


@pytest.mark.parametrize(
    "conditions,match",
    [
        ({"C_IDX": {False: [{"select": "res_id == 12"}]}}, "requires a `group:` field"),
        ({"C_INTCRD": {True: [{"select": "res_id == 12"}]}}, "requires a `group:` field"),
        ({"C_HOT": {True: [{"select": "res_id == 12", "group": "oops"}]}}, "only meaningful for"),
        ({"C_BP": {True: [{"select": "res_id == 12"}]}}, "1-body conditions"),  # 2-body: no selection-pair grammar
        # A quoted `"false"` is truthy in Python, so a plain truthiness test would silently
        # condition on exactly the atoms the author meant to exclude. Only real booleans and
        # the exact strings are accepted; everything else is a spec error rather than a guess.
        ({"C_HOT": {"maybe": [{"select": "res_id == 12"}]}}, "boolean condition value"),
        ({"C_HOT": {1: [{"select": "res_id == 12"}]}}, "boolean condition value"),
        ({"C_HOT": {1.5: [{"select": "res_id == 12"}]}}, "boolean condition value"),
    ],
)
def test_condition_application_errors(lysozyme_cif_path, tmp_path, conditions, match):
    with pytest.raises(ValueError, match=match):
        _build_spec_atom_array(lysozyme_cif_path, tmp_path, conditions)


@pytest.mark.parametrize("value,expected", [("true", True), ("True", True), ("false", False), ("False", False)])
def test_quoted_boolean_condition_values_keep_their_literal_meaning(lysozyme_cif_path, tmp_path, value, expected):
    """JSON object keys are always strings, so `"false"` has to mean false, not "truthy"."""
    arr = _build_spec_atom_array(lysozyme_cif_path, tmp_path, {"C_HOT": {value: [{"select": "res_id == 15"}]}})
    selected = arr.res_id == 6  # Original res_id 15 after contig assembly.
    assert (C_HOT.mask(arr, default="generate")[selected] == expected).all()


def test_mask_only_sequence_conditioning_reveals_the_real_residue_names(lysozyme_cif_path, tmp_path):
    """`C_SEQ: true` must reveal ground-truth `res_name`, never write the literal value.

    `C_SEQ`'s value is read off the structure, like `C_CRD`'s coordinates. Treating it as a
    value-writing condition assigns the *string* ``"True"`` into its str-dtype annotation
    (silently truncated to ``"Tru"``), replacing the whole target sequence with a sentinel
    the model then reads as the known sequence.
    """
    arr = _build_spec_atom_array(lysozyme_cif_path, tmp_path, {"C_SEQ": {True: [{"select": "res_id == 15"}]}})

    selected = arr.res_id == 6  # Original res_id 15 after contig assembly.
    assert C_SEQ.mask(arr, default="generate")[selected].all()
    revealed = C_SEQ.annotation(arr, default="generate")[selected]
    assert (revealed == arr.res_name[selected]).all(), f"expected real res_names, got {set(revealed)}"


def test_unrecognized_example_field_raises(lysozyme_cif_path, tmp_path):
    """`out_dir` (and any other stray key) is never read, so it must not be silently accepted."""
    spec = {"input": str(lysozyme_cif_path), "out_dir": "./somewhere", "contig": "A10-20"}
    with pytest.raises(ValueError, match=r"unrecognized condition_spec field\(s\) \['out_dir'\]"):
        build_annotated_atom_array_from_condition_spec(spec, spec_dir=tmp_path, example_id="e2e")


@pytest.mark.parametrize(
    "spec,match",
    [
        ({"global_args": ["not", "a", "mapping"], "example": {}}, "`global_args` must be a mapping"),
        ({"example": "not a mapping"}, "examples must be mappings"),
    ],
)
def test_split_global_args_rejects_non_mapping_entries(spec, match):
    """Skipping a malformed entry would silently drop a whole example from the run."""
    with pytest.raises(ValueError, match=match):
        split_global_args(spec)


def test_chain_id_collision_across_input_files_raises(lysozyme_cif_path, tmp_path):
    second_path = tmp_path / "6lyz_copy.cif"
    second_path.write_bytes(lysozyme_cif_path.read_bytes())  # same chain_id "A" as the original

    with pytest.raises(ValueError, match="chain_id 'A' appears in both"):
        load_condition_spec_inputs([lysozyme_cif_path, second_path])


def test_multi_input_files_get_independent_molecule_and_index_groups(lysozyme_cif_path, tmp_path):
    """Two input files' molecule/entity ids are each derived relative to that file alone,
    so blindly concatenating them (without re-deriving) would silently alias e.g. both
    files' first molecule into `molecule_iid` 0 -- merging their index groups downstream.
    """
    second = cached_parse("1ehz", hydrogen_policy="remove")["atom_array"]
    second = second[second.chain_id == "A"]
    second.chain_id[:] = "Z"
    second_path = tmp_path / "trna.cif"
    to_cif_file(second, second_path)

    combined = load_condition_spec_inputs([lysozyme_cif_path, second_path])
    a_molecules = set(combined.molecule_iid[combined.chain_id == "A"])
    z_molecules = set(combined.molecule_iid[combined.chain_id == "Z"])
    assert not (a_molecules & z_molecules)
