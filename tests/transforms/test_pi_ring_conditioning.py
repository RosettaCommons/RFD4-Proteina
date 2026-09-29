"""Ring-level handling of pi-stacking conditioning.

The ground truth here comes from atomworks' own interaction annotator, so the rings this module
groups by can be checked directly against the ring definitions the annotator itself used.
"""

import numpy as np
import pytest
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.protonation import ensure_hydrogens
from atomworks.ml.conditions import C_SEQ
from atomworks.ml.transforms.interactions import annotate_interactions, build_atom_index_mapping
from atomworks.ml.utils.testing import cached_parse

from rfproteina.conditions import C_PISTACK
from rfproteina.transforms.design_task_modifiers import _whole_ring_groups, add_pistack_condition


@pytest.fixture(scope="module")
def fad_pistacking():
    """3grs: glutathione reductase Tyr197 stacked against the FAD isoalloxazine.

    This covers both cases at once. Tyr197 is a protein residue with a single ring, while the FAD
    is a ligand carrying five rings -- an adenine and a three-ring fused isoalloxazine -- of which
    the stack covers exactly one. The two rings fused to it are covered only at the two atoms each
    shares with it, so anything that grouped by residue, or that split a fused system, would get
    this wrong.

    Returns:
      The heavy-atom array, atomworks' per-atom pi-stacking mask, and atomworks' own aromatic ring
      definitions, all in the heavy-atom index space.
    """
    data = cached_parse("3grs", **(STANDARD_PARSER_ARGS | {"hydrogen_policy": "remove"}))
    atom_array = as_atom_array_plus(data["atom_array"])
    atom_array = atom_array[np.isin(atom_array.chain_iid, ["A_1", "C_1"])]

    # Mirror AnnotateInteractions: detect on a hydrogenated copy, then map back to the heavy atoms.
    with_h = as_atom_array_plus(ensure_hydrogens(atom_array.copy(), silence_lost_annotation_warnings=True))
    results = annotate_interactions(with_h, interaction_types=("pistacking",), inter_chain_only=True)
    to_heavy = build_atom_index_mapping(with_h, atom_array)

    is_pistacking = np.zeros(len(atom_array), dtype=bool)
    for atom_a, atom_b in results["pistacking"]:
        is_pistacking[[to_heavy[atom_a], to_heavy[atom_b]]] = True
    rings = [to_heavy[ring] for ring in results["_aromatic_rings"]]

    return atom_array, is_pistacking, [ring for ring in rings if (ring >= 0).all()]


def test_detected_rings_match_atomworks_ring_definitions(fad_pistacking):
    """The rings we group by are exactly atomworks' rings that the annotation covers completely.

    Annotation that does not close a ring is dropped rather than conditioned on.
    """
    atom_array, is_pistacking, rings = fad_pistacking

    fully_annotated = [ring for ring in rings if is_pistacking[ring].all()]
    partly_annotated = [ring for ring in rings if is_pistacking[ring].any() and not is_pistacking[ring].all()]

    # One Tyr ring and one FAD ring stack; the FAD rings fused to that one are covered only at the
    # two atoms each shares with it. Without this the test would not exercise the fused case.
    assert len(fully_annotated) == 2
    assert len(partly_annotated) == 2
    assert all(is_pistacking[ring].sum() == 2 for ring in partly_annotated)

    kept, detected = _whole_ring_groups(atom_array, is_pistacking, False, (True,), {})

    assert {frozenset(ring) for ring in detected} == {frozenset(ring) for ring in fully_annotated}
    # Every annotated atom belongs to a whole ring here, so none of the annotation is dropped.
    assert np.array_equal(kept, is_pistacking)
    assert set(np.concatenate(detected).tolist()) == set(np.flatnonzero(is_pistacking).tolist())

    tyr_ring = next(ring for ring in fully_annotated if atom_array.res_name[ring[0]] == "TYR")
    in_tyr = (atom_array.chain_iid == atom_array.chain_iid[tyr_ring[0]]) & (
        atom_array.res_id == atom_array.res_id[tyr_ring[0]]
    )
    unannotated_ring = next(ring for ring in rings if not is_pistacking[ring].any())

    not_a_ring = is_pistacking.copy()
    not_a_ring[unannotated_ring[:4]] = True  # Simulating a partially-resolved ring
    not_a_ring[in_tyr & (atom_array.atom_name == "CB")] = True  # a substituent on an annotated ring
    not_a_ring[np.flatnonzero(atom_array.atom_name == "CA")[0]] = True  # bonded to nothing annotated

    kept, detected = _whole_ring_groups(atom_array, not_a_ring, False, (True,), {})

    assert {frozenset(ring) for ring in detected} == {frozenset(ring) for ring in fully_annotated}
    assert np.array_equal(kept, is_pistacking), "annotation outside a whole ring was not dropped"


@pytest.mark.parametrize("subsample_frac", [0.5, 1.0])
def test_subsampling_reveals_only_complete_rings(fad_pistacking, subsample_frac):
    """Subsampling drops whole rings, so what survives is a union of complete atomworks rings."""
    atom_array, is_pistacking, rings = fad_pistacking
    atom_array = atom_array.copy()

    C_PISTACK.set_annotation(atom_array, is_pistacking)
    C_SEQ.set_annotation(atom_array, C_SEQ.ground_truth_value(atom_array))

    data = {"atom_array": atom_array, "task": {}}
    result = add_pistack_condition(
        data,
        p_add_pistack_condition=1.0,
        p_pistack_subsample=1.0,
        pistack_subsample_frac=subsample_frac,
    )
    revealed = C_PISTACK.annotation(result["atom_array"])

    fully_annotated = [ring for ring in rings if is_pistacking[ring].all()]
    in_a_ring = np.zeros(len(revealed), dtype=bool)
    for ring in fully_annotated:
        in_a_ring[ring] = True

    assert not (revealed & ~in_a_ring).any(), "conditioning survived on an atom outside a complete ring"
    for ring in fully_annotated:
        assert revealed[ring].all() or not revealed[ring].any(), "a ring was revealed only in part"
    assert sum(revealed[ring].all() for ring in fully_annotated) == max(1, int(len(fully_annotated) * subsample_frac))
