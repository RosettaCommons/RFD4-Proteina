import numpy as np
import pytest

from rfproteina.metrics.motif_geometry import score_geometry


def _inputs():
    reference = np.random.default_rng(7).normal(size=(30, 3)) * 5
    ca = np.ones(30, dtype=bool)
    motif = np.zeros(30, dtype=bool)
    motif[:12] = True
    catalytic = np.zeros(30, dtype=bool)
    catalytic[12:16] = True
    groups = np.full(30, -1)
    groups[:6] = 0
    groups[6:12] = 1
    return reference, ca, motif, catalytic, groups


def test_rigid_transform_preserves_all_geometry():
    reference, ca, motif, cat, groups = _inputs()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    predicted = reference @ rotation + [10, 20, 30]
    scores = score_geometry(reference, predicted, reference, ca, motif, cat, groups)
    assert max(scores.values()) < 1e-5


def test_separate_island_motion_is_not_hidden_by_independent_alignment():
    reference, ca, motif, cat, groups = _inputs()
    predicted = reference.copy()
    predicted[6:12] += [8, 0, 0]
    scores = score_geometry(reference, predicted, reference, ca, motif, cat, groups)
    assert scores["motif_bb_rmsd"] > 2
    assert scores["worst_segment_bb_rmsd"] >= scores["motif_bb_rmsd"]


def test_sidechain_only_failure_with_perfect_backbone():
    reference, ca, motif, cat, groups = _inputs()
    ca[cat] = False
    predicted = reference.copy()
    predicted[12] += [4, 0, 0]
    scores = score_geometry(reference, predicted, reference, ca, motif, cat, groups)
    assert scores["motif_bb_rmsd"] < 1e-5
    assert scores["scaffold_ca_rmsd"] < 1e-5
    assert scores["max_catalytic_displacement"] == pytest.approx(4)
    assert scores["max_catalytic_distance_error"] > 1


def test_missing_coordinates_fail_instead_of_shrinking_the_motif():
    reference, ca, motif, cat, groups = _inputs()
    predicted = reference.copy()
    predicted[12] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        score_geometry(reference, predicted, reference, ca, motif, cat, groups)
