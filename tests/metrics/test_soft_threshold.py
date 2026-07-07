"""Unit tests for linear soft threshold scoring."""

import pytest

from proteinfoundation.metrics.soft_threshold import apply_soft_bounds_to_outputs, soft_score_upper_bound


@pytest.mark.parametrize(
    ("value", "upper", "margin", "expected"),
    [
        (4.0, 5.0, 1.0, 1.0),
        (5.5, 5.0, 1.0, 0.5),
        (6.0, 5.0, 1.0, 0.0),
    ],
)
def test_soft_score_upper_bound(value, upper, margin, expected):
    assert soft_score_upper_bound(value, upper, margin) == pytest.approx(expected)


def test_apply_soft_bounds_to_outputs():
    metrics = {"plddt_coi": 79.0, "rmsd": 1.5}
    bounds = [
        {"key": "plddt_coi", "lower_bound": 80.0, "margin": 5.0},
        {"key": "rmsd", "upper_bound": 1.0, "margin": 1.0},
    ]
    out = apply_soft_bounds_to_outputs(metrics, bounds)
    assert out["plddt_coi_soft_score"] == pytest.approx(0.8)
    assert out["rmsd_soft_score"] == pytest.approx(0.5)
    assert out["soft_min_score"] == pytest.approx(0.5)
