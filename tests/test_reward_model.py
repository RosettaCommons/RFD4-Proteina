"""Unit tests for ``RewardModel.aggregate_metrics_output`` and the beam search guard on unrankable rewards."""

import pytest
import torch

from rfproteina.rewards.reward_model import RewardModel
from rfproteina.search.beam_search import _check_rankable

NEG_INF = float("-inf")


def _model() -> RewardModel:
    return RewardModel({"a.x": {"weight": 2.0}, "b.y": {"weight": 0.5, "clip": {"min": 0.0, "max": 10.0}}})


def test_finite_values_are_clipped_and_weighted():
    assert _model().aggregate_metrics_output({"a.x": 1.0, "b.y": 40.0}) == 2.0 * 1.0 + 0.5 * 10.0


def test_missing_key_is_minus_inf():
    assert _model().aggregate_metrics_output({"a.x": 1.0}) == NEG_INF


@pytest.mark.parametrize("key", ["a.x", "b.y"])
def test_nan_metric_is_minus_inf(key):
    """A NaN reward is neither worse nor better than anything, and torch ranks it first in a descending sort."""
    metrics = {"a.x": 1.0, "b.y": 4.0, key: float("nan")}
    assert _model().aggregate_metrics_output(metrics) == NEG_INF


def test_all_nan_candidates_are_reported_as_unrankable():
    """Every candidate NaN is every candidate -inf: say so instead of blaming a missing metric."""
    model = RewardModel({"a.x": {}})
    rewards = torch.tensor([model.aggregate_metrics_output({"a.x": float("nan")})] * 4)

    with pytest.raises(ValueError, match="NaN"):
        _check_rankable(rewards, torch.ones(2, dtype=torch.bool), 15, model.weights)


def test_a_finite_candidate_keeps_the_prune_rankable():
    _check_rankable(torch.tensor([0.5, NEG_INF]), torch.ones(1, dtype=torch.bool), 15, ["a.x"])


def test_nan_candidate_is_not_selected():
    """Beam search keeps the top candidates of ``torch.argsort(rewards, descending=True, stable=True)``."""
    model = RewardModel({"a.x": {}})
    rewards = torch.tensor([model.aggregate_metrics_output({"a.x": v}) for v in (0.5, float("nan"), 0.9, 0.7)])
    assert torch.argsort(rewards, descending=True, stable=True)[:2].tolist() == [2, 3]
