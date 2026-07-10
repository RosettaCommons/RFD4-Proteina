"""Reward models for search-based protein design.

RewardModel is a pure aggregator that composes any Metric instances and weights
them to produce scalar rewards for search.

For specialized metrics (RF3, AF2, TMOL), see proteinfoundation.metrics.
"""

from proteinfoundation.rewards.reward_model import RewardModel

__all__ = ["RewardModel"]
