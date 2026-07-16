"""Reward models for search-based protein design.

RewardModel is a pure aggregator that composes any Metric instances and weights
them to produce scalar rewards for search.

For specialized metrics (RF3, AF2, TMOL), see rfproteina.metrics.
"""

from rfproteina.rewards.reward_model import SEARCH_REWARD_KEY, RewardModel

__all__ = ["SEARCH_REWARD_KEY", "RewardModel"]
