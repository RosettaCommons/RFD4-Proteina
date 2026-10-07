"""Reward models for search-based protein design.

RewardModel combines the outputs of explicitly selected metrics into a scalar
reward for one search checkpoint.

For specialized metrics (RF3, AF2, TMOL), see rfproteina.metrics.
"""

from rfproteina.rewards.reward_model import SEARCH_REWARD_KEY, RewardModel

__all__ = ["SEARCH_REWARD_KEY", "RewardModel"]
