"""Base class for search strategies."""

from abc import ABC, abstractmethod
from typing import Any


class SearchStrategy(ABC):
    """Abstract base class for search strategies.

    Search strategies guide protein generation by exploring the generation space
    with reward models. Different strategies implement different exploration/exploitation
    trade-offs.

    Subclasses must implement:
        - generate(): Main generation method that uses the model and rewards
    """

    @abstractmethod
    def generate(
        self,
        model: Any,  # Proteina instance
        batch: dict,
        reward_model: Any | None = None,  # RewardModel instance
    ) -> list[dict[str, Any]]:
        """Generate structures using the search strategy.

        Args:
            model: Proteina model instance
            batch: Input batch from dataloader
            reward_model: Optional reward model for guided generation

        Returns:
            List of dictionaries containing generated structures and metadata.
            Each dict should have keys like 'generated_atom_array', 'reward', etc.
        """
        raise NotImplementedError
