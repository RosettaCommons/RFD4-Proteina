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

    @property
    def metric_names(self) -> set[str]:
        """Explicit search metrics to initialize for the validation epoch."""
        return set()

    def configure_metrics(self, model: Any) -> None:
        """Validate metric selection before generation, when needed."""
        return None

    @abstractmethod
    def generate(
        self,
        model: Any,  # Proteina instance
        batch: dict,
    ) -> list[dict[str, Any]]:
        """Generate structures using the search strategy.

        Args:
            model: Proteina model instance
            batch: Input batch from dataloader

        Returns:
            List of generated-structure dicts (formatted).
        """
        raise NotImplementedError
