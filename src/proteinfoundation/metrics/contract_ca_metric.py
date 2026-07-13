"""Trivial differentiable guidance metric: contract CA coordinates toward the origin.

A minimal, dependency-free differentiable metric used to exercise and test the guidance
path end to end. Its output ``sq_norm`` is the summed squared norm of the backbone CA
coordinates (a *cost*); give it a negative-weight schedule to contract the protein.
"""

from functools import cached_property

from jaxtyping import Float
from torch import Tensor

from foundry.metrics.metric import Metric


class ContractCAMetric(Metric):
    """Sum of squared backbone-CA norms (negative-weight it to contract the protein)."""

    differentiable = True

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, tuple[str, str]]:
        """Read ``bb_ca`` from the differentiable ``state`` namespace."""
        return {"bb_ca": ("state", "bb_ca")}

    def compute(self, bb_ca: Float[Tensor, "b n 3"]) -> dict[str, Tensor]:
        """Return the summed squared CA norm.

        Args:
          bb_ca: Backbone CA coordinates. Shape: [b, n, 3].

        Returns:
          ``{"sq_norm": <scalar tensor>}`` with grad intact when ``bb_ca`` requires grad.
        """
        return {"sq_norm": (bb_ca**2).sum()}
