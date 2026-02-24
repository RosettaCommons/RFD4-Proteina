"""Base class for refinement strategies."""

from abc import ABC, abstractmethod

from biotite.structure import AtomArray


class RefinementStrategy(ABC):
    """Abstract base class for gradient-based refinement strategies.

    Refinement strategies are STANDALONE utilities that optimize protein structures
    using gradient-based methods. They work on AtomArrays and can be run:

    1. **Standalone**: Via CLI on arbitrary PDB/CIF files
    2. **Integrated**: Called from proteina.py after generation

    Both BindCraft (sequence-only) and Gradient Ascent (latent optimization)
    share this unified interface.

    Subclasses must implement:
        - refine_atom_arrays(): Main refinement method (standalone)
    """

    def _check_differentiable(self, reward_model) -> None:
        """Validate that reward model is differentiable.

        Args:
            reward_model: Reward model to check

        Raises:
            ValueError: If reward model is not differentiable
        """
        if not reward_model.differentiable:
            raise ValueError(
                f"Refinement requires differentiable reward model, "
                f"but got differentiable={reward_model.differentiable}"
            )

    @abstractmethod
    def refine_atom_arrays(
        self,
        atom_arrays: list[AtomArray],
        reward_model,
        autoencoder=None,
    ) -> list[AtomArray]:
        """Refine structures using gradient-based optimization.

        This is the STANDALONE interface that works on AtomArrays.
        Can be called from CLI on arbitrary PDB/CIF files.

        Args:
            atom_arrays: List of AtomArray structures to refine
            reward_model: Differentiable reward model (must have differentiable=True)
            autoencoder: Autoencoder for latent optimization (required for GradientAscent, not for BindCraft)

        Returns:
            List of refined AtomArray structures

        Raises:
            ValueError: If reward_model.differentiable is False
        """
        raise NotImplementedError
