import numpy as np
from atomworks.io.utils.atom_array_plus import (
    AtomArrayPlus,
    stack_atom_array_plus,
)
from atomworks.ml.conditions import C_CRD
from beartype.typing import Any
from biotite.structure import AtomArray, AtomArrayStack
from jaxtyping import Float

from proteinfoundation.conditions import C_HOT
from proteinfoundation.metrics.metric import Metric
from proteinfoundation.utils.common import exists
from proteinfoundation.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def hotspot_obedience(
    predicted_atom_array_stack: AtomArray | AtomArrayStack | AtomArrayPlus | None = None,
    predicted_coords: Float[np.ndarray, "D L 3"] | None = None,
    max_hotspot_distance: float = 7.0,
) -> float:
    """
    Computes hotspot obedience. Hotspots are defined as atoms on the target that are
    within a certain distance of the binder. This metric checks for each hotspot atom
    whether there is a binder atom within the hotspot distance. Result is returned as a
    fraction of hotspot atoms that have been obeyed. If there are no hotspots, the
    metric returns NaN.

    Args:
        ground_truth_atom_array_stack (AtomArray | AtomArrayStack | AtomArrayPlus): The ground truth atom array stack.
        predicted_atom_array_stack (AtomArray | AtomArrayStack | AtomArrayPlus | None): The predicted atom array stack.
        predicted_coords (Float[np.ndarray, "D L 3"] | None): The predicted coordinates.

    Returns:
        hotspot_obedience (float): The fraction of hotspot atoms that have been obeyed.
    """

    if predicted_atom_array_stack is None and predicted_coords is None:
        raise ValueError("Either predicted_atom_array_stack or predicted_coords must be provided")
    if predicted_atom_array_stack is not None and predicted_coords is not None:
        raise ValueError("Only one of predicted_atom_array_stack or predicted_coords should be provided")

    if predicted_atom_array_stack is not None:
        if isinstance(predicted_atom_array_stack, list):
            predicted_atom_array_stack = stack_atom_array_plus(predicted_atom_array_stack)
        else:
            predicted_atom_array_stack = (
                stack_atom_array_plus([predicted_atom_array_stack])
                if not isinstance(predicted_atom_array_stack, AtomArrayStack)
                else predicted_atom_array_stack
            )

    # Extract hotspot mask
    annotation = C_HOT.annotation(predicted_atom_array_stack, default="generate")
    if exists(annotation):
        hotspot_mask = annotation.astype(bool)

    if hotspot_mask is None:
        raise ValueError("No hotspot mask found in ground truth or predicted atom array stack")

    hotspot_atoms = predicted_atom_array_stack[0][hotspot_mask]
    if len(hotspot_atoms) == 0:
        return {"hotspot_obedience": np.nan}

    # Get binder mask
    hotspot_chain_iids = np.unique(hotspot_atoms.chain_iid)
    obeyed_hotspots_count = 0
    coordinate_conditioned_mask = C_CRD.mask(predicted_atom_array_stack, default="generate")
    for hotspot_chain_iid in hotspot_chain_iids:
        current_chain_mask = predicted_atom_array_stack.chain_iid == hotspot_chain_iid
        current_chain_hotspot_mask = hotspot_mask & current_chain_mask

        if predicted_coords is None:
            predicted_coords = predicted_atom_array_stack.coord
        else:
            predicted_coords = np.asarray(predicted_coords)

        # Calculate hotspot obedience
        predicted_binder_atoms = predicted_coords[:, ~current_chain_mask & ~coordinate_conditioned_mask, :][
            :, np.newaxis, :, :
        ]  # [D, 1, L_DIFFUSED_OTHER_CHAINS, 3]
        current_chain_predicted_hotspot_atoms = predicted_coords[:, current_chain_hotspot_mask, :][
            :, :, np.newaxis, :
        ]  # [D, L_HOT_CURRENT_CHAIN, 1, 3]
        binder_distances = np.linalg.norm(
            predicted_binder_atoms - current_chain_predicted_hotspot_atoms, axis=3
        )  # [D, L_HOT_CURRENT_CHAIN, L_DIFFUSED_OTHER_CHAINS]
        obeyed_hotspots_count += np.sum(np.any(binder_distances < max_hotspot_distance, axis=2), axis=-1)

    hotspot_obedience = obeyed_hotspots_count / np.sum(hotspot_mask)

    return {"hotspot_obedience": hotspot_obedience}


class HotspotObedience(Metric):
    """
    Computes how frequently hotspot conditioining is obeyed. Hotspots are defined as
    atoms on the target that are within a certain distance of the binder. This metric
    checks for each hotspot atom whether there is a binder atom within the hotspot
    distance. Result is returned as a fraction of hotspot atoms that have been obeyed.
    If there are no hotspots, the metric returns NaN.

    Args:
        max_hotspot_distance (float): The maximum distance between a hotspot atom and a binder atom.

    Returns:
        hotspot_obedience (float): The fraction of hotspot atoms that have been obeyed.
    """

    def __init__(self, max_hotspot_distance: float, **kwargs):
        super().__init__(**kwargs)
        self.max_hotspot_distance = max_hotspot_distance

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array_stack": "predicted_atom_array",
        }

    def compute(
        self,
        predicted_atom_array_stack: AtomArray | AtomArrayStack | AtomArrayPlus | None = None,
        predicted_coords: Float[np.ndarray, "D L 3"] | None = None,
    ) -> float:
        """
        Computes hotspot obedience (see class docstring for more details).

        Handles both single structures (AtomArray) and stacks of structures (AtomArrayStack).
        Can accept either predicted atom array stack or predicted coordinates directly.

        Args:
            predicted_atom_array_stack: The predicted atom array stack.
            predicted_coords: The predicted coordinates.

        Returns:
            Dictionary with keys 'hotspot_obedience'
        """

        return hotspot_obedience(
            predicted_atom_array_stack=predicted_atom_array_stack,
            predicted_coords=predicted_coords,
            max_hotspot_distance=self.max_hotspot_distance,
        )
