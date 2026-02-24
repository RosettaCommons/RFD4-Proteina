import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray


def concat_atom_arrays_without_conflict(atom_array: AtomArray, atom_array_to_add: AtomArray) -> AtomArray:
    """Concatenate two atom arrays without causing conflicts.

    Args:
        atom_array: The atom array to add the chain to.
        atom_array_to_add: The atom array to add.
    """
    # handle chain_id, chain_iid, pn_unit_id, and pn_unit_iid conflicts
    possible_chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    used_chain_ids = np.unique(atom_array.chain_id)
    unused_chain_ids = [chain_id for chain_id in possible_chain_ids if chain_id not in used_chain_ids]
    chain_rename = {}
    for i, used_chain_id in enumerate(np.unique(atom_array_to_add.chain_id)):
        chain_rename[used_chain_id] = unused_chain_ids[i]

    def _rename_chain_in_annotation(
        aa: AtomArray, annotation: str, chain_rename: dict = chain_rename
    ) -> None:  # does this in place
        annotation_values = aa.get_annotation(annotation)
        annotation_values = [
            annotation_value
            if (letter := annotation_value.split("_")[0]) not in chain_rename
            else annotation_value.replace(letter, chain_rename[letter])
            for annotation_value in annotation_values
        ]
        aa.set_annotation(annotation, annotation_values)

    _rename_chain_in_annotation(atom_array_to_add, "chain_id")
    _rename_chain_in_annotation(atom_array_to_add, "chain_iid")
    _rename_chain_in_annotation(atom_array_to_add, "pn_unit_id")
    _rename_chain_in_annotation(atom_array_to_add, "pn_unit_iid")

    # also need to handle molecule_identity, transformation_id, pn_unit_entity, molecule_id, molecule_iid, chain_entity
    atom_array_to_add.chain_entity += np.max(atom_array.chain_entity) + 1
    atom_array_to_add.pn_unit_entity += np.max(atom_array.pn_unit_entity) + 1
    atom_array_to_add.molecule_entity += np.max(atom_array.molecule_entity) + 1
    atom_array_to_add.molecule_id += np.max(atom_array.molecule_id) + 1
    atom_array_to_add.molecule_iid += np.max(atom_array.molecule_iid) + 1

    atom_array = struc.concatenate([atom_array, atom_array_to_add])
    return atom_array


def add_chain_nearby_without_clash(
    atom_array: AtomArray,
    atom_array_to_add: AtomArray,
    chain_id: str | None = None,
    min_distance: float = 5.0,
    direction: np.ndarray[float, np.dtype[np.float32]] | None = np.array([0.0, 0.0, 1.0]),
) -> AtomArray:
    """Add a chain nearby without causing clashes.

    Works by first putting the new chain on the COM of the original atom array. Then we scoot it away from the original chain by 1.5A at a time
    in the direction of the direction vector. Once the minimum distance between the two chains is reached, we settle on the location

    Args:
        atom_array: The atom array to add the chain to.
        atom_array_to_add: The atom array to add.
        chain_id: The chain ID to grab from atom_array_to_add. If None, will use all the chains in atom_array_to_add.
        min_distance: The minimum distance between the two chains.
        direction: The direction to add the chain in. If None, will use a random direction. Stochasticity is not recommended for testing.

    Returns:
        The new atom array with the chain added.
    """
    orig_com = np.nanmean(atom_array.coord, axis=0)

    atom_array_to_add = atom_array_to_add.copy()
    if chain_id is not None:
        atom_array_to_add = atom_array_to_add[atom_array_to_add.chain_id == chain_id]

    if direction is None:
        direction = np.random.randn(3)
    # normalize direction vector
    direction = direction / np.linalg.norm(direction)

    current_coords = atom_array_to_add.coord - np.nanmean(atom_array_to_add.coord, axis=0) + orig_com
    while (
        np.nanmin(np.linalg.norm(current_coords[np.newaxis, :, :] - atom_array.coord[:, np.newaxis, :], axis=-1))
        < min_distance
    ):
        current_coords += direction * 1.5

    atom_array_to_add.coord = current_coords

    return concat_atom_arrays_without_conflict(atom_array, atom_array_to_add)
