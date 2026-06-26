"""Transforms for metal atoms and their coordination bonds."""

import biotite.structure as struc
import numpy as np
from atomworks.constants import METAL_ELEMENTS
from atomworks.enums import ChainType
from atomworks.io.transforms.atom_array import add_id_and_entity_annotations, add_iid_annotations
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.ml.transforms.base import Transform


class IsolateMetalsToOwnChain(Transform):
    """Isolate metal atoms onto their own chains with coordination bonds.

    Normalizes CCD-style structures (metals on same chain as ligand, single
    bonds) to match OMOL convention (metals on own chain, coordination bonds,
    ``res_name`` = element symbol).

    Reads / writes:
      ``data["atom_array"]`` (:py:class:`biotite.structure.AtomArray`): per-atom
      ``chain_id``, ``res_name``, ``res_id``, optional ``is_polymer`` /
      ``chain_type`` / ``transformation_id`` annotations, and
      ``bonds`` (:py:class:`biotite.structure.BondList`) are mutated in place.
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        metal_mask = np.isin(
            np.char.upper(atom_array.element.astype(str)),
            list(METAL_ELEMENTS),
        )
        if not metal_mask.any():
            return data

        existing_chain_ids = set(np.unique(atom_array.chain_id))
        chain_gen = create_chain_id_generator(existing_chain_ids)

        metal_indices = np.where(metal_mask)[0]
        for idx in metal_indices:
            new_chain = next(chain_gen)
            atom_array.chain_id[idx] = new_chain
            atom_array.res_name[idx] = atom_array.element[idx].upper()
            atom_array.atom_name[idx] = atom_array.element[idx].upper()
            atom_array.res_id[idx] = 1

            if "is_polymer" in atom_array.get_annotation_categories():
                atom_array.is_polymer[idx] = False
            if "chain_type" in atom_array.get_annotation_categories():
                atom_array.chain_type[idx] = ChainType.NON_POLYMER.value

            # Convert bonds to/from this metal atom to COORDINATION
            if atom_array.bonds is not None:
                bond_array = atom_array.bonds.as_array()
                mask = (bond_array[:, 0] == idx) | (bond_array[:, 1] == idx)
                bond_array[mask, 2] = int(struc.BondType.COORDINATION)
                atom_array.bonds = struc.BondList(len(atom_array), bond_array)

        # Re-derive id and entity annotations, excluding coordination bonds so
        # identical metal ions share pn_unit_entity and molecule_entity.
        atom_array = add_id_and_entity_annotations(
            atom_array, overwrite=True, exclude_bond_types={struc.BondType.COORDINATION}
        )
        if "transformation_id" in atom_array.get_annotation_categories():
            atom_array = add_iid_annotations(atom_array, overwrite=True)

        # Move metal atoms to the end so they don't interrupt chain/residue contiguity
        metal_mask_final = np.isin(
            np.char.upper(atom_array.element.astype(str)),
            list(METAL_ELEMENTS),
        )
        reorder = np.concatenate([np.where(~metal_mask_final)[0], np.where(metal_mask_final)[0]])
        atom_array = atom_array[reorder]

        data["atom_array"] = atom_array
        return data


class DropoutCoordinationBonds(Transform):
    """Drop out coordination bonds (bond_order=4) with two-level stochastic masking.

    Level 1: With probability ``p_all``, hide ALL coordination bonds in the example.
    Level 2: If kept, each individual coordination bond is independently dropped
    with probability ``p_individual``.

    Modifies ``bond_mask``, ``bond_order``, and ``is_aromatic`` in-place so all
    downstream features stay consistent. Only added to pipeline during training.

    Reads / writes:
      ``data["feats"]["bond_order"]`` (``int16 [n_tokens, n_tokens]``),
      ``data["feats"]["bond_mask"]`` (``bool  [n_tokens, n_tokens]``),
      ``data["feats"]["is_aromatic"]`` (``bool  [n_tokens, n_tokens]``).
      All three matrices must be symmetric; the transform preserves that.
    """

    COORDINATION_ORDER = 4

    def __init__(self, p_all: float = 0.5, p_individual: float = 0.1):
        self.p_all = p_all
        self.p_individual = p_individual

    def forward(self, data: dict) -> dict:
        bond_order = data["feats"]["bond_order"]  # [n_tokens, n_tokens] int16
        coord_mask = bond_order == self.COORDINATION_ORDER  # [n_tokens, n_tokens] bool

        if not coord_mask.any():
            return data

        # Level 1: drop ALL coordination bonds with probability p_all
        if np.random.rand() < self.p_all:
            drop = coord_mask
        else:
            # Level 2: drop individual coordination bonds with probability p_individual
            drop = coord_mask & (np.random.rand(*coord_mask.shape) < self.p_individual)
            # Symmetrize: if (i,j) is dropped, (j,i) must also be dropped
            drop = drop | drop.T

        data["feats"]["bond_order"][drop] = 0
        data["feats"]["bond_mask"][drop] = False
        data["feats"]["is_aromatic"][drop] = False  # coordination bonds shouldn't be aromatic, but stay safe

        return data
