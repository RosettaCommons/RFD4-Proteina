"""Shared structure export utilities used by flow and AE validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from atomworks.io.utils.atom_array_plus import as_atom_array_plus, concatenate_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.io.utils.standard_annotations import S_SEGMAX, S_SEGMIN
from atomworks.ml.conditions import C_CRD, C_IDX
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
from atomworks.ml.transforms.encoding import atom_array_from_encoding
from atomworks.ml.transforms.filters import remove_nucleic_acid_terminal_oxygen, remove_protein_terminal_oxygen
from atomworks.ml.utils.token import apply_token_wise, get_token_count, get_token_starts
from biotite.structure import AtomArray
from torch._tensor import Tensor

from rfproteina.conditions import C_INTCRD
from rfproteina.inference_utils.insertion_mapping import build_predicted_insertion_mapping
from rfproteina.transforms.utils.conditioning_utils import (
    align_atom_array_to_input_frame,
    insert_internal_coord_conditioned_groups,
)
from rfproteina.utils.coordinate_transfer import copy_generated_coordinates_to_input
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.pdb import _prepare_bondless_atom_array_for_cif
from rfproteina.utils.standard_annotations import SA_CWI

logger = RankedLogger(__name__, rank_zero_only=True)


def process_single_sample_to_atom_array(
    coords: torch.Tensor,
    residue_type: torch.Tensor,
    padding_mask: torch.Tensor,
    input_atom_array: AtomArray,
    bb_ca_only: bool = False,
    atom_mask: torch.Tensor | None = None,
    insertion_logits: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Convert one sample's atom37 tensors into annotated AtomArrays.

    Mirrors the flow-model validation export path in ``Proteina._process_single_sample_to_atom_array``.
    """
    if bb_ca_only:
        input_atom_array = input_atom_array[input_atom_array.atom_name == "CA"]
        input_atom_array.atomize = np.full_like(input_atom_array.atomize, True)

    has_idx_atom_lvl = C_IDX.mask(input_atom_array)
    has_idx_any_token_level = apply_token_wise(input_atom_array, has_idx_atom_lvl, np.any)
    has_idx_all_token_level = apply_token_wise(input_atom_array, has_idx_atom_lvl, np.all)
    if not np.all(has_idx_any_token_level == has_idx_all_token_level):
        raise ValueError("Input AtomArray should have either all or none of the atoms indexed per token")

    indexed_mask: Tensor = padding_mask.clone()
    non_padding_indices = np.where(indexed_mask.cpu().numpy())[0]
    indexed_mask[non_padding_indices[~has_idx_all_token_level]] = False

    coords = coords[indexed_mask]
    residue_type = residue_type[indexed_mask]

    # Select [n_valid, n_valid] sub-matrix (indexed + unindexed, no padding)
    if insertion_logits is not None:
        insertion_logits_for_generated = insertion_logits[padding_mask][:, padding_mask]
    else:
        insertion_logits_for_generated = None

    if atom_mask is not None:
        atom_mask = atom_mask[indexed_mask]
    if torch.any(residue_type == UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]):
        raise ValueError("There should be no mask tokens in generated output")

    unindexed_input_atom_array = input_atom_array[~has_idx_atom_lvl].copy()
    input_atom_array = input_atom_array[has_idx_atom_lvl]

    if get_token_count(input_atom_array) != coords.shape[0]:
        raise ValueError("Token counts in input AtomArray and generated coords do not match after filtering")

    all_zeros = (coords == 0.0).all(dim=-1, keepdim=True)
    coords = coords.masked_fill(all_zeros, float("nan"))
    if atom_mask is not None:
        # Use the atom37 mask aligned with residue_type (decoder atom_mask for predicted
        # seq, batch atom37_mask for true seq). Inferring occupancy from coords alone fails
        # when residue_type and coordinate occupancy disagree (e.g. AE seq head vs struct).
        encoded_mask = atom_mask.bool()
    else:
        encoded_mask = ~torch.isnan(coords[..., 0])

    token_starts = get_token_starts(input_atom_array)
    atomized_token_mask = torch.from_numpy(input_atom_array.atomize[token_starts]).to(residue_type.device)
    if torch.any(atomized_token_mask):
        if not torch.all(residue_type[atomized_token_mask] == -1):
            raise ValueError("Atomized tokens should have -1 residue type")
        if not torch.all(coords[atomized_token_mask, 1, :].isfinite()):
            raise ValueError("Atomized tokens should have CA atom occupied")
        if not torch.all(
            coords[atomized_token_mask][:, torch.arange(coords.shape[1], device=coords.device) != 1, :].isnan()
        ):
            raise ValueError("Non-CA atoms should be NaN for atomized tokens")
        residue_type = residue_type.clone()
        residue_type[atomized_token_mask] = UNIFIED_ATOM37_ENCODING.token_to_idx["<A>"]

    try:
        generated_array = atom_array_from_encoding(
            encoded_coord=coords,
            encoded_mask=encoded_mask,
            encoded_seq=residue_type,
            encoding=UNIFIED_ATOM37_ENCODING,
            token_id=input_atom_array.token_id[token_starts],
            chain_type=input_atom_array.chain_type[token_starts],
            token_is_atom=input_atom_array.atomize[token_starts],
        )
    except AssertionError as exc:
        n_occupied = int(encoded_mask.sum().item())
        raise ValueError(
            "Residue types and atom occupancy are inconsistent for CIF export "
            f"({n_occupied} occupied atom37 slots across {coords.shape[0]} tokens). "
            "When exporting AE reconstructions with a predicted sequence, pass the "
            "decoder atom_mask so occupancy matches the sequence template."
        ) from exc
    generated_array = remove_protein_terminal_oxygen(generated_array)
    generated_array = remove_nucleic_acid_terminal_oxygen(generated_array)
    input_atom_array = remove_protein_terminal_oxygen(input_atom_array)
    input_atom_array = remove_nucleic_acid_terminal_oxygen(input_atom_array)
    if get_token_count(generated_array) != get_token_count(input_atom_array):
        raise ValueError("Token counts in input and generated AtomArrays do not match after removing terminal oxygens")

    updated_array = copy_generated_coordinates_to_input(
        generated_atom_array=generated_array,
        input_atom_array=input_atom_array,
    )

    # Stage the motif-RMSD reference only for conditioned designs; C_CRD.annotation is the absolute overlay.
    crd_mask = C_CRD.mask(updated_array)
    if crd_mask.any() or C_INTCRD.mask(updated_array).any():
        coords_with_insertion = updated_array.coord.copy()
        coords_with_insertion[crd_mask] = C_CRD.annotation(updated_array, default="generate")[crd_mask]
        SA_CWI.set_annotation(updated_array, np.asarray(coords_with_insertion, dtype=np.float32))

    C_INTCRD.set_mask(input_atom_array, C_INTCRD.mask(input_atom_array))
    C_INTCRD.set_mask(updated_array, C_INTCRD.mask(updated_array))
    updated_array = insert_internal_coord_conditioned_groups(updated_array)

    S_SEGMIN.set_mask(updated_array, np.zeros_like(S_SEGMIN.mask(updated_array)))
    S_SEGMAX.set_mask(updated_array, np.zeros_like(S_SEGMAX.mask(updated_array)))
    S_SEGMIN.set_annotation(updated_array, S_SEGMIN.default_annotation(updated_array))
    S_SEGMAX.set_annotation(updated_array, S_SEGMAX.default_annotation(updated_array))

    if not has_idx_all_token_level.all():
        SA_CWI.set_annotation(unindexed_input_atom_array, SA_CWI.default_annotation(unindexed_input_atom_array))
        unindexed_input_atom_array.coord = unindexed_input_atom_array.coord_to_be_noised.copy()
        full_array_with_unindexed = as_atom_array_plus(
            concatenate_atom_array_plus([updated_array, unindexed_input_atom_array])
        )
        try:
            predicted_insertion_mapping = build_predicted_insertion_mapping(
                insertion_logits_for_generated, full_array_with_unindexed
            )
        except (AttributeError, AssertionError, ValueError, RuntimeError) as e:
            logger.warning(
                f"Predicted insertion mapping failed for sample: {e} "
                "Emitting sample without unindexed-region annotations."
            )
            predicted_insertion_mapping = {}
    else:
        full_array_with_unindexed = None
        predicted_insertion_mapping = {}

    return {
        "generated_atom_array": as_atom_array_plus(updated_array),
        "input_atom_array": as_atom_array_plus(input_atom_array),
        "full_array_with_unindexed": full_array_with_unindexed,
        "predicted_insertion_mapping": predicted_insertion_mapping,
    }


def write_input_structure_cif(atom_array: AtomArray, path: str | Path) -> None:
    """Write an input AtomArray to CIF, matching flow validation ``_write_input_cif``."""
    atom_array = _prepare_bondless_atom_array_for_cif(atom_array)
    to_cif_file(atom_array, str(path), include_entity_categories=False)


def write_generated_structure_cif(atom_array: AtomArray, path: str | Path) -> None:
    """Write a generated AtomArray to CIF."""
    atom_array = _prepare_bondless_atom_array_for_cif(align_atom_array_to_input_frame(atom_array))
    to_cif_file(atom_array, str(path), include_entity_categories=False)
