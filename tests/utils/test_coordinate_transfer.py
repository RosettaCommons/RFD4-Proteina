"""Tests for coordinate transfer utilities."""

import numpy as np
import pytest
from atomworks.io.utils.atom_array_plus import insert_atoms
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.ml.conditions import C_SEQ
from atomworks.ml.encoding_definitions import AF2_ATOM37_WITH_ATOMIZATION
from atomworks.ml.transforms.encoding import atom_array_from_encoding, atom_array_to_encoding
from atomworks.ml.transforms.filters import remove_protein_terminal_oxygen
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray
from projects.proteinfoundation.proteinfoundation.utils.coordinate_transfer import (
    copy_generated_coordinates_to_input,
)
from projects.proteinfoundation.tests.pipeline.test_pipeline_invariants import (
    GLOBAL_DESIGN_TASK_KWARGS,
    parse_and_run_pipeline,
)

from proteinfoundation.transforms.design_task import (
    MotifScaffoldTask,
    PPIHotspotWith3DTarget,
    UnconditionalTask,
)


def apply_test_mutations(atom_array: AtomArray, skip_mask: np.ndarray | None = None) -> AtomArray:
    """Apply test mutations: ASP/GLU→ALA (remove atoms), GLY→ALA (add CB), shift +10.

    Args:
      atom_array: The atom array to mutate.
      skip_mask: Boolean mask of atoms to skip (e.g., seq-conditioned tokens).
    """
    atom_array = atom_array.copy()

    if skip_mask is None:
        skip_mask = np.zeros(len(atom_array), dtype=bool)

    # Mutation 1: ASP/GLU → ALA (remove sidechain atoms) - skip seq-conditioned
    res_to_change_asp_glu = np.isin(atom_array.res_name, ["ASP", "GLU"]) & (~atom_array.atomize) & (~skip_mask)
    atom_array.res_name[res_to_change_asp_glu] = "ALA"

    ala_atom_names = ["N", "CA", "C", "O", "CB"]
    keep_mask = np.isin(atom_array.atom_name, ala_atom_names) | ~res_to_change_asp_glu
    atom_array = atom_array[keep_mask]

    # Update skip_mask after filtering
    skip_mask = skip_mask[keep_mask]

    # Mutation 2: GLY → ALA (add CB atoms) - skip seq-conditioned
    res_to_change_gly = np.isin(atom_array.res_name, ["GLY"]) & (~atom_array.atomize) & (~skip_mask)
    atom_array.res_name[res_to_change_gly] = "ALA"

    cb_atoms_to_add = []
    cb_insertion_indices = []

    for token_id in np.unique(atom_array.token_id[res_to_change_gly]):
        token_mask = atom_array.token_id == token_id
        ca_mask = token_mask & (atom_array.atom_name == "CA")
        if not np.any(ca_mask):
            continue

        ca_idx = np.where(ca_mask)[0][0]
        ca_atom = atom_array[ca_mask]

        cb_atom = ca_atom.copy()
        cb_atom.atom_name[0] = "CB"
        cb_atom.coord[0] = ca_atom.coord[0] + np.array([1.5, 0.0, 0.0])

        cb_atoms_to_add.append(cb_atom)
        cb_insertion_indices.append(ca_idx + 1)

    if len(cb_atoms_to_add) > 0:
        atom_array = insert_atoms(atom_array, cb_atoms_to_add, cb_insertion_indices)

    # Mutation 3: Coordinate shift
    atom_array.coord += 10.0

    return atom_array


def encode_decode_atom_array(atom_array: AtomArray) -> AtomArray:
    """Encode and decode atom array to simulate generation."""
    encoded = atom_array_to_encoding(atom_array, AF2_ATOM37_WITH_ATOMIZATION)
    token_starts = get_token_starts(atom_array)

    generated_array = atom_array_from_encoding(
        encoded_coord=encoded["xyz"],
        encoded_mask=encoded["mask"],
        encoded_seq=encoded["seq"],
        encoding=AF2_ATOM37_WITH_ATOMIZATION,
        token_is_atom=encoded["token_is_atom"],
        token_id=atom_array.token_id[token_starts],
        chain_type=atom_array.chain_type[token_starts],
    )

    return remove_protein_terminal_oxygen(generated_array)


# Design task configurations for testing
DESIGN_TASK_CONFIGS = {
    "motif_scaffold": {
        "transform": MotifScaffoldTask(
            min_motif_size=5,
            max_motif_size=20,
            global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS,
        ),
        "frequency": 1.0,
    },
    "ppi_hotspot_with_3d_target": {
        "transform": PPIHotspotWith3DTarget(
            global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS,
        ),
        "frequency": 1.0,
    },
    "unconditional": {
        "transform": UnconditionalTask(global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS),
        "frequency": 1.0,
    },
}


@pytest.mark.parametrize(
    "design_task_config",
    [
        {"motif_scaffold": DESIGN_TASK_CONFIGS["motif_scaffold"]},
        {"ppi_hotspot_with_3d_target": DESIGN_TASK_CONFIGS["ppi_hotspot_with_3d_target"]},
        {"unconditional": DESIGN_TASK_CONFIGS["unconditional"]},
    ],
)
def test_coordinate_transfer_preserves_annotations(design_task_config):
    """Test coordinate transfer preserves annotations across design tasks.

    Verifies:
        - Annotation preservation (minimal loss)
        - Seq-conditioned tokens: sequence unchanged, coordinates changed
        - Non-seq-conditioned tokens: coordinates transferred correctly
        - Token counts match
    """
    task_name = list(design_task_config.keys())[0]

    # Load structure with specific design task
    input_array = parse_and_run_pipeline(
        pdb_id="5ocm",
        design_tasks=design_task_config,
        seed=42,
        crop_size=256,
    )["atom_array"]

    # Save original coordinates and annotations
    original_coords = input_array.coord.copy()
    initial_annotations = set(get_annotation_categories(input_array, n_body="all"))
    initial_seq_mask = input_array.get_annotation(C_SEQ.get_mask_name()).copy()

    # Apply mutations and generate - skip seq-conditioned tokens
    mutated_array = apply_test_mutations(input_array, skip_mask=initial_seq_mask)
    generated_array = encode_decode_atom_array(mutated_array)

    # Perform coordinate transfer
    updated_array = copy_generated_coordinates_to_input(
        generated_atom_array=generated_array,
        input_atom_array=input_array,
    )

    # VERIFICATION 1: Annotation preservation
    final_annotations = set(get_annotation_categories(updated_array, n_body="all"))
    lost_annotations = initial_annotations - final_annotations

    allowed_losses = {
        "coord_to_be_noised",
        "occupancy",
        "b_factor",
        "atom_id",
        "at_crop_boundary",
    }
    unexpected_losses = lost_annotations - allowed_losses
    assert len(unexpected_losses) == 0, f"Lost unexpected annotations: {unexpected_losses}"

    # VERIFICATION 2: Token counts
    assert set(updated_array.token_id) == set(generated_array.token_id), f"[{task_name}] Token IDs must match"
    assert len(updated_array) == len(generated_array), f"[{task_name}] Atom counts must match"

    # VERIFICATION 3: Sequence-conditioned tokens
    seq_conditioned_tokens = np.unique(input_array.token_id[initial_seq_mask])

    for token_id in seq_conditioned_tokens:
        if token_id not in updated_array.token_id:
            continue

        input_mask = input_array.token_id == token_id
        updated_mask = updated_array.token_id == token_id
        generated_mask = generated_array.token_id == token_id

        # Sequence must be unchanged
        assert np.array_equal(
            input_array.res_name[input_mask], updated_array.res_name[updated_mask]
        ), f"[{task_name}] Seq-conditioned token {token_id} sequence changed"

        # Atom names must match
        assert set(input_array.atom_name[input_mask]) == set(
            updated_array.atom_name[updated_mask]
        ), f"[{task_name}] Seq-conditioned token {token_id} atom names changed"

        # Coordinates must differ from original (shifted by +10)
        for atom_name in input_array.atom_name[input_mask]:
            atom_indices = np.where(input_mask & (input_array.atom_name == atom_name))[0]
            original_atom_coords = original_coords[atom_indices]
            updated_coords = updated_array.coord[updated_mask & (updated_array.atom_name == atom_name)]

            assert not np.allclose(
                original_atom_coords, updated_coords
            ), f"[{task_name}] Seq-conditioned token {token_id} atom {atom_name} coords unchanged"

        # Coordinates must match generated
        for atom_name in updated_array.atom_name[updated_mask]:
            updated_coords = updated_array.coord[updated_mask & (updated_array.atom_name == atom_name)]
            generated_coords = generated_array.coord[generated_mask & (generated_array.atom_name == atom_name)]

            assert np.allclose(
                updated_coords, generated_coords, equal_nan=True
            ), f"[{task_name}] Seq-conditioned token {token_id} coords don't match generated"

    # VERIFICATION 4: All non-single-atom tokens have correct coordinates
    for token_id in np.unique(updated_array.token_id):
        updated_mask = updated_array.token_id == token_id
        generated_mask = generated_array.token_id == token_id

        # Skip single-atom tokens (atomized)
        if np.sum(updated_mask) == 1:
            continue

        # Coordinate shapes must match
        assert (
            updated_array.coord[updated_mask].shape == generated_array.coord[generated_mask].shape
        ), f"[{task_name}] Token {token_id} coordinate shape mismatch"

        # All atoms in updated must be in generated with matching coordinates
        for atom_name in updated_array.atom_name[updated_mask]:
            assert (
                atom_name in generated_array.atom_name[generated_mask]
            ), f"[{task_name}] Token {token_id} missing atom {atom_name} in generated"

            updated_coords = updated_array.coord[updated_mask & (updated_array.atom_name == atom_name)]
            generated_coords = generated_array.coord[generated_mask & (generated_array.atom_name == atom_name)]

            assert np.allclose(
                updated_coords, generated_coords, equal_nan=True
            ), f"[{task_name}] Token {token_id} atom {atom_name} coordinate mismatch"
