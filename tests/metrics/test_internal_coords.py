"""Tests for internal coordinate conditioning metrics and utilities."""

import numpy as np
import torch
import pytest
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_CRD
from atomworks.ml.utils.geometry import random_rigid_augmentation
from atomworks.ml.utils.testing import cached_parse
from biotite.structure import superimpose

from proteinfoundation.conditions import C_CRDGRP, C_INTCRD
from proteinfoundation.metrics.gt_structure_comparison import InternalCoordAdherenceMetric
from proteinfoundation.transforms.utils.conditioning_utils import (
    convert_to_internal_coords,
    insert_internal_coord_conditioned_groups,
)


def setup_two_internal_coord_groups(atom_array, apply_rotations=True):
    """Helper to create two internal coordinate groups with different rotations.

    Args:
        atom_array: Input AtomArray
        apply_rotations: If True, apply different random rotations to each group

    Returns:
        Modified atom_array with two coordinate groups set up
    """
    n_atoms = atom_array.array_length()

    # Split into two equal groups
    group1_mask = np.zeros(n_atoms, dtype=bool)
    group2_mask = np.zeros(n_atoms, dtype=bool)
    group1_mask[: n_atoms // 2] = True
    group2_mask[n_atoms // 2 :] = True

    # Set up coordinate groups
    coord_groups = np.full(n_atoms, C_CRDGRP.UNKNOWN_COORDS, dtype=int)
    coord_groups[group1_mask] = 1
    coord_groups[group2_mask] = 2
    C_CRDGRP.set_annotation(atom_array, coord_groups)

    # Set internal coordinate masks
    intcrd_mask = np.zeros(n_atoms, dtype=bool)
    intcrd_mask[group1_mask | group2_mask] = True
    C_INTCRD.set_mask(atom_array, intcrd_mask)

    if apply_rotations:
        # Apply different random rotations to each group (but keep them centered)
        for group_mask, group_id in [(group1_mask, 1), (group2_mask, 2)]:
            group_coords = atom_array.coord[group_mask]
            centered = group_coords - group_coords.mean(axis=0, keepdims=True)
            rotated = random_rigid_augmentation(torch.from_numpy(centered)[None], batch_size=1, s=0)[0].numpy()
            atom_array.coord[group_mask] = rotated

    return atom_array


@pytest.mark.parametrize("pdb_id", ["6lyz"])
@pytest.mark.parametrize(
    "deviation_type,expected_rmsd_range",
    [
        ("none", (0.0, 0.1)),  # Perfect alignment, RMSD should be near zero
        ("small_noise", (0.3, 1.5)),  # Small perturbation, RMSD should be small but non-zero
    ],
)
def test_internal_coord_adherence_metric(deterministic_rng, pdb_id, deviation_type, expected_rmsd_range):
    """Test InternalCoordAdherenceMetric with independent alignment per group.

    Tests that each group is aligned independently by applying different rotations
    to each group and verifying the metric handles them correctly.
    """
    # Load and prepare structure
    atom_array = cached_parse(pdb_id)["atom_array"]
    atom_array = as_atom_array_plus(atom_array)

    # NaNs cause problems, so we zero them out -- unphysical but sufficient for testing
    atom_array.coord = np.nan_to_num(atom_array.coord)

    # Set up two groups with different rotations
    atom_array = setup_two_internal_coord_groups(atom_array, apply_rotations=True)

    # Create predicted structure
    predicted = atom_array.copy()

    if deviation_type == "small_noise":
        # Add small random noise
        noise = np.random.randn(*predicted.coord.shape) * 0.5
        predicted.coord = predicted.coord + noise
    # else: no deviation, predicted is identical to ground truth

    # Compute metric
    metric = InternalCoordAdherenceMetric(metric_type="rmsd", gt_coord_annotation_name="coord")
    result = metric.compute(ground_truth_atom_array_stack=atom_array, predicted_atom_array_stack=predicted)

    # Verify output structure
    assert isinstance(result, dict)
    assert "rmsd" in result  # Overall mean across groups
    assert "rmsd_coord_group_1" in result
    assert "rmsd_coord_group_2" in result

    # Verify RMSD is in expected range
    min_rmsd, max_rmsd = expected_rmsd_range
    assert (
        min_rmsd <= result["rmsd"] <= max_rmsd
    ), f"Overall RMSD {result['rmsd']:.3f} not in expected range [{min_rmsd}, {max_rmsd}]"
    assert min_rmsd <= result["rmsd_coord_group_1"] <= max_rmsd
    assert min_rmsd <= result["rmsd_coord_group_2"] <= max_rmsd


@pytest.mark.parametrize("pdb_id", ["6lyz"])
def test_convert_to_internal_coords(deterministic_rng, pdb_id):
    """Test conversion from absolute to internal coordinate conditioning."""
    atom_array = cached_parse(pdb_id)["atom_array"]
    atom_array = as_atom_array_plus(atom_array)
    n_atoms = atom_array.array_length()

    # NaNs cause problems, so we zero them out -- unphysical but sufficient for testing
    atom_array.coord = np.nan_to_num(atom_array.coord)

    # Set up absolute coordinate conditioning on first half
    abs_coord_mask = np.zeros(n_atoms, dtype=bool)
    abs_coord_mask[: n_atoms // 2] = True
    C_CRD.set_mask(atom_array, abs_coord_mask)

    # Convert first quarter to internal coords
    internal_mask = np.zeros(n_atoms, dtype=bool)
    internal_mask[: n_atoms // 4] = True
    atom_array = convert_to_internal_coords(atom_array, internal_mask)

    # Verify C_CRD mask was updated (internal region removed from absolute coords)
    assert not C_CRD.mask(atom_array)[internal_mask].any()
    assert C_CRD.mask(atom_array)[n_atoms // 4 : n_atoms // 2].all()

    # Verify C_INTCRD mask was set
    assert C_INTCRD.mask(atom_array)[internal_mask].all()

    # Verify unique coordinate group was created
    coord_groups = C_CRDGRP.annotation(atom_array)
    internal_group_id = coord_groups[internal_mask][0]
    assert internal_group_id not in [C_CRDGRP.KNOWN_ABSOLUTE_COORDS, C_CRDGRP.UNKNOWN_COORDS]
    assert np.all(coord_groups[internal_mask] == internal_group_id)

    # Verify internal coords are centered
    internal_coords = C_INTCRD.annotation(atom_array)[internal_mask]
    assert np.allclose(internal_coords.mean(axis=0), 0, atol=1e-3), "Internal coords should be centered"


@pytest.mark.parametrize("pdb_id", ["6lyz"])
def test_insert_internal_coord_conditioned_groups(deterministic_rng, pdb_id):
    """Test alignment and insertion of internal coordinate groups."""
    atom_array = cached_parse(pdb_id)["atom_array"]
    atom_array = as_atom_array_plus(atom_array)
    n_atoms = atom_array.array_length()

    # Create internal coordinate group (first quarter of atoms)
    group_mask = np.zeros(n_atoms, dtype=bool)
    group_mask[: n_atoms // 4] = True

    # Set up coordinate group annotation
    coord_groups = np.full(n_atoms, C_CRDGRP.UNKNOWN_COORDS, dtype=int)
    coord_groups[group_mask] = 1
    C_CRDGRP.set_annotation(atom_array, coord_groups)

    # Set internal coordinate mask
    intcrd_mask = np.zeros(n_atoms, dtype=bool)
    intcrd_mask[group_mask] = True
    C_INTCRD.set_mask(atom_array, intcrd_mask)

    # Store original coords and create centered internal coords
    original_coords = np.nan_to_num(atom_array.coord.copy())
    original_group_coords = original_coords[group_mask]
    centered = original_group_coords - original_group_coords.mean(axis=0, keepdims=True)

    internal_coord_annotation = np.full((n_atoms, 3), np.nan)
    internal_coord_annotation[group_mask] = centered
    C_INTCRD.set_annotation(atom_array, internal_coord_annotation)

    # Create "generated" coords with random rotation/translation
    generated_coords = atom_array.coord.copy()
    generated_coords[group_mask] = random_rigid_augmentation(torch.from_numpy(centered)[None], batch_size=1, s=1.0)[
        0
    ].numpy()
    atom_array.coord = generated_coords
    atom_array.set_annotation("original_input_coord", original_coords)

    # Run insertion
    atom_array = insert_internal_coord_conditioned_groups(atom_array, original_input_coords=original_coords)

    # Verify coords_with_insertion was created
    assert "coords_with_insertion" in atom_array.get_annotation_categories()
    inserted_coords = atom_array.coords_with_insertion[group_mask]

    # Verify inserted coords match aligned original coords
    aligned_original, _ = superimpose(fixed=generated_coords[group_mask], mobile=original_coords[group_mask])
    assert np.allclose(inserted_coords, aligned_original, atol=1e-3)


@pytest.mark.parametrize("pdb_id", ["6lyz"])
def test_coordinate_group_default_annotation(deterministic_rng, pdb_id):
    """Test CoordinateGroup default annotation assigns correct groups."""
    atom_array = cached_parse(pdb_id)["atom_array"]
    atom_array = as_atom_array_plus(atom_array)
    n_atoms = atom_array.array_length()

    # Set up absolute coordinate conditioning on half the atoms
    abs_mask = np.zeros(n_atoms, dtype=bool)
    abs_mask[: n_atoms // 2] = True
    C_CRD.set_mask(atom_array, abs_mask)

    # Set up internal coordinate conditioning on a quarter of atoms
    int_mask = np.zeros(n_atoms, dtype=bool)
    int_mask[n_atoms // 2 : 3 * n_atoms // 4] = True
    C_INTCRD.set_mask(atom_array, int_mask)

    # Get default coordinate group annotation
    coord_groups = C_CRDGRP.default_annotation(atom_array)

    # Verify group assignments
    assert np.all(coord_groups[abs_mask] == C_CRDGRP.KNOWN_ABSOLUTE_COORDS)
    assert np.all(coord_groups[int_mask] == 1)

    unconditioned_mask = ~abs_mask & ~int_mask
    assert np.all(coord_groups[unconditioned_mask] == C_CRDGRP.UNKNOWN_COORDS)


if __name__ == "__main__":
    pytest.main([__file__])
