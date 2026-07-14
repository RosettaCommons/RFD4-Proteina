"""Tests for internal coordinate conditioning metrics and utilities."""

import numpy as np
import pytest
import torch
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_CRD
from atomworks.ml.utils.geometry import random_rigid_augmentation
from atomworks.ml.utils.testing import cached_parse
from biotite.structure import AtomArray, superimpose

from rfproteina.conditions import C_CRDGRP, C_INTCRD
from rfproteina.metrics.gt_structure_comparison import StructureComparisonMetric
from rfproteina.transforms.utils.conditioning_utils import (
    convert_to_internal_coords,
    insert_internal_coord_conditioned_groups,
)
from rfproteina.utils.standard_annotations import SA_CWI


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

    # Populate the internal coordinate annotation with centered per-group coords (the metric substitutes ground-truth coords with this, so the default NaNs would contaminate the RMSD).
    internal_coord_annotation = np.full((n_atoms, 3), np.nan)
    for group_mask in (group1_mask, group2_mask):
        group_coords = atom_array.coord[group_mask]
        internal_coord_annotation[group_mask] = group_coords - group_coords.mean(axis=0, keepdims=True)
    C_INTCRD.set_annotation(atom_array, internal_coord_annotation)

    if apply_rotations:
        # Apply different random rotations to each group (but keep them centered)
        for group_mask, _group_id in [(group1_mask, 1), (group2_mask, 2)]:
            group_coords = atom_array.coord[group_mask]
            centered = group_coords - group_coords.mean(axis=0, keepdims=True)
            rotated = random_rigid_augmentation(torch.from_numpy(centered)[None], batch_size=1, s=0)[0].numpy()
            atom_array.coord[group_mask] = rotated

    return atom_array


@pytest.mark.parametrize("pdb_id", ["6lyz"])
@pytest.mark.parametrize(
    "noise_sigma,expected_rmsd_range",
    [
        (0.0, (0.0, 1e-6)),  # Identical arrays → RMSD = 0 exactly
        (0.5, (0.1, 2.0)),  # Small noise → small RMSD
    ],
)
def test_subdivide_by_annotation_metric(deterministic_rng, pdb_id, noise_sigma, expected_rmsd_range):
    """StructureComparisonMetric with subdivide_by_annotation scores each annotation group."""
    atom_array = cached_parse(pdb_id)["atom_array"]
    atom_array = as_atom_array_plus(atom_array)
    atom_array.coord = np.nan_to_num(atom_array.coord)

    # Assign all atoms to two groups (no rotations — just verify the subdivide path works)
    atom_array = setup_two_internal_coord_groups(atom_array, apply_rotations=False)

    gt = atom_array
    pred = atom_array.copy()
    if noise_sigma > 0:
        pred.coord = pred.coord + np.random.randn(*pred.coord.shape) * noise_sigma

    metric = StructureComparisonMetric(
        subdivide_by_annotation=C_CRDGRP.full_name,
        metric_type="rmsd",
    )
    result = metric.compute(ground_truth_atom_array=gt, predicted_atom_array=pred)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"rmsd"}

    min_rmsd, max_rmsd = expected_rmsd_range
    assert (
        min_rmsd <= result["rmsd"] <= max_rmsd
    ), f"RMSD {result['rmsd']:.6f} not in expected range [{min_rmsd}, {max_rmsd}]"


def test_subdivide_by_annotation_atom_weighted():
    """Aggregation is RMS-weighted by atom count, not a simple average of per-group RMSDs.

    Group 1: 3 atoms, uniform shift = 1.0 Å  → RMSD₁ = 1.0
    Group 2: 7 atoms, uniform shift = 2.0 Å  → RMSD₂ = 2.0

    Correct (RMS atom-weighted):  sqrt((3*1² + 7*2²) / 10) = sqrt(31/10) ≈ 1.7607
    Simple mean of per-group RMSDs:  (1.0 + 2.0) / 2 = 1.5            ← wrong
    Linear atom-weighted mean:        (3*1.0 + 7*2.0) / 10 = 1.7      ← also wrong
    """
    n1, n2 = 3, 7
    n_total = n1 + n2
    d1, d2 = 1.0, 2.0

    # Build a minimal AtomArray: all coords at origin
    atoms = AtomArray(n_total)
    atoms.coord = np.zeros((n_total, 3), dtype=float)
    atoms.chain_id = np.array(["A"] * n_total)
    atoms.res_id = np.arange(1, n_total + 1)
    atoms.res_name = np.array(["ALA"] * n_total)
    atoms.atom_name = np.array(["CA"] * n_total)
    atoms.element = np.array(["C"] * n_total)
    gt = as_atom_array_plus(atoms)
    gt.set_annotation("test_group", np.array([1] * n1 + [2] * n2, dtype=int))

    # Pred: group 1 shifted by d1 along x, group 2 by d2 along x
    pred_coord = np.zeros((n_total, 3), dtype=float)
    pred_coord[:n1, 0] = d1
    pred_coord[n1:, 0] = d2
    pred = gt.copy()
    pred.coord = pred_coord

    metric = StructureComparisonMetric(
        subdivide_by_annotation="test_group",
        metric_type="rmsd",
    )
    result = metric.compute(ground_truth_atom_array=gt, predicted_atom_array=pred)

    rmsd1, rmsd2 = d1, d2
    expected = np.sqrt((n1 * rmsd1**2 + n2 * rmsd2**2) / n_total)  # sqrt(31/10) ≈ 1.7607

    assert set(result.keys()) == {"rmsd"}
    assert (
        abs(result["rmsd"] - expected) < 1e-6
    ), f"Expected RMS atom-weighted RMSD {expected:.6f}, got {result['rmsd']:.6f}."


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
    assert SA_CWI.has_annotation(atom_array)
    inserted_coords = SA_CWI.annotation(atom_array)[group_mask]

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
