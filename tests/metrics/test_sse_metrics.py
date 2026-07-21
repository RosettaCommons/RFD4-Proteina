"""Tests for secondary structure element (SSE) evaluation metrics."""

import numpy as np
import pytest
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.utils.testing import cached_parse

from rfproteina.conditions import C_SEC, SSEnum
from rfproteina.metrics.sse_metrics import (
    SecondaryStructureAccuracy,
    SecondaryStructureDistribution,
    annotate_secondary_structure,
)


@pytest.mark.parametrize("pdb_id", ["6lyz", "1fu2"])
@pytest.mark.parametrize("use_stack", [False, True])
def test_secondary_structure_distribution_metric(deterministic_rng, pdb_id, use_stack):
    """Test SSE distribution metric end-to-end on real structures."""
    # Load structure
    atom_array = cached_parse(pdb_id)["atom_array"]

    # Convert to stack if requested
    if use_stack:
        atom_array = ensure_atom_array_stack(atom_array)

    # Initialize and compute
    metric = SecondaryStructureDistribution()
    result = metric.compute(atom_array=atom_array)

    # Verify return structure
    assert isinstance(result, dict)
    base_keys = {"helix_fraction", "sheet_fraction", "loop_fraction", "num_ss_elements"}
    expected_keys = base_keys | {f"{k}_coi" for k in base_keys}
    assert set(result.keys()) == expected_keys

    # Verify values are valid, both overall and for the chain of interest
    for suffix in ("", "_coi"):
        assert 0 <= result[f"helix_fraction{suffix}"] <= 1, f"Invalid helix_fraction{suffix}"
        assert 0 <= result[f"sheet_fraction{suffix}"] <= 1, f"Invalid sheet_fraction{suffix}"
        assert 0 <= result[f"loop_fraction{suffix}"] <= 1, f"Invalid loop_fraction{suffix}"
        assert result[f"num_ss_elements{suffix}"] >= 0, f"Invalid num_ss_elements{suffix}"

        # Verify fractions sum to ~1.0
        total_fraction = (
            result[f"helix_fraction{suffix}"] + result[f"sheet_fraction{suffix}"] + result[f"loop_fraction{suffix}"]
        )
        assert 0.95 <= total_fraction <= 1.05, f"Fractions don't sum to ~1.0: {total_fraction}"

        # Verify num_ss_elements is reasonable for real proteins
        assert result[f"num_ss_elements{suffix}"] > 0, "Real proteins should have SSE elements"


@pytest.mark.parametrize("pdb_id", ["6lyz", "1fu2"])
def test_annotate_secondary_structure_is_residue_uniform(pdb_id):
    """``sse`` and ``sse_is_valid`` must be uniform within every residue.

    This is the invariant that breaks for partially-atomized PTM residues if
    the annotation is computed/spread at token level instead of residue level.
    """
    atom_array = cached_parse(pdb_id)["atom_array"]
    atom_array = annotate_secondary_structure(atom_array)

    residue_start_stops = get_residue_starts(atom_array, add_exclusive_stop=True)
    sse = atom_array.sse
    is_valid = atom_array.sse_is_valid
    for start, stop in zip(residue_start_stops[:-1], residue_start_stops[1:], strict=False):
        assert (
            np.unique(sse[start:stop]).size == 1
        ), f"sse not uniform within residue at atoms [{start}, {stop}): {np.unique(sse[start:stop])}"
        assert (
            np.unique(is_valid[start:stop]).size == 1
        ), f"sse_is_valid not uniform within residue at atoms [{start}, {stop})"


@pytest.mark.parametrize(
    "conditioning_type,expected_accuracy_range",
    [
        ("perfect", (0.95, 1.0)),  # C_SEC matches structure
        ("imperfect", (0.2, 0.8)),  # Some mismatches (wider range due to randomness)
        ("none", None),  # No C_SEC annotation
    ],
)
@pytest.mark.parametrize("use_stack", [False, True])
def test_secondary_structure_accuracy_metric(deterministic_rng, conditioning_type, expected_accuracy_range, use_stack):
    """Test SSE accuracy metric with different conditioning scenarios."""
    # Load structure
    atom_array = cached_parse("6lyz")["atom_array"]
    atom_array = as_atom_array_plus(atom_array)

    # Create generated array (this is the generated structure with C_SEC annotations)
    generated = atom_array.copy()

    # Add C_SEC conditioning based on type
    if conditioning_type == "perfect":
        # Run P-SEA on atom array to get ground truth SSE
        atom_array = annotate_secondary_structure(
            atom_array, annotation_name="sse", is_valid_annotation_name="sse_is_valid"
        )

        # Use P-SEA values directly for conditioning (already SSEnum values)
        c_sec_values = atom_array.sse.copy()

        # Only condition valid protein atoms
        c_sec_values[~atom_array.sse_is_valid] = SSEnum.NONE

        # Condition a subset (not all atoms)
        conditioning_mask = np.random.rand(len(c_sec_values)) < 0.5
        c_sec_values[~conditioning_mask] = SSEnum.NONE

        C_SEC.set_annotation(generated, c_sec_values)

    elif conditioning_type == "imperfect":
        # Create random conditioning that won't match perfectly
        c_sec_values = np.random.choice(
            [SSEnum.NONE, SSEnum.HELIX, SSEnum.SHEET, SSEnum.LOOP],
            size=len(generated),
            p=[0.5, 0.2, 0.2, 0.1],  # 50% not conditioned
        )
        C_SEC.set_annotation(generated, c_sec_values)

    elif conditioning_type == "none":
        # Don't add C_SEC annotation
        pass

    # Convert to stack if requested
    if use_stack:
        generated = ensure_atom_array_stack(generated)

    # Initialize metric
    metric = SecondaryStructureAccuracy()

    # Compute
    result = metric.compute(atom_array=generated)

    # Verify results based on conditioning type
    if conditioning_type == "none":
        # Should return empty dict when no C_SEC
        assert result == {}, f"Expected empty dict for no C_SEC, got {result}"
    else:
        # Should return accuracy metrics
        assert isinstance(result, dict)
        expected_keys = {
            "overall_accuracy",
            "helix_recall",
            "sheet_recall",
            "loop_recall",
            "n_conditioned_atoms",
        }
        assert set(result.keys()) == expected_keys

        # Check accuracy is in expected range
        if expected_accuracy_range:
            min_acc, max_acc = expected_accuracy_range
            assert (
                min_acc <= result["overall_accuracy"] <= max_acc
            ), f"Accuracy {result['overall_accuracy']} not in range [{min_acc}, {max_acc}]"

        # Verify per-class recalls are valid (0-1 or NaN)
        for key in ["helix_recall", "sheet_recall", "loop_recall"]:
            val = result[key]
            assert np.isnan(val) or (0 <= val <= 1), f"Invalid recall for {key}: {val}"

        # Verify n_conditioned_atoms is positive
        assert result["n_conditioned_atoms"] > 0, "Should have conditioned atoms"
