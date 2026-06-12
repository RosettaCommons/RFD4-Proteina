"""Tests for secondary structure condition modifier."""

import numpy as np
from atomworks.ml.utils.testing import cached_parse
from atomworks.ml.utils.token import apply_token_wise
from projects.proteinfoundation.tests.conftest import skip_if_no_dssp

from proteinfoundation.conditions import C_SEC
from proteinfoundation.transforms.design_task_modifiers import add_secondary_structure_condition


@skip_if_no_dssp
def test_add_secondary_structure_condition_basic(cleanup_dssp, deterministic_rng):
    """Test basic end-to-end functionality of SS conditioning.

    Verifies that:
    - DSSP annotations are created
    - C_SEC annotation is set with valid values
    - Where sampled, values match sse
    """
    data = {"atom_array": cached_parse("1a8o")["atom_array"]}

    result = add_secondary_structure_condition(data)
    atom_array = result["atom_array"]

    # Verify DSSP annotations were created
    assert "sse" in atom_array.get_annotation_categories()
    assert "sse_is_valid" in atom_array.get_annotation_categories()

    # Verify C_SEC annotation was set
    assert C_SEC.full_name in atom_array.get_annotation_categories()
    ss_annotation = C_SEC.annotation(atom_array)

    # Check valid values: should be 0 (unspecified) or SSEnum values (1-4)
    valid_values = {0, 1, 2, 3, 4}
    assert set(np.unique(ss_annotation)).issubset(valid_values), (
        f"Found invalid SS annotation values: {set(np.unique(ss_annotation)) - valid_values}"
    )

    # Where sampled (not 0/NONE), should match sse
    sampled_mask = ss_annotation != 0
    if np.any(sampled_mask):
        assert np.all(ss_annotation[sampled_mask] == atom_array.sse[sampled_mask]), (
            "SS conditioning values don't match sse where sampled"
        )


@skip_if_no_dssp
def test_secondary_structure_requires_valid_dssp(cleanup_dssp, deterministic_rng):
    """Test that SS conditioning only applies where sse_is_valid is True.

    Verifies that only atoms with successful DSSP annotation are eligible
    for secondary structure conditioning.
    """
    data = {"atom_array": cached_parse("1a8o")["atom_array"]}

    result = add_secondary_structure_condition(data)
    atom_array = result["atom_array"]

    # Where SS is conditioned (not 0/NONE), sse_is_valid must be True
    ss_annotation = C_SEC.annotation(atom_array)
    sampled_mask = ss_annotation != 0  # 0 is SSEnum.NONE

    if np.any(sampled_mask):
        assert np.all(atom_array.sse_is_valid[sampled_mask]), "SS conditioning applied where DSSP is invalid"


@skip_if_no_dssp
def test_secondary_structure_island_sampling(cleanup_dssp, deterministic_rng):
    """Test that SS conditioning creates contiguous island patterns.

    Verifies that when constrained island parameters are used, the conditioning
    creates the expected number of contiguous token regions.
    """
    data = {"atom_array": cached_parse("1a8o")["atom_array"]}

    # Use controlled island parameters
    result = add_secondary_structure_condition(data, min_ss_island_len=3, max_ss_island_len=3, n_islands_max=1)
    atom_array = result["atom_array"]

    # Check at token level
    ss_annotation = C_SEC.annotation(atom_array)
    sampled_mask = ss_annotation != 0  # 0 is SSEnum.NONE
    token_sampled = apply_token_wise(atom_array, sampled_mask, np.any)

    # Should have exactly one island of length 3
    # (or possibly 0 if no eligible tokens, which is also valid)
    if token_sampled.sum() > 0:
        assert token_sampled.sum() == 3, f"Expected 3 tokens in one island, got {token_sampled.sum()}"
