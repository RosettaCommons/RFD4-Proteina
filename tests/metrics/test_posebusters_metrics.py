"""Tests for PoseBusters ligand validation metric."""

import pytest
from atomworks.ml.utils.testing import cached_parse

from proteinfoundation.metrics.posebusters_metrics import (
    PoseBustersMetric,
    aggregate_ligand_results,
    validate_ligands,
    validate_single_ligand,
)


def _has_posebusters() -> bool:
    """Check if PoseBusters is installed."""
    try:
        from posebusters import PoseBusters  # noqa: F401

        return True
    except ImportError:
        return False


has_posebusters = _has_posebusters()
skip_if_no_posebusters = pytest.mark.skipif(not has_posebusters, reason="PoseBusters not installed")


def _extract_sa8_ligand():
    """Extract SA8 ligand from 6wtf structure."""
    from atomworks.ml.conditions.annotator import ensure_annotations

    atom_array = cached_parse("6wtf")["atom_array"]
    ensure_annotations(atom_array, "is_ligand")

    # Filter to SA8 ligand only (chain E, res_id 803)
    ligand_mask = atom_array.is_ligand
    ligand_atoms = atom_array[ligand_mask]
    sa8_mask = (ligand_atoms.chain_id == "E") & (ligand_atoms.res_id == 803) & (ligand_atoms.res_name == "SA8")
    return ligand_atoms[sa8_mask]


@skip_if_no_posebusters
def test_posebusters_clean_sa8_passes():
    """Test that clean SA8 ligand from 6wtf passes all PoseBusters checks."""
    sa8_ligand = _extract_sa8_ligand()

    metric = PoseBustersMetric(output_level="aggregate")
    result = metric.compute(atom_array_stack=sa8_ligand)

    # Should have exactly 1 ligand
    assert result["n_ligands"] == 1
    assert result["n_pb_bust_failed"] == 0
    assert result["n_rdkit_conversion_failed"] == 0

    # Should pass all checks (100% pass rate)
    assert result["pass_rate"] == 1.0
    assert result["n_passed"] == 1


@skip_if_no_posebusters
def test_posebusters_mutated_sa8_fails():
    """Test that mutated SA8 ligand fails geometry checks."""
    sa8_ligand = _extract_sa8_ligand().copy()

    # Collapse coordinates to induce bond length/angle failures
    sa8_ligand.coord = sa8_ligand.coord * 0.5

    metric = PoseBustersMetric(output_level="per_check")
    result = metric.compute(atom_array_stack=sa8_ligand)

    # Should have exactly 1 ligand that was validated
    assert result["n_ligands"] == 1
    assert result["n_pb_bust_failed"] == 0
    assert result["n_rdkit_conversion_failed"] == 0

    # Should fail validation (0% pass rate)
    assert result["pass_rate"] == 0.0
    assert result["n_passed"] == 0

    # Specifically bond_lengths and bond_angles should fail
    assert result["bond_lengths_pass_rate"] == 0.0
    assert result["bond_angles_pass_rate"] == 0.0


@skip_if_no_posebusters
@pytest.mark.parametrize("output_level", ["aggregate", "per_check", "per_ligand"])
def test_posebusters_output_levels(output_level: str):
    """Test that different output levels produce expected keys."""
    sa8_ligand = _extract_sa8_ligand()

    metric = PoseBustersMetric(output_level=output_level)
    result = metric.compute(atom_array_stack=sa8_ligand)

    # All levels should have these base keys
    assert "pass_rate" in result
    assert "n_ligands" in result
    assert "n_passed" in result
    assert "n_pb_bust_failed" in result
    assert "n_rdkit_conversion_failed" in result

    if output_level in ("per_check", "per_ligand"):
        # Should have per-check pass rates
        check_keys = [k for k in result if k.endswith("_pass_rate") and k != "pass_rate"]
        assert len(check_keys) > 0, "Expected per-check pass rates"

    if output_level == "per_ligand":
        # Should have per-ligand results
        ligand_keys = [k for k in result if k.startswith("ligand_")]
        assert len(ligand_keys) > 0, "Expected per-ligand results"


def _extract_all_ligands():
    """Extract all ligands from 6wtf structure."""
    from atomworks.ml.conditions.annotator import ensure_annotations

    atom_array = cached_parse("6wtf")["atom_array"]
    ensure_annotations(atom_array, "is_ligand")
    return atom_array[atom_array.is_ligand]


@skip_if_no_posebusters
def test_posebusters_multiple_ligands():
    """Test PoseBusters validation with multiple ligands from 6wtf.

    6wtf contains 4 ligands:
      - SF4 (iron-sulfur cluster): Fails some validation checks (metal complex)
      - B12 (cobalamin/vitamin B12): PoseBusters fails (SVD convergence issue)
      - SA8: Passes validation
      - TRP (modified tryptophan): Passes validation
    """
    all_ligands = _extract_all_ligands()

    metric = PoseBustersMetric(output_level="per_ligand")
    result = metric.compute(atom_array_stack=all_ligands)

    # Should detect all 4 ligands
    assert result["n_ligands"] == 4

    # B12 fails to process (SVD convergence); SF4, SA8, TRP are processed
    assert result["n_pb_bust_failed"] == 1
    assert result["n_rdkit_conversion_failed"] == 0

    # Of the 3 processed, SA8 and TRP pass; SF4 fails validation checks
    assert result["n_passed"] == 2

    # Verify per-ligand results are present
    ligand_ids = [result.get(f"ligand_{i}_id") for i in range(4)]
    assert "C/SF4/801" in ligand_ids
    assert "D/B12/802" in ligand_ids
    assert "E/SA8/803" in ligand_ids
    assert "F/TRP/804" in ligand_ids


@skip_if_no_posebusters
def test_validate_ligands_standalone():
    """Test validate_ligands function directly without metric class."""
    atom_array = cached_parse("6wtf")["atom_array"]

    results = validate_ligands(atom_array)

    # Should return list of per-ligand results
    assert isinstance(results, list)
    assert len(results) == 4

    # Check structure of results
    for r in results:
        assert "ligand_id" in r
        assert "n_atoms" in r
        assert "pb_bust_failed" in r
        assert "rdkit_conversion_failed" in r
        assert "passes_validation" in r


@skip_if_no_posebusters
def test_validate_single_ligand_standalone():
    """Test validate_single_ligand function directly."""
    sa8_ligand = _extract_sa8_ligand()

    result = validate_single_ligand(sa8_ligand, ligand_id="test_sa8")

    assert result["ligand_id"] == "test_sa8"
    assert result["n_atoms"] == len(sa8_ligand)
    assert result["pb_bust_failed"] is False
    assert result["rdkit_conversion_failed"] is False
    assert result["passes_validation"] is True

    # Should have individual check results
    assert "bond_lengths" in result
    assert "bond_angles" in result


def test_aggregate_ligand_results_empty():
    """Test aggregate_ligand_results with empty input."""
    result = aggregate_ligand_results([])

    assert result["n_ligands"] == 0
    assert result["n_passed"] == 0
    assert result["n_pb_bust_failed"] == 0
    assert result["n_rdkit_conversion_failed"] == 0


def test_aggregate_ligand_results_mixed():
    """Test aggregate_ligand_results with mixed pass/fail results."""
    ligand_results = [
        {"ligand_id": "A", "passes_validation": True, "pb_bust_failed": False, "rdkit_conversion_failed": False},
        {"ligand_id": "B", "passes_validation": False, "pb_bust_failed": False, "rdkit_conversion_failed": False},
        {"ligand_id": "C", "passes_validation": True, "pb_bust_failed": True, "rdkit_conversion_failed": False},
    ]

    result = aggregate_ligand_results(ligand_results, output_level="aggregate")

    assert result["n_ligands"] == 3
    assert result["n_pb_bust_failed"] == 1
    assert result["n_rdkit_conversion_failed"] == 0
    # Only A and B were validated (C had pb_bust_failed), A passed, B failed
    assert result["n_passed"] == 1
    assert result["pass_rate"] == 0.5
