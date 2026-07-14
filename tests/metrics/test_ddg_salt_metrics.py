"""Tests for the charge complementarity metric and its helpers."""

import os

import numpy as np
from atomworks.io import parse

from proteinfoundation.conditions import C_CRD

# Get test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BINDER_STRUCTURE_PATH = os.path.join(TEST_DATA_DIR, "5vht_from_file_model_0.cif")


class TestDDGSaltMetrics:
    """End-to-end tests for the charge complementarity metric."""

    def test_charge_metric_on_binder_structure(self):
        """Test ChargeComplementarityMetric and helpers on a real binder-target complex.

        Uses the 5vht structure (2-chain complex with chains A and B). The structure
        must have multiple chains for interface metrics to work.
        """
        from proteinfoundation.metrics.physics.ddg_salt_metrics import (
            ChargeComplementarityMetric,
            get_binder_target_mask,
            get_charged_residue_positions,
        )

        # Load structure using atomworks parser
        parsed = parse(BINDER_STRUCTURE_PATH)
        atom_array = parsed["asym_unit"][0]

        # Verify multi-chain structure
        unique_chains = np.unique(atom_array.chain_id)
        assert len(unique_chains) >= 2, f"Expected multi-chain structure, got chains: {unique_chains}"

        # -------------------------------------------
        # Test binder/target mask extraction
        # -------------------------------------------

        # Set up C_CRD annotation: chain A = binder (generated), chain B = target (conditioned)
        crd_mask = atom_array.chain_id == "B"  # Chain B is conditioned (target)
        C_CRD.set_annotation(atom_array, crd_mask.astype(np.float32))

        binder_mask, target_mask = get_binder_target_mask(atom_array)
        assert binder_mask.sum() > 0, "Should have binder atoms"
        assert target_mask.sum() > 0, "Should have target atoms"
        assert np.all(binder_mask != target_mask), "Binder and target should be mutually exclusive"

        # Test charged residue extraction
        positions, charges, res_ids, chain_ids = get_charged_residue_positions(atom_array)
        # May or may not have charged residues depending on structure
        assert positions.shape[1] == 3 if len(positions) > 0 else True

        # -------------------------------------------
        # Test ChargeComplementarityMetric (no TMol dependency)
        # -------------------------------------------

        charge_metric = ChargeComplementarityMetric(
            min_dist=7.0,
            max_dist=9.0,
            max_bonus=2.0,
            charge_func="linear",
        )

        charge_results = charge_metric.compute(predicted_atom_array=atom_array)

        # Verify return structure
        assert isinstance(charge_results, dict)
        expected_charge_keys = {
            "charge_score",
            "n_salt_bridges",
            "n_charge_clashes",
            "n_charged_residues_binder",
            "n_charged_residues_target",
        }
        assert set(charge_results.keys()) == expected_charge_keys

        # Verify values are valid
        assert isinstance(charge_results["n_salt_bridges"], int)
        assert isinstance(charge_results["n_charge_clashes"], int)
        assert charge_results["n_salt_bridges"] >= 0
        assert charge_results["n_charge_clashes"] >= 0
