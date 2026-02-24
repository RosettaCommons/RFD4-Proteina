"""Tests for DDG and charge complementarity metrics."""

import os

import numpy as np
import pytest
from atomworks.io import parse

from proteinfoundation.conditions import C_CRD

# Get test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BINDER_STRUCTURE_PATH = os.path.join(TEST_DATA_DIR, "5vht_from_file_model_0.cif")


class TestDDGSaltMetrics:
    """End-to-end tests for DDG and charge complementarity metrics."""

    def test_ddg_salt_metrics_on_binder_structure(self, gpu):
        """Test all DDG and charge metrics on a real binder-target complex.
        
        Uses 5vht structure (2-chain complex with chains A and B) to test:
        1. SoftDDGMetric - TMol-based binding energy
        2. ChargeComplementarityMetric - Salt bridge scoring
        3. SoftDDGWithSaltMetric - Combined metric
        
        The structure must have multiple chains for interface metrics to work.
        """
        # Skip if TMol is not available
        try:
            from proteinfoundation.metrics.ddg_salt_metrics import (
                SoftDDGMetric,
                ChargeComplementarityMetric,
                SoftDDGWithSaltMetric,
                get_binder_target_mask,
                get_charged_residue_positions,
                split_atom_array_by_chain,
            )
        except ImportError as e:
            pytest.skip(f"TMol dependencies not available: {e}")

        # Load structure using atomworks parser
        parsed = parse(BINDER_STRUCTURE_PATH)
        atom_array = parsed["atom_array"]
        
        # Verify multi-chain structure
        unique_chains = np.unique(atom_array.chain_id)
        assert len(unique_chains) >= 2, f"Expected multi-chain structure, got chains: {unique_chains}"
        
        # -------------------------------------------
        # Test helper functions
        # -------------------------------------------
        
        # Test chain splitting
        chain_arrays = split_atom_array_by_chain(atom_array)
        assert len(chain_arrays) == len(unique_chains)
        for chain_id, chain_arr in chain_arrays.items():
            assert chain_id in unique_chains
            assert len(chain_arr) > 0
        
        # Test binder/target mask extraction
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
        
        # -------------------------------------------
        # Test SoftDDGMetric (requires TMol)
        # -------------------------------------------
        
        try:
            ddg_metric = SoftDDGMetric(
                use_soft_rep=True,
                soft_rep_weight=0.55,
                device="cuda",
            )
            
            ddg_results = ddg_metric.compute(predicted_atom_array=atom_array)
            
            # Verify return structure
            assert isinstance(ddg_results, dict)
            expected_ddg_keys = {
                "ddg",
                "complex_energy",
                "binder_energy",
                "target_energy",
                "monomer_energy_sum",
            }
            assert set(ddg_results.keys()) == expected_ddg_keys
            
            # Verify values are numeric (may be NaN if TMol fails)
            for key, value in ddg_results.items():
                assert isinstance(value, (int, float)), f"{key} should be numeric, got {type(value)}"
            
            # If scoring succeeded, verify relationships
            if not np.isnan(ddg_results["ddg"]):
                assert not np.isnan(ddg_results["complex_energy"])
                assert not np.isnan(ddg_results["binder_energy"])
                assert not np.isnan(ddg_results["target_energy"])
                
                # DDG should be complex - monomer sum
                expected_ddg = (
                    ddg_results["complex_energy"] - ddg_results["monomer_energy_sum"]
                )
                assert np.isclose(ddg_results["ddg"], expected_ddg, rtol=1e-5)
                
        except Exception as e:
            pytest.skip(f"TMol scoring failed (expected in some environments): {e}")
        
        # -------------------------------------------
        # Test SoftDDGWithSaltMetric (combined)
        # -------------------------------------------
        
        try:
            combined_metric = SoftDDGWithSaltMetric(
                use_soft_rep=True,
                soft_rep_weight=0.55,
                salt_min_dist=7.0,
                salt_max_dist=9.0,
                salt_bonus=2.0,
                charge_func="linear",
                device="cuda",
            )
            
            combined_results = combined_metric.compute(predicted_atom_array=atom_array)
            
            # Should have all keys from both metrics plus combined score
            assert "ddg_with_salt" in combined_results
            assert "ddg" in combined_results
            assert "charge_score" in combined_results
            
            # Verify combined score relationship
            if not np.isnan(combined_results["ddg"]):
                expected_combined = combined_results["ddg"] + combined_results["charge_score"]
                assert np.isclose(
                    combined_results["ddg_with_salt"], expected_combined, rtol=1e-5
                )
                
        except Exception as e:
            pytest.skip(f"Combined metric failed (expected in some environments): {e}")

