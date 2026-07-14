"""Tests for MACE energy metric."""

import os

import pytest

# Skip all tests if mace-torch is not installed
pytest.importorskip("mace")

import numpy as np
from atomworks.io.utils.ccd import atom_array_from_ccd_code
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from mace.calculators import MACECalculator

from proteinfoundation.metrics.physics.mace_energy_metric import compute_mace_energy

# Get MACE model path from environment variable
MACE_MODEL_PATH = os.environ.get("MACE_MODEL_PATH")


class TestComputeMaceEnergy:
    """Tests for the standalone compute_mace_energy function."""

    def test_energy_sensitivity_to_coordinate_noise(self, gpu):
        """Test that MACE energy changes appropriately with coordinate noise."""
        if MACE_MODEL_PATH is None:
            pytest.skip("MACE_MODEL_PATH environment variable not set")

        # Load HEM (heme) from CCD
        hem = atom_array_from_ccd_code("HEM")

        calculator = MACECalculator(
            model_paths=MACE_MODEL_PATH,
            device="cuda",
            default_dtype="float32",
        )

        # Compute reference energy
        reference_energy = compute_mace_energy(hem, calculator, hydrogen_policy="keep")

        # Test with large noise (0.5 Angstrom std)
        hem_large_noise = hem.copy()
        with rng_state(create_rng_state_from_seeds(np_seed=42)):
            hem_large_noise.coord = hem.coord + np.random.normal(0, 0.5, hem.coord.shape)
        large_noise_energy = compute_mace_energy(hem_large_noise, calculator, hydrogen_policy="keep")
        large_energy_diff = abs(large_noise_energy - reference_energy)

        # Test with tiny noise (0.01 Angstrom std)
        hem_tiny_noise = hem.copy()
        with rng_state(create_rng_state_from_seeds(np_seed=42)):
            hem_tiny_noise.coord = hem.coord + np.random.normal(0, 0.01, hem.coord.shape)
        tiny_noise_energy = compute_mace_energy(hem_tiny_noise, calculator, hydrogen_policy="keep")
        tiny_energy_diff = abs(tiny_noise_energy - reference_energy)

        # Large noise should produce larger energy difference than tiny noise; also check thresholds
        assert (
            large_energy_diff > tiny_energy_diff
        ), f"Expected large noise ({large_energy_diff:.4f} eV) > tiny noise ({tiny_energy_diff:.4f} eV)"
        assert large_energy_diff > 1.0, f"Expected large energy diff > 1 eV, got {large_energy_diff:.4f} eV"
        assert tiny_energy_diff < 1.0, f"Expected tiny energy diff < 1 eV, got {tiny_energy_diff:.4f} eV"
