"""Tests for SimpleEMA weight swapping logic.

These tests verify that the EMA callback correctly swaps between training
and EMA weights in various scenarios, including edge cases like checkpoint
saving during validation.
"""

from unittest.mock import Mock

import pytest
import torch

from proteinfoundation.utils.simple_ema import SimpleEMA


class DummyModule:
    """Mock LightningModule with identifiable state dicts.

    This mimics a real PyTorch module where state is stored in actual tensors,
    not in a separate tracking variable. This properly tests whether state_dict()
    returns copies or references.
    """

    def __init__(self):
        # Actual parameter storage (like real PyTorch nn.Parameters)
        # Training weights: all tensors have value 1.0
        self.params = {
            "layer1.weight": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            "layer1.bias": torch.tensor([1.0, 1.0]),
            "layer2.weight": torch.tensor([[1.0]]),
        }

    def state_dict(self):
        """Return state dict - mimics PyTorch's actual behavior."""
        # PyTorch returns a dict with tensor values
        # Question: does it clone or return references?
        return self.params

    def load_state_dict(self, state_dict):
        """Load state dict - mimics PyTorch's actual behavior."""
        # Copy values from state_dict into self.params
        for key in self.params:
            self.params[key].copy_(state_dict[key])


def get_ema_state():
    """Get reference EMA state dict (all 99.0)."""
    return {
        "layer1.weight": torch.tensor([[99.0, 99.0], [99.0, 99.0]]),
        "layer1.bias": torch.tensor([99.0, 99.0]),
        "layer2.weight": torch.tensor([[99.0]]),
    }


def check_state(module):
    """Check which state is currently loaded by examining tensor values."""
    state = module.state_dict()
    first_tensor = next(iter(state.values()))

    if torch.allclose(first_tensor, torch.tensor(1.0)):
        return "training"
    elif torch.allclose(first_tensor, torch.tensor(99.0)):
        return "ema"
    else:
        return f"unknown (values: {first_tensor})"


def assert_state(module, expected_state, message=""):
    """Helper to assert which weights are currently loaded."""
    actual = check_state(module)
    assert actual == expected_state, f"{message}\nExpected: {expected_state}, Got: {actual}"


class TestSimpleEMASwapping:
    """Test suite for EMA weight swapping logic."""

    @pytest.fixture
    def module(self):
        """Create a dummy module with training weights loaded."""
        return DummyModule()

    @pytest.fixture
    def ema_callback(self, module):
        """Create SimpleEMA callback with initialized EMA model."""
        callback = SimpleEMA(decay=0.999, update_every_n=1)

        # Mock the EMA model
        callback.ema_model = Mock()
        callback.ema_model.module.state_dict.return_value = get_ema_state()

        return callback

    @pytest.fixture
    def trainer(self):
        """Create mock trainer."""
        trainer = Mock()
        trainer.global_step = 0
        return trainer

    def test_basic_validation_swap(self, module, ema_callback, trainer):
        """Test basic validation: start → end."""
        assert_state(module, "training", "Initial state")

        # Start validation
        ema_callback.on_validation_start(trainer, module)
        assert_state(module, "ema", "After on_validation_start")
        assert ema_callback._ema_weights_active is True

        # End validation
        ema_callback.on_validation_end(trainer, module)
        assert_state(module, "training", "After on_validation_end")
        assert ema_callback._ema_weights_active is False

    def test_checkpoint_save_during_validation(self, module, ema_callback, trainer):
        """Test checkpoint save during validation using use_training_weights."""
        assert_state(module, "training", "Initial state")

        # Start validation
        ema_callback.on_validation_start(trainer, module)
        assert_state(module, "ema", "After on_validation_start")

        # Checkpoint save happens during validation
        with ema_callback.use_training_weights(module):
            assert_state(module, "training", "Inside use_training_weights context")
            assert ema_callback._ema_weights_active is False

        # After checkpoint context exits
        assert_state(module, "ema", "After use_training_weights context exits")
        assert ema_callback._ema_weights_active is True

        # End validation
        ema_callback.on_validation_end(trainer, module)
        assert_state(module, "training", "After on_validation_end")
        assert ema_callback._ema_weights_active is False

    def test_ema_checkpoint_save_during_validation(self, module, ema_callback, trainer):
        """Test EMA checkpoint save during validation using use_ema_weights."""
        assert_state(module, "training", "Initial state")

        # Start validation
        ema_callback.on_validation_start(trainer, module)
        assert_state(module, "ema", "After on_validation_start")

        # EMA checkpoint save (should be no-op since already EMA)
        with ema_callback.use_ema_weights(module):
            assert_state(module, "ema", "Inside use_ema_weights context")
            assert ema_callback._ema_weights_active is True

        # After checkpoint context exits
        assert_state(module, "ema", "After use_ema_weights context exits")
        assert ema_callback._ema_weights_active is True

        # End validation
        ema_callback.on_validation_end(trainer, module)
        assert_state(module, "training", "After on_validation_end")
        assert ema_callback._ema_weights_active is False

    def test_both_checkpoints_during_validation(self, module, ema_callback, trainer):
        """Test both training and EMA checkpoints saved during validation."""
        assert_state(module, "training", "Initial state")

        # Start validation
        ema_callback.on_validation_start(trainer, module)
        assert_state(module, "ema", "After on_validation_start")

        # Save training checkpoint
        with ema_callback.use_training_weights(module):
            assert_state(module, "training", "Inside use_training_weights")
            assert ema_callback._ema_weights_active is False

        assert_state(module, "ema", "After use_training_weights exits")
        assert ema_callback._ema_weights_active is True

        # Save EMA checkpoint (should be no-op)
        with ema_callback.use_ema_weights(module):
            assert_state(module, "ema", "Inside use_ema_weights")
            assert ema_callback._ema_weights_active is True

        assert_state(module, "ema", "After use_ema_weights exits")
        assert ema_callback._ema_weights_active is True

        # End validation
        ema_callback.on_validation_end(trainer, module)
        assert_state(module, "training", "After on_validation_end")
        assert ema_callback._ema_weights_active is False

    def test_nested_context_managers(self, module, ema_callback, trainer):
        """Test nested context managers (should this even happen?)."""
        assert_state(module, "training", "Initial state")

        # Start validation
        ema_callback.on_validation_start(trainer, module)
        assert_state(module, "ema", "After on_validation_start")

        # Nested contexts
        with ema_callback.use_training_weights(module):
            assert_state(module, "training", "Outer: use_training_weights")

            # Nested use_training_weights (should be no-op)
            with ema_callback.use_training_weights(module):
                assert_state(module, "training", "Inner: use_training_weights")

            assert_state(module, "training", "After inner context")

        assert_state(module, "ema", "After outer context")

        # End validation
        ema_callback.on_validation_end(trainer, module)
        assert_state(module, "training", "After on_validation_end")

    def test_use_ema_weights_from_training(self, module, ema_callback, trainer):
        """Test use_ema_weights when starting from training mode."""
        assert_state(module, "training", "Initial state")
        assert ema_callback._ema_weights_active is False

        # Use EMA weights from training mode
        with ema_callback.use_ema_weights(module):
            assert_state(module, "ema", "Inside use_ema_weights context")
            assert ema_callback._ema_weights_active is True

        # Should restore to training
        assert_state(module, "training", "After use_ema_weights context exits")
        assert ema_callback._ema_weights_active is False

    def test_use_training_weights_from_training(self, module, ema_callback, trainer):
        """Test use_training_weights when already in training mode (should be no-op)."""
        assert_state(module, "training", "Initial state")
        assert ema_callback._ema_weights_active is False

        # Use training weights when already in training (no-op)
        with ema_callback.use_training_weights(module):
            assert_state(module, "training", "Inside use_training_weights context")
            assert ema_callback._ema_weights_active is False

        assert_state(module, "training", "After use_training_weights context exits")
        assert ema_callback._ema_weights_active is False

    def test_multiple_validations(self, module, ema_callback, trainer):
        """Test multiple validation cycles."""
        for i in range(3):
            assert_state(module, "training", f"Before validation {i}")

            ema_callback.on_validation_start(trainer, module)
            assert_state(module, "ema", f"During validation {i}")

            ema_callback.on_validation_end(trainer, module)
            assert_state(module, "training", f"After validation {i}")

    def test_validation_with_checkpoint_multiple_times(self, module, ema_callback, trainer):
        """Test validation + checkpoint save multiple times."""
        for i in range(3):
            assert_state(module, "training", f"Iteration {i}: Initial")

            # Validation starts
            ema_callback.on_validation_start(trainer, module)
            assert_state(module, "ema", f"Iteration {i}: After validation start")

            # Checkpoint saves
            with ema_callback.use_training_weights(module):
                assert_state(module, "training", f"Iteration {i}: Saving training checkpoint")

            assert_state(module, "ema", f"Iteration {i}: After training checkpoint")

            with ema_callback.use_ema_weights(module):
                assert_state(module, "ema", f"Iteration {i}: Saving EMA checkpoint")

            assert_state(module, "ema", f"Iteration {i}: After EMA checkpoint")

            # Validation ends
            ema_callback.on_validation_end(trainer, module)
            assert_state(module, "training", f"Iteration {i}: After validation end")

    def test_original_state_dict_not_corrupted(self, module, ema_callback, trainer):
        """Test that _original_state_dict is not corrupted during checkpoint saves."""
        assert_state(module, "training", "Initial state")

        # Start validation - stores original state
        ema_callback.on_validation_start(trainer, module)
        original_stored = ema_callback._original_state_dict

        # Verify original is training weights
        first_tensor = next(iter(original_stored.values()))
        assert torch.allclose(first_tensor, torch.tensor(1.0)), "_original_state_dict should contain training weights"

        # Checkpoint save during validation
        with ema_callback.use_training_weights(module):
            pass

        # Verify original is still correct
        first_tensor = next(iter(ema_callback._original_state_dict.values()))
        assert torch.allclose(first_tensor, torch.tensor(1.0)), (
            "_original_state_dict corrupted after use_training_weights"
        )

        # End validation
        ema_callback.on_validation_end(trainer, module)
        assert_state(module, "training", "After validation end")
