"""Test fixtures and utilities"""

# Load .env BEFORE any atomworks imports (they read env vars at import time)
from foundry.utils.env import load_dotenv

load_dotenv(override=True)

import os

import pytest
from atomworks.ml.executables import _EXECUTABLES
from atomworks.ml.executables.dssp import DSSPExecutable
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from foundry.testing import configure_pytest, get_test_data_dir, gpu  # noqa: F401

TEST_DATA_DIR = get_test_data_dir(__file__)


def pytest_configure(config):
    """Set up environment variables, paths, etc."""
    configure_pytest(config, __file__)


# DSSP setup for secondary structure tests
DSSP_PATH = os.environ.get("DSSP_PATH", "/path/to/data")


def _has_dssp() -> bool:
    """Check if DSSP is installed and available."""
    if os.path.isfile(DSSP_PATH) and os.access(DSSP_PATH, os.X_OK):
        try:
            DSSPExecutable.get_or_initialize(DSSP_PATH)
            return True
        except Exception:
            return False
    return False


has_dssp = _has_dssp()
skip_if_no_dssp = pytest.mark.skipif(not has_dssp, reason="DSSP is not installed or unavailable")


@pytest.fixture
def cleanup_dssp():
    """Reinitialize DSSPExecutable for test isolation.

    Ensures the DSSP executable singleton is reset between tests.
    """
    if has_dssp:
        DSSPExecutable.reinitialize(DSSP_PATH)
    yield
    _EXECUTABLES.pop(DSSPExecutable.name, None)
    DSSPExecutable._is_initialized = False


@pytest.fixture
def deterministic_rng():
    """Fixture for deterministic random number generation."""
    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)):
        yield
