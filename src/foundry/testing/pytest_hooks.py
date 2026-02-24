"""Shared pytest configuration hooks for foundry tests."""

import pytest
import rootutils
from dotenv import load_dotenv


class _SkipGpuPlugin:
    """Auto-skip tests marked @pytest.mark.gpu when CUDA is unavailable."""

    def pytest_runtest_setup(self, item):
        if item.get_closest_marker("gpu"):
            import torch

            if not torch.cuda.is_available():
                pytest.skip("requires GPU (CUDA not available)")


def configure_pytest(config, conftest_file: str) -> None:
    """Configure pytest for foundry tests.
    Sets up project root and environment variables.
    """
    # Setup the project root
    rootutils.setup_root(conftest_file, indicator=".project-root", pythonpath=True)

    # Setup environment variables
    load_dotenv(override=True)

    # Auto-skip GPU tests when CUDA is unavailable
    config.pluginmanager.register(_SkipGpuPlugin(), "skip_gpu")
