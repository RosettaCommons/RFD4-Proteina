"""Shared pytest configuration hooks for foundry tests."""

import pytest
import rootutils
import torch

from rfproteina.utils.env import load_dotenv


class _GpuMarkerPlugin:
    """Unify GPU gating: treat any test using the ``gpu`` fixture as ``@pytest.mark.gpu``.

    This makes ``@pytest.mark.gpu`` the single mechanism that selects/deselects
    GPU work (``-m gpu``), regardless of whether a test opts in via the marker or
    via the ``gpu`` fixture, so CI marker expressions stay simple.
    """

    def pytest_collection_modifyitems(self, items):
        for item in items:
            if "gpu" in getattr(item, "fixturenames", ()):
                item.add_marker(pytest.mark.gpu)


class _SkipGpuPlugin:
    """Auto-skip tests marked @pytest.mark.gpu when CUDA is unavailable."""

    def pytest_runtest_setup(self, item):
        if item.get_closest_marker("gpu"):
            if not torch.cuda.is_available():
                pytest.skip("requires GPU (CUDA not available)")


def configure_pytest(config, conftest_file: str) -> None:
    """Configure pytest for foundry tests.
    Sets up project root and environment variables.
    """
    # Setup the project root
    rootutils.setup_root(conftest_file, indicator=".project-root", pythonpath=True)

    # Setup environment variables
    load_dotenv()

    # Mark gpu-fixture tests as gpu (single gating mechanism), then auto-skip
    # gpu-marked tests when CUDA is unavailable. Registration is session-global,
    # so register once even when several conftests call configure_pytest during a
    # repo-wide collection.
    _register_once(config, _GpuMarkerPlugin(), "gpu_marker")
    _register_once(config, _SkipGpuPlugin(), "skip_gpu")


def _register_once(config, plugin, name: str) -> None:
    """Register ``plugin`` under ``name`` unless something already holds that name."""
    if not config.pluginmanager.has_plugin(name):
        config.pluginmanager.register(plugin, name)
