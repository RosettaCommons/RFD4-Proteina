"""Shared testing utilities for the tests package."""

import rootutils

from tests.helpers.fixtures import get_test_data_dir, gpu
from tests.helpers.pytest_hooks import configure_pytest

PROJECT_ROOT = rootutils.find_root(search_from=__file__, indicator=".project-root")
CONFIGS_DIR = PROJECT_ROOT / "configs"
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"

__all__ = ["BENCHMARKS_DIR", "CONFIGS_DIR", "PROJECT_ROOT", "configure_pytest", "get_test_data_dir", "gpu"]
