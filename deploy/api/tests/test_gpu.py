"""Opt-in integration test against an already running GPU API."""

import os
from pathlib import Path

import pytest
from smoke_test import run_smoke


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_API_TESTS") != "1", reason="Requires a running GPU API")
def test_gpu_generation_and_recovery(tmp_path: Path) -> None:
    run_smoke(tmp_path)
