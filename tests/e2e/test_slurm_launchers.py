"""Cluster e2e: syntax-check SLURM launcher scripts (no GPU, no job submission).

Foundry launchers call ``sbatch`` directly (no dry-run mode), so we cannot
safely submit jobs in CI. A ``bash -n`` parse is a cheap guard that catches
syntax regressions in the launchers without touching the scheduler.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.e2e.helpers.gpu_e2e import ROOT

SLURM_UTILS = ROOT / "slurm_utils"

LAUNCHERS = [
    "launch_train_latents.sh",
    "launch_train_ae.sh",
    "launch_infer.sh",
    "launch_infer_ae.sh",
    "launch_search.sh",
    "launch_evaluate.sh",
    "launch_external_eval.sh",
]


@pytest.mark.cluster
@pytest.mark.parametrize("script", LAUNCHERS)
def test_launcher_script_parses(script: str) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    path = SLURM_UTILS / script
    if not path.is_file():
        pytest.skip(f"Launcher not found: {path}")
    proc = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Syntax error in {script}:\n{proc.stderr}"
