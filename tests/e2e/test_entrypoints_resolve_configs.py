"""Every hydra entrypoint must find ``configs/`` from wherever it lives under ``src/``.

``train.py`` used a *relative* ``config_path="../configs"``, which silently became
``src/configs`` when the package moved under ``src/`` -- so training died with
``MissingConfigException`` only on a GPU runner, minutes into an allocation.
``--help`` exercises hydra's config-dir resolution in about a second, on CPU.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.e2e.helpers.gpu_e2e import ROOT

ENTRYPOINTS = [
    "src/rfproteina/train.py",
    "src/rfproteina/partial_autoencoder/train.py",
    "src/rfproteina/generate.py",
    "src/rfproteina/evaluate.py",
]


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_entrypoint_resolves_its_config_dir(entrypoint: str):
    script = ROOT / entrypoint
    assert script.is_file(), f"entrypoint missing: {entrypoint}"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = proc.stdout + proc.stderr
    assert (
        "Primary config directory not found" not in combined
    ), f"{entrypoint} cannot find configs/:\n{combined[-500:]}"
