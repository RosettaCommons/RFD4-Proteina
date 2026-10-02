"""Skip guard for the DIGS-only sanity checks.

The tests in this directory inspect the DIGS host environment (checkpoints under
``/projects``, the Apptainer SIF under ``/path/to/shared/storage``, a fresh ``setup.sh``
install), so they only make sense on a DIGS node. They run inside the latent-dev
SIF on the self-hosted DIGS runner, where ``rfproteina`` is importable -- future
DIGS tests are free to import from it. See ``.github/workflows/digs_tests.yaml``.

Off-DIGS (a laptop or another CI site), a stray ``pytest`` still collects these
files, so we force-skip every ``digs``-marked test unless ``CLUSTER=digs`` or
``RUN_DIGS_TESTS=1`` is set. The skip keys off the marker, not the directory, so
it stays correct no matter where the runner invokes pytest from.
"""

import os
import subprocess
from pathlib import Path

import pytest

ENV_DIGS = Path(__file__).resolve().parents[2] / ".ipd" / ".env.digs"


def env_digs(snippet: str) -> str:
    """Return ``snippet``'s stdout with ``.ipd/.env.digs`` sourced.

    Sourcing, not pattern-matching, so ``${DATA_PATH}`` interpolation resolves.
    Unset vars: write ``"${VAR?}"`` and the non-zero exit raises.
    """
    argv = ["bash", "-c", f'set -a; source "{ENV_DIGS}"; set +a; {snippet}']
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.strip()


def running_on_digs() -> bool:
    """Return whether the DIGS sanity checks should actually execute."""
    return os.environ.get("CLUSTER") == "digs" or os.environ.get("RUN_DIGS_TESTS") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``digs``-marked tests unless we are on a DIGS node.

    Args:
      config: Active pytest config (unused).
      items: Collected test items; ``digs``-marked ones are skipped off-DIGS.
    """
    if running_on_digs():
        return
    skip_off_digs = pytest.mark.skip(
        reason="DIGS sanity checks only run on the DIGS cluster (set CLUSTER=digs or RUN_DIGS_TESTS=1)"
    )
    for item in items:
        if item.get_closest_marker("digs"):
            item.add_marker(skip_off_digs)
