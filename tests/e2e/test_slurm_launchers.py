"""Cluster e2e: syntax-check SLURM launcher scripts (no GPU, no job submission).

Foundry launchers call ``sbatch`` directly (no dry-run mode), so we cannot
safely submit jobs in CI. A ``bash -n`` parse is a cheap guard that catches
syntax regressions in the launchers without touching the scheduler.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.e2e.helpers.gpu_e2e import ROOT

SLURM_UTILS = ROOT / "scripts"

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


def test_common_sh_pythonpath_resolves_to_real_dirs():
    """``setup_pythonpath`` must emit directories that exist.

    ``bash -n`` only parses; it cannot catch a path that is syntactically fine but
    points nowhere (e.g. a stale ``projects/proteinfoundation/`` prefix left over
    after the project became its own root). Source the real function and check
    every entry it puts on PYTHONPATH.
    """
    script = f"""
        source {SLURM_UTILS / "common.sh"}
        repo_root=$(find_repo_root "{SLURM_UTILS}")
        setup_pythonpath "$repo_root" >/dev/null
        printf '%s' "$PYTHONPATH"
    """
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

    entries = [p for p in proc.stdout.split(":") if p]
    assert entries, "setup_pythonpath produced an empty PYTHONPATH"
    for path in entries:
        assert Path(path).is_dir(), f"PYTHONPATH entry does not exist: {path}"


def test_entrypoint_scripts_referenced_by_e2e_and_launchers_exist():
    """Every ``*.py`` entrypoint spawned as a subprocess must actually exist.

    The GPU e2e tests invoke ``src/proteinfoundation/{train,generate}.py`` via
    subprocess, but they are GPU-gated and skipped on CPU -- so a stale path (e.g.
    one left behind by the move under ``src/``) surfaces only on a GPU runner,
    minutes in. Assert the paths resolve, cheaply, on CPU.

    Only *quoted string literals* are checked, so prose in docstrings and comments
    does not produce false positives.
    """
    sources = list((ROOT / "tests" / "e2e").rglob("*.py"))
    sources += list((ROOT / "src" / "script_utils").glob("*.py"))
    # .github/ci/*.sh invoke entrypoints too -- a stale path there breaks CI only, and
    # only on a GPU runner (this is how `scripts/af3_daemon.py` survived the move to
    # src/proteinfoundation/af3/ and silently disabled the AF3 daemon).
    sources += list((ROOT / ".github" / "ci").glob("*.sh"))

    # A quoted path that looks like one of our python entrypoints.
    quoted = re.compile(r"[\"']((?:src/)?(?:proteinfoundation|script_utils|scripts)/[\w/]+\.py)[\"']")
    bare = re.compile(r"(?:^|\s)((?:src/)?(?:proteinfoundation|script_utils|scripts)/[\w/]+\.py)")
    referenced = set()
    for f in sources:
        text = f.read_text()
        referenced |= set(quoted.findall(text))
        if f.suffix == ".sh":
            referenced |= set(bare.findall(text))
    assert referenced, "found no entrypoint references to check"

    missing = [p for p in sorted(referenced) if not (ROOT / p).is_file()]
    assert not missing, f"entrypoints referenced but missing: {missing}"
