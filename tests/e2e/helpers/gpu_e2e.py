"""GPU e2e guards, subprocess runners, and training-loss parsing.

Adapted from protein-foundation-models ``tests/e2e/helpers/gpu_e2e.py`` for the
foundry ``proteinfoundation`` project. ``ROOT`` is the project dir
(``projects/proteinfoundation``) since that is the working directory training
and inference subprocesses run from (``store/``, ``inference/`` live there).
"""

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import pytest

from tests.conftest import has_af3

# helpers/ -> e2e/ -> tests/ -> proteinfoundation (project root for train/infer)
ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = Path(os.environ.get("CKPT_PATH", ROOT / "ckpts"))
RUN_E2E = os.getenv("RUN_GPU_E2E_TESTS", "0") == "1"


def get_data_path() -> Path | None:
    """Return ``$DATA_PATH`` as a :class:`~pathlib.Path`, or ``None`` if unset."""
    raw = os.environ.get("DATA_PATH", "").strip()
    return Path(raw) if raw else None


PYTHON = sys.executable

# GPU CI uses RF3 refolding (no AF3 apptainer in the GPU lane).
RF3_ORACLE_OVERRIDE = "metrics/refolding_oracles=rf3"

PrevalOracle = Literal["rf3", "af3", "none"]

# Default RF3 oracle weights (``configs/metrics/refolding_oracles/rf3.yaml``).
_DEFAULT_RF3_CKPT_NAME = "rf3-with-confidence-xaira-epoch-0711-fixed.ckpt"

# CI training smokes use a small batch to limit VRAM and speed up steps.
GPU_E2E_TRAIN_BATCH_SIZE = int(os.getenv("GPU_E2E_TRAIN_BATCH_SIZE", "2"))
GPU_E2E_NGPUS = int(os.getenv("GPU_E2E_NGPUS", "2"))

# Short training smoke shape: 100 optimizer steps + light epoch-0 validation.
GPU_E2E_TRAIN_STEPS = int(os.getenv("GPU_E2E_TRAIN_STEPS", "100"))
# Lightning sanity check: validation batches *total* across all val loaders (not per dataset).
GPU_E2E_SANITY_VAL_STEPS = int(os.getenv("GPU_E2E_SANITY_VAL_STEPS", "0"))
# Epoch validation: batches *per* val dataset (SOTA has ~11; RF3 refolding makes these expensive).
GPU_E2E_LIMIT_VAL_BATCHES = int(os.getenv("GPU_E2E_LIMIT_VAL_BATCHES", "2"))
# Back-compat alias for scripts/docs that still say PREVAL_STEPS.
GPU_E2E_PREVAL_STEPS = GPU_E2E_LIMIT_VAL_BATCHES

# Matches progress-bar / log lines like ``train_loss=1.234``, ``train_loss_step: 26.9``.
TRAIN_LOSS_RE = re.compile(r"train_loss(?:_step|_epoch)?(?:=|\s|:|\")?\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)")


def _detect_gpu_count() -> int:
    count = int(os.getenv("GPU_COUNT", "0") or "0")
    if count:
        return count
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 0
    return len([ln for ln in proc.stdout.splitlines() if ln.strip().startswith("GPU ")])


def require_gpu_e2e(*, min_gpus: int = 0) -> None:
    """Skip unless GPU e2e is explicitly enabled and enough GPUs are present."""
    if not RUN_E2E:
        pytest.skip("Set RUN_GPU_E2E_TESTS=1 to run GPU e2e tests")
    if min_gpus > 0:
        gpu_count = _detect_gpu_count()
        if gpu_count < min_gpus and os.getenv("ALLOW_SINGLE_GPU_E2E", "0") != "1":
            pytest.skip(f"Need at least {min_gpus} GPUs (detected {gpu_count})")


def require_data_file(relative: str) -> Path:
    require_gpu_e2e()
    data_path = get_data_path()
    if data_path is None:
        pytest.skip("Set DATA_PATH for training/inference data")
    path = data_path / relative
    if not path.exists():
        pytest.skip(f"Missing data file: {path}")
    return path


def require_sota_ckpt() -> tuple[Path, str]:
    """Resolve the SOTA checkpoint for inference/search tests.

    Returns ``(ckpt_dir, ckpt_name)``. Honors ``SOTA_CKPT_PATH`` (full file path)
    or falls back to ``$CKPT_PATH/$SOTA_CKPT_NAME``. Skips if not present.
    """
    require_gpu_e2e()
    full = os.environ.get("SOTA_CKPT_PATH")
    if full:
        ckpt = Path(full)
        if not ckpt.is_file():
            pytest.skip(f"SOTA_CKPT_PATH not found: {ckpt}")
        return ckpt.parent, ckpt.name
    name = os.environ.get("SOTA_CKPT_NAME")
    if not name:
        pytest.skip("Set SOTA_CKPT_PATH=/abs/file.ckpt (or CKPT_PATH + SOTA_CKPT_NAME) for inference/search e2e")
    ckpt = CHECKPOINT_DIR / name
    if not ckpt.is_file():
        pytest.skip(f"Missing SOTA checkpoint: {ckpt}")
    return CHECKPOINT_DIR, name


def resolve_rf3_ckpt() -> Path:
    """Resolve the RF3 refolding checkpoint path; skip if not present.

    Honors ``RF3_CKPT_PATH`` (absolute ``.ckpt`` file) or
    ``$DATA_PATH/rf3_checkpoints/$RF3_CKPT_NAME``.

    Returns:
        Absolute path to the RF3 checkpoint file.
    """
    full = os.environ.get("RF3_CKPT_PATH")
    if full:
        ckpt = Path(full)
        if not ckpt.is_file():
            pytest.skip(f"RF3_CKPT_PATH not found: {ckpt}")
        return ckpt
    data_path = get_data_path()
    if data_path is None:
        pytest.skip("Set DATA_PATH for RF3 refolding checkpoint")
    name = os.environ.get("RF3_CKPT_NAME", _DEFAULT_RF3_CKPT_NAME)
    ckpt = data_path / "rf3_checkpoints" / name
    if not ckpt.is_file():
        pytest.skip(
            f"Missing RF3 checkpoint: {ckpt} "
            f"(set RF3_CKPT_PATH=/abs/file.ckpt or place under $DATA_PATH/rf3_checkpoints/)"
        )
    return ckpt


def require_rf3_ckpt() -> Path:
    """GPU e2e gate plus :func:`resolve_rf3_ckpt`."""
    require_gpu_e2e()
    return resolve_rf3_ckpt()


def preval_oracle_overrides(oracle: PrevalOracle) -> list[str]:
    """Hydra overrides for prevalidation refolding oracle selection."""
    if oracle == "rf3":
        return ["metrics/refolding_oracles=rf3"]
    if oracle == "af3":
        return ["metrics/refolding_oracles=af3"]
    return ["~metrics.refolding_oracles"]


def require_preval_oracle(oracle: PrevalOracle) -> None:
    """Skip when the variant's prevalidation oracle is unavailable."""
    require_gpu_e2e()
    if oracle == "rf3":
        require_rf3_ckpt()
    elif oracle == "af3":
        if os.environ.get("AF3_LAUNCHER_SOCKET_TEMPLATE"):
            return
        if not has_af3:
            pytest.skip("AF3 not available (set AF3_APPTAINER_PATH, AF3_SCRIPTS_DIR, or AF3_LAUNCHER_SOCKET_TEMPLATE)")
        scripts_dir = os.environ.get("AF3_SCRIPTS_DIR", "")
        script = os.path.join(scripts_dir, "run_alphafold_from_cif.py")
        if not os.path.isfile(script):
            pytest.skip(f"AF3 script not found: {script}")


def _rmtree_robust(path: Path) -> None:
    if not path.exists():
        return

    def _onerror(func, p, _exc_info):  # type: ignore[no-untyped-def]
        if not os.access(p, os.W_OK):
            os.chmod(p, stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            func(p)
        else:
            raise

    if path.is_dir():
        shutil.rmtree(path, onerror=_onerror)
    else:
        path.unlink(missing_ok=True)


def cleanup_smoke_training_store(run_name: str) -> None:
    """Remove stale Lightning checkpoints from prior smoke runs (avoids resume conflicts)."""
    _rmtree_robust(ROOT / "store" / run_name)
    _rmtree_robust(ROOT / "store" / f"{run_name}_local")


def _terminate_process_tree(pid: int, *, grace_s: float = 5.0) -> None:
    """Ensure Lightning/DDP child processes exit after the parent subprocess returns."""
    if pid <= 0:
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
                time.sleep(0.2)
            except ProcessLookupError:
                return
        if sig == signal.SIGTERM:
            continue
        return


def run_cmd(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a command, capturing output and cleaning up the DDP process group."""
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=merged,
        start_new_session=True,
    )
    stdout, stderr = proc.communicate()
    _terminate_process_tree(proc.pid)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def parse_train_losses(stdout: str) -> list[float]:
    values: list[float] = []
    for match in TRAIN_LOSS_RE.finditer(stdout):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def parse_train_losses_from_csv_logger(out_dir: Path) -> list[float]:
    """Fallback parser for ``train.py +nolog=true`` runs.

    With ``nolog`` set, foundry uses ``CSVLogger(save_dir=out_dir, name="csv_logs")``,
    writing scalars to ``<out_dir>/csv_logs/version_*/metrics.csv``.
    """
    csv_files = sorted(Path(out_dir).glob("csv_logs/version_*/metrics.csv"))
    if not csv_files:
        return []
    losses: list[float] = []
    with open(csv_files[-1], newline="") as fh:
        reader = csv.DictReader(fh)
        col = next((c for c in (reader.fieldnames or []) if c == "train_loss" or c.endswith("/train_loss")), None)
        if col is None:
            return []
        for row in reader:
            raw = row.get(col, "")
            if raw not in ("", None):
                try:
                    losses.append(float(raw))
                except ValueError:
                    continue
    return losses


def run_training_smoke_subprocess(
    *,
    variant_name: str,
    run_name: str,
    cmd: list[str],
    out_dir: Path,
    compile_enabled: bool = False,
) -> None:
    """Run a short DDP training subprocess and assert train loss does not explode."""
    cleanup_smoke_training_store(run_name)
    compile_suffix = "compile" if compile_enabled else "nocompile"
    env = {
        "TORCHINDUCTOR_CACHE_DIR": str(ROOT / ".cache" / "torchinductor" / run_name / compile_suffix),
        "WANDB_MODE": "disabled",
    }
    proc = run_cmd(cmd, env=env)
    assert proc.returncode == 0, f"[{variant_name}] training failed:\n{proc.stderr or proc.stdout}"
    losses = collect_train_losses(proc.stdout + "\n" + proc.stderr, out_dir=out_dir)
    assert_loss_not_exploding(losses)


def collect_train_losses(stdout: str, *, out_dir: Path) -> list[float]:
    """Prefer stdout (progress bar) train_loss values; fall back to the CSV logger."""
    losses = parse_train_losses(stdout)
    if losses:
        return losses
    return parse_train_losses_from_csv_logger(out_dir)


def assert_loss_not_exploding(
    losses: list[float],
    *,
    min_points: int = 5,
    max_value: float = 1e4,
    spike_factor: float = 50.0,
) -> None:
    """Assert short smoke losses stay finite, non-negative, and free of spikes."""
    assert len(losses) >= min_points, f"Expected at least {min_points} train_loss values, got {len(losses)}"
    for i, loss in enumerate(losses):
        assert math.isfinite(loss), f"Non-finite train_loss at step {i}: {loss}"
        assert loss >= 0.0, f"Negative train_loss at step {i}: {loss}"
        assert loss <= max_value, f"Train loss exploded at step {i}: {loss} > {max_value}"
    baseline = statistics.median(losses)
    if baseline > 0:
        for i, loss in enumerate(losses):
            assert loss <= baseline * spike_factor, (
                f"Train loss spike at step {i}: {loss} > {spike_factor}x median {baseline}"
            )


def assert_loss_decreased(losses: list[float], *, min_points: int = 5, rtol: float = 0.0) -> None:
    """Assert the mean of the last 10% of losses is below the first 10%."""
    assert len(losses) >= min_points, f"Expected at least {min_points} train_loss values, got {len(losses)}"
    window = max(1, len(losses) // 10)
    start = sum(losses[:window]) / window
    end = sum(losses[-window:]) / window
    assert end <= start * (1.0 - rtol) + 1e-6, (
        f"Expected loss to decrease: start≈{start:.4f}, end≈{end:.4f}, samples={losses[:3]}...{losses[-3:]}"
    )
