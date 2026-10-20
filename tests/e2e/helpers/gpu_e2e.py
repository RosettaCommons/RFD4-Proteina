"""GPU e2e guards, subprocess runners, and training-loss parsing.

Adapted from protein-foundation-models ``tests/e2e/helpers/gpu_e2e.py`` for the
foundry ``rfproteina`` project. ``ROOT`` is the project dir
(``projects/rfproteina``) since that is the working directory training
and inference subprocesses run from (``store/``, ``inference/`` live there).
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
import shutil
import signal
import stat
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal, NoReturn

import pytest

from tests.conftest import has_af2, has_af3

# helpers/ -> e2e/ -> tests/ -> rfproteina (project root for train/infer)
ROOT = Path(__file__).resolve().parents[3]
RUN_E2E = os.getenv("RUN_GPU_E2E_TESTS", "0") == "1"


def get_data_path() -> Path | None:
    """Return ``$DATA_PATH`` as a :class:`~pathlib.Path`, or ``None`` if unset."""
    raw = os.environ.get("DATA_PATH", "").strip()
    return Path(raw) if raw else None


PrevalOracle = Literal["af3", "af2", "none"]

# CI training smokes use a small batch to limit VRAM and speed up steps.
GPU_E2E_TRAIN_BATCH_SIZE = int(os.getenv("GPU_E2E_TRAIN_BATCH_SIZE", "2"))
GPU_E2E_NGPUS = int(os.getenv("GPU_E2E_NGPUS", "2"))

# Short training smoke shape: 100 optimizer steps + light epoch-0 validation.
GPU_E2E_TRAIN_STEPS = int(os.getenv("GPU_E2E_TRAIN_STEPS", "100"))
# Lightning sanity check: validation batches *total* across all val loaders (not per dataset).
GPU_E2E_SANITY_VAL_STEPS = int(os.getenv("GPU_E2E_SANITY_VAL_STEPS", "0"))
# Epoch validation: batches *per* val dataset (SOTA has ~11; refolding makes these expensive).
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


def _missing_data(reason: str) -> NoReturn:
    """Fail in CI, skip elsewhere.

    A GPU test that skips still reports the lane green while contributing no coverage, so whether
    it ran comes down to which runner the job landed on. In CI that is a failure, not a skip.
    """
    if os.getenv("CI", "").lower() in ("true", "1", "yes"):
        pytest.fail(reason)
    pytest.skip(reason)


def require_data_file(relative: str) -> Path:
    require_gpu_e2e()
    data_path = get_data_path()
    if data_path is None:
        _missing_data("Set DATA_PATH for training/inference data")
    path = data_path / relative
    if not path.exists():
        _missing_data(f"Missing data file: {path}")
    return path


def preval_oracle_overrides(oracle: PrevalOracle) -> list[str]:
    """Hydra overrides for prevalidation refolding oracle selection."""
    if oracle == "af3":
        return ["metrics/refolding_oracles=af3"]
    if oracle == "af2":
        return ["metrics/refolding_oracles=af2"]
    return ["~metrics.refolding_oracles"]


def require_preval_oracle(oracle: PrevalOracle) -> None:
    """Skip when the variant's prevalidation oracle is unavailable."""
    require_gpu_e2e()
    if oracle == "af3":
        if os.environ.get("AF3_LAUNCHER_SOCKET_TEMPLATE"):
            return
        if not has_af3:
            message = "AF3 not available (set AF3_IMAGE and AF3_MODEL_DIR, or AF3_LAUNCHER_SOCKET_TEMPLATE)"
            if os.environ.get("REQUIRE_AF3") == "1":
                pytest.fail(message)
            pytest.skip(message)
        return
    if oracle == "af2":
        if os.environ.get("AF2_LAUNCHER_SOCKET_TEMPLATE"):
            return
        if not has_af2:
            message = "AF2 not available (set AF2_IMAGE and AF2_MODEL_DIR, or AF2_LAUNCHER_SOCKET_TEMPLATE)"
            if os.environ.get("REQUIRE_AF2") == "1":
                pytest.fail(message)
            pytest.skip(message)


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


def run_cmd_with_loss_watcher(
    cmd: list[str],
    *,
    out_dir: Path,
    expected_losses: list[float],
    atol: float,
    mean_atol: float,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    poll_interval: float = 1.0,
) -> None:
    """Run a training command with two-tier loss checking.

    While the subprocess runs, polls ``out_dir/csv_logs/version_*/metrics.csv``
    every ``poll_interval`` seconds and kills the process as soon as any single
    step's loss deviates from ``expected_losses`` by more than ``atol``.

    After a clean exit, also asserts that the mean absolute deviation across the
    full trajectory is within ``mean_atol``, catching systematic drift that stays
    below the per-step limit on every individual step.

    Args:
        cmd: Command list passed to Popen.
        out_dir: Training output directory (contains csv_logs/).
        expected_losses: Per-step baseline losses to compare against.
        atol: Per-step absolute tolerance; process is killed on first exceedance.
        mean_atol: Mean absolute deviation tolerance applied after all steps complete.
        cwd: Working directory for the subprocess.
        env: Additional env vars (merged over os.environ).
        poll_interval: Seconds between CSV polls.
    """
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

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    def _drain(pipe, buf):
        for line in pipe:
            buf.write(line)

    t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_buf), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_buf), daemon=True)
    t_out.start()
    t_err.start()

    fail_error: str | None = None
    checked_steps = 0

    while proc.poll() is None:
        losses = parse_train_losses_from_csv_logger(out_dir)
        for i in range(checked_steps, len(losses)):
            if i < len(expected_losses):
                exp, obs = expected_losses[i], losses[i]
                if abs(exp - obs) > atol:
                    fail_error = (
                        f"step {i}: expected {exp:.6f}, got {obs:.6f} " f"(diff={abs(exp - obs):.2e}, atol={atol:.0e})"
                    )
                    proc.terminate()
                    break
        checked_steps = len(losses)
        if fail_error:
            break
        time.sleep(poll_interval)

    t_out.join()
    t_err.join()
    _terminate_process_tree(proc.pid)

    assert_no_dataset_fallbacks(stdout_buf.getvalue() + stderr_buf.getvalue())

    if fail_error:
        raise AssertionError(
            f"Training loss mismatch — {fail_error}\n"
            f"Subprocess stderr (last 2000 chars):\n{stderr_buf.getvalue()[-2000:]}"
        )

    stderr_text = stderr_buf.getvalue()
    assert proc.returncode == 0, (
        f"Training subprocess failed (exit {proc.returncode}):\n" f"{stderr_text or stdout_buf.getvalue()}"
    )

    # Confirm the expected number of steps were produced, then check mean deviation.
    losses = parse_train_losses_from_csv_logger(out_dir)
    assert len(losses) == len(expected_losses), f"Expected {len(expected_losses)} training steps, got {len(losses)}"
    mean_diff = sum(abs(obs - exp) for obs, exp in zip(losses, expected_losses, strict=False)) / len(losses)
    assert mean_diff <= mean_atol, (
        f"Mean training loss deviation {mean_diff:.4f} exceeds {mean_atol} "
        f"(per-step atol {atol} was not exceeded on any individual step)"
    )


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
        col = next(
            (
                c
                for c in (reader.fieldnames or [])
                if c in ("train_loss", "train/loss_step", "train_loss_step") or c.endswith("/train_loss")
            ),
            None,
        )
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


DATASET_FALLBACK_LINE = "Trying fallback index"


def assert_no_dataset_fallbacks(output: str) -> None:
    """Assert the dataset never substituted an example for one that failed to load.

    ``concat_dataset`` swallows any parse error and returns a different example, so a run can
    silently train on data nobody configured. Workers inherit the subprocess stderr, so counting
    the log line is enough.

    Args:
      output: Combined subprocess stdout+stderr.
    """
    n_fallbacks = output.count(DATASET_FALLBACK_LINE)
    if n_fallbacks:
        causes = re.findall(r"Error \(.*?\) at index [^\n]*", output, flags=re.DOTALL)[:3]
        raise AssertionError(
            f"Dataset fell back {n_fallbacks} times; this run did not train on the configured "
            "data. First causes:\n" + "\n".join(causes)
        )


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
            assert (
                loss <= baseline * spike_factor
            ), f"Train loss spike at step {i}: {loss} > {spike_factor}x median {baseline}"


def assert_loss_decreased(losses: list[float], *, min_points: int = 5, rtol: float = 0.0) -> None:
    """Assert the mean of the last 10% of losses is below the first 10%."""
    assert len(losses) >= min_points, f"Expected at least {min_points} train_loss values, got {len(losses)}"
    window = max(1, len(losses) // 10)
    start = sum(losses[:window]) / window
    end = sum(losses[-window:]) / window
    assert (
        end <= start * (1.0 - rtol) + 1e-6
    ), f"Expected loss to decrease: start≈{start:.4f}, end≈{end:.4f}, samples={losses[:3]}...{losses[-3:]}"
