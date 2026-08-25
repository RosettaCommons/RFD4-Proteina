"""Test fixtures and utilities"""

# Load .env BEFORE any atomworks imports (they read env vars at import time)
from rfproteina.utils.env import load_dotenv

load_dotenv(override=True)

import os

# Container/CI runners may lack passwd entries for the job UID; torch inductor
# cache setup calls getpass.getuser() and fails without these.
os.environ.setdefault("USER", "pytest")
os.environ.setdefault("LOGNAME", "pytest")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor-pytest")
import subprocess

_MASTER_PORT_FALLBACK = 29500


def _pin_gpu_for_xdist_worker() -> None:
    """Give each pytest-xdist worker its own GPU and rendezvous port.

    Unpinned, every worker would default to device 0. Must run before torch creates a CUDA
    context, hence its position above the imports below; training subprocesses inherit the env.
    A no-op outside the GPU lane -- the cpu lane also runs under xdist, but its container has
    no GPUs (or nvidia-smi).
    """
    if os.environ.get("RUN_GPU_E2E_TESTS") != "1":
        return

    worker = os.environ.get("PYTEST_XDIST_WORKER", "")  # "gw0", "gw1", ...; empty when serial
    if not worker.startswith("gw"):
        return
    index = int(worker[2:])

    # Offset the per-run random port (run_gpu_tests.sh), not a constant: two 4-GPU jobs fit on an
    # 8-GPU node, and a fixed base would give both the same rendezvous ports.
    base = int(os.environ.get("MASTER_PORT") or _MASTER_PORT_FALLBACK)
    os.environ["MASTER_PORT"] = str(base + index)

    devices = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if not devices:
        try:
            listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        except (FileNotFoundError, OSError):
            return
        devices = [str(i) for i, ln in enumerate(listed.stdout.splitlines()) if ln.startswith("GPU ")]
    if not devices:
        return

    # Wrapping would silently co-locate two workers on one GPU and OOM mid-run.
    if index >= len(devices):
        raise RuntimeError(
            f"pytest-xdist worker {worker} has no GPU: {len(devices)} visible ({devices}). "
            f"Allocate one GPU per worker (--gres=gpu:N with -n N), or lower GPU_PYTEST_WORKERS."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = devices[index]

    os.environ["AF2_DAEMON_RANK"] = str(index)
    os.environ["AF3_DAEMON_RANK"] = str(index)


_pin_gpu_for_xdist_worker()

import shutil

import pytest
from atomworks.ml.executables import _EXECUTABLES
from atomworks.ml.executables.dssp import DSSPExecutable
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state

from tests.helpers import CONFIGS_DIR, PROJECT_ROOT, configure_pytest, get_test_data_dir, gpu  # noqa: F401

TEST_DATA_DIR = get_test_data_dir(__file__)


def pytest_configure(config):
    """Set up environment variables, paths, etc."""
    configure_pytest(config, __file__)


# DSSP setup for secondary structure tests
DSSP_PATH = os.environ.get("DSSP_PATH", "/path/to/data")


def _has_dssp() -> bool:
    """Check if DSSP is installed and available."""
    if os.path.isfile(DSSP_PATH) and os.access(DSSP_PATH, os.X_OK):
        try:
            DSSPExecutable.get_or_initialize(DSSP_PATH)
            return True
        except Exception:
            return False
    return False


has_dssp = _has_dssp()
skip_if_no_dssp = pytest.mark.skipif(not has_dssp, reason="DSSP is not installed or unavailable")

AF3_IMAGE = os.environ.get("AF3_IMAGE", "/path/to/shared/storage")
AF3_MODEL_DIR = os.environ.get("AF3_MODEL_DIR", "")


def _has_af3() -> bool:
    """Check if AF3 is available: runtime image, apptainer binary, and weights all present."""
    return os.path.isfile(AF3_IMAGE) and shutil.which("apptainer") is not None and bool(AF3_MODEL_DIR)


has_af3 = _has_af3()
skip_if_no_af3 = pytest.mark.skipif(not has_af3, reason="AF3 apptainer image not available")

AF2_IMAGE = os.environ.get("AF2_IMAGE", "")
AF2_MODEL_DIR = os.environ.get("AF2_MODEL_DIR", "")


def _af2_image_available(image: str, backend: str) -> bool:
    """Apply the same image-reference rules as the selected AF2 runtime."""
    if not image:
        return False
    if backend == "apptainer":
        return os.path.isfile(image)

    from rfproteina.af2.af2_apptainer import (
        _normalize_docker_container_image,
        _normalize_pyxis_container_image,
    )

    normalizer = {
        "docker": _normalize_docker_container_image,
        "pyxis": _normalize_pyxis_container_image,
    }.get(backend)
    if normalizer is None:
        return False
    try:
        normalizer(image)
    except (OSError, ValueError):
        return False
    return True


def _has_af2() -> bool:
    """Check if the configured AF2 image, runtime backend, and weights are available."""
    image = os.environ.get("AF2_IMAGE", AF2_IMAGE)
    model_dir = os.environ.get("AF2_MODEL_DIR", AF2_MODEL_DIR)
    backend = os.environ.get("AF2_RUNTIME_BACKEND", "apptainer").strip().lower()
    runtime = {"apptainer": "apptainer", "pyxis": "srun", "docker": "docker"}.get(backend)
    return (
        bool(image)
        and _af2_image_available(image, backend)
        and runtime is not None
        and shutil.which(runtime) is not None
        and bool(model_dir)
    )


has_af2 = _has_af2()
skip_if_no_af2 = pytest.mark.skipif(not has_af2, reason="AF2 runtime is not available")


@pytest.fixture
def cleanup_dssp():
    """Reinitialize DSSPExecutable for test isolation.

    Ensures the DSSP executable singleton is reset between tests.
    """
    if has_dssp:
        DSSPExecutable.reinitialize(DSSP_PATH)
    yield
    _EXECUTABLES.pop(DSSPExecutable.name, None)
    DSSPExecutable._is_initialized = False


@pytest.fixture
def deterministic_rng():
    """Fixture for deterministic random number generation."""
    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)):
        yield
