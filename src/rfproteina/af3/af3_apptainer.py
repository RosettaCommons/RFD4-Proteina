"""AF3 apptainer command builder shared by ``af3_daemon`` (warm-worker spawner)
and :py:class:`~rfproteina.metrics.oracles.af3_oracle.AF3Oracle` (direct fallback
route).

Lives under ``scripts/`` (not under the ``rfproteina`` package) on
purpose: ``af3_daemon.py`` is launched via the host's ``/usr/bin/python3`` from
the slurm launcher, where heavy package init (``import torch``) isn't
available. Importing this module must not trigger any of that.
"""

import os
import tempfile
from pathlib import Path

# Hardcoded everywhere — the daemon JAX-compiles against this value at startup
# and per-call overrides aren't possible without recompiling. Edit here if a
# different value is genuinely needed.
AF3_NUM_RECYCLES = 5

# AF3 attention backend (triton/cudnn/xla). AF3's ``triton`` default over-requests shared memory on the
# Ampere/Ada cards here, failing folds with ``RESOURCE_EXHAUSTED``; ``cudnn`` is robust everywhere, so set
# ``=triton`` only on a datacenter GPU. Lowercased: the container flag is a case-sensitive absl enum.
AF3_FLASH_ATTENTION_IMPLEMENTATION = os.environ.get("AF3_FLASH_ATTENTION_IMPLEMENTATION", "cudnn").strip().lower()

# XLA configuration for the fold worker. ``af3-open.def`` hardcodes this in the
# image's ``%environment``, which is sourced at container start and therefore
# overrides anything the host exports -- so passing it as ``--env`` does nothing.
# Both routes below instead set it *inside* the invocation, after that block has
# run, so the two backends provably share one XLA configuration. Without this the
# native route silently ran with triton GEMM enabled while the container ran with
# it disabled, which makes any container-vs-native comparison a comparison of two
# different computations.
AF3_XLA_FLAGS = os.environ.get("AF3_XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false").strip()

# Default location of the warm-worker entry script (sibling under scripts/).
_DEFAULT_BATCH_SCRIPT = Path(__file__).resolve().parent / "run_alphafold_batch.py"


_BACKENDS = ("apptainer", "native")


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise OSError(f"{name} is not set. Source .ipd/.env.<cluster> before running.")
    return value


def resolve_backend(backend: str | None = None) -> str:
    """Return the AF3 worker backend: ``apptainer`` (default) or ``native``.

    ``native`` skips the container and runs the batch script under
    ``AF3_PYTHON``. Unlike AF2's backend switch there is no ``CONTAINER_TYPE``
    fallback: a host with no apptainer has nothing to infer from, so the native
    route is opt-in via ``AF3_RUNTIME_BACKEND``.
    """
    explicit = backend or os.environ.get("AF3_RUNTIME_BACKEND")
    name = explicit.strip().lower() if explicit else "apptainer"
    if name not in _BACKENDS:
        raise ValueError(f"AF3_RUNTIME_BACKEND must be one of {_BACKENDS}, got {name!r}")
    return name


def required_env_vars(backend: str | None = None) -> tuple[str, ...]:
    """Env vars the selected backend needs before a worker can start.

    Both routes need the weights and the JAX cache; only apptainer needs an
    image, and only native needs an interpreter.
    """
    common = ("AF3_JAX_CACHE_DIR", "AF3_MODEL_DIR")
    if resolve_backend(backend) == "native":
        return ("AF3_PYTHON", *common)
    return ("AF3_IMAGE", *common)


def _worker_args(
    manifest_path: str | Path | None,
    *,
    jax_cache: str,
    weights: str,
    save_full_outputs: bool,
    serve: bool,
) -> list[str]:
    """Return the batch-script arguments shared by every AF3 backend."""
    args = [
        f"--num_recycles={AF3_NUM_RECYCLES}",
        f"--flash_attention_implementation={AF3_FLASH_ATTENTION_IMPLEMENTATION}",
        "--num_diffusion_samples=1",
        "--gpu_device=0",
        f"--jax_compilation_cache_dir={jax_cache}",
        f"--model_dir={weights}",
        f"--save_full_outputs={'true' if save_full_outputs else 'false'}",
    ]
    if serve:
        args.append("--serve")
    else:
        if manifest_path is None:
            raise ValueError("manifest_path is required when serve is False.")
        args.append(f"--manifest={manifest_path}")
    return args


def _deduped_bind_specs(*specs: str) -> list[str]:
    """Return unique ``host:container`` bind specs, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for spec in specs:
        spec = spec.strip()
        if spec and spec not in seen:
            seen.add(spec)
            out.append(spec)
    return out


def _bind_args_for_specs(specs: list[str]) -> list[str]:
    return [arg for spec in specs for arg in ("--bind", spec)]


def _data_bind_args() -> list[str]:
    """Return ``--bind ...`` argv for cluster-specific data mounts.

    Reads ``APPTAINER_BIND_PATHS`` — the project-wide cluster bind list set in
    ``.ipd/.env.<cluster>`` and already consumed by ``exec_core.sh``. Semicolon-
    separated host:container specs, e.g. ``/mnt/data:/mnt/data;/scratch:/scratch``.
    ``/tmp`` is always bound (daemon socket lives there), as is the resolved
    ``tempfile.gettempdir()`` root — on clusters where SLURM sets ``TMPDIR`` to
    a per-job scratch path, that's where ``AF3Oracle`` actually
    writes ``manifest.json``, and it won't be ``/tmp``.
    """
    specs: list[str] = []
    raw = os.environ.get("APPTAINER_BIND_PATHS", "").strip()
    if raw:
        specs.extend(spec.strip() for spec in raw.split(";") if spec.strip())
    specs.append("/tmp:/tmp")
    tmp_root = tempfile.gettempdir()
    specs.append(f"{tmp_root}:{tmp_root}")
    return _bind_args_for_specs(_deduped_bind_specs(*specs))


def _af3_resource_bind_args() -> list[str]:
    """Bind AF3 weight/cache paths into ``af3-open.sif``.

    Without these binds, nested ``apptainer run af3-open.sif`` (e.g. from inside
    latent-dev during GPU e2e) cannot see the host weights or compilation cache.
    """
    specs: list[str] = []
    for env_name in ("AF3_MODEL_DIR", "AF3_JAX_CACHE_DIR"):
        path = os.environ.get(env_name, "").strip().rstrip("/")
        if path:
            specs.append(f"{path}:{path}")
    return _bind_args_for_specs(_deduped_bind_specs(*specs))


def build_apptainer_fold_command(
    manifest_path: str | Path | None,
    gpu_device: int,
    save_full_outputs: bool,
    batch_script: str | Path | None = None,
    serve: bool = False,
) -> list[str]:
    """Build the ``apptainer run af3-open.sif python run_alphafold_batch.py`` argv.

    Reads ``AF3_IMAGE`` / ``AF3_JAX_CACHE_DIR`` /
    ``AF3_MODEL_DIR`` from env (set by ``.ipd/.env.<cluster>``); raises
    :py:class:`OSError` on missing var. Data binds come from
    ``APPTAINER_BIND_PATHS`` (see :py:func:`_data_bind_args`).

    ``num_recycles`` is hardcoded via :py:data:`AF3_NUM_RECYCLES`.

    Args:
        manifest_path: JSON manifest of design payloads. Required when
            ``serve`` is False; ignored (and may be ``None``) when ``serve``
            is True.
        gpu_device: Driver-level GPU index to expose via ``CUDA_VISIBLE_DEVICES``.
        save_full_outputs: Forwarded as ``--save_full_outputs={true,false}``.
        batch_script: Path to ``run_alphafold_batch.py``. Defaults to
            :py:data:`_DEFAULT_BATCH_SCRIPT`.
        serve: If True, build the warm-worker invocation: the apptainer
            subprocess loads AF3 once and reads manifest paths on stdin
            instead of folding a single ``--manifest`` argument and exiting.
    """
    sif = _require("AF3_IMAGE")
    jax_cache = _require("AF3_JAX_CACHE_DIR")
    weights = _require("AF3_MODEL_DIR")
    if batch_script is None:
        batch_script = _DEFAULT_BATCH_SCRIPT
    batch_path = Path(batch_script).resolve()

    cmd = [
        "apptainer",
        "run",
        "--nv",
        "--env",
        f"CUDA_VISIBLE_DEVICES={int(gpu_device)}",
        "--env",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "--env",
        "JAX_PLATFORMS=cuda",
        "--env",
        f"AF3_MODEL_DIR={weights}",
        *_data_bind_args(),
        *_af3_resource_bind_args(),
        "--bind",
        f"{batch_path.parent}:{batch_path.parent}",
        sif,
        "env",
        f"XLA_FLAGS={AF3_XLA_FLAGS}",
        "python",
        str(batch_script),
        *_worker_args(
            manifest_path,
            jax_cache=jax_cache,
            weights=weights,
            save_full_outputs=save_full_outputs,
            serve=serve,
        ),
    ]
    return cmd


def build_native_fold_command(
    manifest_path: str | Path | None,
    gpu_device: int,
    save_full_outputs: bool,
    batch_script: str | Path | None = None,
    serve: bool = False,
) -> list[str]:
    """Build the no-container ``python run_alphafold_batch.py`` argv.

    The container only ever supplied the pinned ``alphafold3`` package and its
    compiled extensions; the adapter in :py:mod:`rfproteina.af3.af3_runtime` and
    the batch script already live on the host. So the native route is the same
    worker, invoked through the interpreter of a second venv that has
    ``alphafold3`` (the OF3-capable fork) and ``jax[cuda12]`` installed.

    Reads ``AF3_PYTHON`` (that venv's interpreter) plus the same
    ``AF3_JAX_CACHE_DIR`` / ``AF3_MODEL_DIR`` as the apptainer route; no
    ``AF3_IMAGE``. Environment is carried by a leading ``env`` so the return
    type stays a plain argv and callers need no ``env=`` plumbing.
    """
    python = _require("AF3_PYTHON")
    jax_cache = _require("AF3_JAX_CACHE_DIR")
    weights = _require("AF3_MODEL_DIR")
    if batch_script is None:
        batch_script = _DEFAULT_BATCH_SCRIPT
    batch_path = Path(batch_script).resolve()

    # NB: deliberately not ``.resolve()``. A virtualenv's ``bin/python`` is a
    # symlink to the base interpreter, and a venv is only active when its own
    # path is the one executed (that is how ``sys.prefix`` is derived, via the
    # adjacent ``pyvenv.cfg``). Resolving the symlink launches the base
    # interpreter instead, silently dropping the venv's site-packages and
    # failing with ``ModuleNotFoundError: No module named 'alphafold3'``.
    # ``abspath`` normalises without following symlinks. ``is_file()`` does
    # follow them, so a dangling link is still rejected here.
    python_path = Path(os.path.abspath(Path(python).expanduser()))
    if not python_path.is_file():
        raise OSError(f"AF3_PYTHON does not point at an interpreter: {python!r}")

    return [
        "env",
        f"CUDA_VISIBLE_DEVICES={int(gpu_device)}",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "JAX_PLATFORMS=cuda",
        f"XLA_FLAGS={AF3_XLA_FLAGS}",
        f"AF3_MODEL_DIR={weights}",
        str(python_path),
        str(batch_path),
        *_worker_args(
            manifest_path,
            jax_cache=jax_cache,
            weights=weights,
            save_full_outputs=save_full_outputs,
            serve=serve,
        ),
    ]


def build_fold_command(
    manifest_path: str | Path | None,
    gpu_device: int,
    save_full_outputs: bool,
    batch_script: str | Path | None = None,
    serve: bool = False,
    backend: str | None = None,
) -> list[str]:
    """Select the Apptainer or native AF3 worker from runtime configuration."""
    builders = {
        "apptainer": build_apptainer_fold_command,
        "native": build_native_fold_command,
    }
    return builders[resolve_backend(backend)](
        manifest_path=manifest_path,
        gpu_device=gpu_device,
        save_full_outputs=save_full_outputs,
        batch_script=batch_script,
        serve=serve,
    )
