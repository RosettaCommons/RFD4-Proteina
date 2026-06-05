"""AF3 apptainer command builder shared by ``af3_daemon`` (warm-worker spawner)
and :py:class:`~proteinfoundation.metrics.af3_oracle.AF3Oracle` (direct fallback
route).

Lives under ``scripts/`` (not under the ``proteinfoundation`` package) on
purpose: ``af3_daemon.py`` is launched via the host's ``/usr/bin/python3`` from
the slurm launcher, where heavy package init (``import torch``) isn't
available. Importing this module must not trigger any of that.
"""

import os
from pathlib import Path

# Hardcoded everywhere — the daemon JAX-compiles against this value at startup
# and per-call overrides aren't possible without recompiling. Edit here if a
# different value is genuinely needed.
AF3_NUM_RECYCLES = 5

# Default location of the warm-worker entry script (sibling under scripts/).
_DEFAULT_BATCH_SCRIPT = Path(__file__).resolve().parent / "run_alphafold_batch.py"


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise OSError(f"{name} is not set. Source .ipd/.env.<cluster> before running.")
    return value


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
    ``/tmp`` is always bound (daemon socket lives there).
    """
    specs: list[str] = []
    raw = os.environ.get("APPTAINER_BIND_PATHS", "").strip()
    if raw:
        specs.extend(spec.strip() for spec in raw.split(";") if spec.strip())
    specs.append("/tmp:/tmp")
    return _bind_args_for_specs(_deduped_bind_specs(*specs))


def _af3_resource_bind_args() -> list[str]:
    """Bind AF3 script/weight/cache paths into ``af3.sif``.

    Without these binds, nested ``apptainer run af3.sif`` (e.g. from inside
    latent-dev during GPU e2e) cannot see host paths like
    ``/path/to/shared/storage``.
    """
    specs: list[str] = []
    for env_name in ("AF3_SCRIPTS_DIR", "AF3_MODEL_DIR", "AF3_JAX_CACHE_DIR"):
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
    """Build the ``apptainer run af3.sif python run_alphafold_batch.py`` argv.

    Reads ``AF3_APPTAINER_PATH`` / ``AF3_SCRIPTS_DIR`` / ``AF3_JAX_CACHE_DIR`` /
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
    sif = _require("AF3_APPTAINER_PATH")
    scripts_dir = _require("AF3_SCRIPTS_DIR")
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
        f"AF3_SCRIPTS_DIR={scripts_dir}",
        "--env",
        f"AF3_MODEL_DIR={weights}",
        *_data_bind_args(),
        *_af3_resource_bind_args(),
        "--bind",
        f"{batch_path.parent}:{batch_path.parent}",
        sif,
        "python",
        str(batch_script),
        f"--num_recycles={AF3_NUM_RECYCLES}",
        "--num_diffusion_samples=1",
        "--gpu_device=0",
        f"--jax_compilation_cache_dir={jax_cache}",
        f"--model_dir={weights}",
        f"--save_full_outputs={'true' if save_full_outputs else 'false'}",
    ]
    if serve:
        cmd.append("--serve")
    else:
        if manifest_path is None:
            raise ValueError("manifest_path is required when serve is False.")
        cmd.append(f"--manifest={manifest_path}")
    return cmd
