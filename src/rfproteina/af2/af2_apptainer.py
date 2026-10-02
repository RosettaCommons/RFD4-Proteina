"""AF2 apptainer command builder shared by ``af2_daemon`` and direct callers.

This module is intentionally lightweight: ``af2_daemon.py`` is launched with
the host Python interpreter, where importing the full ``rfproteina`` package
may not be possible.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# The resident worker compiles against this value at startup. Per-request
# overrides would require recompilation.
AF2_NUM_RECYCLES = 3
DEFAULT_RESIDENT_MODELS = ("model_1_ptm", "model_1_multimer_v3")

_DEFAULT_BATCH_SCRIPT = Path(__file__).resolve().parent / "run_alphafold2_batch.py"
_COMPILE_MODES = {"fast", "tuned", "full"}
_AUTOTUNE_TMP_RELATIVE = Path("xla_gpu_per_fusion_autotune_cache_dir") / "tmp"


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise OSError(f"{name} is not set. Source .ipd/.env.<cluster> before running.")
    return value


def _prepare_jax_cache(cache_dir: str) -> None:
    """Create XLA's derived writable temporary directory before container launch."""
    autotune_tmp = Path(cache_dir) / _AUTOTUNE_TMP_RELATIVE
    try:
        autotune_tmp.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"AF2 JAX cache is not writable at {autotune_tmp}: {exc}") from exc
    if not os.access(autotune_tmp, os.W_OK):
        raise OSError(f"AF2 JAX cache is not writable at {autotune_tmp}")


def _resident_models(override: str | None = None) -> str:
    """Return a non-empty resident set for every worker launch route."""
    return (override or os.environ.get("AF2_RESIDENT_MODELS", "")).strip() or ",".join(DEFAULT_RESIDENT_MODELS)


def _runtime_optimization_args() -> list[str]:
    raw_pallas = os.environ.get("AF2_USE_PALLAS", "true").strip().lower()
    if raw_pallas in {"1", "true", "yes", "on"}:
        use_pallas = "true"
    elif raw_pallas in {"0", "false", "no", "off"}:
        use_pallas = "false"
    else:
        raise ValueError(f"AF2_USE_PALLAS must be a boolean value, got {raw_pallas!r}")
    compile_mode = os.environ.get("AF2_COMPILE_MODE", "tuned").strip().lower()
    if compile_mode not in _COMPILE_MODES:
        raise ValueError(f"AF2_COMPILE_MODE must be one of {', '.join(sorted(_COMPILE_MODES))}, got {compile_mode!r}")
    return [f"--use_pallas={use_pallas}", f"--compile_mode={compile_mode}"]


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


def _data_bind_specs() -> list[str]:
    """Return configured data paths and host temporary directories."""
    specs: list[str] = []
    raw = os.environ.get("APPTAINER_BIND_PATHS", "").strip()
    if raw:
        specs.extend(spec.strip() for spec in raw.split(";") if spec.strip())
    specs.append("/tmp:/tmp")
    tmp_root = tempfile.gettempdir()
    specs.append(f"{tmp_root}:{tmp_root}")
    return _deduped_bind_specs(*specs)


def _data_bind_args() -> list[str]:
    return _bind_args_for_specs(_data_bind_specs())


def _af2_resource_bind_specs() -> list[str]:
    """Return AF2 weight and compilation-cache bind specifications."""
    specs: list[str] = []
    for env_name in ("AF2_MODEL_DIR", "AF2_JAX_CACHE_DIR"):
        path = os.environ.get(env_name, "").strip().rstrip("/")
        if path:
            specs.append(f"{path}:{path}")
    return _deduped_bind_specs(*specs)


def _af2_resource_bind_args() -> list[str]:
    return _bind_args_for_specs(_af2_resource_bind_specs())


def _resolve_backend(backend: str | None = None) -> str:
    explicit = backend or os.environ.get("AF2_RUNTIME_BACKEND")
    if explicit:
        name = explicit.strip().lower()
    else:
        name = "pyxis" if os.environ.get("CONTAINER_TYPE", "").lower() == "docker" else "apptainer"
    if name not in {"apptainer", "pyxis", "docker"}:
        raise ValueError(f"AF2_RUNTIME_BACKEND must be 'apptainer', 'pyxis', or 'docker', got {name!r}")
    return name


def _worker_args(
    manifest_path: str | Path | None,
    *,
    jax_cache: str,
    weights: str,
    resident_models: str,
    serve: bool,
) -> list[str]:
    """Return the AF2 batch-script arguments shared by all container backends."""
    args = [
        f"--num_recycles={AF2_NUM_RECYCLES}",
        "--gpu_device=0",
        f"--jax_compilation_cache_dir={jax_cache}",
        f"--model_dir={weights}",
        f"--resident_models={resident_models}",
        *_runtime_optimization_args(),
    ]
    if serve:
        args.append("--serve")
    else:
        if manifest_path is None:
            raise ValueError("manifest_path is required when serve is False.")
        args.append(f"--manifest={manifest_path}")
    return args


def _normalize_pyxis_container_image(image: str) -> str:
    """Normalize a Pyxis Docker reference or existing local squashfs image."""
    original = image
    if image.startswith("docker://"):
        image = image[len("docker://") :]

    is_clear_reference = _is_clear_docker_reference(original)
    path = Path(image).expanduser()
    is_local = (
        image.startswith(("/", "./", "../", "~"))
        or path.is_file()
        or (path.suffix in {".sif", ".sqsh"} and not is_clear_reference)
    )
    if is_local:
        if path.suffix == ".sif":
            raise ValueError(f"Pyxis requires a squashfs image, not an Apptainer .sif archive: {image!r}")
        if not path.is_file():
            raise ValueError(f"Pyxis local container image does not exist: {image!r}")
        return str(path.resolve())

    if "#" in image:
        return image
    if "/" in image:
        registry, path_and_tag = image.split("/", 1)
        return f"{registry}#{path_and_tag}"
    return image


def _is_clear_docker_reference(image: str) -> bool:
    """Return whether ``image`` has syntax that unambiguously identifies an OCI reference.

    Archive-suffixed names are treated as local unless they use ``docker://``,
    contain an explicit tag or digest, or name a registry host (``localhost``,
    a dotted host, or a host with a port). Explicit local prefixes take
    precedence over registry-host detection. This conservative ambiguity rule
    rejects missing local archives early without using filesystem existence.
    """
    if image.startswith(("/", "./", "../", "~")):
        return False
    if image.startswith("docker://") or "@" in image:
        return True
    repository = image.rsplit("/", 1)[-1]
    if ":" in repository:
        return True
    if "/" not in image:
        return False
    host = image.split("/", 1)[0]
    return host == "localhost" or "." in host or ":" in host


def _normalize_docker_container_image(image: str) -> str:
    """Accept Docker/OCI references while rejecting local Apptainer archives."""
    is_clear_reference = _is_clear_docker_reference(image)
    if image.startswith("docker://"):
        image = image[len("docker://") :]
    if not is_clear_reference and Path(image).suffix in {".sqsh", ".sif"}:
        raise ValueError(
            f"Docker backend requires an OCI/Docker image reference, not a local .sqsh or .sif path: {image!r}"
        )
    return image


def build_apptainer_fold_command(
    manifest_path: str | Path | None,
    gpu_device: int,
    batch_script: str | Path | None = None,
    serve: bool = False,
    resident_models: str | None = None,
) -> list[str]:
    """Build the AF2 one-shot or resident-worker apptainer command."""
    sif = _require("AF2_IMAGE")
    jax_cache = _require("AF2_JAX_CACHE_DIR")
    _prepare_jax_cache(jax_cache)
    weights = _require("AF2_MODEL_DIR")
    resident_models = _resident_models(resident_models)
    if batch_script is None:
        batch_script = _DEFAULT_BATCH_SCRIPT
    batch_path = Path(batch_script).resolve()

    # Apptainer --nv injects driver libs into /.singularity.d/libs, but this
    # image does not set LD_LIBRARY_PATH (unlike AF3), so JAX cannot load CUDA.
    cmd = [
        "apptainer",
        "exec",
        "--nv",
        "--env",
        f"CUDA_VISIBLE_DEVICES={int(gpu_device)}",
        "--env",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "--env",
        "JAX_PLATFORMS=cuda,cpu",
        "--env",
        "LD_LIBRARY_PATH=/.singularity.d/libs",
        "--env",
        "PYTHONPATH=/app/alphafold",
        "--env",
        f"AF2_MODEL_DIR={weights}",
        "--env",
        f"AF2_RESIDENT_MODELS={resident_models}",
        *_data_bind_args(),
        *_af2_resource_bind_args(),
        "--bind",
        f"{batch_path.parent}:{batch_path.parent}",
        sif,
        "/opt/conda/bin/python",
        str(batch_path),
        *_worker_args(
            manifest_path,
            jax_cache=jax_cache,
            weights=weights,
            resident_models=resident_models,
            serve=serve,
        ),
    ]
    return cmd


def build_pyxis_fold_command(
    manifest_path: str | Path | None,
    gpu_device: int,
    batch_script: str | Path | None = None,
    serve: bool = False,
    resident_models: str | None = None,
) -> list[str]:
    """Build a Docker/Pyxis ``srun --container-image`` AF2 worker command."""
    image = _normalize_pyxis_container_image(_require("AF2_IMAGE"))
    jax_cache = _require("AF2_JAX_CACHE_DIR")
    _prepare_jax_cache(jax_cache)
    weights = _require("AF2_MODEL_DIR")
    resident_models = _resident_models(resident_models)
    if batch_script is None:
        batch_script = _DEFAULT_BATCH_SCRIPT
    batch_path = Path(batch_script).resolve()

    mount_specs = [
        *_data_bind_specs(),
        *_af2_resource_bind_specs(),
        "/tmp:/tmp",
        f"{batch_path.parent}:{batch_path.parent}",
    ]
    if not serve:
        if manifest_path is None:
            raise ValueError("manifest_path is required when serve is False.")
        manifest_file = Path(manifest_path).resolve()
        mount_specs.append(f"{manifest_file.parent}:{manifest_file.parent}")
    mounts = _deduped_bind_specs(*mount_specs)
    node = _require("SLURMD_NODENAME")
    cmd = [
        "srun",
        "--overlap",
        "--exact",
        "--ntasks=1",
        "--nodes=1",
        "--cpu-bind=none",
        "--kill-on-bad-exit=1",
        f"--nodelist={node}",
    ]
    cmd.extend(
        [
            f"--container-image={image}",
            f"--container-mounts={','.join(mounts)}",
            "/usr/bin/env",
            f"CUDA_VISIBLE_DEVICES={int(gpu_device)}",
            "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "JAX_PLATFORMS=cuda,cpu",
            "PYTHONPATH=/app/alphafold",
            f"AF2_MODEL_DIR={weights}",
            f"AF2_RESIDENT_MODELS={resident_models}",
            "/opt/conda/bin/python",
            str(batch_path),
            *_worker_args(
                manifest_path,
                jax_cache=jax_cache,
                weights=weights,
                resident_models=resident_models,
                serve=serve,
            ),
        ]
    )
    return cmd


def build_docker_fold_command(
    manifest_path: str | Path | None,
    gpu_device: int,
    batch_script: str | Path | None = None,
    serve: bool = False,
    resident_models: str | None = None,
) -> list[str]:
    """Build a local Docker AF2 worker command.

    The Pallas AF2 image ships ``ENTRYPOINT ["/app/run_alphafold.sh"]``, which
    absl-parses argv and rejects our batch-script flags. Override the entrypoint
    to the packaged interpreter so ``docker run`` matches Apptainer ``exec`` /
    Pyxis command replacement.
    """
    image = _normalize_docker_container_image(_require("AF2_IMAGE"))
    jax_cache = _require("AF2_JAX_CACHE_DIR")
    _prepare_jax_cache(jax_cache)
    weights = _require("AF2_MODEL_DIR")
    resident_models = _resident_models(resident_models)
    if batch_script is None:
        batch_script = _DEFAULT_BATCH_SCRIPT
    batch_path = Path(batch_script).resolve()

    mount_specs = [
        *_data_bind_specs(),
        *_af2_resource_bind_specs(),
        f"{batch_path.parent}:{batch_path.parent}",
    ]
    if not serve:
        if manifest_path is None:
            raise ValueError("manifest_path is required when serve is False.")
        manifest_file = Path(manifest_path).resolve()
        mount_specs.append(f"{manifest_file.parent}:{manifest_file.parent}")
    mounts = _deduped_bind_specs(*mount_specs)
    # Match host uid/gid so jax_cache and fold outputs are not root-owned on the
    # bind mounts. Skip only when explicitly requested (e.g. rootless edge cases).
    skip_user = os.environ.get("AF2_DOCKER_SKIP_USER", "").strip().lower() in {"1", "true", "yes", "on"}
    cmd = [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--init",
        "--gpus",
        f"device={int(gpu_device)}",
        # JAX scratch / XLA collectives need more than Docker's 64 MiB default.
        "--shm-size=8g",
        "--entrypoint",
        "/opt/conda/bin/python",
    ]
    if not skip_user:
        cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    cmd.extend(
        [
            *[arg for spec in mounts for arg in ("--volume", spec)],
            "--env",
            "CUDA_VISIBLE_DEVICES=0",
            "--env",
            "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "--env",
            "JAX_PLATFORMS=cuda,cpu",
            "--env",
            "PYTHONPATH=/app/alphafold",
            "--env",
            f"HOME={tempfile.gettempdir()}",
            "--env",
            f"AF2_MODEL_DIR={weights}",
            "--env",
            f"AF2_RESIDENT_MODELS={resident_models}",
            image,
            str(batch_path),
            *_worker_args(
                manifest_path,
                jax_cache=jax_cache,
                weights=weights,
                resident_models=resident_models,
                serve=serve,
            ),
        ]
    )
    return cmd


def build_container_fold_command(
    manifest_path: str | Path | None,
    gpu_device: int,
    batch_script: str | Path | None = None,
    serve: bool = False,
    backend: str | None = None,
    resident_models: str | None = None,
) -> list[str]:
    """Select Apptainer, Pyxis, or Docker from container configuration."""
    resolved = _resolve_backend(backend)
    builders = {
        "apptainer": build_apptainer_fold_command,
        "pyxis": build_pyxis_fold_command,
        "docker": build_docker_fold_command,
    }
    builder = builders[resolved]
    return builder(
        manifest_path=manifest_path,
        gpu_device=gpu_device,
        batch_script=batch_script,
        serve=serve,
        resident_models=resident_models,
    )
