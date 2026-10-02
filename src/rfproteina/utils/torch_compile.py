"""Helpers for torch.compile / CUDA graph integration."""

from __future__ import annotations

import getpass
import logging
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from threading import get_ident

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning_utilities.core.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)

# Base path for the torch.compile "Mega-Cache" blob. A single file avoids the NFS
# stale-file-handle failures of a shared cache *directory*. The real path is derived from
# this by :py:func:`_blob_path`, which scopes the blob to the current user and torch build
# -- never point this at a literal file shared across either.
_BLOB_ENV = "TORCH_COMPILE_CACHE_BLOB"
_COMPILE_GUIDANCE_SHOWN = False


@contextmanager
def report_compile_progress(batch_size: int, length: int) -> Iterator[None]:
    """Report Inductor code generation once, leaving disk and in-memory cache hits quiet.

    Inductor's compilation-start log occurs after its graph-cache lookup. Dynamo's
    compile callbacks also run on disk-cache hits, so they cannot distinguish this work.
    """
    compiler_logger = logging.getLogger("torch._inductor.compile_fx")
    previous_level = compiler_logger.level
    effective_level = compiler_logger.getEffectiveLevel()
    thread_id = get_ident()
    reported = False

    def report(record: logging.LogRecord) -> bool:
        global _COMPILE_GUIDANCE_SHOWN
        nonlocal reported
        if not reported and get_ident() == thread_id and "torchinductor compiling " in record.getMessage():
            if rank_zero_only.rank == 0:
                message = f"Compiling model for padded length {length} tokens (batch size {batch_size})."
                if not _COMPILE_GUIDANCE_SHOWN:
                    message += " This may take a few minutes."
                    if os.environ.get(_BLOB_ENV):
                        message += "\nCached shapes can be reused across runs with compatible settings and hardware."
                    else:
                        message += (
                            "\nThis shape may need recompiling in later runs; "
                            "set TORCH_COMPILE_CACHE_BLOB to a persistent file path to preserve the cache."
                        )
                logger.info(message, extra={"always_visible": True})
                _COMPILE_GUIDANCE_SHOWN = True
            reported = True
        return record.levelno >= effective_level

    compiler_logger.addFilter(report)
    compiler_logger.setLevel(min(effective_level, logging.INFO))
    try:
        yield
    finally:
        compiler_logger.removeFilter(report)
        compiler_logger.setLevel(previous_level)


def _sanitized_user() -> str:
    """Return the current username with filesystem-unsafe characters replaced.

    Matches torch's own sanitization of :py:func:`getpass.getuser` when building the
    ``torchinductor_<user>`` cache dir.
    """
    return re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())


def configure_inductor_cache_dir() -> None:
    """Point Inductor at a per-rank, node-local cache dir unless one is already set.

    Must be called before the first ``torch.compile``; Inductor reads the env var when it builds a
    cache path. A cluster env file or launcher that already exports ``TORCHINDUCTOR_CACHE_DIR`` wins.
    """
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        # Already set, don't override it
        return

    # Default to a node-local temp dir with the username and rank in the path. Fall back through the
    # SLURM, torchrun, and Lightning rank vars so non-SLURM multi-process runs never share a cache dir.
    rank = os.environ.get("SLURM_PROCID") or os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{_sanitized_user()}_rank{rank}"
    logger.info(f"Inductor cache dir set to {os.environ['TORCHINDUCTOR_CACHE_DIR']}")


def torch_compile_kwargs(cfg: Mapping, *, default_fullgraph: bool = False) -> dict:
    """Build ``torch.compile`` kwargs from a config node.

    Args:
      cfg: A config node exposing ``.get`` (dict or OmegaConf).
      default_fullgraph: Value used when ``torch_compile_fullgraph`` is absent.
    """
    kwargs: dict = {"fullgraph": bool(cfg.get("torch_compile_fullgraph", default_fullgraph)), "dynamic": False}
    mode = cfg.get("torch_compile_mode", None)
    if mode:
        kwargs["mode"] = mode
    return kwargs


def maybe_mark_cudagraph_step_begin() -> None:
    """Mark a training-step boundary for torch.compile CUDA graphs.

    No-op on CPU-only runs and in environments where inductor cache setup fails
    (e.g. pytest-xdist workers without passwd entries).
    """
    if not torch.cuda.is_available():
        return
    compiler = getattr(torch, "compiler", None)
    if compiler is None or not hasattr(compiler, "cudagraph_mark_step_begin"):
        return
    try:
        compiler.cudagraph_mark_step_begin()
    except (AssertionError, ImportError, KeyError, OSError, RuntimeError):
        return


def _blob_path() -> str | None:
    """Return the blob path for this user and torch build, or ``None`` if unconfigured.

    Cached artifacts bake in absolute paths from the run that compiled them (Inductor
    cache dirs, autotune ``.best_config`` targets), so a blob is only reusable by the
    user who wrote it: another user's run would autotune into *their* directories and
    die with ``PermissionError``. Artifacts are also tied to the exact torch build --
    a foreign blob either fails to unpickle or silently misses, and would then be
    overwritten. Both are avoided by giving every (user, torch build) its own file:
    ``<dir>/<user>/<name>.<torch_version>``.

    Deriving the user here rather than interpolating ``${USER}`` in the cluster env
    files keeps the scoping intact when ``USER`` is unset (``getpass.getuser`` falls
    back to ``LOGNAME``/``pwd``), where ``${USER}`` would collapse to a shared path.
    """
    base = os.environ.get(_BLOB_ENV)
    if not base:
        return None
    directory, name = os.path.split(base)
    return os.path.join(directory, _sanitized_user(), f"{name}.{torch.__version__}")


def load_compile_cache() -> None:
    """Load this run's torch.compile Mega-Cache blob (see :py:func:`_blob_path`) if present."""
    path = _blob_path()
    if not path or not os.path.exists(path) or not hasattr(torch.compiler, "load_cache_artifacts"):
        return
    try:
        with open(path, "rb") as f:
            torch.compiler.load_cache_artifacts(f.read())
        logger.info(f"Loaded torch.compile cache from {path}")
    except Exception as e:  # a partial/corrupt blob must never break startup
        logger.warning(f"Ignoring unreadable torch.compile cache {path}: {e}")


def save_compile_cache() -> None:
    """Serialize compiled artifacts to this run's blob (single-file atomic write)."""
    path = _blob_path()
    if not path or not hasattr(torch.compiler, "save_cache_artifacts"):
        return
    out = torch.compiler.save_cache_artifacts()
    if out is None:
        return
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(out[0])
        os.chmod(tmp, 0o644)  # independent of the writer's umask
        os.replace(tmp, path)
    except OSError as e:  # never fail a finished run over a cache write
        logger.warning(f"Could not save torch.compile cache to {path}: {e}")
        if os.path.exists(tmp):
            os.unlink(tmp)
        return
    logger.info(f"Saved torch.compile cache to {path}")


def setup_run_compile(*, inference: bool = False) -> None:
    """Configure torch.compile for a run and warm its local Inductor cache."""
    global _COMPILE_GUIDANCE_SHOWN
    _COMPILE_GUIDANCE_SHOWN = False
    torch._inductor.config.coordinate_descent_tuning = True
    configure_inductor_cache_dir()
    if inference:
        blob = os.environ.get(_BLOB_ENV)
        if blob and not blob.endswith(".inference"):
            os.environ[_BLOB_ENV] = f"{blob}.inference"
    load_compile_cache()


class SaveCompileCacheCallback(Callback):
    """Persist the torch.compile Mega-Cache once, post-warmup, from global rank 0."""

    def __init__(self, after_step: int = 150):
        self.after_step = after_step
        self._done = False

    def on_train_batch_end(self, trainer: L.Trainer, *args, **kwargs) -> None:
        if self._done or trainer.global_rank != 0 or trainer.global_step < self.after_step:
            return
        save_compile_cache()
        self._done = True
