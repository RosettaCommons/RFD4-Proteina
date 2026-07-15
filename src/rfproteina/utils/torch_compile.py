"""Helpers for torch.compile / CUDA graph integration."""

from __future__ import annotations

import getpass
import logging
import mmap
import os
import re
import tempfile

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)

# Path to a per-user torch.compile "Mega-Cache" blob. A single file avoids the NFS
# stale-file-handle failures of a shared cache *directory*. Per-user (not shared): the
# blob's FX-graph entries bake in absolute ``/tmp/torchinductor_<user>`` paths, so a
# blob shared across users poisons one user's cache with another's paths.
_BLOB_ENV = "TORCH_COMPILE_CACHE_BLOB"

# Subdir of the Inductor cache holding cached FX graphs (``TORCHINDUCTOR_FX_GRAPH_CACHE``).
_FXGRAPH_SUBDIR = "fxgraph"

# Matches a baked ``/tmp/torchinductor_<user>/`` cache-dir *path* (anchored on the
# surrounding slashes so it never matches internal torch identifiers like
# ``torchinductor_worker_logpath``). Group 1 is the full ``<user>`` component (``[^/]+``
# so unusual usernames are captured whole, never truncated to look foreign).
_INDUCTOR_DIR_RE = re.compile(rb"/torchinductor_([^/]+)/")


def _sanitized_user() -> str:
    """Return the current username with filesystem-unsafe characters replaced.

    Matches torch's own sanitization of :py:func:`getpass.getuser` when building the
    ``torchinductor_<user>`` cache dir.
    """
    return re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())


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


def _inductor_cache_dir() -> str:
    """Return the Inductor cache root, matching torch's default resolution.

    Mirrors :py:func:`torch._inductor.runtime.cache_dir_utils.cache_dir` without the
    side effects (env mutation / ``makedirs``) of calling that private helper.
    """
    explicit = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if explicit:
        return explicit
    return os.path.join(tempfile.gettempdir(), f"torchinductor_{_sanitized_user()}")


def purge_foreign_inductor_cache() -> None:
    """Delete FX-graph cache entries that bake in another user's cache dir.

    CoreWeave nodes are long-lived, so ``/tmp/torchinductor_<user>`` persists across
    jobs. A cross-user Mega-Cache blob (now per-user) previously seeded the FX-graph
    cache with kernels whose autotune paths point at another user's
    ``/tmp/torchinductor_<other>``. Reusing such a kernel makes coordinate-descent
    autotuning write ``.best_config`` into that foreign dir → ``PermissionError`` →
    a crashed rank → an eventual NCCL timeout. Removing the poisoned entries forces a
    clean recompile against the current user's cache dir. Own entries are kept.
    """
    own = _sanitized_user().encode()
    fxdir = os.path.join(_inductor_cache_dir(), _FXGRAPH_SUBDIR)
    if not os.path.isdir(fxdir):
        return

    removed = 0
    for root, _dirs, files in os.walk(fxdir):
        for name in files:
            fp = os.path.join(root, name)
            try:
                with open(fp, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    foreign = any(m.group(1) != own for m in _INDUCTOR_DIR_RE.finditer(mm))
            except (OSError, ValueError):
                continue  # unreadable or empty file (empty cannot be mmap'd) → leave it
            if foreign:
                try:
                    os.remove(fp)
                    removed += 1
                except OSError:
                    pass  # concurrent rank already removed it, or racing writer

    if removed:
        logger.info(f"Purged {removed} cross-user Inductor FX-graph cache entries under {fxdir}")


def load_compile_cache() -> None:
    """Load the torch.compile Mega-Cache blob (``$TORCH_COMPILE_CACHE_BLOB``) if present."""
    try:
        purge_foreign_inductor_cache()  # drop kernels baked with another user's cache dir
    except Exception as e:  # cache cleanup (e.g. getpass.getuser) must never break startup
        logger.warning(f"Skipping foreign Inductor cache purge: {e}")
    path = os.environ.get(_BLOB_ENV)
    if not path or not os.path.exists(path) or not hasattr(torch.compiler, "load_cache_artifacts"):
        return
    try:
        with open(path, "rb") as f:
            torch.compiler.load_cache_artifacts(f.read())
        logger.info(f"Loaded torch.compile cache from {path}")
    except Exception as e:  # a partial/corrupt blob must never break startup
        logger.warning(f"Ignoring unreadable torch.compile cache {path}: {e}")


def save_compile_cache() -> None:
    """Serialize compiled artifacts to ``$TORCH_COMPILE_CACHE_BLOB`` (single-file atomic write)."""
    path = os.environ.get(_BLOB_ENV)
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
