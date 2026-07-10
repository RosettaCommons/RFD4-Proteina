"""Helpers for torch.compile / CUDA graph integration."""

from __future__ import annotations

import logging
import os

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)

# Path to a single torch.compile "Mega-Cache" blob (shared across runs/users). A single
# file avoids the NFS stale-file-handle failures of a shared cache *directory*.
_BLOB_ENV = "TORCH_COMPILE_CACHE_BLOB"


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


def load_compile_cache() -> None:
    """Load the torch.compile Mega-Cache blob (``$TORCH_COMPILE_CACHE_BLOB``) if present."""
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
