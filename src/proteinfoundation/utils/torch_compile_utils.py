"""Helpers for torch.compile / CUDA graph integration."""

from __future__ import annotations

import torch


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
