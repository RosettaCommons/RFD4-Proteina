from __future__ import annotations

import logging

import torch


def apply_torch_deterministic(logger: logging.Logger) -> None:
    """Enable fully deterministic CUDA operations for reproducible runs.

    Must be called after model init — model.__init__ may call
    _configure_flash_attention_backends() which re-enables Flash SDP.
    """
    logger.warning(
        "torch_deterministic=true: disabling Flash/mem-efficient SDP and locking cuDNN "
        "algorithm selection. This can significantly reduce throughput."
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
