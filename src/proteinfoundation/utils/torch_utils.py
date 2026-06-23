from __future__ import annotations

import logging

import torch


def apply_deterministic_training(logger: logging.Logger) -> None:
    """Enable fully deterministic CUDA operations for reproducible training.

    Must be called after model init — model.__init__ may call
    _configure_flash_attention_backends() which re-enables Flash SDP.
    """
    logger.warning(
        "deterministic_training=true: disabling Flash/mem-efficient SDP and locking cuDNN "
        "algorithm selection. This can significantly reduce training throughput."
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
