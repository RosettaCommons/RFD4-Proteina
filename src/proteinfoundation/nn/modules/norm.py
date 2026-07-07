"""Config-selectable LayerNorm/RMSNorm factory for the score-network trunk."""

import logging

import torch

logger = logging.getLogger(__name__)

# Process-global norm selection, set once before the model is built. Assumes
# single-threaded model construction (set_norm_type is called before make_norm).
_NORM_TYPE = "layer_norm"


def set_norm_type(norm_type: str | None) -> None:
    """Set the process-global norm type: ``layer_norm`` (default), ``rmsnorm_torch``, ``rmsnorm_apex``."""
    global _NORM_TYPE
    _NORM_TYPE = norm_type or "layer_norm"
    logger.info(f"Score-network norm type: {_NORM_TYPE!r}")


def make_norm(
    normalized_shape: int,
    *,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
    bias: bool = True,
) -> torch.nn.Module:
    """Build a norm layer of the configured type; signature mirrors :py:class:`torch.nn.LayerNorm`.

    ``bias`` is ignored by the RMSNorm variants (RMSNorm has no bias by construction).
    """
    if _NORM_TYPE == "layer_norm":
        return torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    if _NORM_TYPE == "rmsnorm_torch":
        return torch.nn.RMSNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    if _NORM_TYPE == "rmsnorm_apex":
        # Fail loudly rather than silently falling back to a slower kernel.
        from apex.normalization import FusedRMSNorm

        return FusedRMSNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    raise ValueError(f"Unknown norm_type {_NORM_TYPE!r}; expected 'layer_norm', 'rmsnorm_torch', or 'rmsnorm_apex'.")
