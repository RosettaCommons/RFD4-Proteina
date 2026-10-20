"""Select the inference backend. The native network is never touched."""

from collections.abc import Mapping
from typing import Any

BACKENDS = ("native", "bioir")


def inference_backend(cfg: Mapping) -> str:
    """Return ``native`` (the default compile/eager path) or ``bioir``."""
    backend = cfg.get("inference_backend", "native")
    if backend not in BACKENDS:
        raise ValueError(f"inference_backend must be one of {BACKENDS}, got {backend!r}")
    return backend


def configure_bioir(model: Any, cfg: Mapping) -> None:
    """Route generation through the full BioIR workflow; the native blocks stay as loaded."""
    options = dict(cfg.get("bioir") or {})
    for key in ("max_graphs", "cuda_graph_max_tokens"):
        value = options.get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"bioir.{key} must be null or a positive integer")
    if (options.get("max_graphs") or 2) < 2:
        raise ValueError("bioir.max_graphs needs at least 2 slots, for a full batch and its singleton tail")
    if any(value is not None for value in (cfg.get("nn") or {}).values()):
        raise ValueError("nn.* overrides tune the native backend; unset them or use inference_backend=native")
    if getattr(model, "guidance_model", None) is not None:
        raise ValueError("Differentiable guidance is not qualified against BioIR; use inference_backend=native")
    if cfg.get("adapter_path"):
        raise ValueError("LoRA adapters are not converted by BioIR; use inference_backend=native")
    args = model.inf_cfg.args
    if args.get("guidance_w", 1.0) != 1.0:
        raise ValueError("Classifier-free guidance is not qualified against BioIR; use inference_backend=native")
    if args.get("save_trajectory_every", 0) or args.get("dump_noisy_trajectories", False):
        raise ValueError("Trajectory export is not bridged to BioIR; use inference_backend=native")
    from rfproteina.nn.bioir_full import release_full_bioir, require_bioir

    require_bioir(model)
    release_full_bioir(model)  # A reused model may still hold an earlier conversion.
    model._full_bioir_options = options


def configure_native(model: Any) -> None:
    """Undo any earlier BioIR selection on a reused model, so it dispatches and holds weights natively."""
    if getattr(model, "_full_bioir_options", None) is not None:
        from rfproteina.nn.bioir_full import release_full_bioir

        release_full_bioir(model)
