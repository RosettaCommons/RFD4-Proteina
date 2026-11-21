"""Run the native CLI while auditing actual gradient finiteness and residue scope.

This adapter forwards PyTorch's overloaded gradient API unchanged, inspects its
real returned gradients before sanitization, and audits the final guidance force.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from atomworks.ml.conditions import C_SEQ

from rfproteina.encoding import UNIFIED_ATOM37_ENCODING
from rfproteina.generate import main_hydra
from rfproteina.guidance.guidance_model import GuidanceModel
from rfproteina.model_masks import compute_ae_masks
from rfproteina.surface_engineering import audit_guidance_delta


def main() -> None:
    """Preserve native generation, then require finite nonzero scoped guidance."""
    overrides = dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)
    indices = json.loads(overrides["generation.args.sequence_guidance_token_indices"])
    output = Path(overrides["out_dir"])
    output.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, float]] = []
    raw_finite: list[bool] = []
    original_grad = torch.autograd.grad
    original_call = GuidanceModel.__call__

    # Any only forwards the third-party overloaded autograd signature and its
    # heterogeneous extra_info metadata; inspected tensors are validated below.
    def audited_grad(*args: Any, **kwargs: Any) -> tuple[torch.Tensor | None, ...]:
        values = cast(tuple[torch.Tensor | None, ...], original_grad(*args, **kwargs))
        finite = all(value is None or bool(torch.isfinite(value).all()) for value in values)
        raw_finite.append(finite)
        if not finite:
            raise ValueError("Native surface guidance produced nonfinite gradients before sanitization")
        return values

    def audited_call(
        self: GuidanceModel, x_1_pred: dict[str, torch.Tensor], extra_info: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        delta = original_call(self, x_1_pred, extra_info)
        if delta:
            batch = extra_info["batch"]
            mutable = torch.zeros_like(batch["padding_mask"])
            mutable[:, indices] = True
            _, emitted = compute_ae_masks(batch)
            fixed = batch[C_SEQ.full_name] != UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]
            mutable &= emitted & batch["padding_mask"] & ~fixed
            observations.append(audit_guidance_delta(delta, mutable))
        return delta

    completed = False
    try:
        with patch.object(torch.autograd, "grad", audited_grad), patch.object(GuidanceModel, "__call__", audited_call):
            # Hydra exposes a zero-argument CLI around the typed cfg function.
            cast(Callable[[], None], main_hydra)()
        completed = True
    finally:
        passed = (
            completed and bool(raw_finite) and all(raw_finite) and any(row["gradient_l1"] > 0 for row in observations)
        )
        receipt = {
            "passed": passed,
            "raw_grad_calls": len(raw_finite),
            "raw_all_finite": all(raw_finite),
            "active_calls": len(observations),
            "nonzero_calls": sum(row["gradient_l1"] > 0 for row in observations),
            "max_outside_force": max((row["outside_max"] for row in observations), default=0.0),
            "observations": observations,
        }
        (output / "guidance_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not passed:
        raise RuntimeError("No finite nonzero native guidance was observed; do not admit this arm")


if __name__ == "__main__":
    main()
