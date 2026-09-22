"""Muon updates grouped by matrix shape, with native optimizer checkpoint layout."""

import math
from collections.abc import Callable

import torch
from torch import Tensor


def supports_grouped_muon(parameter: Tensor) -> bool:
    """Whether ordinary FP32 matrix storage supports grouped CPU/CUDA updates."""
    return (
        type(parameter) in (Tensor, torch.nn.Parameter)
        and parameter.dtype == torch.float32
        and parameter.device.type in ("cpu", "cuda")
        and parameter.layout == torch.strided
        and parameter.ndim == 2
        and min(parameter.shape) > 0
    )


def _orthogonalize(updates: Tensor, coefficients: tuple[float, float, float], steps: int, eps: float) -> Tensor:
    """Apply Muon's BF16 Newton-Schulz arithmetic independently to [B, M, N]."""
    if steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if len(coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = coefficients
    transposed = updates.shape[-2] > updates.shape[-1]
    updates = updates.bfloat16()
    if transposed:
        updates = updates.transpose(-1, -2)
    updates.div_(updates.norm(dim=(-2, -1), keepdim=True).clamp(min=eps))
    for _ in range(steps):
        gram = torch.bmm(updates, updates.transpose(-1, -2))
        polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        updates = torch.baddbmm(updates, polynomial, updates, beta=a)
    return updates.transpose(-1, -2) if transposed else updates


class GroupedMuon(torch.optim.Muon):
    """Batch equal-shaped FP32 Muon matrices within each parameter group.

    Inherits Muon's constructor, closure contract, hooks, and checkpoint format.
    Parameters and momentum buffers retain their identities and storage layouts.
    BF16 model parameters and distributed tensor subclasses require native Muon;
    BF16 mixed precision with ordinary FP32 master parameters is supported.
    """

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Evaluate the optional closure once, then update parameters without gradients."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            buckets = {}
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if parameter.is_complex():
                    raise RuntimeError("Muon does not support complex parameters")
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if not supports_grouped_muon(parameter):
                    raise RuntimeError("GroupedMuon requires ordinary FP32 CPU/CUDA matrices; use native Muon instead")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient, memory_format=torch.preserve_format)
                key = (parameter.shape, parameter.device)
                buckets.setdefault(key, []).append((parameter, gradient, state["momentum_buffer"]))
            lr, momentum = float(group["lr"]), group["momentum"]
            for (shape, _), entries in buckets.items():
                parameters, gradients, buffers = map(list, zip(*entries, strict=True))
                torch._foreach_lerp_(buffers, gradients, 1 - momentum)
                updates = torch._foreach_lerp(gradients, buffers, momentum) if group["nesterov"] else buffers
                updates = _orthogonalize(
                    torch.stack(updates), group["ns_coefficients"], group["ns_steps"], group["eps"]
                )
                if group["adjust_lr_fn"] in (None, "original"):
                    ratio = math.sqrt(max(1, shape[0] / shape[1]))
                elif group["adjust_lr_fn"] == "match_rms_adamw":
                    ratio = 0.2 * math.sqrt(max(shape))
                else:
                    ratio = 1.0
                torch._foreach_mul_(parameters, 1 - lr * group["weight_decay"])
                torch._foreach_add_(parameters, list(updates.unbind()), alpha=-lr * ratio)
        return loss
