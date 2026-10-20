# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.experimental.symbolic_shapes import has_static_value

from rfproteina.nn.modules.openfold_modules.primitives import LayerNorm, Linear
from rfproteina.nn.modules.openfold_modules.tensor_utils import permute_final_dims

try:
    import cuequivariance_ops_torch
    from cuequivariance_ops_torch import triangle_multiplicative_update

    # A failed Triton import leaves stub functions and records the traceback here.
    _CUEQ_UNAVAILABLE = getattr(cuequivariance_ops_torch, "IMPORT_EXCEPTION", None)
except (ImportError, OSError) as error:
    _CUEQ_UNAVAILABLE = f"cuequivariance-ops-torch>=0.13 is required ({error})"
if os.environ.get("DISABLE_CUEQUIVARIANCE", "").strip().lower() in ("1", "true", "yes"):
    _CUEQ_UNAVAILABLE = "DISABLE_CUEQUIVARIANCE is set"


@cache
def _warn_once(message: str) -> None:
    logging.getLogger(__name__).warning(message)


def _cat_optional(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    """Row-wise concatenation of two optional bias vectors (both present or both absent)."""
    return None if a is None or b is None else torch.cat([a, b])


def _bias_or_zeros(module: nn.Module) -> torch.Tensor:
    """The module's bias, or zeros matching its weight when it learns none."""
    return module.weight.new_zeros(module.weight.shape[0]) if module.bias is None else module.bias


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """

    _direction: str  # "outgoing" or "incoming"; set by each subclass with its contraction.

    def __init__(self, c_z: int, c_hidden: int, bias: bool = True, backend: str = "torch"):
        """
        Args:
            c_z:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            bias:
                Whether projections and layer normalizations learn additive biases
            backend:
                ``torch`` or ``cuequivariance``
        """
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.set_backend(backend)

        self.linear_a_p = Linear(self.c_z, self.c_hidden, bias=bias)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, bias=bias, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden, bias=bias)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, bias=bias, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, bias=bias, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, bias=bias, init="final")

        self.layer_norm_in = LayerNorm(self.c_z, bias=bias)
        self.layer_norm_out = LayerNorm(self.c_hidden, bias=bias)

        self.sigmoid = nn.Sigmoid()

    def set_backend(self, backend: str) -> None:
        """Select the runtime backend without changing checkpoint parameters."""
        if backend not in ("torch", "cuequivariance"):
            raise ValueError(f"nn.tri_mult_backend must be 'torch' or 'cuequivariance', got {backend!r}")
        if backend == "cuequivariance" and _CUEQ_UNAVAILABLE is not None:
            _warn_once(f"cuEquivariance is unavailable; using torch. Reason: {_CUEQ_UNAVAILABLE}")
            backend = "torch"
        self.backend = backend

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("This method needs to be overridden")

    def _cueq_parameters(self) -> dict[str, torch.Tensor | None]:
        """Keyword parameters of ``cuequivariance_ops_torch.triangle_multiplicative_update``.

        The a and b projections are packed row-wise into the op's ``[2 * c_hidden, c_z]`` input weights.
        Absent projection biases stay ``None`` so the op selects its bias-free kernels; absent LayerNorm
        biases are passed as zeros because the op's own default assumes ``c_hidden == c_z``.
        """
        return {
            "norm_in_weight": self.layer_norm_in.weight,
            "norm_in_bias": _bias_or_zeros(self.layer_norm_in),
            "p_in_weight": torch.cat([self.linear_a_p.weight, self.linear_b_p.weight]),
            "p_in_bias": _cat_optional(self.linear_a_p.bias, self.linear_b_p.bias),
            "g_in_weight": torch.cat([self.linear_a_g.weight, self.linear_b_g.weight]),
            "g_in_bias": _cat_optional(self.linear_a_g.bias, self.linear_b_g.bias),
            "norm_out_weight": self.layer_norm_out.weight,
            "norm_out_bias": _bias_or_zeros(self.layer_norm_out),
            "p_out_weight": self.linear_z.weight,
            "p_out_bias": self.linear_z.bias,
            "g_out_weight": self.linear_g.weight,
            "g_out_bias": self.linear_g.bias,
        }

    def _plain_linears(self) -> bool:
        """Whether every projection is an unwrapped Linear; wrapped ones (e.g. LoRA) need their own forward."""
        linears = (self.linear_a_p, self.linear_a_g, self.linear_b_p, self.linear_b_g, self.linear_z, self.linear_g)
        return all(type(linear) is Linear for linear in linears)

    def _gated_projections(self, z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked, gated a and b; plain projections use two grouped GEMMs with autograd and one without."""
        if not self._plain_linears():
            a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z)) * mask
            return a, self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z)) * mask

        def linear(modules: tuple[Linear, ...]) -> torch.Tensor:
            bias = None if modules[0].bias is None else torch.cat([m.bias for m in modules])
            return F.linear(z, torch.cat([m.weight for m in modules]), bias)

        payload, gate = (self.linear_a_p, self.linear_b_p), (self.linear_a_g, self.linear_b_g)
        if torch.is_grad_enabled():
            payload, gate = linear(payload), linear(gate)
        else:
            payload, gate = linear(payload + gate).chunk(2, dim=-1)
        return (payload * self.sigmoid(gate) * mask).chunk(2, dim=-1)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        if (
            self.backend == "cuequivariance"
            and z.is_cuda
            and z.dim() == 4
            and mask.shape == z.shape[:-1]
            and has_static_value(z.shape[0])
            and has_static_value(z.shape[1])
            and self._plain_linears()
        ):
            parameters = self._cueq_parameters()
            tensors = [t for t in (z, *parameters.values()) if t is not None]
            training = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
            autocast_bf16 = torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
            bf16 = z.dtype == torch.bfloat16 or autocast_bf16
            # Eager no-grad calls can need both cuEq stages, which the public predicate does not report.
            if (training or torch.compiler.is_compiling()) and _cueq_supported(
                int(z.shape[0]), int(z.shape[1]), self.c_z, self.c_hidden, z.device.index, training, bf16
            ):
                # No precision override: precision="bfloat16" rounds the pair and every weight to BF16 before
                # the LayerNorms, which doubled the parameter-gradient error against FP32 (measured). Under
                # BF16 autocast the default keeps the LayerNorms on the FP32 pair (documented cuEq behaviour).
                return triangle_multiplicative_update(
                    z, self._direction, mask, eps=self.layer_norm_in.eps, **parameters
                )

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a, b = self._gated_projections(z, mask)
        x = self._combine_projections(a, b)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        z = x * g

        return z


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """Outgoing triangle multiplicative update."""

    _direction = "outgoing"

    def _combine_projections(
        self,
        a: torch.Tensor,  # [*, N_i, N_k, C]
        b: torch.Tensor,  # [*, N_j, N_k, C]
    ) -> torch.Tensor:
        # [*, C, N_i, N_j]
        p = torch.matmul(
            permute_final_dims(a, (2, 0, 1)),
            permute_final_dims(b, (2, 1, 0)),
        )

        # [*, N_i, N_j, C]
        return permute_final_dims(p, (1, 2, 0))


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """Incoming triangle multiplicative update."""

    _direction = "incoming"

    def _combine_projections(
        self,
        a: torch.Tensor,  # [*, N_k, N_i, C]
        b: torch.Tensor,  # [*, N_k, N_j, C]
    ) -> torch.Tensor:
        # [*, C, N_i, N_j]
        p = torch.matmul(
            permute_final_dims(a, (2, 1, 0)),
            permute_final_dims(b, (2, 0, 1)),
        )

        # [*, N_i, N_j, C]
        return permute_final_dims(p, (1, 2, 0))
