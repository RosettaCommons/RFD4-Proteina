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

    _CUEQ_UNAVAILABLE = getattr(cuequivariance_ops_torch, "IMPORT_EXCEPTION", None)
    _CUEQ_SPLIT = getattr(cuequivariance_ops_torch, "triangle_multiplicative_update_split_parameters", None)
    _CUEQ_PACKED = getattr(cuequivariance_ops_torch, "triangle_multiplicative_update", None)
    _CUEQ_TENSOR_SUPPORTED = getattr(cuequivariance_ops_torch, "triangle_multiplicative_update_is_supported", None)
    _CUEQ_TRAIN_SUPPORTED = getattr(
        cuequivariance_ops_torch, "triangle_multiplicative_update_training_shape_is_supported", None
    )
    _CUEQ_INFERENCE_SUPPORTED = getattr(
        cuequivariance_ops_torch, "triangle_multiplicative_update_inference_shape_is_supported", None
    )
    if not callable(_CUEQ_PACKED):
        _CUEQ_UNAVAILABLE = "cuEquivariance triangle multiplication API is missing"
except (ImportError, OSError) as error:
    _CUEQ_UNAVAILABLE = f"cuequivariance-ops-torch-cu13>=0.11.1 is required ({error})"
    _CUEQ_SPLIT = _CUEQ_PACKED = _CUEQ_TRAIN_SUPPORTED = _CUEQ_INFERENCE_SUPPORTED = _CUEQ_TENSOR_SUPPORTED = None
if os.environ.get("DISABLE_CUEQUIVARIANCE", "").strip().lower() in ("1", "true", "yes"):
    _CUEQ_UNAVAILABLE = "DISABLE_CUEQUIVARIANCE is set"

_CUEQ_SUPPORTED: dict[tuple, bool] = {}


@cache
def _warn_once(message: str) -> None:
    logging.getLogger(__name__).warning(message, extra={"always_visible": True})


@torch.compiler.assume_constant_result
def _cueq_supported(batch: int, n: int, c_z: int, c_hidden: int, device: int, training: bool, bf16: bool) -> bool:
    """Whether cuEquivariance fuses this static BF16 shape; warns once per shape otherwise."""
    key = (batch, n, c_z, c_hidden, device, training, bf16)
    if key not in _CUEQ_SUPPORTED:
        mode = "training" if training else "inference"
        query = _CUEQ_TRAIN_SUPPORTED if training else _CUEQ_INFERENCE_SUPPORTED
        if query is not None:
            supported = bf16 and query(batch, n, c_z, c_hidden, device)
        else:
            # The public 0.11 API has no shape predicate. Restrict it to the
            # SM90 shape family used by BioIR's RFD4 integration.
            supported = (
                bf16
                and n > 256
                and n % 16 == 0
                and c_z == 384
                and c_hidden == 256
                and torch.cuda.get_device_capability(device) == (9, 0)
            )
        _CUEQ_SUPPORTED[key] = supported
        if not _CUEQ_SUPPORTED[key]:
            _warn_once(
                f"cuEquivariance has no fused {mode} triangle multiplication for batch={batch}, n={n}, "
                f"c_z={c_z}, c_hidden={c_hidden}, bf16={bf16}; using the torch backend."
            )
    return _CUEQ_SUPPORTED[key]


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """

    _direction: str  # "outgoing" or "incoming"; set by each subclass with its contraction.

    def __init__(self, c_z: int, c_hidden: int, bias: bool = True, backend: str = "cuequivariance"):
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
        """Select shared training/inference kernels without changing trained parameters."""
        if backend not in ("torch", "cuequivariance"):
            raise ValueError(f"Triangle backend must be torch or cuequivariance, got {backend!r}")
        self.configured_backend = backend
        self.backend = backend
        if backend == "cuequivariance" and _CUEQ_UNAVAILABLE:
            _warn_once(
                f"cuEquivariance is unavailable ({str(_CUEQ_UNAVAILABLE).strip().splitlines()[-1]}); using torch."
            )
            self.backend = "torch"

    def _cueq_forward(self, z: torch.Tensor, mask: torch.Tensor, parameters: list[torch.Tensor | None]) -> torch.Tensor:
        """Request BF16 compute while retaining FP32 parameter leaves and their gradients."""
        niw, nib, apw, apb, agw, agb, bpw, bpb, bgw, bgb, now, nob, pow, pob, gow, gob = parameters
        return _CUEQ_PACKED(
            z,
            direction=self._direction,
            mask=mask,
            norm_in_weight=niw,
            norm_in_bias=nib,
            p_in_weight=torch.cat((apw, bpw)),
            p_in_bias=None if apb is None else torch.cat((apb, bpb)),
            g_in_weight=torch.cat((agw, bgw)),
            g_in_bias=None if agb is None else torch.cat((agb, bgb)),
            norm_out_weight=now,
            norm_out_bias=nob,
            p_out_weight=pow,
            p_out_bias=pob,
            g_out_weight=gow,
            g_out_bias=gob,
            eps=self.layer_norm_in.eps,
            precision="bfloat16",
        )

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("This method needs to be overridden")

    def _cueq_parameters(self) -> list[torch.Tensor | None]:
        """Preserve absent projection biases; supply correctly sized normalization zeros."""
        modules = (self.layer_norm_in, self.linear_a_p, self.linear_a_g, self.linear_b_p, self.linear_b_g)
        modules += (self.layer_norm_out, self.linear_z, self.linear_g)
        parameters = []
        for module in modules:
            bias = module.bias
            if bias is None and module in (self.layer_norm_in, self.layer_norm_out):
                bias = module.weight.new_zeros(module.weight.shape)
            parameters.extend((module.weight, bias))
        return parameters

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
            training = torch.is_grad_enabled() and any(t.requires_grad for t in (z, *parameters) if t is not None)
            autocast_bf16 = torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
            bf16 = z.dtype == torch.bfloat16 or autocast_bf16
            supported = _cueq_supported(
                int(z.shape[0]), int(z.shape[1]), self.c_z, self.c_hidden, z.device.index, training, bf16
            )
            if supported:
                return self._cueq_forward(z, mask, parameters)

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
