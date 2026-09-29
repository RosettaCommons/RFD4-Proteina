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

import math
from collections.abc import Callable

import numpy as np

# import deepspeed
import torch
import torch.nn as nn
from scipy.stats import truncnorm


def _prod(nums: list) -> int:
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape: tuple, fan: str = "fan_in") -> float:
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights: torch.Tensor, scale: float = 1.0, fan: str = "fan_in") -> None:
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights: torch.Tensor) -> None:
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights: torch.Tensor) -> None:
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights: torch.Tensor) -> None:
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights: torch.Tensor) -> None:
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights: torch.Tensor) -> None:
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights: torch.Tensor) -> None:
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    ) -> None:
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super().__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class LayerNorm(nn.Module):
    def __init__(self, c_in: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _d = x.dtype
        # if(_d is torch.bfloat16 and not deepspeed.utils.is_initialized()):
        #     with torch.cuda.amp.autocast(enabled=False):
        #         out = nn.functional.layer_norm(
        #             x,
        #             self.c_in,
        #             self.weight.to(dtype=_d),
        #             self.bias.to(dtype=_d),
        #             self.eps
        #         )
        # else:
        out = nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )

        return out
