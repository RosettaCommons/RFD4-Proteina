# MIT License

# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

"""
Coordinate transformation utilities for protein structures.
Removed openfold dependencies - only keeping the actually-used simple functions.
"""

import torch
from jaxtyping import Bool
from torch import Tensor

nm_to_ang_scale = 10.0
ang_to_nm = lambda trans: trans / nm_to_ang_scale
nm_to_ang = lambda trans: trans * nm_to_ang_scale


def get_atom37_ca_mask(n: int, device: torch.device) -> Bool[Tensor, "n 37"]:
    """
    Returns an atom37 mask that only keeps CA.

    Args:
        n: number of residues

    Returns:
        Boolean mask of shape [n, 37] where all zeros except [:, 1] which
        is ones (indicates CA).
    """
    mask = torch.zeros((n, 37), device=device, dtype=torch.bool)  # [n, 37]
    mask[:, 1] = True
    return mask.bool()


def get_atom37_bb3_mask(n: int, device: torch.device) -> Bool[Tensor, "n 37"]:
    """
    Returns an atom37 mask that only keeps [N CA C].

    Args:
        n: number of residues

    Returns:
        Boolean mask of shape [n, 37] where all zeros except [:, [0, 1, 2]] which
        is ones (indicates [N CA C]).
    """
    mask = torch.zeros((n, 37), device=device, dtype=torch.bool)  # [n, 37]
    mask[:, 0] = True
    mask[:, 1] = True
    mask[:, 2] = True
    return mask.bool()


def get_atom37_bb3o_mask(n: int, device: torch.device) -> Bool[Tensor, "n 37"]:
    """
    Returns an atom37 mask that only keeps [N CA C O].

    Args:
        n: number of residues

    Returns:
        Boolean mask of shape [n, 37] where all zeros except [:, [0, 1, 2, 4]] which
        is ones (indicates [N CA C O]).
    """
    mask = torch.zeros((n, 37), device=device, dtype=torch.bool)  # [n, 37]
    mask[:, 0] = True
    mask[:, 1] = True
    mask[:, 2] = True
    mask[:, 4] = True
    return mask


def trans_nm_to_atom37(ca_coors_nm):
    """
    Converts CA positions (in nm) into atom37 representation (in Å).

    Args:
        ca_coors: CA coordinates in nm, shape [*, N, 3]

    Returns:
        Coordinates in atom37 representation (in Å)
    """
    return trans_ang_to_atom37(nm_to_ang(ca_coors_nm))


def trans_ang_to_atom37(ca_coors):
    """
    Converts CA positions (in Å) into atom37 representation.

    Creates a [*, N, 37, 3] tensor of zeros and places the CA coordinates
    at position 1 (the CA index in atom37 representation).

    Args:
        ca_coors: CA coordinates in Å, shape [*, N, 3]

    Returns:
        Coordinates in atom37 representation, shape [*, N, 37, 3]
    """
    original_shape = ca_coors.shape  # [*, N, 3]
    atom37_shape = list(original_shape[:-1]) + [37, original_shape[-1]]  # [*, N, 37, 3]
    ca_coors_atom37 = torch.zeros(atom37_shape, dtype=ca_coors.dtype, device=ca_coors.device)  # [*, N, 37, 3]
    ca_coors_atom37[..., 1, :] = ca_coors  # Sets correct positions for CA [*, N, 37, 3]
    return ca_coors_atom37
