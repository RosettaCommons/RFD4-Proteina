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

"""
Minimal vendored utilities from OpenFold for torsion angle computation.
Only includes what's needed for atom37_to_torsion_angles function.
"""

import torch
import torch.nn.functional as F

from proteinfoundation.utils import constants as rc


def batched_gather(data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0) -> torch.Tensor:
    """
    Gather data along a specified dimension using indices.

    Args:
        data: Input tensor
        inds: Indices tensor
        dim: Dimension to gather along
        no_batch_dims: Number of batch dimensions

    Returns:
        Gathered tensor
    """
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s, device=data.device)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[tuple(ranges)]


class Rotation:
    """
    A 3D rotation matrix representation.
    Simplified version of OpenFold's Rotation class.
    """

    def __init__(self, rot_mats: torch.Tensor) -> None:
        """
        Args:
            rot_mats: [..., 3, 3] rotation matrices
        """
        if rot_mats.shape[-2:] != (3, 3):
            raise ValueError("Rotation matrices must be [..., 3, 3]")
        self._rot_mats = rot_mats

    @staticmethod
    def identity(shape: tuple, device: torch.device, dtype: torch.dtype = torch.float32) -> "Rotation":
        """Create identity rotations."""
        rots = torch.eye(3, dtype=dtype, device=device)
        rots = rots.view(*((1,) * len(shape)), 3, 3)
        rots = rots.expand(*shape, -1, -1)
        return Rotation(rot_mats=rots)

    def get_rot_mats(self) -> torch.Tensor:
        """Return rotation matrices."""
        return self._rot_mats

    def compose(self, r: "Rotation") -> "Rotation":
        """
        Compose this rotation with another.

        Args:
            r: Another Rotation object

        Returns:
            Composed rotation
        """
        new_rots = torch.matmul(self._rot_mats, r.get_rot_mats())
        return Rotation(rot_mats=new_rots)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to points.

        Args:
            pts: [..., 3] points

        Returns:
            Rotated points
        """
        return torch.matmul(self._rot_mats, pts.unsqueeze(-1)).squeeze(-1)

    def invert(self) -> "Rotation":
        """Invert the rotation."""
        return Rotation(rot_mats=self._rot_mats.transpose(-1, -2))


class Rigid:
    """
    A class representing a rigid transformation (rotation + translation).
    Simplified version of OpenFold's Rigid class.
    """

    def __init__(self, rots: "Rotation", trans: torch.Tensor) -> None:
        """
        Args:
            rots: Rotation object or None
            trans: [..., 3] translation vectors or None
        """
        if rots is None:
            raise ValueError("rots cannot be None")
        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(shape: tuple, device: torch.device, dtype: torch.dtype = torch.float32) -> "Rigid":
        """Create identity transformations."""
        return Rigid(
            Rotation.identity(shape, device, dtype),
            torch.zeros(*shape, 3, device=device, dtype=dtype),
        )

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        p_xy_plane: torch.Tensor,
        eps: float = 1e-8,
    ) -> "Rigid":
        """
        Construct Rigid transformations from 3 points that define a coordinate frame.

        Args:
            p_neg_x_axis: [..., 3] point on negative x-axis
            origin: [..., 3] origin point
            p_xy_plane: [..., 3] point in xy-plane
            eps: Small epsilon for numerical stability

        Returns:
            Rigid transformation
        """
        # Create vectors
        v1 = p_neg_x_axis - origin  # x-axis direction (will be negated)
        v2 = p_xy_plane - origin  # y-axis direction (in xy-plane)

        # Normalize
        e1 = F.normalize(v1, dim=-1, eps=eps)

        # Gram-Schmidt to make e2 orthogonal to e1
        dot = torch.sum(e1 * v2, dim=-1, keepdim=True)
        v2 = v2 - dot * e1
        e2 = F.normalize(v2, dim=-1, eps=eps)

        # e3 is cross product
        e3 = torch.cross(e1, e2, dim=-1)

        # Build rotation matrix: columns are the basis vectors
        # Note: OpenFold uses a specific convention
        rots = torch.stack([e1, e2, e3], dim=-1)  # [..., 3, 3]

        return Rigid(Rotation(rot_mats=rots), origin)

    def get_rots(self) -> Rotation:
        """Return rotation component."""
        return self._rots

    def get_trans(self) -> torch.Tensor:
        """Return translation component."""
        return self._trans

    def compose(self, r: "Rigid") -> "Rigid":
        """
        Compose this transformation with another.

        Args:
            r: Another Rigid object

        Returns:
            Composed transformation
        """
        new_rots = self._rots.compose(r.get_rots())
        new_trans = self._rots.apply(r.get_trans()) + self._trans
        return Rigid(new_rots, new_trans)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to points.

        Args:
            pts: [..., 3] points

        Returns:
            Transformed points
        """
        rotated = self._rots.apply(pts)
        return rotated + self._trans

    def invert(self) -> "Rigid":
        """Invert the transformation."""
        inv_rots = self._rots.invert()
        inv_trans = inv_rots.apply(-self._trans)
        return Rigid(inv_rots, inv_trans)

    def to_tensor_4x4(self) -> torch.Tensor:
        """
        Convert to 4x4 homogeneous transformation matrix.

        Returns:
            [..., 4, 4] transformation matrix
        """
        rots = self._rots.get_rot_mats()
        trans = self._trans

        # Create 4x4 matrix
        tensor = torch.zeros(*rots.shape[:-2], 4, 4, device=rots.device, dtype=rots.dtype)

        # Fill in rotation and translation
        tensor[..., :3, :3] = rots
        tensor[..., :3, 3] = trans
        tensor[..., 3, 3] = 1.0

        return tensor


def get_chi_atom_indices() -> list:
    """
    Returns atom indices needed to compute chi angles for all residue types.

    Returns:
        A list of shape [residue_types=21, chis=4, atoms=4]. The residue types are
        in the order specified in rc.restypes + unknown residue type at the end.
        For chi angles which are not defined on the residue, the positions indices
        are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue

    return chi_atom_indices


def atom37_to_torsion_angles(
    protein: dict,
    prefix: str = "",
) -> dict:
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        protein: Dict containing:
            * (prefix)aatype: [*, N_res] residue indices
            * (prefix)all_atom_positions: [*, N_res, 37, 3] atom positions (in atom37 format)
            * (prefix)all_atom_mask: [*, N_res, 37] atom position mask

    Returns:
        The same dictionary updated with the following features:
        * "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2]): Torsion angles
        * "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2]): Alternate torsion angles
        * "(prefix)torsion_angles_mask" ([*, N_res, 7]): Torsion angles mask
    """
    aatype = protein[prefix + "aatype"]
    all_atom_positions = protein[prefix + "all_atom_positions"]
    all_atom_mask = protein[prefix + "all_atom_mask"]

    aatype = torch.clamp(aatype, max=20)

    pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
    prev_all_atom_positions = torch.cat([pad, all_atom_positions[..., :-1, :, :]], dim=-3)

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(prev_all_atom_mask[..., 1:3], dim=-1) * torch.prod(all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
    psi_mask = torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype) * all_atom_mask[..., 4]

    chi_atom_indices = torch.as_tensor(get_chi_atom_indices(), device=aatype.device)

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2]))

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype)
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack([fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1)

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    )

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        rc.chi_pi_periodic,
    )[aatype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*aatype.shape, 3),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )

    alt_torsion_angles_sin_cos = torsion_angles_sin_cos * mirror_torsion_angles[..., None]

    protein[prefix + "torsion_angles_sin_cos"] = torsion_angles_sin_cos
    protein[prefix + "alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos
    protein[prefix + "torsion_angles_mask"] = torsion_angles_mask

    return protein
