"""Public API for Contact Molecular Surface computation.

Provides high-level functions that accept raw numpy arrays or biotite
AtomArrays and return CMS values.  Supports both protein-protein and
protein-small molecule interfaces.

When ``device`` is provided, the PyTorch backend (``TorchCMS``) is used
instead of the NumPy reference implementation for faster computation.

Example::

    from proteinfoundation.metrics.contact_ms import (
        calculate_contact_ms,
        extract_cms_inputs,
        get_radii_from_names,
    )

    # NumPy backend (default)
    cms, per_atom_cms, calc = calculate_contact_ms(
        binder_xyz, binder_radii, target_xyz, target_radii
    )

    # PyTorch backend on GPU
    cms, per_atom_cms, _ = calculate_contact_ms(
        binder_xyz, binder_radii, target_xyz, target_radii, device="cuda"
    )

    # From biotite AtomArray
    xyz, radii = extract_cms_inputs(atom_array)
"""

from __future__ import annotations

import numpy as np
import torch
from biotite.structure import AtomArray
from jaxtyping import Float

from .gpu_cms import TorchCMS
from .radii import get_radii_from_names
from .surface_calculator import MolecularSurfaceCalculator


def _to_tensor(
    x: np.ndarray, device: str, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a numpy array to a torch tensor on the given device."""
    return torch.as_tensor(x, dtype=dtype, device=device)


def calculate_contact_ms(
    binder_xyz: Float[np.ndarray, "Nb 3"],
    binder_radii: Float[np.ndarray, "Nb"],
    target_xyz: Float[np.ndarray, "Nt 3"],
    target_radii: Float[np.ndarray, "Nt"],
    *,
    device: str | None = None,
) -> tuple[float, Float[np.ndarray, "Nt"], MolecularSurfaceCalculator | None]:
    """Compute contact molecular surface of binder against target.

    CMS has units of Å² and is a distance-weighted surface area.

    Args:
        binder_xyz: Binder heavy-atom coordinates, shape ``(Nb, 3)``.
        binder_radii: Binder radii from :func:`get_radii_from_names`, shape ``(Nb,)``.
        target_xyz: Target heavy-atom coordinates, shape ``(Nt, 3)``.
        target_radii: Target radii, shape ``(Nt,)``.
        device: If provided (e.g. ``"cuda"`` or ``"cpu"``), use the PyTorch
                ``TorchCMS`` backend.  The returned calculator is ``None``
                in this case.

    Returns:
        ``(cms, per_atom_target_cms, calc)`` where *calc* is the
        ``MolecularSurfaceCalculator`` instance (``None`` when using
        the PyTorch backend).  The calculator can be reused to score the
        binder-side CMS via ``calc.compute_contact_ms(target_side=False)``.
    """
    if device is not None:
        calc_t = TorchCMS(device=device)
        cms_t, per_atom_t = calc_t.compute(
            _to_tensor(binder_xyz, device), _to_tensor(binder_radii, device),
            _to_tensor(target_xyz, device), _to_tensor(target_radii, device),
        )
        return float(cms_t), per_atom_t.cpu().numpy(), None

    calc = MolecularSurfaceCalculator()
    calc.add_binder_and_target(binder_xyz, binder_radii, target_xyz, target_radii)
    cms, per_atom_target_cms = calc.compute()
    return cms, per_atom_target_cms, calc


def calculate_maximum_possible_contact_ms(
    xyz: Float[np.ndarray, "N 3"],
    radii: Float[np.ndarray, "N"],
    *,
    device: str | None = None,
) -> tuple[float, Float[np.ndarray, "N"], MolecularSurfaceCalculator | None]:
    """Compute maximum possible CMS (total surface area) for a molecule.

    Useful for normalising CMS values in small-molecule design.

    Args:
        xyz: Atom coordinates, shape ``(N, 3)``.
        radii: Atom radii, shape ``(N,)``.
        device: If provided, use the PyTorch ``TorchCMS`` backend.

    Returns:
        ``(max_cms, per_atom_cms, calc)`` where *calc* is ``None`` when
        using the PyTorch backend.
    """
    if device is not None:
        calc_t = TorchCMS(device=device)
        total, per_atom_t = calc_t.compute_max_cms(
            _to_tensor(xyz, device), _to_tensor(radii, device),
        )
        return float(total), per_atom_t.cpu().numpy(), None

    calc = MolecularSurfaceCalculator()
    calc.add_molecule(0, xyz, radii)
    cms, per_atom_cms = calc.compute_max_cms()
    return cms, per_atom_cms, calc


def extract_cms_inputs(
    atom_array: AtomArray,
    mask: np.ndarray | None = None,
) -> tuple[Float[np.ndarray, "N 3"], Float[np.ndarray, "N"]]:
    """Extract heavy-atom coordinates and CMS radii from a biotite AtomArray.

    Filters to heavy atoms, looks up radii from the SC radii table, and
    removes atoms with no matching radius.

    Args:
        atom_array: Input structure.
        mask: Optional boolean mask to select a subset of atoms first.

    Returns:
        ``(xyz, radii)`` where both have only atoms with valid (> 0) radii.
    """
    if mask is not None:
        atom_array = atom_array[mask]

    heavy_mask = ~np.isin(atom_array.element, ["H", "D"])
    atom_array = atom_array[heavy_mask]

    xyz = atom_array.coord
    res_names = list(atom_array.res_name)
    atom_names = [n.strip() for n in atom_array.atom_name]

    radii = get_radii_from_names(res_names, atom_names)

    valid = radii > 0
    return xyz[valid], radii[valid]
