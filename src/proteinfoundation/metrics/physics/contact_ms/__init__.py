"""Contact Molecular Surface (CMS) metric.

Distance-weighted interfacial surface area for protein-protein and
protein-small molecule interfaces.  Primary backend is ``TorchCMS``
(GPU and CPU); a NumPy reference implementation is available for
golden-value regression tests.

Quick start::

    from proteinfoundation.metrics.physics.contact_ms import (
        calculate_contact_ms,
        extract_cms_inputs,
    )

    binder_xyz, binder_radii = extract_cms_inputs(atom_array, binder_mask)
    target_xyz, target_radii = extract_cms_inputs(atom_array, target_mask)

    # PyTorch on GPU
    cms, per_atom, _ = calculate_contact_ms(
        binder_xyz,
        binder_radii,
        target_xyz,
        target_radii,
        device="cuda",
    )

    # NumPy fallback (omit device)
    cms, per_atom, calc = calculate_contact_ms(
        binder_xyz,
        binder_radii,
        target_xyz,
        target_radii,
    )
"""

from .api import (
    calculate_contact_ms,
    calculate_maximum_possible_contact_ms,
    extract_cms_inputs,
)
from .gpu_cms import TorchCMS
from .metric import ContactMolecularSurfaceMetric
from .radii import get_radii_from_names

__all__ = [
    "ContactMolecularSurfaceMetric",
    "TorchCMS",
    "calculate_contact_ms",
    "calculate_maximum_possible_contact_ms",
    "extract_cms_inputs",
    "get_radii_from_names",
]
