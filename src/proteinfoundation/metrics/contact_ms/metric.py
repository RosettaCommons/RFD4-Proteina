"""ContactMolecularSurfaceMetric for the proteinfoundation evaluation pipeline.

Computes distance-weighted interfacial surface area between binder and
target molecules. Based on Lawrence & Colman (1993).

Always uses the PyTorch backend (``TorchCMS``) which runs on both GPU and
CPU and is faster than the legacy NumPy implementation in both cases.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import torch
from biotite.structure import AtomArray
from jaxtyping import Bool

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger

from .api import calculate_maximum_possible_contact_ms, extract_cms_inputs
from .gpu_cms import TorchCMS

logger = RankedLogger(__name__, rank_zero_only=True)


def _get_binder_target_mask(
    atom_array: AtomArray,
) -> tuple[Bool[np.ndarray, N], Bool[np.ndarray, N]]:
    """Identify binder vs target atoms.

    Uses ``chain_iid`` annotation if available (chain 0 = binder),
    otherwise falls back to ``chain_id`` (first chain = binder).
    """
    if hasattr(atom_array, "chain_iid"):
        binder_mask = atom_array.chain_iid == 0
        target_mask = atom_array.chain_iid != 0
        return binder_mask, target_mask

    unique_chains = np.unique(atom_array.chain_id)
    if len(unique_chains) >= 2:
        binder_mask = atom_array.chain_id == unique_chains[0]
        return binder_mask, ~binder_mask

    return np.ones(len(atom_array), dtype=bool), np.zeros(len(atom_array), dtype=bool)


class ContactMolecularSurfaceMetric(Metric):
    """Contact Molecular Surface (CMS) metric for protein interfaces.

    CMS is a distance-weighted interfacial surface area (units: Å²).
    Unlike SASA, it measures how tightly two surfaces pack:
    ``CMS = Σ area x exp(-0.5 x distance^2)``.

    Always uses the PyTorch backend (``TorchCMS``), which is faster than
    the legacy NumPy implementation on both GPU and CPU.

    Args:
        compute_binder_side: Also compute binder-side CMS. Defaults to ``True``.
        compute_max_target_cms: Compute maximum possible target CMS for
            normalisation. Defaults to ``False``.
        device: ``"auto"`` (GPU if available), ``"cuda"``, or ``"cpu"``.

    Example:
        >>> metric = ContactMolecularSurfaceMetric()
        >>> results = metric.compute(predicted_atom_array=structure)
        >>> print(f"CMS: {results['contact_ms']:.1f} Å²")
    """

    def __init__(
        self,
        compute_binder_side: bool = True,
        compute_max_target_cms: bool = False,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.compute_binder_side = compute_binder_side
        self.compute_max_target_cms = compute_max_target_cms
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"predicted_atom_array": "predicted_atom_array"}

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Compute CMS for the predicted structure.

        Args:
            predicted_atom_array: Multi-chain structure with binder + target.

        Returns:
            Dictionary with CMS metrics:
                - ``contact_ms``: Target-side CMS (Å²)
                - ``contact_ms_binder``: Binder-side CMS (Å², if enabled)
                - ``max_contact_ms_target``: Max possible target CMS (Å², if enabled)
                - ``contact_ms_fraction``: CMS / max CMS (0-1, if enabled)
        """
        binder_mask, target_mask = _get_binder_target_mask(predicted_atom_array)
        binder_xyz, binder_radii = extract_cms_inputs(predicted_atom_array, binder_mask)
        target_xyz, target_radii = extract_cms_inputs(predicted_atom_array, target_mask)

        if len(binder_xyz) == 0 or len(target_xyz) == 0:
            return self._default_metrics()

        dev = self._device
        calc = TorchCMS(device=dev)
        b_xyz = torch.as_tensor(binder_xyz, dtype=torch.float32, device=dev)
        b_rad = torch.as_tensor(binder_radii, dtype=torch.float32, device=dev)
        t_xyz = torch.as_tensor(target_xyz, dtype=torch.float32, device=dev)
        t_rad = torch.as_tensor(target_radii, dtype=torch.float32, device=dev)

        if self.compute_binder_side:
            (cms_t, _), (binder_cms_t, _) = calc.compute_both_sides(
                b_xyz,
                b_rad,
                t_xyz,
                t_rad,
            )
            cms = float(cms_t)
            results: dict[str, float] = {
                "contact_ms": cms,
                "contact_ms_binder": float(binder_cms_t),
            }
        else:
            cms_t, _ = calc.compute(b_xyz, b_rad, t_xyz, t_rad)
            cms = float(cms_t)
            results = {"contact_ms": cms}

        if self.compute_max_target_cms:
            max_cms, _, _ = calculate_maximum_possible_contact_ms(
                target_xyz,
                target_radii,
                device=dev,
            )
            results["max_contact_ms_target"] = max_cms
            results["contact_ms_fraction"] = cms / max_cms if max_cms > 0 else 0.0

        return results

    def _default_metrics(self) -> dict[str, float]:
        results: dict[str, float] = {"contact_ms": 0.0}
        if self.compute_binder_side:
            results["contact_ms_binder"] = 0.0
        if self.compute_max_target_cms:
            results["max_contact_ms_target"] = 0.0
            results["contact_ms_fraction"] = 0.0
        return results
