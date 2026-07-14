"""TMol block-pair interface metric for arbitrary region pairs.

One general metric covers every interface case; express project-specific presets
(ligand ddG, PPI H-bonds, electrostatics, …) as Hydra configs that instantiate it
with the regions (and, optionally, ``term_names`` / ``term_cutoffs``) you care about
— see ``configs/metrics/inline_metrics/tmol_block_pair_metrics.yaml``.

Each region is an ``atom_selection_from_spec`` spec (query string or field filters),
e.g. ``{"query": "res_name == 'lpr'"}`` or ``{"chain_id": "A"}``. When both regions are
omitted, the binder is the chain of interest and the target is every other atom (see
:func:`~proteinfoundation.metrics.physics.tmol_ddg_core.build_pose_and_masks`).
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Literal

import torch
from biotite.structure import AtomArray

from proteinfoundation.metrics.metric import Metric
from proteinfoundation.metrics.physics.tmol_ddg_core import (
    build_pose_and_masks,
    create_score_function,
    relax_pose_stack,
    score_block_pair,
)
from proteinfoundation.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class TMolBlockPairMetric(Metric):
    """TMol interface ddG, plus optional per-term energies and interaction counts.

    Always returns ``interface_ddg``. Additionally:

    - ``term_names`` (e.g. ``["hbond"]``) adds each term's summed cross-region energy,
      keyed by that term's own name (e.g. ``{"hbond": ...}``).
    - ``term_cutoffs`` (e.g. ``{"n_interface_hbonds": ("hbond", -0.3)}``) adds the count
      of cross-region block-pairs at or under the given cutoff for that term, keyed by
      whatever output key you choose.

    ``relax="fast_relax"`` relaxes the *entire* structure (TMol FastRelax: repeated
    rotamer packing + ramped-repulsion minimization) before scoring, since a single
    cheap minimization is known to get stuck in clashy local minima on raw generated
    structures. ``cartesian=True`` relaxes in Cartesian space (recommended — it can
    remove backbone strain that torsion-space relax can't reach); ``cartesian=False``
    relaxes torsions only.

    ``raise_on_dropped_residues`` (default ``True``) raises if any residue in a scored
    region is silently dropped while TMol builds the PoseStack (waters, unsupported
    ligands, residues missing backbone atoms); set ``False`` to warn and score the
    survivors instead — e.g. when an inconsequential PTM is expected to drop.
    """

    def __init__(
        self,
        region_a: dict[str, Any] | list[dict[str, Any]] | None = None,
        region_b: dict[str, Any] | list[dict[str, Any]] | None = None,
        device: str = "cuda",
        relax: Literal["none", "fast_relax"] = "none",
        cartesian: bool = True,
        num_repeats: int = 2,
        no_opt_h: bool = False,
        use_soft_rep: bool = False,
        term_names: list[str] | tuple[str, ...] | None = None,
        term_cutoffs: dict[str, tuple[str, float]] | None = None,
        raise_on_dropped_residues: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.region_a = region_a
        self.region_b = region_b
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU for %s", self.__class__.__name__)
            requested = torch.device("cpu")
        self.device = requested
        self.relax = relax
        self.cartesian = cartesian
        self.num_repeats = num_repeats
        self.no_opt_h = no_opt_h
        self.use_soft_rep = use_soft_rep
        self.term_names = list(term_names) if term_names is not None else None
        self.term_cutoffs = term_cutoffs
        self.raise_on_dropped_residues = raise_on_dropped_residues
        self._sfxn = None

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"predicted_atom_array": "predicted_atom_array"}

    def setup(self) -> None:
        self._sfxn = create_score_function(self.device, use_soft_rep=self.use_soft_rep)

    def teardown(self) -> None:
        self._sfxn = None

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        if self._sfxn is None:
            self.setup()
        pose_stack, mask_a, mask_b = build_pose_and_masks(
            atom_array=predicted_atom_array,
            device=self.device,
            no_opt_h=self.no_opt_h,
            region_a=self.region_a,
            region_b=self.region_b,
            raise_on_dropped_residues=self.raise_on_dropped_residues,
        )
        if self.relax == "fast_relax":
            pose_stack = relax_pose_stack(
                pose_stack, self._sfxn, cartesian=self.cartesian, num_repeats=self.num_repeats
            )

        return score_block_pair(
            pose_stack=pose_stack,
            mask_a=mask_a,
            mask_b=mask_b,
            sfxn=self._sfxn,
            term_names=self.term_names,
            term_cutoffs=self.term_cutoffs,
        )
