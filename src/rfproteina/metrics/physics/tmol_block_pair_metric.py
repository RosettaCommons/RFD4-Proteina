"""TMol block-pair interface metric for arbitrary region pairs.

Instantiate via Hydra configs (see ``configs/metrics/inline_metrics/tmol_block_pair_metrics.yaml``).
Regions are ``atom_selection_from_spec`` specs; when both are omitted, binder is the chain of interest.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Literal

import torch
from biotite.structure import AtomArray
from tmol.io.pose_stack_from_biotite import build_context_from_biotite

from rfproteina.metrics.metric import Metric
from rfproteina.metrics.physics.tmol_ddg_core import (
    build_pose_and_masks,
    create_score_function,
    relax_pose_stack,
    score_block_pair,
)
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class TMolBlockPairMetric(Metric):
    """TMol interface ddG with optional per-term energies and interaction counts.

    Returns ``interface_ddg``; ``term_names`` / ``term_cutoffs`` add per-term energies and counts.
    ``reuse_context=True`` caches the ligand build context across structures that share chemistry;
    set ``False`` when ligands differ sample-to-sample. ``raise_on_dropped_residues`` (default
    ``True``) errors if a scored residue is dropped from the PoseStack.
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
        reuse_context: bool = True,
        ligand_params_files: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.region_a = region_a
        self.region_b = region_b
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU for %s", self.__class__.__name__)
            requested = torch.device("cpu")
        elif requested.type == "cuda" and requested.index is None:
            # TMol compares devices exactly; torch.device("cuda") != torch.device("cuda:0").
            requested = torch.device("cuda", torch.cuda.current_device())
        self.device = requested
        self.relax = relax
        self.cartesian = cartesian
        self.num_repeats = num_repeats
        self.no_opt_h = no_opt_h
        self.use_soft_rep = use_soft_rep
        self.term_names = list(term_names) if term_names is not None else None
        self.term_cutoffs = term_cutoffs
        self.raise_on_dropped_residues = raise_on_dropped_residues
        self.reuse_context = reuse_context
        self.ligand_params_files = list(ligand_params_files) if ligand_params_files is not None else None
        self._sfxn = None
        self._context = None

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"predicted_atom_array": "predicted_atom_array"}

    def setup(self) -> None:
        # Context needs a structure for ligand detection; clear caches so a new epoch starts fresh.
        self._sfxn = None
        self._context = None

    def teardown(self) -> None:
        self._sfxn = None
        self._context = None

    def _ensure_context_and_sfxn(self, atom_array: AtomArray) -> None:
        """Build (and optionally cache) the PoseBuildContext and ligand-aware score function."""
        if self.reuse_context and self._context is not None and self._sfxn is not None:
            return
        # Commit both only on full success so a failed sfxn build cannot leave a half-cached context.
        context = build_context_from_biotite(
            atom_array,
            self.device,
            prepare_ligands=True,
            strict_ligands=True,
            ligand_params_files=self.ligand_params_files,
            sample_proton_chi=False,
        )
        sfxn = create_score_function(
            self.device,
            use_soft_rep=self.use_soft_rep,
            param_db=context.parameter_database,
        )
        self._context = context
        self._sfxn = sfxn

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        self._ensure_context_and_sfxn(predicted_atom_array)
        pose_stack, mask_a, mask_b = build_pose_and_masks(
            atom_array=predicted_atom_array,
            device=self.device,
            no_opt_h=self.no_opt_h,
            region_a=self.region_a,
            region_b=self.region_b,
            raise_on_dropped_residues=self.raise_on_dropped_residues,
            context=self._context,
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
