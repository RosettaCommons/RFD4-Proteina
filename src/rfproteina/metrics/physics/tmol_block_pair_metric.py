"""TMol block-pair interface metric for arbitrary region pairs.

Instantiate via Hydra configs (see ``configs/metrics/inline_metrics/tmol_block_pair_metrics.yaml``).
Regions are AtomWorks query strings (see :py:func:`~rfproteina.utils.mask_spec.resolve_mask`);
when both are omitted, binder is the chain of interest.
"""

from __future__ import annotations

import importlib
from functools import cached_property
from typing import Any, Literal

import torch
from atomworks.io.utils.ccd import check_ccd_codes_are_available
from biotite.structure import AtomArray
from tmol.io import build_context_from_biotite

from rfproteina.metrics.metric import Metric
from rfproteina.metrics.physics.tmol_ddg_core import (
    build_pose_and_masks,
    create_score_function,
    pack_shell,
    prepare_tmol_atom_array,
    region_query_to_block_mask,
    relax_pose_stack,
    relax_shell,
    score_block_pair,
    shell_block_mask,
)
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.mask_spec import resolve_mask

logger = RankedLogger(__name__, rank_zero_only=True)


def _precompiled_tmol_scoring_available() -> bool:
    """Return whether TMol loaded its precompiled LJ/LK scoring operators."""
    cpp_lib = importlib.import_module("tmol._cpp_lib")
    cpp_lib._ensure_loaded()
    if not cpp_lib.has_cuda():
        return False
    compiled = importlib.import_module("tmol.score.ljlk.potentials._compiled")
    return all(callable(getattr(compiled, name, None)) for name in ("ljlk_pose_scores", "ljlk_rotamer_scores"))


class TMolBlockPairMetric(Metric):
    """TMol interface ddG with optional per-term energies and interaction counts.

    Returns ``interface_ddg``; ``term_names`` / ``term_cutoffs`` add per-term energies and counts.
    ``reuse_context=True`` caches the ligand build context across structures that share chemistry;
    set ``False`` when ligands differ sample-to-sample. ``raise_on_dropped_residues`` (default
    ``True``) errors if a scored residue is dropped from the PoseStack.
    """

    def __init__(
        self,
        region_a: str | None = None,
        region_b: str | None = None,
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
        if requested.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; set device='cpu' for explicit CPU execution")
            cuda_major = (torch.version.cuda or "").split(".")[0]
            if cuda_major not in {"12", "13"}:
                raise RuntimeError(f"TMol does not provide CUDA operators for torch CUDA {torch.version.cuda!r}")
            try:
                operators_available = _precompiled_tmol_scoring_available()
            except Exception as error:
                raise RuntimeError(
                    f"TMol CUDA operators are unavailable; install the 'tmol[cu{cuda_major}]' extra"
                ) from error
            if not operators_available:
                raise RuntimeError(f"TMol CUDA operators are unavailable; install the 'tmol[cu{cuda_major}]' extra")
            if requested.index is None:
                # TMol compares devices exactly; torch.device("cuda") != torch.device("cuda:0").
                requested = torch.device("cuda", torch.cuda.current_device())
        self.device = requested
        if relax not in ("none", "fast_relax"):
            raise ValueError(f"relax must be 'none' or 'fast_relax', got {relax!r}")
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
            prepare_tmol_atom_array(atom_array),
            self.device,
            prepare_ligands=True,
            strict_ligands=True,
            ligand_params_files=self.ligand_params_files,
        )
        sfxn = create_score_function(
            self.device,
            use_soft_rep=self.use_soft_rep,
            param_db=context.parameter_database,
        )
        self._context = context
        self._sfxn = sfxn

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        if self.relax == "none":
            return self._compute(predicted_atom_array)
        # Relaxation needs ordinary tensors and autograd even inside validation's inference mode.
        with torch.inference_mode(False), torch.enable_grad():
            return self._compute(predicted_atom_array)

    def _compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
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


def _site_label(atom_array: AtomArray, query: str) -> str:
    """Chain+res_id label (e.g. ``"A50"``) for the single residue ``query`` selects."""
    idx = int(resolve_mask(atom_array, query).nonzero()[0][0])
    return f"{atom_array.chain_id[idx]}{atom_array.res_id[idx]}"


class TMolMutationDdgMetric(TMolBlockPairMetric):
    """Change in residue-vs-binder interface energy from point mutation(s) on the target.

    For every ``(site, mutate_to)`` in the cross product of ``mutation_selection`` x ``mutate_to``,
    relaxes a shell around the site twice -- once as-is, once mutated (backbone fixed, shell
    repacked + minimized) -- and reports the block-pair energy between ``region_a`` (default: the
    mutated site) and ``region_b`` (default: the binder chain of interest). The wild-type leg
    depends only on the site, so it is relaxed once per site and reused across that site's
    mutations. Identical relax on both legs, so each delta isolates its mutation; it is a relaxed
    interaction-energy change, not a rigorous bound-minus-unbound binding ddG.

    Output keys are suffixed per pair: ``interface_ddg_wt__<site>``,
    ``interface_ddg_mut__<site>__<mutate_to>``, ``interface_ddg_delta__<site>__<mutate_to>``,
    where ``<site>`` is the mutated residue's chain+res_id (e.g. ``A50``).

    Args:
        mutation_selection: AtomWorks query, or list of queries, each selecting one residue to
            mutate.
        mutate_to: Three-letter residue name, or list of names, to mutate each site to.
        shell_radius: Residues with any atom within this many Angstroms of the mutated residue are
            repacked and minimized; everything else is frozen. Defaults to 8.0.
    """

    def __init__(
        self,
        mutation_selection: str | list[str],
        mutate_to: str | list[str],
        *,
        shell_radius: float = 8.0,
        region_a: str | None = None,
        region_b: str | None = None,
        num_repeats: int = 2,
        **kwargs: Any,
    ):
        super().__init__(region_a=region_a, region_b=region_b, relax="fast_relax", num_repeats=num_repeats, **kwargs)
        self.mutation_selections = (
            [mutation_selection] if isinstance(mutation_selection, str) else list(mutation_selection)
        )
        self.mutate_tos = [mutate_to] if isinstance(mutate_to, str) else list(mutate_to)
        # Fixed at construction, so validate once here rather than per scored structure.
        check_ccd_codes_are_available(self.mutate_tos, mode="raise")
        self.shell_radius = shell_radius

    def _compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        self._ensure_context_and_sfxn(predicted_atom_array)
        # Build the pose once; the binder region is shared across all sites, so resolve it once.
        # Pass region_b for both regions when it is set (mask_a is discarded) so the chain-of-interest
        # inference -- which needs >= 2 chains -- runs only when region_b is omitted.
        pose, mask_a, mask_b = build_pose_and_masks(
            atom_array=predicted_atom_array,
            device=self.device,
            no_opt_h=self.no_opt_h,
            region_a=None if self.region_b is None else self.region_b,
            region_b=self.region_b,
            raise_on_dropped_residues=self.raise_on_dropped_residues,
            context=self._context,
        )
        mask_binder = mask_a if self.region_b is None else mask_b
        resolve = lambda query: region_query_to_block_mask(  # noqa: E731
            pose, predicted_atom_array, query, self.device, raise_on_dropped_residues=self.raise_on_dropped_residues
        )

        out: dict[str, float] = {}
        for site in self.mutation_selections:
            mask_residue = resolve(self.region_a) if self.region_a is not None else resolve(site)
            if (mask_residue & mask_binder).any():
                # Overlap means we'd score the residue against its own chain, not an interface.
                raise ValueError(
                    "Mutated-residue region overlaps the binder region; interface ddG would be intra-chain."
                )
            mutation_mask = resolve(site)
            n_mutated = int(mutation_mask.sum().item())
            if n_mutated != 1:
                raise ValueError(f"Each mutation_selection must select one residue; {site!r} selected {n_mutated}")
            shell = shell_block_mask(pose, predicted_atom_array, site, self.shell_radius, self.device)
            label = _site_label(predicted_atom_array, site)

            # Wild-type leg depends only on the site -> relax once and reuse across this site's mutations.
            wt = relax_shell(
                pack_shell(pose.clone(), self._sfxn, shell), self._sfxn, shell, num_repeats=self.num_repeats
            )
            ddg_wt = score_block_pair(wt, mask_residue, mask_binder, self._sfxn)["interface_ddg"]
            out[f"interface_ddg_wt__{label}"] = ddg_wt

            for mutate_to in self.mutate_tos:
                packed = pack_shell(pose.clone(), self._sfxn, shell, mutate_to=mutate_to, mutation_mask=mutation_mask)
                mut = relax_shell(packed, self._sfxn, shell, num_repeats=self.num_repeats)
                ddg_mut = score_block_pair(mut, mask_residue, mask_binder, self._sfxn)["interface_ddg"]
                out[f"interface_ddg_mut__{label}__{mutate_to}"] = ddg_mut
                out[f"interface_ddg_delta__{label}__{mutate_to}"] = ddg_mut - ddg_wt
        return out
