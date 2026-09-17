"""Utilities for running post-design oracle, refolding, and pass-criteria metrics.

The three MetricManager roles used here are:
- oracle  → runs a structure-prediction oracle (RF3, AF3); logs confidence scores to
             the metrics dict but has no access to the refolded structure itself.
- refolded → operates on (designed, refolded) structure pairs to compute comparison metrics (RMSD, etc)
- pass    → derived metrics that require outputs from prior metric managers; typically
             boolean pass/fail criteria.
"""

import itertools
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Hashable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import torch.distributed as dist
from atomworks.common import exists
from atomworks.io.parser import parse as _parse_structure
from atomworks.io.utils.atom_array_plus import as_atom_array
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.conditions import C_CRD, C_DIS
from atomworks.ml.preprocessing.utils.structure_utils import (
    get_atom_mask_from_cell_list,
)
from biotite.structure import AtomArray, AtomArrayStack, CellList, annotate_sse, gyration_radius
from biotite.structure.io.xtc import XTCFile
from biotite.structure.residues import spread_residue_wise
from omegaconf import OmegaConf

from rfproteina.inference_utils.foldseek import chain_cif_dir
from rfproteina.metrics.metric import MetricManager
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput, OracleMetric
from rfproteina.transforms.unindexing import annotate_unindexed_insertion
from rfproteina.transforms.utils.conditioning_utils import align_atom_array_to_input_frame
from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.common import design_output_dir, ensure_dir
from rfproteina.utils.coordinate_transfer import remove_terminal_oxygen_atoms
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.parse_presets import STANDARD_ANNOTATIONS_ONLY
from rfproteina.utils.structure_files import find_structure_file

logger = RankedLogger(__name__, rank_zero_only=True)

# Name under which a refolding metric can address the refold of a design no
# pre_oracle_transform rewrote, whichever oracle a given benchmark folds it with
# (see :py:func:`run_oracle_and_refolding_metrics`). Transformed refolds are
# addressed by their save_id.
UNTRANSFORMED_REFOLD = "untransformed"

# The two tag families a refolding metric gates on, both keyed by the addresses under which
# :py:func:`run_oracle_and_refolding_metrics` hands out refolds:
#
# - `exists_refold:{address}` -- this design has that refold. Require it wherever a metric *reads*
#   one by name, so a design missing it is skipped rather than raising.
# - `is_refold:{address}` -- the refold this scoring pass is scoring is that one. Require it to
#   scope a metric to the kind of refold it was written for; `is_refold:untransformed` is often
#   desirable to avoid issues where a pre-oracle transform leads to a shape mismatch
EXISTS_REFOLD_TAG_PREFIX = "exists_refold:"
IS_REFOLD_TAG_PREFIX = "is_refold:"


class ValOutput(TypedDict, total=False):
    """One entry on :py:attr:`Proteina.validation_output_generation`."""

    example_id: str
    designed_structure_path: str
    output_base_dir: str
    metrics_tags: str  # comma-joined, e.g. "unindexed,3d_coordinate_conditioning,ame_refolding"
    dataset_name: str | None
    # In-memory copy of the just-generated structure, set by Proteina before
    # writing the CIF. When present, some oracles may skip re-parsing the CIF
    designed_atom_array: AtomArray


def run_metrics_for_result(
    result: dict,
    batch: dict,
    batch_item_idx: int,
    metric_manager: MetricManager,
    output_base_dir: Path | None,
    example_id: str | None = None,
    save_input_structure: bool = True,
    metrics_tags_override: list[str] | None = None,
) -> dict:
    """Run a metric manager against one (input, generated) pair.

    Builds ``extra_info`` (example_id, metrics_tags, chain_of_interest,
    save_input_structure) and dispatches to ``metric_manager``.
    ``metrics_tags_override`` overrides the per-batch tags (used by reward
    scoring). ``chain_of_interest`` is always echoed back in the result.

    Returns ``{}`` if the metrics fail, mirroring :py:func:`run_oracle_and_refolding_metrics`.
    """
    if output_base_dir is not None:
        output_base_dir = Path(output_base_dir)
        ensure_dir(output_base_dir)

    if metrics_tags_override is not None:
        metrics_tags = metrics_tags_override
    else:
        metrics_tags_batch = batch.get("metrics_tags_batch", [])
        metrics_tags = metrics_tags_batch[batch_item_idx] if batch_item_idx < len(metrics_tags_batch) else []

    try:
        coi, coi_unambiguous = get_chain_of_interest(result["generated_atom_array"])

        metrics_result = metric_manager(
            network_input=None,
            network_output=None,
            extra_info={
                "example_id": example_id if example_id is not None else result.get("example_id", "sample"),
                "metrics_tags": metrics_tags,
                "output_base_dir": output_base_dir,
                "chain_of_interest": coi if coi_unambiguous else None,
                "save_input_structure": save_input_structure,
                "full_array_with_unindexed": result.get("full_array_with_unindexed"),
                "predicted_insertion_mapping": result.get("predicted_insertion_mapping") or {},
            },
            ground_truth_atom_array=result["input_atom_array"],
            predicted_atom_array=result["generated_atom_array"],
        )

        # Ensure chain_of_interest is always in the output regardless of which metrics are configured.
        # Only rf3_metric echoes it back; other metrics (af2, design) do not.
        if not any(k.endswith("chain_of_interest") for k in metrics_result):
            metrics_result["chain_of_interest"] = coi if coi_unambiguous else None

        return metrics_result

    except Exception as e:
        logger.error(f"Metrics failed for {example_id}: {e}")
        return {}


def normalize_to_atom_array_list(
    x: "AtomArrayStack | AtomArray | list[AtomArray]",
) -> list[AtomArray]:
    """Normalize input to a list of AtomArrays."""
    if isinstance(x, list):
        return x
    if isinstance(x, AtomArrayStack):
        return list(x)
    return [x]


def unwrap_atom_array(x: "AtomArray | AtomArrayStack") -> AtomArray:
    """Unwrap a single-structure input to a bare AtomArray."""
    if isinstance(x, AtomArrayStack):
        assert (
            len(x) == 1
        ), f"Expected a single-element AtomArrayStack, got depth {len(x)}. Metrics expect one structure at a time."
        return x[0]
    return x


def get_ss_metrics_and_rg(
    atom_array: AtomArray, ss_conditioning: dict[str, np.ndarray] | None = None
) -> dict[str, float]:
    """Compute secondary structure metrics and the radius of gyration for a given input file.

    Args:
        atom_array (AtomArray): Input AtomArray
        ss_conditioning (dict[str, np.ndarray] | None): Dictionary mapping the keys "helix", "sheet", "loop" to the
            corresponding conditioning arrays. If None, secondary structure adherence is not computed.

    NOTE: Biotite computes secondary structures using the P-SEA algorithm:
    G. Labesse, N. Colloc'h, J. Pothier, J. Mornon,
    “P-SEA: a new efficient assignment of secondary structure from Ca trace of proteins,”
    Bioinformatics, vol. 13, pp. 291-295, June 1997. doi: 10.1093/bioinformatics/13.3.291
    """
    # Compute secondary structure
    sse_array = annotate_sse(atom_array)
    sse_array_prot_only = sse_array[sse_array != ""]

    # Basic compositional statistics
    pdb_helix_percent = np.mean(sse_array_prot_only == "a")
    pdb_strand_percent = np.mean(sse_array_prot_only == "b")
    pdb_coil_percent = np.mean(sse_array_prot_only == "c")
    pdb_ss_percent = pdb_helix_percent + pdb_strand_percent

    # Number of disjoint helices or sheets
    num_structural_elements = 0
    for k, _ in itertools.groupby(sse_array):
        if k not in ["", "c"]:
            num_structural_elements += 1

    if ss_conditioning is not None:
        ss_adherence_dict = {}
        atom_level_sse_array = spread_residue_wise(atom_array, input_data=sse_array)
        for ss_annot, ss_type in zip(["a", "b", "c"], ["helix", "sheet", "loop"], strict=False):
            metric_name = f"{ss_type}_conditioning_adherence"
            expected_indices = np.where(ss_conditioning[ss_type])[0]

            if len(expected_indices) > 0:
                ss_adherence = (atom_level_sse_array[expected_indices] == ss_annot).mean()
                ss_adherence_dict[metric_name] = ss_adherence
            else:
                # Would be misleading to give a numerical value if no conditioning of this type was provided
                ss_adherence_dict[metric_name] = np.nan

    # Compute radius of gyration
    radius_of_gyration = gyration_radius(as_atom_array(atom_array))

    # Return output metrics
    output_metrics = {
        "non_loop_fraction": pdb_ss_percent,
        "loop_fraction": pdb_coil_percent,
        "helix_fraction": pdb_helix_percent,
        "sheet_fraction": pdb_strand_percent,
        "num_ss_elements": num_structural_elements,
        "radius_of_gyration": radius_of_gyration,
    }

    if ss_conditioning is not None:
        output_metrics.update(ss_adherence_dict)

    return output_metrics


def get_hotspot_contacts(atom_array: AtomArray, hotspot_mask: np.ndarray, distance_cutoff: float = 4.5) -> float:
    """Get the number of inter-chain contacts between diffused atoms and hotspots within a distance cutoff."""

    cell_list = CellList(atom_array, cell_size=distance_cutoff)
    hotspot_array = atom_array[hotspot_mask]

    # Compute all contacts with hotspots
    full_contacting_atom_mask = get_atom_mask_from_cell_list(
        hotspot_array.coord, cell_list, len(atom_array), distance_cutoff
    )  # (n_hotspots, n_atoms)

    # We only count interchain contacts
    interchain_mask = hotspot_array.pn_unit_iid[:, None] != atom_array.pn_unit_iid[None]
    interchain_contacts_mask = full_contacting_atom_mask & interchain_mask

    # We only count contacts to diffused atoms
    has_coord = C_CRD.mask(atom_array, default="generate")
    has_dist = C_DIS.mask(atom_array, default="generate").as_dense_array(False).any(axis=0)
    is_motif_atom = has_coord | has_dist
    diffused_interchain_contacts_mask = interchain_contacts_mask[:, ~is_motif_atom]

    contacted_hotspots_mask = np.any(diffused_interchain_contacts_mask, axis=1)  # (n_hotspots,)

    return float(contacted_hotspots_mask.mean())


def run_oracle_refolding_and_pass_criteria(
    atom_array: AtomArray,
    example_id: str,
    output_dir: Path | None,
    oracle_mm: MetricManager | None,
    refold_mm: MetricManager | None,
    pass_mm: MetricManager | None,
    metrics_tags: list[str] | None = None,
    prior_metrics: dict | None = None,
) -> dict:
    """Run oracle + refolding + pass criteria for one designed structure.

    Wraps :py:func:`run_oracle_and_refolding_metrics` and optionally applies pass criteria,
    merging prior design metrics so pass criteria can reference them.

    Args:
        atom_array: Designed structure, already parsed and charge-annotated.
        example_id: Identifier used for file naming and resumability.
        output_dir: Base output directory for oracle CIFs and confidence JSONs.
        oracle_mm: Oracle MetricManager (RF3Oracle or AF3Oracle). May be ``None``.
        refold_mm: Refolding MetricManager (structure comparison metrics). May be ``None``.
        pass_mm: Pass-criteria MetricManager. May be ``None``.
        metrics_tags: Tags controlling which metrics activate.
        prior_metrics: Design-phase metrics for this example, merged into the
            pass-criteria ``metrics`` context so pass criteria can reference them.

    Returns:
        Merged dict with oracle + refolding + pass-criteria outputs, or an empty dict
        if an unrecoverable error occurs.
    """
    result = run_oracle_and_refolding_metrics(
        atom_array=atom_array,
        example_id=example_id,
        output_dir=output_dir,
        oracle_mm=oracle_mm,
        refold_mm=refold_mm,
        metrics_tags=metrics_tags,
    )

    if exists(pass_mm):
        pass_output = pass_mm(
            network_input=None,
            network_output=None,
            extra_info={
                "example_id": example_id,
                "metrics_tags": metrics_tags or [],
                "metrics": {**(prior_metrics or {}), **result},
            },
        )
        result.update(pass_output)

    return result


def run_oracle_and_refolding_metrics(
    atom_array: AtomArray,
    example_id: str,
    output_dir: Path | None,
    oracle_mm: MetricManager | None,
    refold_mm: MetricManager | None,
    metrics_tags: list[str] | None = None,
) -> dict:
    """Run oracle → refolding metrics for one designed structure.

    Pass criteria are intentionally excluded — the caller is responsible for
    running pass_mm and merging its output so that pass criteria can reference
    design-phase metrics alongside oracle/refolding results.

    Refolding metrics are handed the design as ``ground_truth_atom_array`` and the
    refold being scored as ``predicted_atom_array``, plus *every* refold of this
    design, addressable two ways: by origin, as ``refold.{oracle_key}`` /
    ``refold.{oracle_key}.{save_id}``, and -- when exactly one refold is of that
    kind -- by kind, as ``refold.{save_id}`` or ``refold.untransformed``. A metric
    whose ``input_kwarg_names`` names two of those compares two refolds against
    each other rather than a design against its refold.

    Every address is injected as a tag twice over — ``exists_refold:{address}`` for
    each refold this design has, and ``is_refold:{address}`` for the one the current
    pass is scoring — so a metric can require what it reads and scope itself to what
    it was written for. See :py:data:`EXISTS_REFOLD_TAG_PREFIX` /
    :py:data:`IS_REFOLD_TAG_PREFIX`.

    Args:
        atom_array: Designed structure, already parsed and annotated.
        example_id: Identifier used for file naming and resumability.
        output_dir: Base output directory for oracle CIFs and confidence JSONs.
        oracle_mm: Oracle MetricManager (RF3Oracle or AF3Oracle). May be ``None``.
        refold_mm: Refolding MetricManager (structure comparison metrics). May be ``None``.
        metrics_tags: Tags controlling which metrics activate.

    Returns:
        Merged dict with oracle + refolding outputs keyed by ``example_id``,
        or an empty dict if an unrecoverable error occurs.
    """

    try:
        atom_array = remove_terminal_oxygen_atoms(atom_array.copy())
        coi, coi_unambiguous = get_chain_of_interest(atom_array)

        extra_info: dict = {
            "example_id": example_id,
            "output_base_dir": output_dir,
            "chain_of_interest": coi if coi_unambiguous else None,
            "metrics_tags": metrics_tags or [],
        }

        oracle_output: dict = {}
        if oracle_mm is not None:
            oracle_output = oracle_mm(
                network_input=None,
                network_output=None,
                extra_info=extra_info,
                predicted_atom_array=atom_array,
            )

        refolding_output: dict = {}
        if refold_mm is not None and oracle_mm is not None and output_dir is not None:
            active_tags = set(metrics_tags or [])
            # Read every refold this design has before scoring any of them
            refolds: dict[str, AtomArray] = {}
            scoring_passes: list[tuple[str, str | None, str]] = []  # (label, save_id, oracle_name)
            by_kind: dict[str, list[str]] = {}  # alias -> labels of the refolds claiming it
            for oracle_key, oracle_metric in oracle_mm.metrics.items():
                if not oracle_metric.applies_to_tags(active_tags):
                    continue
                for save_id in oracle_metric.list_save_ids() or [None]:
                    label = oracle_key if save_id is None else f"{oracle_key}.{save_id}"
                    cif_path = oracle_metric.existing_refolded_cif_path(output_dir, example_id, save_id)
                    if cif_path is None:
                        logger.warning(f"{example_id}: refolded CIF missing for {label}: {cif_path}")
                        continue
                    try:
                        refolded_aa = _parse_structure(cif_path, config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]
                    except Exception as e:
                        logger.warning(f"{example_id}: failed to parse refolded CIF {cif_path}: {e}")
                        continue
                    refolds[label] = remove_terminal_oxygen_atoms(refolded_aa)
                    scoring_passes.append((label, save_id, oracle_metric.oracle_name))
                    by_kind.setdefault(save_id if save_id is not None else UNTRANSFORMED_REFOLD, []).append(label)

            # Also address a refold by what it *is* -- e.g. "untransformed" as a
            # convenience alias, independent of the baseline oracle. An alias that
            # collides with an oracle's own label would shadow that oracle's refold,
            # so the by-origin name wins and the alias is simply not offered.
            alias_of: dict[str, str] = {}  # label -> its by-kind alias, where one is offered
            for alias, labels in by_kind.items():
                if len(labels) == 1 and alias not in refolds:
                    refolds[alias] = refolds[labels[0]]
                    alias_of[labels[0]] = alias
                else:
                    logger.warning(
                        f"{example_id}: refold alias {alias!r} is claimed by {labels} or collides with an oracle's "
                        f"own name, so neither `refold.{alias}` nor `is_refold:{alias}` is offered; address those "
                        "refolds by oracle instead."
                    )

            refold_kwargs = {f"refold.{label}": refold for label, refold in refolds.items()}
            # One exists_refold:{address} tag per refold this design actually has, mirroring the
            # refold.{address} kwargs above. A metric that reads a refold gates itself on the
            # matching tag and is skipped when the refold is absent, rather than raising an error.
            exists_refold_tags = {f"{EXISTS_REFOLD_TAG_PREFIX}{label}" for label in refolds}
            ran_refolding = False
            for label, save_id, oracle_name in scoring_passes:
                # Inject oracle:{name} so refolding metrics can scope themselves to a specific
                # oracle when several are active at once, and is_refold:{address} -- every address
                # of the refold *this* pass scores -- so a metric can scope itself to the kind of
                # refold it is written for, `is_refold:untransformed` being the common case.
                is_refold_tags = {
                    f"{IS_REFOLD_TAG_PREFIX}{address}"
                    for address in (label, alias_of.get(label))
                    if address is not None
                }
                oracle_scoped_tags = active_tags | exists_refold_tags | is_refold_tags | {f"oracle:{oracle_name}"}
                if not any(m.applies_to_tags(oracle_scoped_tags) for m in refold_mm.metrics.values()):
                    continue
                sub_result = refold_mm(
                    network_input=None,
                    network_output=None,
                    extra_info={"example_id": example_id, "metrics_tags": list(oracle_scoped_tags)},
                    ground_truth_atom_array=atom_array,
                    predicted_atom_array=refolds[label],
                    **refold_kwargs,
                )
                for key, value in sub_result.items():
                    metric_name, sep, output_key = key.partition(".")
                    # MetricManager stamps a bare "example_id"; the caller sets that itself.
                    if not sep:
                        continue
                    # The save_id goes *after* the metric name, matching the oracle
                    # side: the aggregation pass resolves the owning metric (and so
                    # its configured reductions) from the first key segment.
                    if save_id is not None:
                        key = f"{metric_name}.{save_id}.{output_key}"
                    refolding_output[key] = value
                ran_refolding = True
            if not ran_refolding:
                logger.warning(
                    f"{example_id}: refolding metrics skipped -- either none applied or no refolded structures were"
                    " found"
                )
        elif refold_mm is not None:
            logger.warning(f"{example_id}: refolding metrics configured but no oracle output dir — skipping")

        return {"example_id": example_id, **oracle_output, **refolding_output}

    except Exception as e:
        logger.error(f"Refolding pipeline/metrics failed for {example_id}: {e}")
        return {}


def _build_applicable_oracle_jobs(
    val_outputs: list[ValOutput],
    oracle_metric: OracleMetric,
) -> list[OracleJobInput]:
    """Filter ``val_outputs`` to those this oracle should refold and parse their CIFs.

    Returns the list of :class:`OracleJobInput` that
    :py:meth:`OracleMetric.fold_batch` accepts.
    """
    jobs: list[OracleJobInput] = []
    for v in val_outputs:
        tags = {t for t in v.get("metrics_tags", "").split(",") if t}
        if not oracle_metric.applies_to_tags(tags):
            continue
        output_base_dir = v.get("output_base_dir")
        if not output_base_dir:
            continue
        atom_array = v.get("designed_atom_array")
        if atom_array is None:
            logger.warning(f"fold_batch: no designed_atom_array for {v.get('example_id', 'sample')}, skipping.")
            continue
        atom_array = unwrap_atom_array(atom_array)
        jobs.append(
            OracleJobInput(
                atom_array=atom_array,
                example_id=v.get("example_id", "sample"),
                out_dir=Path(output_base_dir),
            )
        )
    return jobs


def apply_pre_oracle_filters(
    val_outputs: list[ValOutput],
    pre_oracle_filter_mm: MetricManager,
    delete_failing: bool = True,
) -> list[ValOutput]:
    """Filter designs using pre-oracle PassCriterionMetric criteria.

    Runs each design's merged design-phase metrics through ``pre_oracle_filter_mm``.
    Designs where any applicable filter returns ``pass: 0.0`` are excised from
    the list. When ``delete_failing`` is ``True``, their CIF files are also
    deleted from disk for definitive failures (all metrics present). Designs for
    which no filter is applicable (tag mismatch) are always kept.

    This is called after :py:func:`drain_generated_structure` merges design
    metrics into each :class:`ValOutput` entry, so the full design-metric dict
    is available on each entry.

    Args:
        val_outputs: Per-design entries from ``validation_output_generation``
            with design-phase metrics already merged in.
        pre_oracle_filter_mm: MetricManager whose entries are
            :py:class:`~rfproteina.metrics.oracles.pass_criterion_metric.PassCriterionMetric`
            instances. Tag-based gating controls which filters apply per design.
        delete_failing: If ``True``, delete structure files on disk for designs
            that fail with all metrics present (definitive failures). Defaults to
            ``True``. Set to ``False`` when evaluating pre-existing structures
            where deletion would be destructive.

    Returns:
        Filtered list containing only designs that pass all applicable criteria.
    """

    def _filter_result(filter_output: dict) -> tuple[bool, bool]:
        """Return (should_filter, should_delete) for a design's filter output.

        A design is filtered (excluded from the oracle) if any applicable filter
        returned ``pass: 0.0``.  Files are only deleted when the failure was
        definitive — i.e. all required metrics were present.  Missing-metric
        failures are filtered but files are kept (indeterminate, not a hard fail).
        """
        prefixes = {k.rsplit(".", 1)[0] for k in filter_output if k.endswith(".pass")}
        should_filter = False
        should_delete = False
        for prefix in prefixes:
            if filter_output[f"{prefix}.pass"] == 0.0:
                should_filter = True
                if filter_output.get(f"{prefix}.all_metrics_present") == 1.0:
                    should_delete = True
        return should_filter, should_delete

    passing: list[ValOutput] = []
    n_filtered = 0
    dirs_to_clean: set[tuple[Path, str]] = set()
    for entry in val_outputs:
        example_id = entry.get("example_id", "unknown")
        metrics_tags_str = entry.get("metrics_tags", "")
        metrics_tags = [t for t in metrics_tags_str.split(",") if t] if metrics_tags_str else []

        filter_output = pre_oracle_filter_mm(
            network_input=None,
            network_output=None,
            extra_info={
                "example_id": example_id,
                "metrics_tags": metrics_tags,
                "metrics": {k: v for k, v in entry.items() if not isinstance(v, AtomArray)},
            },
        )

        should_filter, should_delete = _filter_result(filter_output)
        if should_filter:
            if delete_failing and should_delete:
                designed_path = entry.get("designed_structure_path")
                if designed_path:
                    try:
                        Path(designed_path).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"pre-oracle filter: could not delete {designed_path}: {e}")

                output_base_dir = entry.get("output_base_dir")
                if output_base_dir:
                    dirs_to_clean.add((Path(output_base_dir), example_id))
                chain_path = entry.get("designed_chain_path")
                if chain_path:
                    try:
                        Path(chain_path).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"pre-oracle filter: could not delete chain CIF {chain_path}: {e}")
            n_filtered += 1
        else:
            passing.append(entry)

    # Only the design's own directory is cleaned up; the foldseek chain directory is
    # shared across designs and left for the run to finish with.  design_output_dir refuses
    # ids that do not resolve to a direct child of the output root -- an rmtree there would
    # take every other design in the run with it.
    for base_dir, design_id in dirs_to_clean:
        try:
            shutil.rmtree(design_output_dir(base_dir, design_id), ignore_errors=True)
        except ValueError as e:
            logger.error(f"pre-oracle filter: skipping cleanup for example_id {design_id!r} -- {e}")

    if n_filtered:
        logger.info(f"pre-oracle filter: removed {n_filtered}/{len(val_outputs)} designs before oracle.")

    return passing


def fold_designs_for_oracles(
    val_outputs: list[ValOutput], oracle_mm: MetricManager, on_incomplete: str = "raise"
) -> None:
    """Fold this rank's designs with each oracle, then assert (across ranks) that every
    dispatched design produced a refold. ``on_incomplete``: 'raise' (default) | 'warn'.
    """
    n_dispatched = n_refolded = 0
    for _, oracle_metric in oracle_mm.metrics.items():
        jobs = _build_applicable_oracle_jobs(val_outputs, oracle_metric)
        # A pre_oracle_transform may fan each design out into several outputs
        n_dispatched += len(jobs) * len(oracle_metric.list_save_ids() or [None])
        jobs = oracle_metric.apply_transform(jobs)
        if jobs:
            n_refolded += oracle_metric.fold_batch(jobs)

    if dist.is_initialized() and dist.get_world_size() > 1:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
        counts = torch.tensor([n_dispatched, n_refolded], device=device)
        dist.all_reduce(counts)
        n_dispatched, n_refolded = int(counts[0]), int(counts[1])

    if n_refolded >= n_dispatched:
        return
    msg = (
        f"Validation refold coverage {n_refolded}/{n_dispatched}: {n_dispatched - n_refolded} designs "
        f"dispatched to refolding produced no structure (dropped or failed folds)."
    )
    if on_incomplete == "warn":
        logger.warning(msg)
    else:
        raise RuntimeError(msg)


def compute_oracle_metrics_for_val_output(
    val_output: ValOutput,
    oracle_mm: MetricManager | None,
    refold_mm: MetricManager | None,
    pass_mm: MetricManager | None,
    prior_design_metrics_by_id: dict[str, dict],
) -> dict | None:
    """Run oracle + refolding + pass_criteria for one design.

    Reads the designed structure, calls :py:func:`run_oracle_refolding_and_pass_criteria`
    to merge the oracle cache hit, refolding metrics, and pass criteria, and
    annotates the result with the designed-structure path and dataset name.

    Returns ``None`` if the designed structure is missing on disk.
    """
    example_id = val_output.get("example_id", "unknown")
    designed_path = val_output.get("designed_structure_path")
    output_base_dir = val_output.get("output_base_dir")
    metrics_tags_str = val_output.get("metrics_tags", "")
    metrics_tags = [t for t in metrics_tags_str.split(",") if t] if metrics_tags_str else []

    atom_array = val_output.get("designed_atom_array")
    if atom_array is None:
        logger.warning(f"Oracle loop: no designed_atom_array for {example_id}, skipping.")
        return None
    atom_array = unwrap_atom_array(atom_array)

    output_dir = Path(output_base_dir) if output_base_dir else None
    result = run_oracle_refolding_and_pass_criteria(
        atom_array=atom_array,
        example_id=example_id,
        output_dir=output_dir,
        oracle_mm=oracle_mm,
        refold_mm=refold_mm,
        pass_mm=pass_mm,
        metrics_tags=metrics_tags,
        prior_metrics=prior_design_metrics_by_id.get(example_id),
    )
    if result is None:
        return None

    result["designed_structure_path"] = designed_path
    result["dataset_name"] = val_output.get("dataset_name")
    return result


# ---------------------------------------------------------------------------
# Process-pool worker plumbing. The pool runs `_stage_generated_structure`
# (per-structure metrics + CIF writes) during validation and `_stage_post_fold`
# (refold + pass criteria) at epoch end. Workers are spawn-context.
# ---------------------------------------------------------------------------

_WORKER_MMS: dict[str, MetricManager | None] = {}


def _metrics_worker_init(mm_cfgs: dict[str, Any], val_dataset_tags: Any) -> None:
    """Worker initializer: rebuild the four MMs from configs.
    Skips ``setup()`` for the oracle MM (would load GPU weights)."""
    for name, cfg in mm_cfgs.items():
        if cfg is None:
            _WORKER_MMS[name] = None
            continue
        mm = MetricManager.instantiate_from_hydra(cfg)
        if name != "oracle":
            mm.on_validation_epoch_start(val_dataset_tags=val_dataset_tags)
        _WORKER_MMS[name] = mm


def _write_chain_only_cif_if_unambiguous(
    atom_array: AtomArray, example_id: str, output_dir: Path, cif_suffix: str = ".cif"
) -> Path | None:
    """Write a chain-of-interest-only CIF into ``output_dir`` for foldseek.

    New files use ``{example_id}_chain{cif_suffix}``; existing encodings are retained. Callers should pass
    :py:func:`~rfproteina.inference_utils.foldseek.chain_cif_dir` as ``output_dir`` so chain
    CIFs land under the dedicated ``foldseek_input_chains/`` tree.
    """
    coi, coi_unambiguous = get_chain_of_interest(atom_array)
    if not coi_unambiguous:
        return None
    chain_attr = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
    coi_array = atom_array[atom_array.get_annotation(chain_attr) == coi]

    # Foldseek's mmCIF parser requires _atom_site.occupancy + _atom_site.B_iso_or_equiv
    coi_array = coi_array.copy()
    coi_array.set_annotation("occupancy", np.ones(coi_array.array_length(), dtype=np.float32))
    coi_array.set_annotation("b_factor", np.zeros(coi_array.array_length(), dtype=np.float32))

    foldseek_input_path = output_dir / f"{example_id}_chain{cif_suffix}"
    foldseek_input_path = find_structure_file(foldseek_input_path) or foldseek_input_path
    to_cif_file(coi_array, foldseek_input_path, save_standard_annotations=False, extra_fields=[])
    return foldseek_input_path


def _stage_generated_structure(payload: dict, generated_structure_mm: MetricManager | None = None) -> dict:
    """Worker: per-structure metrics + final disk writes for one example.

    - Annotate unindexed-region insertion, SA_CRDSRC, and conditioned-reference coords
    - Run the per-structure MetricManager via ``run_metrics_for_result``.
    - Write the final CIF (post-unindex, or aligned variant from the MM)
      and a chain-only PDB for foldseek.
    - Optionally write input CIF + trajectory snapshots (offloaded here so
      validation_step doesn't block on disk I/O).
    """
    if generated_structure_mm is None:
        generated_structure_mm = _WORKER_MMS.get("generated_structure")
    example_id = payload.get("example_id", "sample")
    if generated_structure_mm is None:
        return {"example_id": example_id}

    array_to_annotate = payload.get("full_array_with_unindexed") or payload["generated_atom_array"]
    token_mapping = payload.get("predicted_insertion_mapping") or {}
    generated_array = annotate_unindexed_insertion(array_to_annotate, token_mapping)

    result = {
        "generated_atom_array": generated_array,
        "input_atom_array": payload.get("input_atom_array"),
        "full_array_with_unindexed": payload.get("full_array_with_unindexed"),
        "predicted_insertion_mapping": token_mapping,
        "example_id": example_id,
    }
    output_base_dir = Path(payload["output_base_dir"]) if payload.get("output_base_dir") else None
    save_input = bool(payload.get("save_input", True))
    metrics_result = run_metrics_for_result(
        result=result,
        batch={},
        batch_item_idx=0,
        metric_manager=generated_structure_mm,
        output_base_dir=output_base_dir,
        example_id=example_id,
        save_input_structure=save_input,
        metrics_tags_override=payload.get("metrics_tags") or [],
    )

    cif_suffix = payload.get("output_cif_suffix", ".cif")
    gen_path = payload.get("generated_output_path")
    # When reusing a structure already on disk (eval-only pass), the design CIF is the source
    # of truth and must not be rewritten; only derived artifacts (chain CIF) are regenerated.
    reuse_existing_structure = bool(payload.get("reuse_existing_structure", False))
    foldseek_input_path: Path | None = None
    if gen_path is not None:
        gen_path = Path(gen_path)
        if not reuse_existing_structure:
            try:
                ensure_dir(gen_path.parent)
                # Stage on the same filesystem; keep the basename used for the CIF data block.
                with tempfile.TemporaryDirectory(dir=gen_path.parent, prefix=".generated-cif-") as staging_dir:
                    staged_path = Path(staging_dir) / gen_path.name
                    to_cif_file(align_atom_array_to_input_frame(generated_array), staged_path)
                    os.replace(staged_path, gen_path)
            except Exception as e:
                logger.warning(f"generated-structure worker: failed to save CIF for {example_id}: {e}")
        # Per-design chain-only CIF is only consumed by foldseek clustering
        if payload.get("save_chain_of_interest", True):
            try:
                ds = payload.get("dataset_name")
                task = payload.get("task")
                chains_dir = (
                    chain_cif_dir(output_base_dir, dataset_name=ds, task=task) if output_base_dir else gen_path.parent
                )
                ensure_dir(chains_dir)
                foldseek_input_path = _write_chain_only_cif_if_unambiguous(
                    generated_array, example_id, chains_dir, cif_suffix
                )
            except Exception as e:
                logger.warning(
                    f"generated-structure worker: failed to pre-extract chain-only CIF for {example_id}: {e}"
                )
                foldseek_input_path = None

    if save_input:
        _write_input_cif(payload, example_id)
    _write_trajectories_if_present(payload, example_id)

    metrics_result["designed_atom_array"] = generated_array
    if foldseek_input_path is not None:
        metrics_result["designed_chain_path"] = str(foldseek_input_path.absolute())
    return metrics_result


def _write_input_cif(payload: dict, example_id: str) -> None:
    """Save the input AtomArray into the design's output directory."""
    input_atom_array = payload.get("input_atom_array")
    output_base_dir = payload.get("output_base_dir")
    if input_atom_array is None or output_base_dir is None:
        return
    try:
        save_path = (
            design_output_dir(output_base_dir, example_id)
            / f"{example_id}_input{payload.get('output_cif_suffix', '.cif')}"
        )
        ensure_dir(save_path.parent)
        to_cif_file(input_atom_array, save_path)
    except Exception as e:
        logger.warning(f"generated-structure worker: failed to save input CIF for {example_id}: {e}")


def _write_trajectories_if_present(payload: dict, example_id: str) -> None:
    """Write trajectory snapshots. ``payload['trajectories']`` maps key → (array, step_labels).

    step_labels is an int array → XTC (fixed CA-per-residue atom count); ``None`` → multi-model CIF.
    """
    trajectories = payload.get("trajectories") or {}
    output_base_dir = payload.get("output_base_dir")
    if not trajectories or output_base_dir is None:
        return
    traj_dir = design_output_dir(output_base_dir, example_id) / "trajectories"
    ensure_dir(traj_dir)
    cif_suffix = payload.get("output_cif_suffix", ".cif")

    for traj_key, (traj, step_labels) in trajectories.items():
        if traj is None:
            continue
        try:
            if step_labels is not None:
                # XTC; write a CA-only CIF as the topology reference.
                generated_array = payload["generated_atom_array"]
                predicted_bb_ca = generated_array[generated_array.atom_name == "CA"]
                to_cif_file(predicted_bb_ca, traj_dir / f"{example_id}_{traj_key}_bb_ca{cif_suffix}")
                xtc_file = XTCFile()
                xtc_file.set_structure(traj, time=step_labels)
                xtc_file.write(str(traj_dir / f"{example_id}_{traj_key}.xtc"))
            else:
                to_cif_file(traj, traj_dir / f"{example_id}_{traj_key}{cif_suffix}")
        except Exception as e:
            logger.warning(f"generated-structure worker: failed to save {traj_key} for {example_id}: {e}")


def _stage_post_fold(payload: dict) -> dict | None:
    """Worker: oracle-cache read + refolding metrics + pass criteria for one design."""
    val_output = payload["val_output"]
    design_metrics = payload.get("design_metrics") or {}
    return compute_oracle_metrics_for_val_output(
        val_output,
        oracle_mm=_WORKER_MMS.get("oracle"),
        refold_mm=_WORKER_MMS.get("refold"),
        pass_mm=_WORKER_MMS.get("pass"),
        prior_design_metrics_by_id={val_output.get("example_id", ""): design_metrics},
    )


# ---------------------------------------------------------------------------
# Metrics pool: per-rank ProcessPoolExecutor wrapping the staged workers.
# ---------------------------------------------------------------------------


def auto_pick_metrics_workers(cfg_exp: Any) -> int:
    """Pick metrics-pool worker count from cfg.
    -1 → ``min(8, available - 1 - val_dl_workers)``; ≥0 honoured as explicit cap."""
    configured = int(OmegaConf.select(cfg_exp, "dataloaders.metrics.num_workers", default=-1))
    if configured >= 0:
        return configured
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    val_dl_workers = int(OmegaConf.select(cfg_exp, "dataloaders.val.dataloader_params.num_workers", default=0) or 0)
    n = max(0, min(8, available - 1 - val_dl_workers))
    logger.info(f"Metrics pool: auto-picked {n} workers (available={available}, val_dl={val_dl_workers}).")
    return n


def resolve_mm_cfgs_from_metrics(metrics_cfg: Any) -> dict[str, Any]:
    """Resolve MetricManager subtrees from a metrics sub-config.

    Returns four subtrees intentionally: ``pre_oracle_filters`` is excluded
    because it runs inline and does not need worker re-instantiation.

    Args:
        metrics_cfg: The ``cfg_exp.metrics`` sub-config (or None).

    Returns:
        Dict with keys ``generated_structure``, ``oracle``, ``refold``, ``pass``.
    """

    def _resolve(key: str) -> Any:
        if metrics_cfg is None:
            return None
        sub = metrics_cfg.get(key)
        return OmegaConf.to_container(sub, resolve=True) if sub is not None else None

    return {
        "generated_structure": _resolve("inline_metrics"),
        "oracle": _resolve("refolding_oracles"),
        "refold": _resolve("refolding_metrics"),
        "pass": _resolve("pass_criteria"),
    }


def _completed(value: Any) -> Future:
    """Pre-completed Future wrapping ``value`` (uniform handling for inline results)."""
    fut: Future = Future()
    fut.set_result(value)
    return fut


class MetricsPool:
    """Per-rank worker pool for per-structure + post-fold metrics.

    ``num_workers <= 0`` disables the pool and runs every stage inline.
    """

    def __init__(self, num_workers: int, mm_cfgs: dict[str, Any]) -> None:
        self.num_workers = num_workers
        self._mm_cfgs = mm_cfgs
        self._executor: ProcessPoolExecutor | None = None
        self._futures: list[tuple[Hashable, Future]] = []

    def start(self, val_dataset_tags: Any) -> None:
        """Lazily create the executor (idempotent)."""
        if self.num_workers <= 0 or self._executor is not None:
            return
        self._executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_metrics_worker_init,
            initargs=(self._mm_cfgs, val_dataset_tags),
        )
        logger.info(f"Metrics pool: started with {self.num_workers} spawn workers")

    def shutdown(self) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._executor = None

    def submit_generated_structure(
        self, design_key: Hashable, payload: dict, fallback_mm: MetricManager | None
    ) -> None:
        """Dispatch one generated structure; runs inline when the pool is disabled."""
        if self._executor is not None:
            fut = self._executor.submit(_stage_generated_structure, payload)
        else:
            fut = _completed(_stage_generated_structure(payload, fallback_mm))
        self._futures.append((design_key, fut))

    def drain_generated_structure(self) -> dict[Hashable, dict]:
        """Drain submitted futures into one ``{design_key: metrics}`` map.

        Re-raises the first worker exception. ``_futures`` is cleared first
        so the pool is left in a clean state for re-use even on failure.
        """
        if not self._futures:
            return {}
        t0 = time.perf_counter()
        futures = self._futures
        self._futures = []
        out: dict[Hashable, dict] = {}
        try:
            for design_key, fut in futures:
                out[design_key] = fut.result()
        finally:
            dt = time.perf_counter() - t0
            logger.info(f"[metrics-pool] drain: n={len(futures)} workers={self.num_workers} wall={dt:.2f}s")
        return out

    def map_post_fold(self, payloads: list[dict], inline_fn: Callable[[dict], dict | None]) -> Iterator[dict | None]:
        """Run ``_stage_post_fold`` on each payload — pool when available, else inline."""
        n = len(payloads)
        t0 = time.perf_counter()
        mode = "pool" if self._executor is not None and payloads else "inline"
        if self._executor is not None and payloads:
            yield from self._executor.map(_stage_post_fold, payloads, chunksize=1)
        else:
            for p in payloads:
                yield inline_fn(p)
        dt = time.perf_counter() - t0
        logger.info(
            f"[metrics-pool] post-fold {mode}: n={n} workers={self.num_workers} "
            f"wall={dt:.2f}s avg={dt / max(n, 1) * 1000:.0f}ms/ex"
        )
