#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../scripts/rfproteina_exec.sh" training "$0" "$@"'

"""Evaluate pre-generated structures: design metrics → oracle refolding → pass criteria.

Usage::

    python rfproteina/evaluate.py inputs=/path/to/structures metrics_tags=[unconditional_refolding]
"""

import os
import warnings
from contextlib import nullcontext
from pathlib import Path

# Retain import warnings until the resolved config selects verbosity.
with (
    warnings.catch_warnings(record=True)
    if __name__ in ("__main__", "__mp_main__")
    else nullcontext([]) as _startup_warnings
):
    import hydra
    import pandas as pd
    import rootutils
    from atomworks.common import exists
    from atomworks.io.parser import parse
    from atomworks.ml.utils.misc import hash_sequence
    from omegaconf import DictConfig

    from rfproteina.inference_utils.foldseek import chain_cif_dir
    from rfproteina.metrics import MetricManager
    from rfproteina.metrics.utils.metrics_utils import (
        _write_chain_only_cif_if_unambiguous,
        apply_pre_oracle_filters,
        run_oracle_refolding_and_pass_criteria,
    )
    from rfproteina.proteina import _clean_metrics_dict
    from rfproteina.utils.chain import get_chain_of_interest
    from rfproteina.utils.common import design_output_dir
    from rfproteina.utils.ddp import RankedLogger
    from rfproteina.utils.env import load_dotenv
    from rfproteina.utils.logging import print_config_tree
    from rfproteina.utils.parse_presets import STANDARD_ANNOTATIONS_ONLY
    from rfproteina.utils.structure_files import (
        find_structure_file,
        iter_structure_files,
        structure_file_stem,
    )

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    from rfproteina.utils.verbosity import configured_verbosity, queue_startup_warnings

queue_startup_warnings(_startup_warnings)

logger = RankedLogger(__name__, rank_zero_only=True)
_config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")


def _find_input_structure(structure_file: Path) -> Path | None:
    """Locate the design's input CIF, which inference saves alongside the generated CIF."""
    base_name = structure_file_stem(structure_file).removesuffix("_generated")
    input_path = structure_file.parent / f"{base_name}_input.cif"
    return find_structure_file(input_path)


@configured_verbosity(default=True)
def run_evaluate(cfg: DictConfig) -> None:
    """Run evaluation on all structure files in ``cfg.inputs``."""
    cif_suffix = ".cif.zst" if cfg.get("compress_outputs", False) else ".cif"
    metrics_tags_raw = cfg.get("metrics_tags")
    if isinstance(metrics_tags_raw, str):
        metrics_tags: list[str] | None = [t.strip() for t in metrics_tags_raw.split(",") if t.strip()]
    elif metrics_tags_raw is not None:
        metrics_tags = list(metrics_tags_raw)
    else:
        metrics_tags = None

    inputs_dir = Path(cfg.inputs)
    structure_files = iter_structure_files(inputs_dir, include_pdb=True)
    if not structure_files:
        raise ValueError(f"No structure files found in {cfg.inputs}")

    task_csv_path = cfg.get("task_csv")
    filename_to_task: dict[str, str] = {}
    if task_csv_path is not None:
        task_df = pd.read_csv(task_csv_path)
        if "designed_structure_path" not in task_df.columns or "task" not in task_df.columns:
            raise ValueError(f"task_csv {task_csv_path} must have 'designed_structure_path' and 'task' columns")
        filename_to_task = dict(zip(task_df["designed_structure_path"], task_df["task"], strict=False))
        logger.info(f"Loaded task assignments for {len(filename_to_task)} files from {task_csv_path}")
    else:
        logger.warning("No task_csv provided; all structures will be assigned task='all'")

    job_id = cfg.get("job_id")
    if exists(job_id):
        total = len(structure_files)
        gen_njobs = cfg.get("gen_njobs", 1)
        chunk_size_abs = (total + gen_njobs - 1) // gen_njobs
        start = job_id * chunk_size_abs
        structure_files = structure_files[start : min(start + chunk_size_abs, total)]
    if not structure_files:
        raise ValueError("No structure files remain after chunking.")

    out_dir = Path(cfg.out_dir)
    samples_dir = out_dir / "evaluation_samples"
    metrics_dir = out_dir / "evaluation_metrics"
    job_suffix = f"_job{job_id}" if exists(job_id) else ""

    def _mm(key: str) -> MetricManager | None:
        sub = cfg.metrics.get(key)
        return MetricManager.instantiate_from_hydra(metrics_cfg=sub) if exists(sub) else None

    design_mm = _mm("inline_metrics")
    oracle_mm = _mm("refolding_oracles")
    if oracle_mm is not None:
        for oracle in oracle_mm.metrics.values():
            oracle.output_cif_suffix = cif_suffix
    refold_mm = _mm("refolding_metrics")
    pass_mm = _mm("pass_criteria")
    pre_oracle_filter_mm = _mm("pre_oracle_filters")

    dataset_name = inputs_dir.name
    design_results: list[dict] = []
    parsed_arrays: dict[str, tuple[Path, object]] = {}

    if design_mm is not None:
        design_mm.on_validation_epoch_start(val_dataset_tags=[set(metrics_tags or [])])

    for structure_file in structure_files:
        rel = structure_file.relative_to(inputs_dir)
        # Inputs come from a nested tree flattened into one output dir, so basenames repeat
        # (af3/design.cif vs boltz/design.cif); the path hash is what keeps them distinct.
        stem = structure_file_stem(rel)
        identity = rel.with_name(stem + ".cif")
        example_id = f"{stem}_{hash_sequence(str(identity), 8)}"
        task = filename_to_task.get(str(structure_file), "all")
        try:
            atom_array = parse(structure_file, config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]
        except Exception as e:
            logger.warning(f"Failed to parse {structure_file}: {e}")
            continue
        parsed_arrays[example_id] = (structure_file, atom_array)
        result: dict = {
            "example_id": example_id,
            "dataset_name": dataset_name,
            "task": task,
            "designed_structure_path": str(structure_file),
            "output_base_dir": str(samples_dir),
            "metrics_tags": ",".join(metrics_tags or []),
        }
        # Symlink the input structure into evaluation_samples/ so the output directory mirrors
        # inference_samples/ and is self-contained.  Using the original filename avoids a
        # double _generated suffix when processing inference outputs.
        try:
            example_dir = design_output_dir(samples_dir, example_id)
            example_dir.mkdir(parents=True, exist_ok=True)
            link = example_dir / structure_file.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(structure_file.absolute())
        except Exception as e:
            logger.warning(f"evaluate: failed to symlink structure for {example_id}: {e}")
        # Write chain-of-interest CIF under foldseek_input_chains/{ds_hash}/{task}/ so that
        # _partition_groups can discover and route the correct files to each foldseek job.
        try:
            chains_dir = chain_cif_dir(samples_dir, dataset_name=dataset_name, task=task)
            chains_dir.mkdir(parents=True, exist_ok=True)
            chain_cif = _write_chain_only_cif_if_unambiguous(atom_array, example_id, chains_dir, cif_suffix)
            if chain_cif is not None:
                result["designed_chain_path"] = str(chain_cif)
        except Exception as e:
            logger.warning(f"evaluate: failed to write chain CIF for {example_id}: {e}")
        if design_mm is not None:
            try:
                input_path = _find_input_structure(structure_file)
                ground_truth_aa = (
                    parse(input_path, config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]
                    if input_path is not None
                    else None
                )
                coi, coi_unambiguous = get_chain_of_interest(atom_array)
                result |= design_mm(
                    network_input=None,
                    network_output=None,
                    extra_info={
                        "example_id": example_id,
                        "output_base_dir": samples_dir,
                        "chain_of_interest": coi if coi_unambiguous else None,
                        "metrics_tags": metrics_tags or [],
                    },
                    ground_truth_atom_array=ground_truth_aa,
                    predicted_atom_array=atom_array,
                )
            except Exception as e:
                logger.warning(f"Design metrics failed for {example_id}: {e}")

        design_results.append(result)

    if design_mm is not None:
        design_mm.on_validation_epoch_end()

    if pre_oracle_filter_mm is not None:
        pre_oracle_filter_mm.on_validation_epoch_start(val_dataset_tags=[set(metrics_tags or [])])
        design_results = apply_pre_oracle_filters(
            design_results,
            pre_oracle_filter_mm,
            delete_failing=cfg.get("delete_failing_pre_oracle", False),
        )
        pre_oracle_filter_mm.on_validation_epoch_end()
        surviving_ids = {m["example_id"] for m in design_results if m.get("example_id")}
        parsed_arrays = {eid: v for eid, v in parsed_arrays.items() if eid in surviving_ids}

    samples_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    refolding_results: list[dict] = []
    design_by_id = {m.get("example_id"): m for m in design_results if m.get("example_id")}
    # Pass the tags so metrics that cannot fire are never set up -- an unused AF3
    # oracle would otherwise start its warm worker and hold ~10 GB.
    for mm in filter(None, [oracle_mm, refold_mm, pass_mm]):
        mm.on_validation_epoch_start(val_dataset_tags=[set(metrics_tags or [])])
    # Sort by token_count ascending to avoid unnecessary oracle recompilation.
    sorted_arrays = sorted(
        parsed_arrays.items(),
        key=lambda kv: design_by_id.get(kv[0], {}).get("token_count", 0),
    )
    for example_id, (structure_file, atom_array) in sorted_arrays:
        result = run_oracle_refolding_and_pass_criteria(
            atom_array=atom_array,
            example_id=example_id,
            output_dir=samples_dir,
            oracle_mm=oracle_mm,
            refold_mm=refold_mm,
            pass_mm=pass_mm,
            metrics_tags=metrics_tags,
            prior_metrics=design_by_id.get(example_id),
        )
        if result is None:
            continue
        result["designed_structure_path"] = str(structure_file)
        refolding_results.append(result)
    for mm in filter(None, [oracle_mm, refold_mm, pass_mm]):
        mm.on_validation_epoch_end()

    # Design rows may carry only metadata (e.g. when every design metric is disabled), but that
    # metadata -- output_base_dir, designed_chain_path, task -- is what routes Foldseek clustering
    # downstream, so keep those rows and let the merge below fold them into the refolding rows.
    cleaned_design = [c for m in design_results if (c := _clean_metrics_dict(m, keep_metadata_only=True)) is not None]
    cleaned_refold = [c for m in refolding_results if (c := _clean_metrics_dict(m)) is not None]
    if cleaned_design or cleaned_refold:
        by_id: dict = {}
        for m in cleaned_design + cleaned_refold:
            if eid := m.get("example_id"):
                by_id.setdefault(eid, {}).update(m)
        combined = pd.DataFrame(list(by_id.values())).round(4)
        combined.to_csv(metrics_dir / f"per_example_metrics{job_suffix}.csv", index=False)


@hydra.main(version_base="1.3", config_path=_config_path, config_name="evaluate")
@configured_verbosity(default=True)
def main_hydra(cfg: DictConfig) -> None:
    load_dotenv()
    print_config_tree(cfg, resolve=True)
    run_evaluate(cfg)


if __name__ == "__main__":
    main_hydra()
