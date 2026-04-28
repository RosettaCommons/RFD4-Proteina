#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../.ipd/shebang/latent_exec.sh" "$0" "$@"'

"""Evaluate pre-generated structures: design metrics → oracle refolding → pass criteria.

Usage::

    python proteinfoundation/evaluate.py inputs=/path/to/structures metrics_tags=[unconditional_refolding]
"""

import os
from pathlib import Path

import hydra
import pandas as pd
import rootutils
from atomworks.common import exists
from atomworks.io.parser import parse
from atomworks.io.transforms.atom_array import add_charge_from_ccd_codes
from atomworks.io.utils.io_utils import CIF_LIKE_EXTENSIONS
from omegaconf import DictConfig

from foundry.metrics import MetricManager
from foundry.utils.ddp import RankedLogger
from foundry.utils.env import load_dotenv
from foundry.utils.logging import print_config_tree, silence_warnings
from proteinfoundation.metrics.metrics_utils import run_oracle_refolding_and_pass_criteria
from proteinfoundation.proteina import _clean_metrics_dict
from proteinfoundation.utils.chain_utils import get_chain_of_interest
from proteinfoundation.utils.parse_presets import STANDARD_ANNOTATIONS_ONLY

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
logger = RankedLogger(__name__, rank_zero_only=True)
_config_path = os.path.join(os.environ["PROJECT_ROOT"], "projects/proteinfoundation/configs")


def _iter_structure_files(directory: Path) -> list[Path]:
    files = []
    for suffix in CIF_LIKE_EXTENSIONS:
        files.extend(directory.rglob(f"*{suffix}"))
    return sorted(set(files))


def _find_input_structure(structure_file: Path) -> Path | None:
    base_name = structure_file.name.split(".")[0].removesuffix("_generated")
    input_path = structure_file.parent.parent / "input" / f"{base_name}_input.cif"
    return input_path if input_path.exists() else None


def run_evaluate(cfg: DictConfig) -> None:
    """Run evaluation on all structure files in ``cfg.inputs``."""
    metrics_tags_raw = cfg.get("metrics_tags")
    if isinstance(metrics_tags_raw, str):
        metrics_tags: list[str] | None = [t.strip() for t in metrics_tags_raw.split(",") if t.strip()]
    elif metrics_tags_raw is not None:
        metrics_tags = list(metrics_tags_raw)
    else:
        metrics_tags = None

    structure_files = _iter_structure_files(Path(cfg.inputs))
    if not structure_files:
        raise ValueError(f"No structure files found in {cfg.inputs}")

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
    job_suffix = f"_job{job_id}" if exists(job_id) else ""

    def _mm(key: str) -> MetricManager | None:
        sub = cfg.metrics.get(key)
        return MetricManager.instantiate_from_hydra(metrics_cfg=sub) if exists(sub) else None

    design_mm = _mm("inline_metrics")
    oracle_mm = _mm("refolding_oracles")
    refold_mm = _mm("refolding_metrics")
    pass_mm = _mm("pass_criteria")

    design_results: list[dict] = []
    parsed_arrays: dict[str, tuple[Path, object]] = {}

    if design_mm is not None:
        design_mm.on_validation_epoch_start()

    for structure_file in structure_files:
        example_id = structure_file.name.split(".")[0]
        try:
            atom_array = parse(structure_file, config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]
        except Exception as e:
            logger.warning(f"Failed to parse {structure_file}: {e}")
            continue
        parsed_arrays[example_id] = (structure_file, atom_array)
        result = {"example_id": example_id, "designed_structure_path": str(structure_file)}
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
                        "output_base_dir": out_dir,
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

    out_dir.mkdir(parents=True, exist_ok=True)
    refolding_results: list[dict] = []
    design_by_id = {m.get("example_id"): m for m in design_results if m.get("example_id")}
    for mm in filter(None, [oracle_mm, refold_mm, pass_mm]):
        mm.on_validation_epoch_start()
    # Sort by n_tokens ascending to avoid unnecessary oracle recompilation.
    sorted_arrays = sorted(
        parsed_arrays.items(),
        key=lambda kv: design_by_id.get(kv[0], {}).get("general_metrics.n_tokens", 0),
    )
    for example_id, (structure_file, atom_array) in sorted_arrays:
        result = run_oracle_refolding_and_pass_criteria(
            atom_array=add_charge_from_ccd_codes(atom_array),
            example_id=example_id,
            output_dir=out_dir,
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

    cleaned_design = [c for m in design_results if (c := _clean_metrics_dict(m)) is not None]
    cleaned_refold = [c for m in refolding_results if (c := _clean_metrics_dict(m)) is not None]
    if cleaned_design or cleaned_refold:
        by_id: dict = {}
        for m in cleaned_design + cleaned_refold:
            if eid := m.get("example_id"):
                by_id.setdefault(eid, {}).update(m)
        combined = pd.DataFrame(list(by_id.values())).round(4)
        combined.to_csv(out_dir / f"per_example_metrics{job_suffix}.csv", index=False)


@hydra.main(version_base="1.3", config_path=_config_path, config_name="evaluate")
def main_hydra(cfg: DictConfig) -> None:
    load_dotenv()
    silence_warnings()
    print_config_tree(cfg, resolve=True)
    run_evaluate(cfg)


if __name__ == "__main__":
    main_hydra()
