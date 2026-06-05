"""Checkpoint-gated inference / offline-validation smokes via generate.py.

These run the real ``proteinfoundation/generate.py`` Hydra entrypoint as a
subprocess against the SOTA checkpoint, on a small MCSA validation subset (and a
minimal beam-search smoke), keeping everything cheap enough for CI on 2 GPUs.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.helpers.gpu_e2e import (
    GPU_E2E_NGPUS,
    PYTHON,
    RF3_ORACLE_OVERRIDE,
    ROOT,
    require_gpu_e2e,
    require_rf3_ckpt,
    require_sota_ckpt,
    run_cmd,
)

_GENERATE_SCRIPT = "proteinfoundation/generate.py"


def _has_cif_outputs(out_dir: Path) -> bool:
    return any(out_dir.rglob("*.cif"))


def run_mcsa_validation_smoke(run_name: str, *, num_replicates: int = 2) -> None:
    """Offline validation on the MCSA subset using the SOTA checkpoint."""
    require_gpu_e2e(min_gpus=GPU_E2E_NGPUS)
    require_rf3_ckpt()
    ckpt_dir, ckpt_name = require_sota_ckpt()

    out_dir = ROOT / "inference" / run_name
    cmd = [
        PYTHON,
        _GENERATE_SCRIPT,
        "inference_experiment=inference_on_val_set",
        "+datasets/val@generation=mcsa/mcsa_subset",
        f"ckpt_path={ckpt_dir}",
        f"ckpt_name={ckpt_name}",
        RF3_ORACLE_OVERRIDE,
        f"num_replicates={num_replicates}",
        f"run_name_={run_name}",
        f"out_dir={out_dir}",
        "+single=true",
    ]
    proc = run_cmd(cmd, env={"WANDB_MODE": "disabled"})
    assert proc.returncode == 0, f"MCSA validation failed:\n{proc.stderr or proc.stdout}"
    assert _has_cif_outputs(out_dir), f"No designed structures written under {out_dir}"


def run_beam_search_smoke(run_name: str) -> None:
    """Minimal beam search to exercise search + refolding on a small benchmark."""
    require_gpu_e2e(min_gpus=GPU_E2E_NGPUS)
    require_rf3_ckpt()
    ckpt_dir, ckpt_name = require_sota_ckpt()

    out_dir = ROOT / "inference" / run_name
    cmd = [
        PYTHON,
        _GENERATE_SCRIPT,
        "inference_experiment=inference_beam_example",
        RF3_ORACLE_OVERRIDE,
        f"ckpt_path={ckpt_dir}",
        f"ckpt_name={ckpt_name}",
        "generation.args.nsteps=20",
        "search.strategy.step_checkpoints=[0,10,20]",
        "search.strategy.n_branch=2",
        "search.strategy.beam_width=2",
        "subset_to_examples=monomer-short",
        f"run_name_={run_name}",
        f"out_dir={out_dir}",
        "+single=true",
    ]
    proc = run_cmd(cmd, env={"WANDB_MODE": "disabled"})
    assert proc.returncode == 0, f"Beam search smoke failed:\n{proc.stderr or proc.stdout}"
    assert _has_cif_outputs(out_dir), f"No designed structures written under {out_dir}"
