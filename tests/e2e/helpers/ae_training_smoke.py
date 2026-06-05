"""AE training smokes: short DDP runs that assert train loss does not explode.

Each variant runs ``experiment_ae=smoke`` for a small number of optimizer steps plus
a handful of sanity validation steps, on multiple GPUs with a small batch.
Four variants cover Muon/Adam × compile/no-compile (no refolding oracle).
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.e2e.helpers.gpu_e2e import (
    GPU_E2E_NGPUS,
    GPU_E2E_PREVAL_STEPS,
    GPU_E2E_TRAIN_BATCH_SIZE,
    GPU_E2E_TRAIN_STEPS,
    PYTHON,
    ROOT,
    require_data_file,
    require_gpu_e2e,
    run_training_smoke_subprocess,
)

_TRAIN_SCRIPT = "proteinfoundation/partial_autoencoder/train.py"


@dataclass(frozen=True)
class AeTrainVariant:
    name: str
    optimizer: str  # "muon" | "adam"
    compile: bool


AE_TRAIN_VARIANTS: tuple[AeTrainVariant, ...] = (
    AeTrainVariant("ae_muon_nocompile", "muon", False),
    AeTrainVariant("ae_adam_nocompile", "adam", False),
    AeTrainVariant("ae_muon_compile", "muon", True),
    AeTrainVariant("ae_adam_compile", "adam", True),
)


def _variant_overrides(variant: AeTrainVariant) -> list[str]:
    overrides: list[str] = [f"nn_ae.use_torch_compile={'true' if variant.compile else 'false'}"]
    if variant.optimizer == "adam":
        overrides.append("opt.muon_param_patterns=[]")
    return overrides


def run_ae_training_smoke(variant: AeTrainVariant) -> None:
    require_gpu_e2e(min_gpus=GPU_E2E_NGPUS)
    require_data_file("latent_data/train_datasets/afdb_monomer/afdb_monomer.parquet")

    run_name = f"ae_smoke_{variant.name}"
    out_dir = ROOT / "store" / run_name

    cmd = [
        PYTHON,
        _TRAIN_SCRIPT,
        "experiment_ae=smoke",
        "+single=true",
        "+nolog=true",
        "+show_prog_bar=true",
        f"out_dir={out_dir}",
        f"hardware.ngpus_per_node_={GPU_E2E_NGPUS}",
        "hardware.nnodes_=1",
        f"opt.steps_per_epoch={GPU_E2E_TRAIN_STEPS}",
        "opt.max_epochs=1",
        f"opt.num_sanity_val_steps={GPU_E2E_PREVAL_STEPS}",
        f"opt.limit_val_batches={GPU_E2E_PREVAL_STEPS}",
        f"dataloaders.train.dataloader_params.batch_size={GPU_E2E_TRAIN_BATCH_SIZE}",
        "log.log_wandb=false",
        "log.checkpoint=false",
        "pretrain_ckpt_path=null",
        *_variant_overrides(variant),
    ]

    run_training_smoke_subprocess(
        variant_name=variant.name,
        run_name=run_name,
        cmd=cmd,
        out_dir=out_dir,
        compile_enabled=variant.compile,
    )
