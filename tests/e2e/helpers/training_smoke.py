"""SOTA training smokes: short DDP runs that assert train loss does not explode.

Each variant runs ``experiment=sota`` for a small number of optimizer steps plus
a short epoch-0 validation pass (2 batches per val dataset by default), on 2 GPUs.
Prevalidation oracle is chosen per variant (RF3 or AF3).
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.e2e.helpers.gpu_e2e import (
    GPU_E2E_LIMIT_VAL_BATCHES,
    GPU_E2E_NGPUS,
    GPU_E2E_SANITY_VAL_STEPS,
    GPU_E2E_TRAIN_BATCH_SIZE,
    GPU_E2E_TRAIN_STEPS,
    PYTHON,
    ROOT,
    PrevalOracle,
    assert_loss_not_exploding,
    cleanup_smoke_training_store,
    collect_train_losses,
    get_data_path,
    preval_oracle_overrides,
    require_data_file,
    require_gpu_e2e,
    require_preval_oracle,
    run_cmd,
)

# Path to the train entrypoint (run with cwd=ROOT=projects/proteinfoundation).
_TRAIN_SCRIPT = "proteinfoundation/train.py"


@dataclass(frozen=True)
class TrainVariant:
    name: str
    optimizer: str  # "muon" | "adam"
    compile: bool
    preval_oracle: PrevalOracle


SOTA_TRAIN_VARIANTS: tuple[TrainVariant, ...] = (
    TrainVariant("muon_nocompile_rf3", "muon", False, "rf3"),
    TrainVariant("muon_compile_af3", "muon", True, "af3"),
)


def _variant_overrides(variant: TrainVariant) -> list[str]:
    overrides: list[str] = [f"nn.use_torch_compile={'true' if variant.compile else 'false'}"]
    # sota inherits non-empty muon_param_patterns; adam = route nothing to Muon.
    if variant.optimizer == "adam":
        overrides.append("opt.muon_param_patterns=[]")
    return overrides


def _sota_train_dataset_overrides() -> list[str]:
    """Drop train sources that are absent on the runner so smokes still run on AFDB-only."""
    data_path = get_data_path()
    if data_path is None:
        return []
    overrides: list[str] = []
    plinder_parquet = data_path / "train_datasets/plinder/2026_05_20_plinder_systems.parquet"
    omol_lmdb = data_path / "train_datasets/omol"
    has_plinder = plinder_parquet.is_file()
    has_omol = omol_lmdb.is_dir()
    if not has_plinder:
        overrides.append("~datasets.train.plinder")
    if not has_omol:
        overrides.append("~datasets.train.omol")
    if not overrides:
        return []
    # Renormalize the default AFDB/OMOL mix when PLINDER (0.1) is unavailable.
    if has_omol and not has_plinder:
        overrides.extend(
            [
                "datasets.train.afdb_monomer.probability=0.7778",
                "datasets.train.omol.probability=0.2222",
            ]
        )
    else:
        overrides.append("datasets.train.afdb_monomer.probability=1.0")
    return overrides


def run_sota_training_smoke(variant: TrainVariant) -> None:
    require_gpu_e2e(min_gpus=GPU_E2E_NGPUS)
    require_preval_oracle(variant.preval_oracle)
    # sota mix needs AFDB/OMOL/PLINDER; AFDB parquet is the cheapest presence check.
    require_data_file("latent_data/train_datasets/afdb_monomer/afdb_monomer.parquet")

    run_name = f"sota_smoke_{variant.name}"
    out_dir = ROOT / "store" / run_name
    cleanup_smoke_training_store(run_name)

    env = {
        "TORCHINDUCTOR_CACHE_DIR": str(ROOT / ".cache" / "torchinductor" / run_name),
        "WANDB_MODE": "disabled",
    }

    cmd = [
        PYTHON,
        _TRAIN_SCRIPT,
        "experiment=sota",
        "+single=true",
        "+nolog=true",
        "+show_prog_bar=true",
        f"out_dir={out_dir}",
        f"hardware.ngpus_per_node_={GPU_E2E_NGPUS}",
        "hardware.nnodes_=1",
        f"opt.steps_per_epoch={GPU_E2E_TRAIN_STEPS}",
        "opt.max_epochs=1",
        f"opt.num_sanity_val_steps={GPU_E2E_SANITY_VAL_STEPS}",
        "opt.val_start_epoch=0",
        f"+opt.limit_val_batches={GPU_E2E_LIMIT_VAL_BATCHES}",
        *preval_oracle_overrides(variant.preval_oracle),
        f"dataloaders.batch_size={GPU_E2E_TRAIN_BATCH_SIZE}",
        "log.log_wandb=false",
        "log.checkpoint=false",
        "pretrain_ckpt_path=null",
        *_variant_overrides(variant),
        *_sota_train_dataset_overrides(),
    ]

    proc = run_cmd(cmd, env=env)
    assert proc.returncode == 0, f"[{variant.name}] training failed:\n{proc.stderr or proc.stdout}"

    losses = collect_train_losses(proc.stdout + "\n" + proc.stderr, out_dir=out_dir)
    assert_loss_not_exploding(losses)
