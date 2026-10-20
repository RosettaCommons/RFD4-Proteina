"""Training smoke: train briefly, assert no parameter is silently disconnected.

A feature behind an all-false mask gets an exactly-zero grad, so `grad is None` never fires.
"""

import csv
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from tests.e2e.helpers.gpu_e2e import (
    ROOT,
    assert_loss_not_exploding,
    assert_no_dataset_fallbacks,
    cleanup_smoke_training_store,
    require_data_file,
    require_gpu_e2e,
    run_cmd,
)
from tests.e2e.helpers.regression_baselines import deterministic_env

# LOWER BOUND on the rows each parameter must reach with a non-zero gradient; the test fails only
# when a parameter falls BELOW its floor (something got disconnected). A count above the floor is
# fine, so the sample-dependent atom rows can't cause false alarms. Absent name => every row.
# Keyed by feature name, not feats_seq position, so reordering cannot retarget an assertion.
MIN_ALIVE_ROWS_BY_FEATURE: dict[tuple[str, str, str], int] = {
    # atomworks collate feeds `is_atomized`, so atomized tokens from the PDB mix exercise these.
    # Only the charge/element rows present in the sample light up (not every row); dropping below
    # this floor would mean the atom features regressed to the old atom37-collate dummy (`0.0*w`).
    ("seq", "atom_charge", "embedding.weight"): 6,  # of 11
    ("seq", "atomic_element", "embedding.weight"): 21,  # of 128
    # Every CATH embedding must receive a gradient, regardless of the sampled conditioning route.
    ("cond", "cath_code", "embedding_C.weight"): 1,  # of 6
    ("cond", "cath_code", "embedding_A.weight"): 1,  # of 44
    ("cond", "cath_code", "embedding_T.weight"): 1,  # of 1473
    # The release samples SS conditioning; helix, sheet, and coil must all receive gradients.
    ("seq", "secondary_structure", "ss_embedding.weight"): 3,  # of 4; row 0 is padding
    # At least protein/RNA/DNA/atomized out of seven token types must be exercised.
    ("seq", "token_type", "embedding.weight"): 4,
}

_FACTORY_PREFIX = {"seq": "nn.init_repr_factory", "cond": "nn.cond_factory"}


def _resolve_min_alive_rows() -> dict[str, int]:
    """Resolve feature names to `feat_creators.<i>` parameter names via the sota config."""
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="training_proteina", overrides=["experiment=sota"])
    feat_order = {"seq": list(cfg.nn.feats_seq), "cond": list(cfg.nn.feats_cond_seq)}

    resolved: dict[str, int] = {}
    for (factory, feature, suffix), n_alive in MIN_ALIVE_ROWS_BY_FEATURE.items():
        order = feat_order[factory]
        assert feature in order, f"{feature!r} is no longer in feats_{factory}; update the expectations"
        resolved[f"{_FACTORY_PREFIX[factory]}.feat_creators.{order.index(feature)}.{suffix}"] = n_alive
    return resolved


GRAD_TRAIN_STEPS = 50
GRAD_BATCH_SIZE = 16
RUN_NAME = "e2e_grad_coverage"


@pytest.mark.gpu
def test_gradient_coverage(tmp_path: Path) -> None:
    """Train briefly on one GPU; assert the loss is sane and every parameter row gets a gradient."""
    require_gpu_e2e(min_gpus=1)
    # The AFDB parquet remains part of the coverage-oriented smoke mixture below.
    require_data_file("train_datasets/afdb_monomer/afdb_genie3_32_512_annotated.parquet")
    min_alive_rows = _resolve_min_alive_rows()

    out_dir = ROOT / "store" / f"{RUN_NAME}_local"
    coverage_json = tmp_path / "grad_coverage.json"
    cleanup_smoke_training_store(RUN_NAME)

    cmd = [
        "src/rfproteina/train.py",
        "experiment=sota",
        "training.exploration_k=1",  # Spend the batch on distinct examples for feature coverage.
        f"run_name_={RUN_NAME}",
        "+single=true",  # forces 1 GPU, 1 node
        "+nolog=true",
        "_max_tokens=128",
        "datasets.train.afdb_monomer.probability=0.36",
        "datasets.train.omol.probability=0.20",
        "datasets.train.plinder.probability=0.20",
        f"opt.steps_per_epoch={GRAD_TRAIN_STEPS}",
        f"dataloaders.batch_size={GRAD_BATCH_SIZE}",
        "nn.use_torch_compile=false",
        "nn.token_dim=384",
        "nn.pair_repr_dim=64",
        "nn.tri_mult_c=64",
        "nn.update_pair_repr_every_n=3",
        "nn.recycle.grad=last",
        "training.n_recycle_max=2",
        "opt.max_epochs=1",
        "++opt.num_sanity_val_steps=0",
        "++opt.limit_val_batches=0",
        "log.checkpoint=false",
        "log.log_wandb=false",
        "log.log_every_n_steps=1",
        "seed=42",
        # Spawn keeps each worker's S3 client and connection pool process-local.
        "dataloaders.num_workers=12",
        "dataloaders.prefetch_factor=1",
        "dataloaders.multiprocessing_context=spawn",
        "dataloaders.train.dataloader_params.persistent_workers=false",
        # Keep worker assignment and augmentation RNG independent of loading speed.
        "dataloaders.train.dataloader_params.in_order=true",
        "~metrics/refolding_oracles",
        "+callbacks.grad_coverage._target_=rfproteina.utils.training_analysis.GradientCoverageCallback",
        f"+callbacks.grad_coverage.output_path={coverage_json}",
        f"++out_dir={out_dir}",
    ]
    proc = run_cmd(cmd, env=deterministic_env())
    assert proc.returncode == 0, f"Training failed:\n{proc.stderr or proc.stdout}"

    output = proc.stdout + "\n" + proc.stderr
    assert_no_dataset_fallbacks(output)  # a substituted example would invalidate the counts
    # Exclude the chiral penalty from magnitude checks, not from training or gradient checks.
    with next(out_dir.glob("csv_logs/version_*/metrics.csv")).open() as fh:
        losses = [
            float(row["train/loss_step"]) - float(row["train/loss_signed_volume_step"])
            for row in csv.DictReader(fh)
            if row["train/loss_step"]
        ]
    assert_loss_not_exploding(losses, min_points=GRAD_TRAIN_STEPS)

    coverage = json.loads(coverage_json.read_text())
    assert coverage, "gradient coverage callback wrote no parameters"

    mismatches = {}
    for name, cov in coverage.items():
        floor = min_alive_rows.get(name, cov["n_rows"])
        if cov["alive_rows"] < floor:
            mismatches[name] = {"min_expected": floor, **cov}
    assert not mismatches, (
        "Gradient coverage below floor: a parameter reached fewer rows than required, so something "
        f"is disconnected. Understand it before lowering the floor:\n{json.dumps(mismatches, indent=2)}"
    )
