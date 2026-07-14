"""Training smoke: train briefly, assert no parameter is silently disconnected.

A feature behind an all-false mask gets an exactly-zero grad, so `grad is None` never fires.
"""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from tests.e2e.helpers.gpu_e2e import (
    PYTHON,
    ROOT,
    assert_loss_not_exploding,
    assert_no_dataset_fallbacks,
    cleanup_smoke_training_store,
    collect_train_losses,
    require_data_file,
    require_gpu_e2e,
    run_cmd,
)

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
    # Conditioning off, so only the null index is looked up. Below 1 (i.e. 0) would mean severed.
    ("cond", "cath_code", "embedding_C.weight"): 1,  # of 6
    ("cond", "cath_code", "embedding_A.weight"): 1,  # of 44
    ("cond", "cath_code", "embedding_T.weight"): 1,  # of 1473
    # This recipe never conditions on SS (p_add_secondary_structure_condition=0.0 across the whole
    # release/hero lineage), so only padding_idx=0 is looked up -> 0 alive. Floor 0 is vacuous by
    # design; raise it to 3 if SS conditioning is enabled in the recipe.
    ("seq", "secondary_structure", "ss_embedding.weight"): 0,  # of 4; SS conditioning off in this recipe
    # of 5; protein/RNA/DNA/atomized all appear in the sota mix, `unknown` never does.
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


GRAD_TRAIN_STEPS = 100  # x batch 8 = 800 examples, enough to saturate the counts below
GRAD_BATCH_SIZE = 8
RUN_NAME = "e2e_grad_coverage"


@pytest.mark.gpu
def test_gradient_coverage(tmp_path: Path) -> None:
    """Train briefly on one GPU; assert the loss is sane and every parameter row gets a gradient."""
    require_gpu_e2e(min_gpus=1)
    require_data_file("train_datasets/afdb_monomer/afdb_monomer.parquet")
    min_alive_rows = _resolve_min_alive_rows()

    out_dir = ROOT / "store" / f"{RUN_NAME}_local"
    coverage_json = tmp_path / "grad_coverage.json"
    cleanup_smoke_training_store(RUN_NAME)

    cmd = [
        PYTHON,
        "src/proteinfoundation/train.py",
        "experiment=sota",
        f"run_name_={RUN_NAME}",
        "+single=true",  # forces 1 GPU, 1 node
        "+nolog=true",
        f"opt.steps_per_epoch={GRAD_TRAIN_STEPS}",
        f"dataloaders.batch_size={GRAD_BATCH_SIZE}",
        "nn.use_torch_compile=false",
        "opt.max_epochs=1",
        "++opt.num_sanity_val_steps=0",
        "++opt.limit_val_batches=0",
        "log.checkpoint=false",
        "log.log_wandb=false",
        "log.log_every_n_steps=1",
        "seed=42",
        # sota's 16 workers OOMs the lane's 64G, and 0 fixes the sampled batches.
        "dataloaders.num_workers=0",
        "dataloaders.prefetch_factor=null",
        "dataloaders.multiprocessing_context=null",
        "dataloaders.train.dataloader_params.persistent_workers=false",
        "~metrics/refolding_oracles",
        "+callbacks.grad_coverage._target_=proteinfoundation.utils.training_analysis_utils.GradientCoverageCallback",
        f"+callbacks.grad_coverage.output_path={coverage_json}",
        f"++out_dir={out_dir}",
    ]
    proc = run_cmd(cmd, env={"WANDB_MODE": "disabled"})
    assert proc.returncode == 0, f"Training failed:\n{proc.stderr or proc.stdout}"

    output = proc.stdout + "\n" + proc.stderr
    assert_no_dataset_fallbacks(output)  # a substituted example would invalidate the counts
    assert_loss_not_exploding(collect_train_losses(output, out_dir=out_dir))

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
