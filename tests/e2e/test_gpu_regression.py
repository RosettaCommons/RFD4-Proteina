"""Four H200 regression tests:

1. SOTA training — 25 optimizer steps from random init; losses match baseline.
2. AE training — 25 optimizer steps from random init; losses match baseline.
3. Inference — generate one MCSA example; designed CIF matches baseline.
4. Beam search — unindexed-motif search; per-design motif RMSDs match baseline.

A separate GPU-suite preflight checks that both checkpoints selected by
configs/inference.yaml exist on disk. Inference uses a generate.py subprocess
through a module fixture.

Generate baselines with src/script_utils/generate_e2e_baselines.py and commit
these to tests/e2e/baselines/. Missing training baselines fail; missing inference
or beam-search baselines skip.

Run:
    RUN_GPU_E2E_TESTS=1 pytest tests/e2e/test_gpu_regression.py -m regression -v
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.e2e.helpers.gpu_e2e import (
    ROOT,
    cleanup_smoke_training_store,
    require_gpu_e2e,
    run_cmd,
    run_cmd_with_loss_watcher,
)
from tests.e2e.helpers.regression_baselines import (
    BASELINE_RMSD_ATOL_TIGHT,
    BASELINES_DIR,
    BEAM_SEARCH_OVERRIDES,
    BEAM_SEARCH_RMSD_ATOL,
    INFERENCE_OVERRIDES,
    LOSS_ATOL_MEAN,
    LOSS_ATOL_PER_STEP,
    REGRESSION_EXAMPLE_ID,
    all_atom_rmsd,
    beam_search_motif_rmsds,
    deterministic_env,
    find_beam_search_cifs,
    find_inference_cif,
    load_baseline_losses,
    require_h200,
    require_sota_ae_ckpt,
    require_sota_ckpt,
)

# ── Training helpers ──────────────────────────────────────────────────────────


def _run_training(
    run_name: str,
    script: str,
    overrides: list[str],
    expected_losses: list[float],
) -> None:
    out_dir = ROOT / "store" / run_name
    cleanup_smoke_training_store(run_name)
    env = deterministic_env()
    cmd = (
        [script]
        + overrides
        + [
            f"++out_dir={out_dir}",
            "+single=true",
            "+nolog=true",
            "hardware.ngpus_per_node_=1",
            "hardware.nnodes_=1",
            "opt.steps_per_epoch=25",
            "opt.max_epochs=1",
            "callbacks.coordinated_gc.interval_batches=10",
            "++opt.num_sanity_val_steps=0",
            "log.checkpoint=false",
            "log.log_wandb=false",
            "log.log_every_n_steps=1",
            "seed=42",
            "torch_deterministic=true",
            "dataloaders.num_workers=0",
            "dataloaders.prefetch_factor=null",
            "dataloaders.multiprocessing_context=null",
            "dataloaders.train.dataloader_params.persistent_workers=false",
            "dataloaders.train.dataloader_params.in_order=true",
            # Skip end-of-epoch validation — only training losses are regression-tested.
            "++opt.limit_val_batches=0",
        ]
    )
    run_cmd_with_loss_watcher(
        cmd,
        out_dir=out_dir,
        expected_losses=expected_losses,
        atol=LOSS_ATOL_PER_STEP,
        mean_atol=LOSS_ATOL_MEAN,
        env=env,
    )


# ── Inference helpers ─────────────────────────────────────────────────────────


def _run_inference(run_name: str) -> Path:
    """Run generate.py for one MCSA example; return the out_dir."""
    out_dir = ROOT / "inference" / run_name
    shutil.rmtree(out_dir, ignore_errors=True)

    cmd = [
        "src/rfproteina/generate.py",
        *INFERENCE_OVERRIDES,
        "inference_experiment=inference_on_val_set",
        "+datasets/val@generation=mcsa/mcsa_subset",
        f"generation.dataset.subset_to_keys=[{REGRESSION_EXAMPLE_ID}]",
        "~metrics/refolding_oracles",
        "generation.dataset.num_replicates=1",
        f"run_name_={run_name}",
        f"out_dir={out_dir}",
        "seed=42",
        "torch_deterministic=true",
        # No compilation for regression tests
        "compile_lengths=null",
        "+single=true",
    ]
    proc = run_cmd(cmd, env=deterministic_env())
    assert proc.returncode == 0, f"[{run_name}] inference failed:\n{proc.stderr or proc.stdout}"
    return out_dir


def _run_beam_search(run_name: str) -> Path:
    """Run beam search inference; return the out_dir."""
    out_dir = ROOT / "inference" / run_name
    shutil.rmtree(out_dir, ignore_errors=True)

    cmd = [
        "src/rfproteina/generate.py",
        *BEAM_SEARCH_OVERRIDES,
        f"run_name_={run_name}",
        f"out_dir={out_dir}",
        "+single=true",
    ]
    proc = run_cmd(cmd, env=deterministic_env())
    assert proc.returncode == 0, f"[{run_name}] beam search failed:\n{proc.stderr or proc.stdout}"
    return out_dir


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
def test_sota_ckpt_staged_for_gpu_tests() -> None:
    """The paired SOTA checkpoints are staged before GPU tests run."""
    require_sota_ckpt()
    require_sota_ae_ckpt()


@pytest.mark.gpu
@pytest.mark.regression
@pytest.mark.slow
def test_sota_training_regression() -> None:
    """Check 25 seeded optimizer steps against baseline losses with periodic full GC."""
    require_gpu_e2e(min_gpus=1)
    require_h200()

    expected = load_baseline_losses("sota_training")
    _run_training(
        run_name="regression_sota_training",
        script="src/rfproteina/train.py",
        overrides=[
            # The live recipe, matching generate_e2e_baselines.py. A data change moves these
            # losses on purpose: regenerate the baseline when it does.
            "experiment=sota",
            # No pretrain_ckpt_path: this smoke exercises the sota *recipe* from random init;
            # baselines are anchored to random-init losses (seed=42).
            "nn.use_torch_compile=false",
        ],
        expected_losses=expected,
    )


@pytest.mark.gpu
@pytest.mark.regression
@pytest.mark.slow
def test_ae_training_regression() -> None:
    """Compare current AE losses from random initialization with periodic full GC."""
    require_gpu_e2e(min_gpus=1)
    require_h200()

    expected = load_baseline_losses("ae_training")
    _run_training(
        run_name="regression_ae_training",
        script="src/rfproteina/partial_autoencoder/train.py",
        overrides=[
            # The live recipe, matching generate_e2e_baselines.py.
            "experiment_ae=ae_sota",
            "dataloaders.train.dataloader_params.batch_size=2",
            # No compile for regression tests
            "nn_ae.use_torch_compile=false",
        ],
        expected_losses=expected,
    )


# ── Module fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def inference_out() -> Path:
    """Run generate.py for inference regression; return out_dir."""
    require_gpu_e2e(min_gpus=1)
    require_h200()
    require_sota_ckpt()
    if not (BASELINES_DIR / "inference" / "designed.cif.gz").is_file():
        pytest.skip("Inference baselines not generated yet.")
    return _run_inference("regression_inference")


# ── Inference & beam search tests ─────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.regression
@pytest.mark.slow
def test_inference_regression(inference_out: Path) -> None:
    """Generate M0664_2dhn; designed CIF matches baseline."""
    baseline_dir = BASELINES_DIR / "inference"
    designed = find_inference_cif(inference_out, REGRESSION_EXAMPLE_ID, "generated")
    rmsd_d = all_atom_rmsd(designed, baseline_dir / "designed.cif.gz")
    assert (
        rmsd_d <= BASELINE_RMSD_ATOL_TIGHT
    ), f"Designed structure all-atom RMSD {rmsd_d:.4f} Å exceeds {BASELINE_RMSD_ATOL_TIGHT} Å"


@pytest.mark.gpu
@pytest.mark.regression
@pytest.mark.slow
def test_beam_search_regression() -> None:
    """Minimal beam search on an unindexed motif; per-design motif RMSDs match baseline."""
    require_gpu_e2e(min_gpus=1)
    require_h200()
    require_sota_ckpt()

    baseline_path = BASELINES_DIR / "beam_search" / "metrics.json"
    if not baseline_path.is_file():
        pytest.skip("Beam search baselines not generated yet.")
    expected = json.loads(baseline_path.read_text())["motif_rmsd"]

    out_dir = _run_beam_search("regression_beam_search")
    observed = beam_search_motif_rmsds(out_dir)

    assert len(observed) == len(expected), f"Expected {len(expected)} scored designs, got {observed}"
    for obs, exp in zip(observed, expected, strict=True):
        assert abs(obs - exp) <= BEAM_SEARCH_RMSD_ATOL, (
            f"Beam search motif RMSD {obs:.4f} Å differs from baseline {exp:.4f} Å by more than "
            f"{BEAM_SEARCH_RMSD_ATOL} Å"
        )

    # Final designs keep the checkpoint id they were scored under, rather than being re-decoded
    # and re-scored under a fresh id.
    for cif in find_beam_search_cifs(out_dir):
        assert "_bsc" in cif.name, f"Final beam design was renamed on the way out: {cif.name}"
