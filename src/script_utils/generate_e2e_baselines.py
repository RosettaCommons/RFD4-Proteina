#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../.ipd/shebang/latent_exec.sh" "$0" "$@"'
"""Generate regression-test baselines for the four GPU regression tests.

Run this script on DIGS, on a GPU node with the squashfs datasets mounted
(inside apptainer, or with `.ipd/.env.digs` sourced):

    cd /path/to/rfproteina   # repo root
    python src/script_utils/generate_e2e_baselines.py

By default the inference/beam baselines use the SOTA checkpoint from configs/inference.yaml
(the single source of truth, shared with generate.py). Pass ``--sota-ckpt`` to regenerate them
against a candidate checkpoint before promoting it in inference.yaml.

For each test this script:
  1. Runs the subprocess twice (same seed, same command).
  2. Asserts outputs match (reproducibility check).
  3. Writes the baseline files to tests/e2e/baselines/<name>/.

After running, commit the updated baselines:
    git add tests/e2e/baselines/
    git commit -m "test: update regression baselines for new SOTA checkpoint"

Options:
    --tests NAMES   Comma-separated subset to regenerate (default: all).
                    Choices: sota_training,ae_training,inference,beam_search
    --sota-ckpt     Optional candidate checkpoint; default = inference.yaml's SOTA ckpt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import rootutils

from tests.e2e.helpers.gpu_e2e import parse_train_losses_from_csv_logger
from tests.e2e.helpers.regression_baselines import (
    BASELINES_DIR,
    BEAM_SEARCH_EXAMPLE,
    LOSS_ATOL_MEAN,
    LOSS_ATOL_PER_STEP,
    REGRESSION_EXAMPLE_ID,
    SOTA_CKPT_PATH,
    all_atom_rmsd,
    deterministic_env,
    find_beam_search_cifs,
    find_inference_cif,
)

# ── Repository / path constants ──────────────────────────────────────────────

# ROOT is the repo root (cwd for all subprocess calls).
ROOT = Path(rootutils.find_root(indicator=".project-root", search_from=__file__))
PYTHON = sys.executable

# ── Subprocess helpers ────────────────────────────────────────────────────────


def _run(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd[:4])} ...")
    print(f"{'='*60}")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=deterministic_env(),
        capture_output=False,  # show stdout/stderr live for monitoring
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")
    return proc


def _run_captured(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    """Run capturing output (for parsing losses from stdout/stderr)."""
    print(f"\n  {label} [capturing output] ...")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=deterministic_env(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout)
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")
    return proc


# ── CIF / metrics helpers — imported from tests.e2e.helpers.regression_baselines ──


# ── Reproducibility check ─────────────────────────────────────────────────────


def _assert_losses_reproducible(losses_a: list[float], losses_b: list[float], label: str) -> None:
    if len(losses_a) != len(losses_b):
        raise RuntimeError(
            f"[{label}] Run 1 produced {len(losses_a)} losses, " f"run 2 produced {len(losses_b)} — not reproducible!"
        )
    diffs = [abs(a - b) for a, b in zip(losses_a, losses_b, strict=False)]
    max_diff = max(diffs) if diffs else 0.0
    if max_diff > LOSS_ATOL_PER_STEP:
        bad = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(losses_a, losses_b, strict=False))
            if abs(a - b) > LOSS_ATOL_PER_STEP
        ]
        msg = "; ".join(f"step {i}: {a:.6f} vs {b:.6f}" for i, a, b in bad[:5])
        raise RuntimeError(f"[{label}] Training NOT reproducible! max diff={max_diff:.2e}. {msg}")
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    if mean_diff > LOSS_ATOL_MEAN:
        raise RuntimeError(
            f"[{label}] Training NOT reproducible! mean diff={mean_diff:.4f} exceeds {LOSS_ATOL_MEAN} "
            f"(per-step max {max_diff:.2e} was within {LOSS_ATOL_PER_STEP})"
        )
    print(f"  ✓ [{label}] Losses reproducible (max_diff={max_diff:.2e}, mean_diff={mean_diff:.4f})")


def _assert_cif_reproducible(
    path_a: Path,
    path_b: Path,
    label: str,
    atol: float = 0.05,
) -> None:
    rmsd = all_atom_rmsd(path_a, path_b)
    if rmsd > atol:
        raise RuntimeError(f"[{label}] CIF NOT reproducible! all-atom RMSD={rmsd:.4f} Å > {atol} Å")
    print(f"  ✓ [{label}] CIF reproducible (all-atom RMSD={rmsd:.4f} Å)")


# ── Per-test generators ───────────────────────────────────────────────────────


def generate_training_baseline(
    name: str,
    script: str,
    extra_overrides: list[str],
    out_dir: Path,
) -> None:
    base_cmd = [
        PYTHON,
        script,
        *extra_overrides,
        "+single=true",
        "+nolog=true",
        "hardware.ngpus_per_node_=1",
        "hardware.nnodes_=1",
        "opt.steps_per_epoch=25",
        "opt.max_epochs=1",
        "++opt.num_sanity_val_steps=0",
        "log.checkpoint=false",
        "log.log_wandb=false",
        "log.log_every_n_steps=1",
        "seed=42",
        "torch_deterministic=true",
        "++opt.limit_val_batches=0",  # skip val dataloader (matches the test)
        # num_workers=0: parallel workers deliver batches in OS-scheduler-dependent
        # order; two runs with the same seed see different data.  Single-threaded
        # loading is the only way to guarantee identical batches across runs.
        "dataloaders.num_workers=0",
        "dataloaders.prefetch_factor=null",
        "dataloaders.multiprocessing_context=null",
        "dataloaders.train.dataloader_params.persistent_workers=false",
    ]

    # Run 1
    run1_out = out_dir.parent / (out_dir.name + "_run1")
    shutil.rmtree(run1_out, ignore_errors=True)
    _run_captured([*base_cmd, f"++out_dir={run1_out}"], f"{name} run 1")
    losses_1 = parse_train_losses_from_csv_logger(run1_out)
    if not losses_1:
        raise RuntimeError(f"[{name}] No losses parsed from run 1 CSV logger output")

    # Run 2 (verify reproducibility)
    run2_out = out_dir.parent / (out_dir.name + "_run2")
    shutil.rmtree(run2_out, ignore_errors=True)
    _run_captured([*base_cmd, f"++out_dir={run2_out}"], f"{name} run 2 (reproducibility check)")
    losses_2 = parse_train_losses_from_csv_logger(run2_out)
    if not losses_2:
        raise RuntimeError(f"[{name}] No losses parsed from run 2 CSV logger output")

    _assert_losses_reproducible(losses_1, losses_2, name)

    # Write baseline
    baseline_file = BASELINES_DIR / name / "losses.json"
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.write_text(
        json.dumps({"seed": 42, "n_steps": len(losses_1), "n_gpus": 1, "losses": losses_1}, indent=2)
    )
    print(f"  ✓ [{name}] Wrote {baseline_file} ({len(losses_1)} steps)")

    # Clean up
    shutil.rmtree(run1_out, ignore_errors=True)
    shutil.rmtree(run2_out, ignore_errors=True)


def _ckpt_overrides(sota_ckpt: Path | None) -> list[str]:
    """generate.py ckpt overrides for a --sota-ckpt override, or [] to use inference.yaml's default."""
    if sota_ckpt is None:
        return []
    return [f"ckpt_path={sota_ckpt.parent}", f"ckpt_name={sota_ckpt.name}"]


def generate_inference_baseline(sota_ckpt: Path | None) -> None:
    """Generate inference baseline: run generate.py twice, verify reproducibility, save designed.cif."""
    name = "inference"

    gen_cmd = [
        PYTHON,
        "src/rfproteina/generate.py",
        "inference_experiment=inference_on_val_set",
        "+datasets/val@generation=mcsa/mcsa_subset",
        f"generation.dataset.subset_to_keys=[{REGRESSION_EXAMPLE_ID}]",
        *_ckpt_overrides(sota_ckpt),  # empty => inference.yaml's default SOTA ckpt
        "~metrics/refolding_oracles",
        "generation.dataset.num_replicates=1",
        "compile_lengths=null",  # no compile in regression (inference config has no compile_autoencoder key)
        "seed=42",
        "torch_deterministic=true",
        "+single=true",
    ]

    run1_dir = ROOT / "inference" / f"{name}_run1"
    shutil.rmtree(run1_dir, ignore_errors=True)
    _run([*gen_cmd, f"run_name_={name}_run1", f"out_dir={run1_dir}"], f"{name} run 1")
    designed_1 = find_inference_cif(run1_dir, REGRESSION_EXAMPLE_ID, "generated")

    run2_dir = ROOT / "inference" / f"{name}_run2"
    shutil.rmtree(run2_dir, ignore_errors=True)
    _run([*gen_cmd, f"run_name_={name}_run2", f"out_dir={run2_dir}"], f"{name} run 2 (reproducibility check)")
    designed_2 = find_inference_cif(run2_dir, REGRESSION_EXAMPLE_ID, "generated")

    _assert_cif_reproducible(designed_1, designed_2, f"{name}/designed", atol=0.05)

    # Write baseline
    bl_dir = BASELINES_DIR / name
    bl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(designed_1, bl_dir / "designed.cif")
    print(f"  ✓ [{name}] Wrote baselines to {bl_dir}")

    shutil.rmtree(run1_dir, ignore_errors=True)
    shutil.rmtree(run2_dir, ignore_errors=True)


def generate_beam_search_baseline(sota_ckpt: Path | None) -> None:
    name = "beam_search"

    base_cmd = [
        PYTHON,
        "src/rfproteina/generate.py",
        "inference_experiment=inference_beam_regression",
        "~metrics/refolding_oracles",
        *_ckpt_overrides(sota_ckpt),  # empty => inference.yaml's default SOTA ckpt
        "generation.args.nsteps=20",
        "compile_lengths=null",  # no compile in regression (inference config has no compile_autoencoder key)
        "search.strategy.step_checkpoints=[0,10,20]",
        "search.strategy.n_branch=2",
        "search.strategy.beam_width=2",
        f"subset_to_examples={BEAM_SEARCH_EXAMPLE}",
        # Pin val_set_name_override so the hash prefix in output filenames is
        # stable across machines and mount points.
        "generation.val_set_name_override=beam_example",
        "seed=42",
        "torch_deterministic=true",
        "+single=true",
    ]

    # Run 1
    run1_dir = ROOT / "inference" / f"{name}_run1"
    shutil.rmtree(run1_dir, ignore_errors=True)
    _run([*base_cmd, f"run_name_={name}_run1", f"out_dir={run1_dir}"], f"{name} run 1")
    cifs_1 = find_beam_search_cifs(run1_dir)

    # Run 2 (reproducibility check — every CIF must match its run-1 counterpart)
    run2_dir = ROOT / "inference" / f"{name}_run2"
    shutil.rmtree(run2_dir, ignore_errors=True)
    _run([*base_cmd, f"run_name_={name}_run2", f"out_dir={run2_dir}"], f"{name} run 2 (reproducibility check)")
    cifs_2 = find_beam_search_cifs(run2_dir)

    if len(cifs_1) != len(cifs_2):
        raise RuntimeError(f"[{name}] Run 1 produced {len(cifs_1)} CIFs, run 2 produced {len(cifs_2)}")
    for c1, c2 in zip(cifs_1, cifs_2, strict=False):
        _assert_cif_reproducible(c1, c2, f"{name}/{c1.name}", atol=0.05)

    # Write baselines — store every generated CIF with its original filename.
    bl_dir = BASELINES_DIR / name
    bl_dir.mkdir(parents=True, exist_ok=True)
    # Remove any stale baseline CIFs before writing fresh ones.
    for stale in bl_dir.glob("*.cif"):
        stale.unlink()
    for cif in cifs_1:
        shutil.copy2(cif, bl_dir / cif.name)

    (bl_dir / "metrics.json").write_text(json.dumps({"seed": 42, "n_gpus": 1, "n_cifs": len(cifs_1)}, indent=2))
    print(f"  ✓ [{name}] Wrote {len(cifs_1)} CIF(s) + metrics.json to {bl_dir}")

    shutil.rmtree(run1_dir, ignore_errors=True)
    shutil.rmtree(run2_dir, ignore_errors=True)


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_TESTS = ["sota_training", "ae_training", "inference", "beam_search"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tests",
        default=",".join(ALL_TESTS),
        help="Comma-separated list of tests to generate baselines for.",
    )
    parser.add_argument(
        "--sota-ckpt",
        type=Path,
        default=None,
        help="Optional: regenerate inference/beam baselines against this checkpoint instead of "
        "inference.yaml's default SOTA (for validating a candidate before promoting it).",
    )
    args = parser.parse_args()

    tests = [t.strip() for t in args.tests.split(",")]
    unknown = [t for t in tests if t not in ALL_TESTS]
    if unknown:
        parser.error(f"Unknown test(s): {unknown}. Valid: {ALL_TESTS}")

    # Effective ckpt: the --sota-ckpt override, else inference.yaml's default (SOTA_CKPT_PATH).
    # generate.py inherits inference.yaml's default when no override is passed; we still validate
    # existence up front so a missing file fails fast instead of mid-run.
    needs_sota = "inference" in tests or "beam_search" in tests
    effective_ckpt = args.sota_ckpt or SOTA_CKPT_PATH
    if needs_sota and effective_ckpt is None:
        sys.exit("DATA_PATH is not set; pass --sota-ckpt or export DATA_PATH")
    if needs_sota and not effective_ckpt.is_file():
        sys.exit(f"SOTA checkpoint not found: {effective_ckpt}")

    ckpt_source = "inference.yaml default" if args.sota_ckpt is None else "--sota-ckpt override"
    print(f"\nGenerating baselines for: {tests}")
    print(f"  SOTA ckpt : {effective_ckpt}  ({ckpt_source})")
    print(f"  Output    : {BASELINES_DIR}\n")

    if "sota_training" in tests:
        store_base = ROOT / "store" / "regression_sota_baseline"
        generate_training_baseline(
            name="sota_training",
            script="src/rfproteina/train.py",
            extra_overrides=[
                "experiment=sota",
                # This smoke trains 100 steps from random init -- it exercises the
                # sota *recipe*, not the sota checkpoint (those are covered by the
                # inference/beam baselines).
                # torch.compile selects CUDA kernels at trace time; disable so the
                # forward pass uses the same eager kernels across runs.
                "nn.use_torch_compile=false",
                "compile_autoencoder=false",  # frozen AE encoder also compiles by default
            ],
            out_dir=store_base,
        )

    if "ae_training" in tests:
        store_base = ROOT / "store" / "regression_ae_baseline"
        generate_training_baseline(
            name="ae_training",
            script="src/rfproteina/partial_autoencoder/train.py",
            extra_overrides=[
                "experiment_ae=ae_sota",
                "dataloaders.train.dataloader_params.batch_size=2",
                "+nn_ae.use_torch_compile=false",  # no compile in regression
            ],
            out_dir=store_base,
        )

    if "inference" in tests:
        generate_inference_baseline(sota_ckpt=args.sota_ckpt)

    if "beam_search" in tests:
        generate_beam_search_baseline(sota_ckpt=args.sota_ckpt)

    print("\n✓ Done. Commit the updated baselines:")
    print(f"  git add {BASELINES_DIR}")
    print("  git commit -m 'test: update regression baselines for new SOTA checkpoint'")


if __name__ == "__main__":
    main()
