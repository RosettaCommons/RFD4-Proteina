# GPU end-to-end tests

Thin test modules + fat helpers that exercise the real training and inference
entrypoints (`proteinfoundation/train.py`, `proteinfoundation/partial_autoencoder/train.py`,
`proteinfoundation/generate.py`) as subprocesses on GPUs. Modeled on
`protein-foundation-models/tests/e2e`.

## What runs

| Test | Marker(s) | What it does |
|------|-----------|--------------|
| `test_gpu_training_sota.py` | `gpu, e2e, training_smoke, slow` | Flow model: `experiment=sota` for 10 steps + 2 prevalidation steps, 2 GPU, batch size 2. Variants: `muon_nocompile_rf3`, `muon_compile_af3`. |
| `test_gpu_training_ae.py` | `gpu, e2e, training_smoke, slow` | Autoencoder: `experiment_ae=smoke` for 10 steps + 2 sanity val steps, 2 GPU, batch size 2. Variants: `ae_muon_nocompile`, `ae_adam_nocompile`, `ae_muon_compile`, `ae_adam_compile`. |
| `test_gpu_inference_mcsa.py` | `gpu, e2e, slow` | Offline MCSA-subset validation + minimal beam search via the SOTA checkpoint. Checkpoint-gated. |
| `test_slurm_launchers.py` | `cluster` | `bash -n` syntax check of SLURM launcher scripts including `launch_train_ae.sh` and `launch_infer_ae.sh` (no GPU, no job submission). |

GPU CI splits prevalidation oracle coverage across the two training smokes
(RF3 on `muon_nocompile_rf3`, AF3 on `muon_compile_af3`). Inference smokes still
pin RF3. See `tests/test_oracles.py` for full oracle unit/integration tests.

## Gating

All GPU e2e tests skip unless explicitly enabled:

```bash
export RUN_GPU_E2E_TESTS=1          # required; helpers skip otherwise
export GPU_COUNT=2                  # or detected via nvidia-smi
export DATA_PATH=/path/to/data
# Inference/search only (else those two tests skip).
# CI: set SOTA_CKPT_PATH via workflow/repository variable when the hero ckpt is available.
export SOTA_CKPT_PATH=/abs/path/to/sota.ckpt
export RF3_CKPT_PATH=/abs/path/to/rf3.ckpt   # optional; default under $DATA_PATH/rf3_checkpoints/
```

For the `muon_compile_af3` training smoke, AF3 prevalidation also needs the cluster
AF3 env vars (defaults in `.ipd/.env.digs` on digs):

```bash
export AF3_APPTAINER_PATH=/path/to/shared/storage
export AF3_SCRIPTS_DIR=/path/to/shared/storage
export AF3_JAX_CACHE_DIR=/path/to/data
export AF3_MODEL_DIR=/path/to/shared/storage
# Host-side daemons (outside latent-dev); training talks to them via:
export AF3_LAUNCHER_SOCKET_TEMPLATE=/tmp/af3_${SLURM_JOB_ID:-$$}/af3_daemon_rank{rank}.sock
```

On digs CI, `run_gpu_tests.sh` sources `.ipd/.env.digs`, launches AF3 daemons on the
compute node with host `/usr/bin/python3`, and passes `AF3_LAUNCHER_SOCKET_TEMPLATE`
plus the `AF3_*` paths into the latent-dev apptainer. Locally, source the same env
file and start daemons before pytest:

```bash
source .ipd/.env.digs   # from repo root
eval "$(/usr/bin/python3 projects/proteinfoundation/scripts/af3_daemon.py launch \
  --num-gpus 2 --log-dir /tmp/af3_e2e --background)"
# then run pytest inside latent-dev with AF3_LAUNCHER_SOCKET_TEMPLATE exported
```

If AF3 is unavailable, `muon_compile_af3` skips; `muon_nocompile_rf3` still runs.

Single-GPU boxes can force the 2-GPU tests to run with
`ALLOW_SINGLE_GPU_E2E=1` (DDP world size still follows `hardware.ngpus_per_node_`).

## Running locally (inside the latent-dev apptainer)

```bash
cd projects/proteinfoundation
RUN_GPU_E2E_TESTS=1 GPU_COUNT=2 DATA_PATH=$DATA_PATH \
  pytest -m "gpu or e2e or training_smoke" -v tests/e2e

# Just the flow training smokes:
RUN_GPU_E2E_TESTS=1 pytest -m training_smoke -v tests/e2e/test_gpu_training_sota.py

# Just the AE training smokes:
RUN_GPU_E2E_TESTS=1 pytest -m training_smoke -v tests/e2e/test_gpu_training_ae.py

# Single flow variant:
pytest -v "tests/e2e/test_gpu_training_sota.py::test_sota_training_smoke_2gpu[muon_nocompile_rf3]"

# Single AE variant:
pytest -v "tests/e2e/test_gpu_training_ae.py::test_ae_training_smoke_2gpu[ae_muon_nocompile]"
```

## CI

Pipeline (`.github/workflows/lint_and_test.yaml`): **lint** (ruff on changed files) → **CPU**
(proteinfoundation / rf3 / foundry matrix, xdist) → **GPU** (2-GPU srun training smokes +
inference/regression).

Both CPU and GPU lanes source [`.ipd/.env.digs`](../../../.ipd/.env.digs) inside their
run scripts and use the same **`LATENT_APPTAINER_SIF`** plus **`PDB_MIRROR_PATH`** /
**`CCD_MIRROR_PATH`**. GPU additionally passes **`DATA_PATH`** explicitly into apptainer.

| Lane | Script | Pytest marker filter |
|------|--------|---------------------|
| CPU | `.github/ci/run_tests.sh` | `not gpu/e2e/training_smoke/benchmark/very_slow` (+ `@cluster` launcher checks) |
| GPU | `.github/ci/run_gpu_tests.sh` | `gpu or e2e or training_smoke` |

**Compile coverage** (6 training smokes total): each model runs Muon/Adam ×
compile/no-compile. Compile-on variants assert subprocess output contains
`Enabling torch.compile` in addition to finite loss:

| Compile on | Flow | AE |
|------------|------|-----|
| Muon | `muon_compile_af3` (skips if AF3 unavailable) | `ae_muon_compile` |
| Adam | — | `ae_adam_compile` |

Flow smokes inherit fixed train padding (I=256) from `experiment=sota`; AE smokes use
`fixed_dimensions: {I: 256}` in `experiment_ae/smoke`.

**Known skips (GPU job can still pass green):** missing `SOTA_CKPT_PATH` (MCSA/beam search),
missing AF3 env/daemons (`muon_compile_af3`), optional metric deps. Override fail-fast
with `GPU_PYTEST_EXTRA_ARGS=""` when debugging multiple failures.

These run in the `gpu` job of `.github/workflows/lint_and_test.yaml`
(ordering: lint -> cpu test -> gpu). The workflow wraps the whole script in
`srun --gres=gpu:2 -p gpu-train ... bash .github/ci/run_gpu_tests.sh`, so
apptainer (and the overlay squash stub it creates) run on the allocated compute
node -- mirroring the cpu lane and digs' `exec_core.sh`. The CPU lane excludes
`gpu/e2e/training_smoke` markers; the cheap `cluster` launcher syntax checks
(no GPU) run in the CPU lane.

See [docs/testing/GPU_E2E_DATA_REQUIREMENTS.md](../../docs/testing/GPU_E2E_DATA_REQUIREMENTS.md)
for data/checkpoint requirements.

## PR summary (gpu-ci vs latent-main)

This branch adds a **3-stage CI pipeline** on digs: lint (changed files) → CPU pytest
matrix → single 2-GPU job. Tests split by markers: CPU runs everything except
`gpu` / `e2e` / `training_smoke` / `very_slow`; GPU runs `gpu or e2e or training_smoke`.

**Training oracle matrix** (prevalidation coverage split across two smokes):

| Variant | Optimizer | Compile | Oracle |
|---------|-----------|---------|--------|
| `muon_nocompile_rf3` | Muon | off | RF3 |
| `muon_compile_af3` | Muon | on | AF3 (host-side daemons in CI) |

**Known skips:** missing `SOTA_CKPT_PATH` (inference/beam search), AF3 unavailable
(`muon_compile_af3`), missing data files. Set `SOTA_CKPT_PATH` via GitHub Actions
variable on digs when the hero checkpoint is available.

**RF3 regression baselines** refreshed after `remove_ccds` parsing fix and checkpoint
pinning (`RF3_REGRESSION_CKPT_PATH`); `RMSD_TOLERANCE` raised to 0.45Å for 8vkf
run-to-run variance. Refresh command documented in
`models/rf3/tests/test_inference_regression.py`.

**Submodule:** `lib/atomworks` bumped to `f034cf01` (required for GPU lane imports).

## Tuning knobs (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `GPU_E2E_NGPUS` | 2 | GPUs per training/inference run |
| `GPU_E2E_TRAIN_BATCH_SIZE` | 2 | batch size (flow: `dataloaders.batch_size`; AE: `dataloaders.train.dataloader_params.batch_size`) |
| `GPU_E2E_TRAIN_STEPS` | 10 | `opt.steps_per_epoch` (one epoch) |
| `GPU_E2E_PREVAL_STEPS` | 2 | sanity / prevalidation batches |
| `GPU_E2E_STREAM_OUTPUT` | 0 | set to `1` to tee train/infer subprocess logs to the terminal (debugging) |
| `RF3_CKPT_PATH` | — | abs RF3 oracle `.ckpt`; else `$DATA_PATH/rf3_checkpoints/$RF3_CKPT_NAME` |
| `RF3_CKPT_NAME` | `rf3-with-confidence-xaira-epoch-0711-fixed.ckpt` | basename under `rf3_checkpoints/` |
| `AF3_APPTAINER_PATH` | digs: `/path/to/shared/storage` | AF3 apptainer image (`muon_compile_af3` only) |
| `AF3_SCRIPTS_DIR` | digs: `/path/to/shared/storage` | must contain `run_alphafold_from_cif.py` |
| `AF3_JAX_CACHE_DIR` | digs: `/path/to/data` | JAX compilation cache for AF3 daemon |
| `AF3_MODEL_DIR` | digs: `/path/to/shared/storage` | AF3 model weights |
| `AF3_LAUNCHER_SOCKET_TEMPLATE` | set by `af3_daemon.py launch` | Unix socket template for daemon route (required at train time) |
