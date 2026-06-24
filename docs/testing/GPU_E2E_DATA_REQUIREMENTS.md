# GPU e2e data & checkpoint requirements

The GPU e2e suite (`tests/e2e/`) runs real training/inference, so it needs the
SOTA training data and (for inference) a checkpoint. On the digs CI runner these
come from the cluster env file `.ipd/.env.digs` (sourced by
`.github/ci/run_gpu_tests.sh`) and the squashfs mounts it defines.

## Container & runtime

| Asset | Source |
|-------|--------|
| Apptainer SIF | `LATENT_APPTAINER_SIF` in `.ipd/.env.digs` |
| Partition / GPUs | `srun --gres=gpu:2 -p gpu-train` |
| `DATA_PATH` | `/path/to/data` (from `.ipd/.env.digs`) |
| PDB/CCD mirrors | `PDB_MIRROR_PATH`, `CCD_MIRROR_PATH` (from `.ipd/.env.digs`) |

## Training smokes (`experiment=sota`)

`sota` is a thin alias of `ncorley/muon_v9_lossweights_25` and uses the
AFDB / OMOL / PLINDER mix. Prevalidation oracle coverage is split across the
four training-smoke variants:

| Variant | Prevalidation oracle |
|---------|---------------------|
| `muon_nocompile_rf3` | RF3 (`metrics/refolding_oracles=rf3`) |
| `muon_compile_af3` | AF3 (`metrics/refolding_oracles=af3`; host-side AF3 daemons in CI) |
| `adam_nocompile_norefolding`, `adam_compile_norefolding` | none (`~metrics.refolding_oracles`) |

| Asset | Path (under `DATA_PATH`) | Mounted at |
|-------|--------------------------|------------|
| AFDB monomer parquet | `train_datasets/afdb_monomer/afdb_monomer.parquet` | (read directly) |
| AFDB monomer squash | `train_datasets/afdb_monomer/data/afdb_512_extracted.squashfs` | `/squash/afdb_monomer` |
| PLINDER parquet | `train_datasets/plinder/2026_05_20_plinder_systems.parquet` | (read directly) |
| PLINDER squash | `train_datasets/plinder/data/2026_05_06_plinder_systems.sqfs` | `/squash/plinder` |
| OMOL | `train_datasets/omol` (LMDB) | (read directly) |
| rf3 oracle ckpt | `rf3_checkpoints/rf3-with-confidence-xaira-epoch-0711-fixed.ckpt` | (read directly; RF3 variant only) |

AF3 prevalidation additionally needs host-side AF3 env vars from
`.ipd/.env.digs`:

| Var | digs default |
|-----|--------------|
| `AF3_APPTAINER_PATH` | `/path/to/shared/storage` |
| `AF3_SCRIPTS_DIR` | `/path/to/shared/storage` |
| `AF3_JAX_CACHE_DIR` | `/path/to/data` |
| `AF3_MODEL_DIR` | `/path/to/shared/storage` |

CI (`run_gpu_tests.sh`) sources those vars, launches AF3 daemons on the compute
node with host `/usr/bin/python3`, passes `AF3_LAUNCHER_SOCKET_TEMPLATE` and the
`AF3_*` paths into latent-dev, and the `muon_compile_af3` variant skips when AF3
is unavailable.

The training tests skip if `train_datasets/afdb_monomer/afdb_monomer.parquet`
is missing (cheapest presence check for the mix).

## AE training smokes (`experiment_ae=smoke`)

`smoke` is aligned with flow smoke under `configs/experiment_ae/smoke.yaml`: AFDB
genie3 monomer training, crop 512, dynamic collate, optional unconditional val.
CI runs Muon/Adam × compile/no-compile on 2 GPUs (no refolding oracle).

| Variant | Optimizer | Compile |
|---------|-----------|---------|
| `ae_muon_nocompile` | Muon (default `muon_param_patterns`) | off |
| `ae_adam_nocompile` | Adam (`opt.muon_param_patterns=[]`) | off |
| `ae_muon_compile` | Muon | on (`nn_ae.use_torch_compile=true`) |
| `ae_adam_compile` | Adam | on |

| Asset | Path (under `DATA_PATH`) |
|-------|--------------------------|
| AFDB monomer parquet | `latent_data/train_datasets/afdb_monomer/afdb_monomer.parquet` |
| AFDB monomer squash | `latent_data/train_datasets/afdb_monomer/data/afdb_512_extracted.squashfs` (mounted at `/squash/afdb_monomer` on ORD) |
| Val benchmark | `benchmarks/monomer.json` (in repo; unconditional val, limited batches in e2e) |

AE smokes skip if `latent_data/train_datasets/afdb_monomer/afdb_monomer.parquet`
is missing.

## Inference / validation / search (checkpoint-gated)

| Asset | How to provide |
|-------|----------------|
| SOTA checkpoint | `SOTA_CKPT_PATH=/abs/file.ckpt` (or `CKPT_PATH` + `SOTA_CKPT_NAME`) |
| MCSA benchmark | `benchmarks/mcsa_40.json` (in repo; used by `mcsa_subset` val config) |
| Search benchmark | `benchmarks/monomer.json` (in repo; used by `inference_test_search`) |

In CI, set `SOTA_CKPT_PATH` via the repository/organization Actions variable when
the digs hero checkpoint is available (e.g.
`/path/to/data`). If unset, the MCSA
validation and search tests skip (training smokes still run).

## Enabling

```bash
export RUN_GPU_E2E_TESTS=1
export GPU_COUNT=2
export DATA_PATH=/path/to/data
export SOTA_CKPT_PATH=/abs/path/to/sota.ckpt   # optional, for inference/search
```
