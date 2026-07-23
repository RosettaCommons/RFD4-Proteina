# Developer Guide

This guide is for **developers**: setting up a training environment on any cluster, continuing
training of the released model on your own data, and understanding datasets, design tasks, metrics,
and the SLURM launchers.

For inference and design usage, see the [main README](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/README.md). For coding standards, see
[CLAUDE.md](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/CLAUDE.md).

## Table of Contents

1. [Setup (any cluster, no container)](#setup-any-cluster-no-container)
2. [Continue-Training Walkthrough](#continue-training-walkthrough)
3. [Training Essentials](#training-essentials)
4. [Datasets and Dataloaders](#datasets-and-dataloaders)
5. [Design Tasks (Conditional Training)](#design-tasks-conditional-training)
6. [SLURM Job Launching](#slurm-job-launching)
7. [Evaluation and Metrics](#evaluation-and-metrics)
8. [Advanced Topics](#advanced-topics)
9. [Troubleshooting](#troubleshooting)

---

## Setup (any cluster, no container)

Training runs in the same plain `uv` environment as inference — no apptainer, squashfs, or shared
filesystem required. Install and configure exactly as in the [README Setup](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/README.md#setup):

```bash
bash setup.sh                      # choose "5) other" at the cluster prompt
source .venv/bin/activate

export CLUSTER=local
export DATA_PATH="$PWD/data_path"
mkdir -p "$DATA_PATH"
printf 'DATA_PATH=%s\n' "$DATA_PATH" > .ipd/.env.local
```

> **Why `CLUSTER=local` + `.ipd/.env.local`?** Both `train.py` and `generate.py` call
> `load_dotenv()` at startup, which requires an env file for the current `CLUSTER` to exist and
> otherwise raises. It loads with `override=False`, so any variable you export yourself wins. On the
> configured clusters (`digs`, `coreweave`, `gb200`, `ord`) the committed `.ipd/.env.<cluster>` files fill
> in shared paths and the container/squashfs machinery; on a fresh cluster a one-line `local` file
> plus your own exports is all you need.

---

## Continue-Training Walkthrough

This walkthrough finetunes the released denoiser on a tiny, self-contained dataset — enough to
prove the training loop runs end-to-end on a fresh cluster. It uses no squashfs and no container.

### 1. Download the model and the test dataset

`download_example_data.sh` with `--with-test-dataset` pulls both checkpoints (denoiser +
autoencoder) and the ~50-system `test_dataset` (parquet + a plain directory of CIFs) into
`$DATA_PATH`:

```bash
./scripts/download_example_data.sh "$DATA_PATH" --with-test-dataset
```

This lays out:

```
$DATA_PATH/
├── latent_checkpoints/
│   ├── ldm_checkpoints/arch_sweep_2026_07_03_pair_trimult_every2_720k-ema.ckpt   # denoiser
│   └── ae_checkpoints/AE_ucond_512.ckpt                                          # autoencoder
└── test_dataset/
    ├── test_systems.parquet
    └── cifs/systems/<xx>/<id>/<id>.cif
```

The autoencoder is required for latent-diffusion training and is read from
`$DATA_PATH/latent_checkpoints/ae_checkpoints/AE_ucond_512.ckpt` by default.

> **CCD / PDB mirrors (optional).** The tutorial needs neither. For broader work that parses
> arbitrary ligands or PDB entries, a CCD mirror is published at
> `https://YOUR_DATASET_STORE/train_datasets/ccd_mirror/2026_01_06_ccd/2026_01_06_ccd.sqfs`
> (a squashfs — mount it or `unsquashfs` it and point `CCD_MIRROR_PATH` at the directory). The full
> PDB mirror is too large to host; fetch structures from [RCSB](https://www.rcsb.org/) or point
> `PDB_MIRROR_PATH` at your own copy.

### 2. Run the finetune

Pass the released denoiser (downloaded to `$DATA_PATH` above) as `pretrain_ckpt_path` — it's the same
checkpoint inference uses, named in [`configs/inference.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/inference.yaml):

```bash
python src/rfproteina/train.py \
    experiment=tutorial/finetune_test_dataset \
    pretrain_ckpt_path=$DATA_PATH/latent_checkpoints/ldm_checkpoints/arch_sweep_2026_07_03_pair_trimult_every2_720k-ema.ckpt \
    +single=true +nolog=true
```

- **`experiment=tutorial/finetune_test_dataset`** wires the tiny dataset ([`configs/experiment/tutorial/finetune_test_dataset.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/experiment/tutorial/finetune_test_dataset.yaml)),
  single GPU, and short epochs. Validation runs the checked-in `val/unconditional` set (unconditional
  monomer generation) with the AF3 refolding tag dropped, so it steps and validates without a container.
- **`pretrain_ckpt_path`** starts a new run from the released weights (resets the optimizer/scheduler/
  step/epoch); use `resume_ckpt_path=<ckpt>` to restore full Lightning state instead. The SOTA
  checkpoint lives only in `inference.yaml` — advance it there and this command follows.
- **`+single=true`** drops the SLURM/DDP plugins; **`+nolog=true`** disables WandB and checkpoint
  writing.

You should see the model build (~169M trainable + a frozen ~256M autoencoder), the pretrained
checkpoint load, and training steps run over the tiny dataset. That confirms the data pipeline,
autoencoder, and model all work on your cluster.

> **Architecture must match the checkpoint.** `train.py` builds the model from the `nn` config, not
> from the checkpoint, so the two must agree. The released denoiser uses a triangle-multiplication
> pair-update architecture that the default `nn` config does not — so the tutorial experiment
> selects a matching one ([`configs/nn/local_latents_score_nn_160M_atom37_trimult.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/nn/local_latents_score_nn_160M_atom37_trimult.yaml),
> extracted from the checkpoint's own hyper-parameters) via `override /nn`. To finetune a different
> checkpoint, select (or add) the `nn` config it was trained with; a mismatch surfaces as
> `size mismatch` / `Missing key(s)` errors on `load_state_dict`.

> **Tip:** the tutorial disables `torch.compile` for a fast first run; re-enable it
> (`nn.use_torch_compile=True compile_autoencoder=True`) for real training. Add
> `opt.steps_per_epoch=3 opt.max_epochs=1` to stop after just a few steps.

### How the no-squashfs dataset works

The production `plinder` dataset reads CIFs from a squashfs mount
(`${SQUASH_MOUNT_PATH,/squash}/plinder`). The tutorial dataset
([`configs/datasets/train/test_dataset.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/datasets/train/test_dataset.yaml)) instead
points `loader.base_path` directly at `${DATA_PATH}/test_dataset/cifs`, a plain directory. Each
parquet `path` row (`systems/<xx>/<id>/<id>.cif`) resolves against that base — no mounting required.
To build your own dataset the same way, mirror this config with your parquet + CIF directory.

### With or without the apptainer shebang

`train.py` and `generate.py` have a shebang that, when the file is *executed* (`./train.py ...`),
re-runs it inside an apptainer container. Invoking them as an argument to Python
(`python src/rfproteina/train.py ...`, as above) bypasses the shebang entirely and runs in your
`uv` venv. On an configured cluster with the SIF present, the shebang path also works; on a fresh cluster,
always use the `python <script>` form.

---

## Training Essentials

### Entry points

| Model | Script | Config name |
| --- | --- | --- |
| Latent diffusion (main model) | `src/rfproteina/train.py` | `training_proteina` |
| Autoencoder | `src/rfproteina/partial_autoencoder/train.py` | `training_ae` |

```bash
# Diffusion, quick single-GPU test
python src/rfproteina/train.py experiment=<name> +single=true +nolog=true

# Autoencoder
python src/rfproteina/partial_autoencoder/train.py --config-name training_ae
```

### Config composition

`configs/training_proteina.yaml` composes dataloaders, model (`nn/…`), metrics, callbacks, and an
`experiment`, then an optional `debug` group. Experiments live in `configs/experiment/` (e.g.
`sota.yaml`, `unconditional_afdb_256.yaml`, `tutorial/finetune_test_dataset.yaml`, plus
user-namespaced folders). `datasets.train` is mandatory and supplied by the experiment.

### Checkpoint loading (mutually exclusive)

| Key | Behavior |
| --- | --- |
| `pretrain_ckpt_path` | Start a **new** run from saved weights (resets optimizer/scheduler/step/epoch/RNG). Use for finetuning on new data. |
| `resume_ckpt_path` | **Resume** an interrupted run (restores full Lightning state). An existing `last.ckpt` in the run dir is auto-detected. |

Two knobs apply only on the `pretrain_ckpt_path` path (see the commented examples in
`configs/training_proteina.yaml`):

- **`weight_loading`** (`rfproteina.utils.weights.WeightLoadingConfig`) — per-parameter copy /
  reinit / zero-init / copy-and-zero-pad policies. Null → `strict=False`.
- **`parameter_freezing`** (`rfproteina.utils.weights.ParameterFreezingConfig`) — freeze subsets of
  parameters (e.g. the trunk).

### Minimal / debug flags

- `+single=true` — single GPU, no SLURM/DDP.
- `+nolog=true` — no WandB, no checkpoint writing.
- `debug=gpu` or `debug=cpu` — preset short-run configs (small `steps_per_epoch`, `num_workers=0`,
  progress bar; see `configs/debug/`).
- Epoch length is `opt.steps_per_epoch` (not the full dataset).

---

## Datasets and Dataloaders

Dataset configs live under `configs/datasets/train/` (and `configs/datasets/val/`). Parquet-backed
sets inherit from `configs/datasets/train/base_pandas.yaml` (`atomworks.ml.datasets.PandasDataset`);
the production `plinder` set is `configs/datasets/train/plinder/plinder.yaml`. All paths use
`${oc.env:DATA_PATH}` (and, for squashfs-backed CIFs, `${oc.env:SQUASH_MOUNT_PATH,/squash}`), so
nothing is hardcoded.

A dataset config specifies:

- `dataset.data` — the parquet index (row metadata + a `path` column for structures).
- `dataset.loader.base_path` — where the CIFs live (a squashfs mount, or a plain dir as in
  `test_dataset.yaml`).
- `dataset.transform.design_tasks` — a design-task preset (see below).
- `weights` — optional sampling weights (e.g. inverse cluster size).

To add a dataset, copy an existing config, point `data`/`base_path` at your files, choose a
design-task preset, and reference it from an experiment via
`- /datasets/train/<your_dataset>@datasets.train.<key>`.

---

## Design Tasks (Conditional Training)

Design tasks control **what information the model sees** during training, enabling conditional
generation. They are configured per dataset as relative-weight frequencies via a preset from
`configs/design_tasks/presets/` (e.g. `plinder_tasks`, `unconditional_only`, `folding_only`,
`motif_pretraining`, …).

| Task | Sees | Produces | Use case |
| --- | --- | --- | --- |
| `unconditional` | Noise | Structure + sequence | General generation |
| `folding` | Sequence | Structure | Structure prediction |
| `inverse_folding` | Structure | Sequence | Sequence design |
| `motif_scaffold` | Partial structure | Complete protein | Scaffolding |
| `ppi_hotspot_*` | Target + hotspots | Binder | Protein binder design |
| `tip_atom*` | Ligand contact residues | Protein | Ligand-site design |

Override a frequency temporarily on the CLI, e.g.:

```bash
python src/rfproteina/train.py experiment=<name> \
    datasets.train.<key>.dataset.transform.design_tasks.unconditional.frequency=80.0
```

---

## SLURM Job Launching

> The `scripts/launch_*.sh` launchers target the configured clusters and require their container/SLURM
> infrastructure (apptainer or docker/enroot, `.ipd/.env.<cluster>`, and a per-user
> `scripts/.user_info_<cluster>`). On a fresh cluster without that stack, run `train.py`/`generate.py`
> directly as shown above.

Each cluster needs a `scripts/.user_info_<cluster>` file (copy the matching
`scripts/.user_info_<cluster>_example` template) holding SLURM settings. `setup.sh` creates one when
you pick a known cluster. Key launchers:

| Workflow | Script |
| --- | --- |
| Train latent diffusion | `scripts/launch_train_latents.sh <cluster> [hydra args]` |
| Train autoencoder | `scripts/launch_train_ae.sh <cluster>` |
| Inference / offline validation | `scripts/launch_infer.sh <cluster> [--njobs N] [hydra args]` |
| Parameter sweeps | `scripts/launch_sweep.sh <cluster> --sweep-config <yaml>` |
| Reward-guided search | `scripts/launch_search.sh <cluster>` |

`launch_infer.sh` splits inputs across job-array tasks (`--njobs`), then a dependent job
concatenates per-job metric CSVs (Foldseek clustering on by default; `--no-foldseek` to disable).
Pass any trailing `key=value` args straight through to the underlying script. To load a config from
outside the repo tree, pass `--config-path <dir> --config-name <stem>` (the config must inherit the
base config and add a `hydra.searchpath` entry — see `configs/inference_experiment/TEMPLATE.yaml`).
For offline validation on a val set, run it with `inference_experiment=inference_on_val_set
+datasets/val@generation=<val_dataset>` (val configs live in `configs/datasets/val/`).

---

## Evaluation and Metrics

Metrics subclass `rfproteina.metrics.metric.Metric` and score generated structures. They run inline
during inference (`metrics_tags=[...]`) and during training validation. Refolding-based metrics use
the **AF3 oracle**, which needs an apptainer (see the [README](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/README.md#refolding--search-requires-af3)).
See the [metrics reference](metrics.md) for the full reference of what's already enabled and its tags.

**Offline evaluation** of already-generated structures (Hydra-based, like `generate.py`):

```bash
python src/rfproteina/evaluate.py \
    inputs=/path/to/structures metrics_tags='[unconditional_refolding]' \
    out_dir=/path/to/eval_output
```

`out_dir` is mandatory here (unlike `generate.py`, which derives one) — the metrics CSVs are
written under `<out_dir>/evaluation_metrics/`.

**Custom metrics/rewards** — add Python metric classes under `src/rfproteina/metrics/custom_metrics/`
and register them in `configs/metrics/custom_metrics/` (both git-ignored; `setup.sh` scaffolds the
entry config). Search rewards are configured inline in the search experiment config (see
`configs/inference_experiment/inference_beam_example.yaml`). See
the [custom metrics guide](custom_metrics.md) for the full walkthrough.

---

## Advanced Topics

**LoRA finetuning** — set a positive `lora.r` in the config to enable low-rank adaptation
(`lora_alpha`, `lora_dropout`, `train_bias`, `exclude_keys`); null `r` = full-model training.

**Flow matching** — modality-specific time distributions control the diffusion schedule, with
separate parameters for `bb_ca` (backbone) and `local_latents` (sidechain); see the `loss.t_distribution`
block in `configs/training_proteina.yaml`.

**EMA** — an exponential-moving-average callback (`rfproteina.utils.simple_ema.SimpleEMA`)
maintains averaged weights; the released denoiser is an EMA checkpoint.

---

## Troubleshooting

| Issue | Solution |
| --- | --- |
| `No .env file found for cluster '<x>'` | `export CLUSTER=local` and create `.ipd/.env.local` (see [Setup](#setup-any-cluster-no-container)) |
| `No autoencoder checkpoint path provided` / AE not found | Download it (`download_example_data.sh`) or set `autoencoder_ckpt_path=` |
| Parquet / CIF not found | Check `DATA_PATH`; for custom data confirm `loader.base_path` + the parquet `path` column resolve |
| Training not resuming | Ensure a `last.ckpt` exists in the run's `checkpoints/`, or pass `resume_ckpt_path=` |
| Out of memory | Lower `dataloaders.batch_size` or `opt.accumulate_grad_batches` |
| Segfault in rdkit/posebusters | uv's Python needs `LD_PRELOAD` set to the bundled `libstdc++.so.6` (see `Makefile`; `make test` sets it) |
| SLURM launcher fails on a fresh cluster | The launchers need the container/SLURM stack — run `python src/rfproteina/train.py ...` directly instead |
