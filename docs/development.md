# Developer guide

This guide is for **developers**. It covers setting up a training environment on any cluster,
continuing training of the released model on your own data, and understanding datasets, design
tasks, metrics and the SLURM launchers.

For inference and design usage, see the [main
README](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/README.md). For coding
standards, see
[AGENTS.md](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/AGENTS.md).

:::{admonition} On a configured compute cluster?
:class: ipd-cluster
`CLUSTER=digs` is already provisioned. The shared paths in `.ipd/.env.digs` get sourced for you, so
the `CLUSTER=local` setup below is only for machines outside IPD. Drop the `python` and call the
entry scripts directly, as `./src/rfproteina/train.py` and `./src/rfproteina/generate.py`, so their
shebang runs them inside the apptainer image. Do this on an interactive GPU. For multi GPU training
use `scripts/launch_train_latents.sh` for the main model or `scripts/launch_train_ae.sh` for the
autoencoder; both enter the container for you.
:::

## Setup on any cluster, with no container

Training runs in the same plain `uv` environment as inference. No apptainer, no squashfs, no shared
filesystem. Install and configure exactly as the [README
setup](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/README.md#setup) describes.

```bash
bash setup.sh                      # choose "5) other" at the cluster prompt
source .venv/bin/activate

export CLUSTER=local
export DATA_PATH="$PWD/data_path"
mkdir -p "$DATA_PATH"
printf 'DATA_PATH=%s\n' "$DATA_PATH" > .ipd/.env.local
```

> **Why `CLUSTER=local` and `.ipd/.env.local`?** Both `train.py` and `generate.py` call
> `load_dotenv()` at startup, which needs an env file for the current `CLUSTER` to exist and raises
> when it does not. It loads with `override=False`, so any variable you export yourself wins. On the
> configured clusters, meaning `digs`, `coreweave`, `gb200` and `ord`, the committed `.ipd/.env.<cluster>`
> files fill in shared paths and the container and squashfs machinery. On a fresh cluster, a one
> line `local` file plus your own exports is all you need.

## Continue training walkthrough

This walkthrough finetunes the released denoiser on a tiny self contained dataset, which is enough
to prove the training loop runs end to end on a fresh cluster. It uses no squashfs and no
container.

### 1. Download the model and the test dataset

Run `download_example_data.sh` with `--with-test-dataset` and it pulls both checkpoints, the
denoiser and the autoencoder, plus the roughly 50 system `test_dataset`, which is a parquet and a
plain directory of CIFs, all into `$DATA_PATH`. Everything comes from the public bucket
**`YOUR_DATASET_STORE`** at `https://YOUR_DATASET_STORE`, over anonymous
HTTPS. No AWS CLI, no credentials, no IPD account. That bucket is the only one reachable from
outside. The internal `YOUR_DATASET_STORE*` buckets return `403` without credentials.

```bash
./scripts/download_example_data.sh "$DATA_PATH" --with-test-dataset
```

This lays out:

```
$DATA_PATH/
├── flow_checkpoints/<release denoiser>.ckpt
├── ae_checkpoints/<release autoencoder>.ckpt
└── test_dataset/
    ├── test_systems.parquet
    └── cifs/systems/<xx>/<id>/<id>.cif
```

See `configs/inference.yaml` for the current release pair.

> **CCD and PDB mirrors, both optional.** The tutorial needs neither. For broader work that parses
> arbitrary ligands or PDB entries, we publish a CCD mirror at
> `https://YOUR_DATASET_STORE/train_datasets/ccd_mirror/2026_01_06_ccd/2026_01_06_ccd.sqfs`.
> It is a squashfs, so mount it or run `unsquashfs` on it and point `CCD_MIRROR_PATH` at the
> directory. The full PDB mirror is too large to host, so fetch structures from
> [RCSB](https://www.rcsb.org/) or point `PDB_MIRROR_PATH` at your own copy.

### 2. Run the finetune

Pass the released denoiser, which you downloaded to `$DATA_PATH` above, as `pretrain_ckpt_path`.
It is the same checkpoint inference uses, named in
[`configs/inference.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/inference.yaml).

```bash
python src/rfproteina/train.py \
    experiment=tutorial/finetune_test_dataset \
    pretrain_ckpt_path=/path/to/denoiser.ckpt \
    +single=true +nolog=true
```

- **`experiment=tutorial/finetune_test_dataset`** wires up the tiny dataset from
  [`configs/experiment/tutorial/finetune_test_dataset.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/experiment/tutorial/finetune_test_dataset.yaml),
  on a single GPU with short epochs. Validation runs the checked in `val/unconditional` set, meaning
  unconditional monomer generation, with the AF3 refolding tag dropped, so it steps and validates
  without a container.
- **`pretrain_ckpt_path`** starts a new run from the released weights, resetting the optimizer,
  scheduler, step and epoch. To restore full Lightning state instead, use `resume_ckpt_path=<ckpt>`.
  Read the current filename from `inference.yaml` rather than copying a checkpoint name into this guide.
- **`+single=true`** drops the SLURM and DDP plugins. **`+nolog=true`** disables WandB and
  checkpoint writing.

You should see the model and frozen autoencoder build, then the pretrained checkpoint load and
training steps run over the tiny dataset. That confirms the data pipeline, autoencoder and model
all work on your cluster.

> **Your architecture must match the checkpoint.** `train.py` builds the model from the `nn` config,
> not from the checkpoint, so the two have to agree. The default
> [`configs/nn/local_latents_score_nn_160M_atom37.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/nn/local_latents_score_nn_160M_atom37.yaml)
> matches the September 15 release checkpoint. To finetune a different checkpoint, select or
> add the `nn` config it was trained with. A mismatch shows up as `size mismatch` or `Missing
> key(s)` errors on `load_state_dict`.

> **Tip.** The tutorial disables `torch.compile` for a fast first run. Turn it back on with
> `execution.training.flow=torch execution.training.encoder=torch execution.training.decoder=torch` for real training. Add
> `opt.steps_per_epoch=3 opt.max_epochs=1` to stop after a few steps.

### How the dataset works without squashfs

The production `plinder` dataset reads CIFs from a squashfs mount at
`${SQUASH_MOUNT_PATH,/squash}/plinder`. The tutorial dataset in
[`configs/datasets/train/test_dataset.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/datasets/train/test_dataset.yaml)
points `loader.base_path` straight at `${DATA_PATH}/test_dataset/cifs`, a plain directory. Each
parquet `path` row, like `systems/<xx>/<id>/<id>.cif`, resolves against that base, so you mount
nothing. To build your own dataset the same way, mirror this config with your parquet and CIF
directory.

### With or without the apptainer shebang

`train.py` and `generate.py` carry a shebang that re-runs the file inside an apptainer container
when you *execute* it, as `./train.py ...`. Passing them to Python instead, as
`python src/rfproteina/train.py ...` above, bypasses the shebang and runs in your `uv` venv. On an
configured cluster with the SIF present, the shebang path works too. On a fresh cluster, always use the
`python <script>` form.

## Training essentials

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

`configs/training_proteina.yaml` composes dataloaders, the model under `nn/`, metrics, callbacks
and an `experiment`, then an optional `debug` group. Experiments live in `configs/experiment/`,
including `sota.yaml`, `unconditional_afdb_256.yaml`, `tutorial/finetune_test_dataset.yaml` and user
namespaced folders. `datasets.train` is mandatory, and the experiment supplies it.

### Checkpoint loading, and why the two keys are mutually exclusive

| Key | Behavior |
| --- | --- |
| `pretrain_ckpt_path` | Starts a **new** run from saved weights, resetting the optimizer, scheduler, step, epoch and RNG. Use this to finetune on new data. |
| `resume_ckpt_path` | **Resumes** an interrupted run, restoring full Lightning state. An existing `last.ckpt` in the run directory gets detected automatically. |

Two knobs apply only on the `pretrain_ckpt_path` path. `configs/training_proteina.yaml` has
commented examples of both.

- **`weight_loading`**, a `rfproteina.utils.weights.WeightLoadingConfig`, sets per parameter
  policies for copy, reinit, zero init and copy with zero pad. Leave it null and you get
  `strict=True`, so any key or shape mismatch between the checkpoint and the configured
  architecture is a hard error. Set it, with an explicit `fallback_policy`, to load a checkpoint
  across an architecture change.
- **`parameter_freezing`**, a `rfproteina.utils.weights.ParameterFreezingConfig`, freezes subsets of
  parameters, such as the trunk.

### Minimal and debug flags

- `+single=true` gives you a single GPU, with no SLURM or DDP.
- `+nolog=true` turns off WandB and checkpoint writing.
- `debug=gpu` and `debug=cpu` are preset short run configs, with a small `steps_per_epoch`,
  `num_workers=0` and a progress bar. See `configs/debug/`.
- Epoch length comes from `opt.steps_per_epoch`, not from the full dataset.

## Datasets and dataloaders

Dataset configs live under `configs/datasets/train/` and `configs/datasets/val/`. Parquet backed
sets inherit from `configs/datasets/train/backend_pandas.yaml`, which is
`atomworks.ml.datasets.PandasDataset`, and the production `plinder` set is
`configs/datasets/train/plinder/plinder.yaml`. Every path uses `${oc.env:DATA_PATH}`, and squashfs
backed CIFs use `${oc.env:SQUASH_MOUNT_PATH,/squash}`, so nothing is hardcoded.

A dataset config specifies four things.

- `dataset.data` is the parquet index, holding row metadata and a `path` column for structures.
- `dataset.loader.base_path` is where the CIFs live, either a squashfs mount or a plain directory as
  in `test_dataset.yaml`.
- `dataset.transform.design_tasks` names a design task preset, covered below.
- `weights` holds optional sampling weights, such as inverse cluster size.

To add a dataset, copy an existing config, point `data` and `base_path` at your files, choose a
design task preset, and reference it from an experiment with
`- /datasets/train/<your_dataset>@datasets.train.<key>`.

## Design tasks, for conditional training

Design tasks control **what information the model sees** during training, which is what makes
conditional generation possible. You configure them per dataset as relative weight frequencies,
through a preset from `configs/design_tasks/presets/` such as `plinder_tasks`,
`unconditional_only`, `folding_only` or `motif_pretraining`.

| Task | Sees | Produces | Use case |
| --- | --- | --- | --- |
| `unconditional` | Noise | Structure + sequence | General generation |
| `folding` | Sequence | Structure | Structure prediction |
| `inverse_folding` | Structure | Sequence | Sequence design |
| `motif_scaffold` | Partial structure | Complete protein | Scaffolding |
| `ppi_hotspot_*` | Target + hotspots | Binder | Protein binder design |
| `tip_atom*` | Ligand contact residues | Protein | Ligand site design |

To override a frequency temporarily, do it on the command line.

```bash
python src/rfproteina/train.py experiment=<name> \
    datasets.train.<key>.dataset.transform.design_tasks.unconditional.frequency=80.0
```

## Launching SLURM jobs

> The `scripts/launch_*.sh` launchers target the configured clusters and need their container and SLURM
> infrastructure, meaning apptainer or docker and enroot, `.ipd/.env.<cluster>`, and a per user
> `scripts/.user_info_<cluster>`. On a fresh cluster without that stack, run `train.py` or
> `generate.py` directly as shown above.

Each cluster needs a `scripts/.user_info_<cluster>` file holding SLURM settings. Copy the matching
`scripts/.user_info_<cluster>_example` template, or let `setup.sh` create one for you when you pick
a known cluster. Here are the main launchers.

| Workflow | Script |
| --- | --- |
| Train latent diffusion | `scripts/launch_train_latents.sh <cluster> [hydra args]` |
| Train autoencoder | `scripts/launch_train_ae.sh <cluster>` |
| Inference / offline validation | `scripts/launch_generate.sh <cluster> [--njobs N] [hydra args]` |
| Parameter sweeps | `scripts/launch_sweep.sh <cluster> --sweep-config <yaml>` |
| Reward guided search | `scripts/launch_search.sh <cluster>` |

`launch_generate.sh` splits inputs across job array tasks with `--njobs`, then a dependent job
concatenates the per job metric CSVs. Foldseek clustering is on by default, and `--no-foldseek`
turns it off. Any trailing `key=value` args pass straight through to the underlying script. To load
a config from outside the repo tree, pass `--config-path <dir> --config-name <stem>`. That config
must inherit the base config and add a `hydra.searchpath` entry, as
`configs/inference_experiment/TEMPLATE.yaml` shows. For offline validation on a val set, run
`inference_experiment=inference_on_val_set +datasets/val@generation=<val_dataset>`, with the val
configs in `configs/datasets/val/`.

## Evaluation and metrics

Metrics subclass `rfproteina.metrics.metric.Metric` and score generated structures. They run inline
during inference through `metrics_tags=[...]`, and during training validation. Refolding based
metrics use the **AF3 oracle**, which needs an apptainer. See the
[running guide](workflow/running.md#evaluating-results).
The [metrics reference](workflow/metrics.md) lists everything already enabled and its tags.

To evaluate structures you already generated, run `evaluate.py`. It is Hydra based, like
`generate.py`.

```bash
python src/rfproteina/evaluate.py \
    inputs=/path/to/structures metrics_tags='[unconditional_refolding]' \
    out_dir=/path/to/eval_output
```

`out_dir` is required here, unlike `generate.py`, which derives one. The metrics CSVs land under
`<out_dir>/evaluation_metrics/`.

**Custom metrics and rewards.** Add Python metric classes under
`src/rfproteina/metrics/custom_metrics/` and register them in `configs/metrics/custom_metrics/`.
Both are git ignored, and `setup.sh` scaffolds the entry config. Search rewards get configured
inline in the search experiment config, as
`configs/inference_experiment/inference_beam_example.yaml` shows. The [custom metrics
guide](workflow/custom-metrics.md) has the full walkthrough.

**Custom sampling transforms.** Same arrangement, for per replicate condition sampling. Drop Python
classes into the git ignored `src/rfproteina/transforms/sampling/` and reference them by `_target_`
from your experiment config. See [sampling transforms](workflow/running.md#sampling-transforms).

## Advanced topics

**LoRA finetuning.** Set a positive `lora.r` in the config to enable low rank adaptation, alongside
`lora_alpha`, `lora_dropout`, `train_bias` and `exclude_keys`. A null `r` means full model training.

**Flow matching.** Modality specific time distributions control the diffusion schedule, with
separate parameters for `bb_ca`, the backbone, and `local_latents`, the sidechain. See the
`loss.t_distribution` block in `configs/training_proteina.yaml`.

**EMA.** An exponential moving average callback, `rfproteina.utils.simple_ema.SimpleEMA`, maintains
averaged weights. The released denoiser is an EMA checkpoint.

## Troubleshooting

| Issue | Solution |
| --- | --- |
| `No .env file found for cluster '<x>'` | `export CLUSTER=local` and create `.ipd/.env.local`. See [Setup](#setup-on-any-cluster-with-no-container) |
| `No autoencoder checkpoint path provided`, or the AE is missing | Download it with `download_example_data.sh`, or set `autoencoder_ckpt_path=` |
| Parquet or CIF not found | Check `DATA_PATH`. For custom data, confirm `loader.base_path` and the parquet `path` column resolve |
| Training not resuming | Make sure a `last.ckpt` exists in the run's `checkpoints/`, or pass `resume_ckpt_path=` |
| Out of memory | Lower `dataloaders.batch_size` or `opt.accumulate_grad_batches` |
| Segfault in rdkit or posebusters | uv's Python needs `LD_PRELOAD` pointed at the bundled `libstdc++.so.6`, as in the `Makefile`. `make test` sets it for you |
| SLURM launcher fails on a fresh cluster | The launchers need the container and SLURM stack. Run `python src/rfproteina/train.py ...` directly instead |
