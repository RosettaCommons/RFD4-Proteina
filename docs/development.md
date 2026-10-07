# Developer guide

This guide is for **developers**. It covers setting up a training environment on any cluster,
continuing training of the released model on your own data, and understanding datasets, design
tasks, metrics and the SLURM launchers.

For inference and design usage, see the [main
README](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/README.md). For coding
standards, see
[AGENTS.md](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/AGENTS.md).

## Setup

Install the native Python environment and configure `.env.local` as described in the
[installation guide](installation.rst). For host tools, notebooks, and SLURM config resolution,
install the optional development environment as well:

```bash
bash setup.sh
bash setup.sh --venv
source .venv/bin/activate
export DATA_PATH="${DATA_PATH:-$PWD/data_path}"  # match the path configured in .env.local
```

Use the installed `rfd4-*` commands from the activated Python environment.
Existing public environment files and private cluster profiles are preserved by setup.

## Continue training walkthrough

This walkthrough finetunes the released denoiser on a tiny self contained dataset, which is enough
to prove the training loop runs end to end on a fresh cluster. It uses the Python environment and plain CIF files, without squashfs.

### 1. Download the model and the test dataset

Run `download_example_data.sh` with `--with-test-dataset` and it pulls both checkpoints, the
denoiser and the autoencoder, plus the roughly 50 system `test_dataset`, which is a parquet and a
plain directory of CIFs, all into `$DATA_PATH`. Everything comes from the public bucket
**`YOUR_DATASET_STORE`** at `https://YOUR_DATASET_STORE`, over anonymous
HTTPS. No AWS CLI or credentials are required.

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
rfd4-train \
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
> `nn.use_torch_compile=True compile_autoencoder=True` for real training. Add
> `opt.steps_per_epoch=3 opt.max_epochs=1` to stop after a few steps.

### How the dataset works without squashfs

The production `plinder` dataset reads CIFs from a squashfs mount at
`${SQUASH_MOUNT_PATH,/squash}/plinder`. The tutorial dataset in
[`configs/datasets/train/test_dataset.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/datasets/train/test_dataset.yaml)
points `loader.base_path` straight at `${DATA_PATH}/test_dataset/cifs`, a plain directory. Each
parquet `path` row, like `systems/<xx>/<id>/<id>.cif`, resolves against that base, so you mount
nothing. To build your own dataset the same way, mirror this config with your parquet and CIF
directory.

### Entry points

Run `rfd4-train` and `rfd4-generate` from the activated Python environment.

## Training essentials

### Entry points

| Model | Command | Config name |
| --- | --- | --- |
| Latent diffusion (main model) | `rfd4-train` | `training_proteina` |
| Autoencoder | `rfd4-train-ae` | `training_ae` |

```bash
# Diffusion, quick single-GPU test
rfd4-train experiment=<name> +single=true +nolog=true

# Autoencoder
rfd4-train-ae --config-name training_ae
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
sets use `atomworks.ml.datasets.PandasDataset`. File-backed sets inherit
`configs/datasets/train/backend_pandas.yaml`; S3 blob sets use
`configs/datasets/sources/backend_blob.yaml`, shared by PDB and distillation sources. Paths use `DATA_PATH` for local data or
`DATASET_STORE_ROOT` for streaming data, and `SQUASH_MOUNT_PATH` for SquashFS mounts.

A dataset config specifies four things.

- `dataset.data` is the parquet index, holding row metadata and a `path` column for structures.
- `dataset.loader.base_path` is where the CIFs live, either a squashfs mount or a plain directory as
  in `test_dataset.yaml`. Blob loaders instead specify `storage: blob` and `blob_dir`.
- `dataset.transform.design_tasks` names a design task preset, covered below.
- `weights` holds optional sampling weights, such as inverse cluster size.

To add a dataset, copy an existing config, point `data` and `base_path` at your files, choose a
design task preset, and reference it from an experiment with
`- /datasets/train/<your_dataset>@datasets.train.<key>`.

The blob backend defines the reader, record IDs, parser, and cache defaults. Metadata memory
mapping defaults to false; new AE streaming sources explicitly enable it. PDB adds its
quality filters and metadata columns in `pdb_common.yaml`; `pdb_interface_common.yaml` adds
interface-specific query-chain mappings and filters. These are dataset rules, not different
streaming implementations. AtomWorks resolves each record through `blob/index.parquet`, fetches
its byte range from `blob/data/`, decompresses it, then parses the requested assembly. PDB's
sampling weights normalize alternative conformers separately from this loading step.

### Published AE streaming datasets

Use `datasets=ae_streaming` with the AE training entry point to select the following mix
and 384-token crops/padding. All sources live under
`${DATASET_STORE_ROOT}/train_datasets/streaming-latest/`:

| Source | Directory | Weight |
| --- | --- | --- |
| PDB interfaces | `pdb/2026_09_19_tmol_plip_cath` | 25% |
| AFDB Genie3 subset | `synthetic/afdb_genie3_subset` | 20% |
| TED dimers | `synthetic/ted_dimers` | 15% |
| Synthetic PTMs | `synthetic/2026_09_21_ptm_esmfold2` | 10% |
| PDB PTM environments | `pdb/2026_09_19_ptm_crop` | 5% |
| PDB glycans | `pdb/2026_09_19_glycan` | 15% |
| Protein–DNA complexes | `synthetic/2026_09_22_protein_dna_esmfold2` | 5% |
| RNA monomers | `synthetic/2026_09_21_rna_monomers` | 5% |

Each directory's README describes its production method. PTM/glycan sources require the
published strict audit and in-vocabulary flags; experimental sources also require the training
split and pre-2024 deposition. Glycans load the audited query plus directly bonded residues.
Those flags describe the published 100-PTM/20-glycan panel, not the model's configured vocabulary;
selecting this mix does not expand model tokens. The recipe configures training sources only;
validation datasets must be supplied separately.

PDB retains its cluster weights and conformer normalization. RNA requires
`source_training_eligible`; TED requires `InterfaceLength > 10`, `AvgIntPAE < 10`,
`AvgIntPlddt > 70`, and `n_res_a + n_res_b <= 768` before cropping.
Parsed records are cached under `${LOCAL_DRIVE_MOUNT:-/tmp}/${USER}/atomworks_parse_cache`.
Object-store-backed profiles must supply `AWS_ENDPOINT_URL`; retain it when launching there.
Re-Glyco is not included until its streaming conversion and audit are published.

The independent `train/ae/antibody_antigen_streaming` leaf targets
`synthetic/2026_09_18_antibody_antigen_esmfold2`. Its default filters are
`overall_plddt >= 0.75` (confidence is stored on a 0–1 scale), `pae_min_ab_ag < 2`, and
`n_contact_pairs >= 25`. It is not assigned weight in the mix above. Confirm the destination
publication is complete before enabling it; the corresponding original publication retains
3,792 of 75,576 structures after these filters.

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
rfd4-train experiment=<name> \
    datasets.train.<key>.dataset.transform.design_tasks.unconditional.frequency=80.0
```

## Launching SLURM jobs

The `scripts/launch_*.sh` launchers require SLURM and use the installed Python
environment. Omit the cluster argument. Configure scheduler resources in the
optional `scripts/.user_info_local`, using `scripts/user_info.example` as a
starting point. See {ref}`slurm-installation` for scheduler settings.

| Workflow | Script |
| --- | --- |
| Train latent diffusion | `scripts/launch_train_latents.sh [hydra args]` |
| Train autoencoder | `scripts/launch_train_ae.sh` |
| Inference / offline validation | `scripts/launch_generate.sh [--njobs N] [hydra args]` |
| Parameter sweeps | `scripts/launch_sweep.sh --sweep-config <yaml>` |
| Reward guided search | `scripts/launch_generate.sh inference_experiment=<search-config>` |

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
rfd4-evaluate \
    inputs=/path/to/structures metrics_tags='[unconditional_refolding]' \
    out_dir=/path/to/eval_output
```

`out_dir` is required here, unlike `generate.py`, which derives one. The metrics CSVs land under
`<out_dir>/evaluation_metrics/`.

**Custom metrics and rewards.** Add Python metric classes under
`src/rfproteina/metrics/custom_metrics/` and register them in `configs/metrics/custom_metrics/`.
Both are git ignored, and `setup.sh --venv` scaffolds the entry config. Search rewards get configured
inline in the search experiment config, as
`configs/inference_experiment/inference_beam_example.yaml` shows. The [custom metrics
guide](workflow/custom-metrics.md) has the full walkthrough.

**Custom sampling transforms.** Same arrangement, for per replicate condition sampling. Drop Python
classes into the git ignored `src/rfproteina/transforms/sampling/` and reference them by `_target_`
from your experiment config. See [sampling transforms](workflow/running.md#sampling-transforms).

## Advanced topics

**LoRA finetuning.** Set `lora.r=8` to enable PEFT adapters on the denoiser's
`to_qkv` and `to_out_node` attention projections. A null rank retains full-model training.
The base and autoencoder stay frozen. Select other denoiser projections with
`lora.target_modules`; keep `lora.train_bias=none`. Use `nn.attention_type=torch`
because fused cuEquivariance attention reads weights directly and bypasses adapters.
Start with `nn.use_torch_compile=false` while validating a new recipe.

Use a training experiment matching the base checkpoint's architecture and conditioning,
and pass its released EMA weights via `pretrain_ckpt_path`. For a full training checkpoint,
the EMA callback weights are selected, matching generation. Base weights load strictly
before PEFT injection; a new adapter run starts fresh optimizer and EMA state.
`resume_ckpt_path` restores a full PEFT Lightning checkpoint, including optimizer and EMA
state, with the same adapter configuration. LoRA uses PEFT exclusively. To migrate a
loralib checkpoint, merge its adapters with the original implementation and start a new
PEFT run from that merged base; adapter weights and optimizer state are not interchangeable.

The optimizer is the same as full training: Muon for eligible 2D parameters and Adam
for non-matrix/excluded parameters, using the existing `opt` configuration. Attention-only
adapters normally need only Muon. Tune adapter learning rate and `lora_alpha/r` for the
task; the optimizer family does not imply that dense-training hyperparameters are optimal.

Export an adapter from a completed EMA checkpoint:

```python
from rfproteina.proteina import Proteina
from rfproteina.utils.lora import save_lora_adapter

model = Proteina.load_from_checkpoint(
    "path/to/finetuned-ema.ckpt", map_location="cpu", weights_only=False,
    load_training_metrics=False,
)
save_lora_adapter(model, "adapter")
```

This writes `adapter_config.json` and `adapter_model.safetensors` using PEFT's
in-place injection and adapter-state APIs. This follows the adapter workflow used by
[HF Diffusers](https://huggingface.co/docs/diffusers/training/lora) and
[BioNeMo ESM2](https://github.com/NVIDIA-BioNeMo/bionemo-framework/tree/main/recipes/esm2_peft_te);
adapters remain specific to the RFD4 architecture and base weights.
The export contains the weights currently loaded in the model; a raw training checkpoint
exports raw adapters. Full Lightning checkpoints remain the training-resume format.

For generation, select the **exact original base/AE checkpoint pair**, then add
`adapter_path=/absolute/path/to/adapter compile_lengths=null` to `rfd4-generate`.
Adapter keys and shapes are checked strictly. Base identity cannot be inferred from
tensor shapes: record the base checkpoint, AE, model config and dataset revision with
the adapter when sharing it. A downloaded HF repository can be passed as this local
directory; no Hub publishing or `from_pretrained` model facade is required.

**Flow matching.** Modality specific time distributions control the diffusion schedule, with
separate parameters for `bb_ca`, the backbone, and `local_latents`, the sidechain. See the
`loss.t_distribution` block in `configs/training_proteina.yaml`.

**EMA.** An exponential moving average callback, `rfproteina.utils.simple_ema.SimpleEMA`, maintains
averaged weights. The released denoiser is an EMA checkpoint.

## Troubleshooting

| Issue | Solution |
| --- | --- |
| `No autoencoder checkpoint path provided`, or the AE is missing | Download it with `download_example_data.sh`, or set `autoencoder_ckpt_path=` |
| Parquet or CIF not found | Check `DATA_PATH`. For custom data, confirm `loader.base_path` and the parquet `path` column resolve |
| Training not resuming | Make sure a `last.ckpt` exists in the run's `checkpoints/`, or pass `resume_ckpt_path=` |
| Out of memory | Lower `dataloaders.batch_size` or `opt.accumulate_grad_batches` |
| Segfault in rdkit or posebusters | uv's Python needs `LD_PRELOAD` pointed at the bundled `libstdc++.so.6`, as in the `Makefile`. `make test` sets it for you |
| SLURM launcher fails on a fresh cluster | Check SLURM availability and scheduler resources; use an installed Python environment |
