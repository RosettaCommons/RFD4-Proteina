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
[installation guide](../installation.rst). For host tools, notebooks, and SLURM config resolution,
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

Run `download_assets.sh` with `--with-test-dataset` to download the configured
model weights and the tiny parquet/CIF dataset into `$DATA_PATH/test_dataset`.
The dataset comes from the pinned `nvidia/RFD4-Proteina` model repository with the
default checkpoint configuration. Authenticate with `HF_TOKEN` or `hf auth login`
if access is restricted. The finetuning command below loads the portable HF bundle separately.

```bash
./scripts/download_assets.sh "$DATA_PATH" --with-test-dataset --skip-of3
```

> **CCD and PDB mirrors are optional.** The tutorial needs neither. For broader
> use, configure your own `CCD_MIRROR_PATH` / `PDB_MIRROR_PATH`, or fetch structures
> from [RCSB](https://www.rcsb.org/).

### 2. Run the finetune

Initialize a new run from the portable bundle:

```bash
rfd4-train experiment=tutorial/finetune_test_dataset \
    hf_ckpt_path=nvidia/RFD4-Proteina \
    hf_ckpt_revision=a06e2509c7033b010fc43614e9fd840303588502 \
    pretrain_ckpt_path=null +single=true log.log_wandb=false
```

The bundle supplies defaults for `nn`, `product_flowmatcher`, autoencoder settings and
the training path-length cutoff. The experiment supplies the data, optimizer and logging
settings. To change a bundle setting, put the corresponding nested value under
`hf_bundle_overrides` in the experiment YAML. Weight loading stays strict by default;
set the existing `weight_loading` policy explicitly if a changed architecture needs
new or resized parameters. LoRA still requires an exactly matching base model.
Native checkpoints remain supported
through `pretrain_ckpt_path`, with a matching model configuration. Resume full
optimizer, scheduler and EMA state with `resume_ckpt_path=<checkpoint>` or the
run's automatic `last.ckpt`; resume does not need the original HF bundle.

> **Tip.** The tutorial disables `torch.compile` for a fast first run. Turn it back on with
> `nn.use_torch_compile=True compile_autoencoder=True` for real training. The bundle does
> not override the experiment's compilation setting. Add
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
| PDB interfaces | `pdb/2026_10_03_tmol_plip_cath` | 25% |
| AFDB Genie3 subset | `synthetic/afdb_genie3_subset` | 20% |
| TED dimers | `synthetic/ted_dimers` | 15% |
| Synthetic PTMs | `synthetic/ncaa/2026_09_21_ptm_esmfold2` | 10% |
| PDB PTM environments | `pdb/2026_09_19_ptm_crop` | 5% |
| PDB glycans | `pdb/2026_09_19_glycan` | 15% |
| Protein–DNA complexes | `synthetic/2026_09_22_protein_dna_esmfold2` | 5% |
| RNA monomers | `synthetic/2026_09_21_rna_monomers` | 5% |

Each directory's S3-root README describes its production method and exact quality criteria.
PTM/glycan sources require quality and in-vocabulary flags; experimental sources also require the training
split and pre-2024 deposition. Glycans load the audited query plus directly bonded residues.
Those flags describe the published 100-PTM/20-glycan panel, not the model's configured vocabulary;
selecting this mix does not expand model tokens. See the source configs for filters and
the selected training recipe for validation datasets.

RNA requires conversion/roundtrip, clash and Ribonanza flags plus a confidence limit.
PDB retains its cluster weights and conformer normalization.
TED requires `InterfaceLength > 10`, `AvgIntPAE < 10`,
`AvgIntPlddt > 70`; AE additionally requires `n_res_a + n_res_b <= 768` before cropping to 384 tokens.
Parsed records are cached under `${LOCAL_DRIVE_MOUNT:-/tmp}/${USER}/atomworks_parse_cache`.
Re-Glyco is a published candidate but is not selected by this recipe.

The shared `sources/antibody_interfaces_distill` source targets
`synthetic/2026_09_18_antibody_antigen_esmfold2`. Its default filters are
`overall_plddt >= 0.75` (confidence is stored on a 0–1 scale), `pae_min_ab_ag < 2`, and
`n_contact_pairs >= 25`. It is not assigned weight in the AE mix above.

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
during inference through `metrics_tags=[...]`, and during training validation. Refolding metrics
use the configured AF3, OpenFold3 or AlphaFold 2 oracle and its separately installed runtime.
See [oracle setup](../how-to/refolding.md) and the
[metrics guide](../how-to/evaluate-with-metrics.md#running-metrics-inline).
The [metrics reference](../reference/metrics.md) lists everything already enabled and its tags.

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
guide](../how-to/advanced/custom-metrics.md) has the full walkthrough.

**Custom sampling transforms.** Same arrangement, for per replicate condition sampling. Drop Python
classes into the git ignored `src/rfproteina/transforms/sampling/` and reference them by `_target_`
from your experiment config. See [sampling transforms](../how-to/run-the-model.md#sampling-transforms).

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
| `No autoencoder checkpoint path provided`, or the AE is missing | Use a portable HF bundle, or set `autoencoder_ckpt_path=` for native checkpoints |
| Parquet or CIF not found | Check `DATA_PATH`. For custom data, confirm `loader.base_path` and the parquet `path` column resolve |
| Training not resuming | Make sure a `last.ckpt` exists in the run's `checkpoints/`, or pass `resume_ckpt_path=` |
| Out of memory | Lower `dataloaders.batch_size` or `opt.accumulate_grad_batches` |
| Segfault in rdkit or posebusters | uv's Python needs `LD_PRELOAD` pointed at the bundled `libstdc++.so.6`, as in the `Makefile`. `make test` sets it for you |
| SLURM launcher fails on a fresh cluster | Check SLURM availability and scheduler resources; use an installed Python environment |

## Portable models

`Proteina.from_pretrained(repo_id_or_directory, revision=...)` loads the model and
paired autoencoder from `config.json` and `model.safetensors`. Authentication and
caching use the normal Hugging Face settings. `model.save_pretrained(directory)`
exports the currently loaded weights; load an EMA checkpoint first to export EMA.

Generation and new fine-tuning runs accept `hf_ckpt_path` and
`hf_ckpt_revision` for a Hub repository or local exported bundle. Native inference
uses `ckpt_path`/`ckpt_name`; native training initialization uses
`pretrain_ckpt_path`, and training resume uses `resume_ckpt_path`. Sampling
in Python can reuse a loaded model with `run_generation(cfg, model=model)`.
Full training resume uses a native Lightning checkpoint, including optimizer and
EMA state, and does not need the original bundle. Bundle configurations retain
model/AE architecture and scientific featurization settings, excluding dataset
locations, metrics and logging configuration.
