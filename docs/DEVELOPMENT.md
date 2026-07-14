# Developer Guide

This document covers training, SLURM job launching, dataset configuration, design task implementation, metrics, and the full project architecture.

For user-facing inference and design documentation, see the [main README](../README.md).

---

## Table of Contents

1. [Setup](#setup)
2. [Datasets and Dataloaders](#datasets-and-dataloaders)
3. [Design Tasks (Conditional Training)](#design-tasks-conditional-training)
4. [Model Training](#model-training)
5. [SLURM Job Launching](#slurm-job-launching)
6. [Evaluation and Metrics](#evaluation-and-metrics)
7. [Adding New Rewards and Metrics](#adding-new-rewards-and-metrics)
8. [Advanced Topics](#advanced-topics)
9. [File Structure Reference](#file-structure-reference)
10. [Troubleshooting](#troubleshooting)

---

## Setup

### Installation

Run `setup.sh` from the repo root on a new machine:

```bash
bash projects/proteinfoundation/setup.sh
```

This installs `uv`, initialises the atomworks submodule, creates the virtualenv, installs all dependencies, and appends `CLUSTER`/`CONTAINER_TYPE` exports to your shell rc file.

Then activate the environment:

```bash
source .venv/bin/activate
```

**Environment Variables**

Shared cluster paths (`DATA_PATH`, `CACHE_DIR`, `CCD_MIRROR_PATH`, etc.) live in `.ipd/.env.<cluster>` (committed to git) and are auto-sourced via `scripts/.user_info_<cluster>`. No per-user `.env` file is needed.

| Variable | Description | Required |
|----------|-------------|----------|
| `DATA_PATH` | Base directory for training data (parquet files) | Yes |
| `CACHE_DIR` | Cache directory for parsed structures | Yes |
| `CCD_MIRROR_PATH` | CCD mirror for ligand parsing | Yes |
| `PDB_MIRROR_PATH` | PDB mirror directory | Optional |

**Expected Data Structure:**

```
${DATA_PATH}/
├── latent_train_parquet_files/
│   ├── mgnify_distillation_big.parquet
│   ├── omol_all_7M_parsed_2025_11_26.parquet
│   ├── ted_dimers.parquet
│   └── afdb_monomer.parquet
├── latent_checkpoints/ae_checkpoints/
│   └── AE_ucond_512.ckpt
└── ccd_mirror/
```

### Test Suite

```bash
pytest tests/                              # All tests
pytest tests/ -v                           # Verbose output
pytest tests/metrics/test_sse_metrics.py   # Specific test module
```

---

## Datasets and Dataloaders

### Available Datasets

| Dataset | Use Case | Chains | Size | Default Tasks |
|---------|----------|--------|------|---------------|
| `mgnify` | General protein generation | Single | ~7.5M | 80% motif, 10% uncond, 5% fold, 5% inv-fold |
| `omol` | Small molecule generation | N/A | ~7M | Unconditional (no tasks) |
| `afdb_dimer` | Binder design, PPI | Multi | ~500K | 80% PPI hotspot, 10% uncond, 5% fold, 5% inv-fold |
| `afdb_monomer` | Single-chain generation | Single | ~5M | Unconditional, motif scaffolding |
| `pdb_interfaces` | Multi-chain PPI | Multi | ~100K | PPI hotspot, unconditional |

**Usage:**

```bash
python proteinfoundation/train.py experiment=mgnify_training
python proteinfoundation/train.py experiment=omol_training
python proteinfoundation/train.py experiment=afdb_dimer_training
```

### Dataset Configuration

Datasets are configured via Hydra YAML files in `configs/datasets/train/distill/` using environment variables for portability:

```yaml
defaults:
  - ../base@dataset
  - /design_tasks/presets/motif_pretraining@dataset.transform.design_tasks
  - _self_

dataset:
  name: mgnify_distillation
  data: ${oc.env:DATA_PATH}/latent_train_parquet_files/mgnify_distillation_big.parquet
  columns_to_load:
    - example_id
    - path
    - ptm
  filters:
    - "ptm > 0.75"
```

All datasets use `${oc.env:DATA_PATH}` and `${oc.env:CACHE_DIR}` sourced from `.ipd/.env.<cluster>` -- no hardcoded paths.

### Adding a New Dataset

**Step 1:** Create dataset config in `configs/datasets/train/distill/my_dataset.yaml`

```yaml
defaults:
  - ../base@dataset
  - /design_tasks/presets/unconditional_only@dataset.transform.design_tasks
  - _self_

probability: 1.0

dataset:
  name: my_dataset
  data: ${oc.env:DATA_PATH}/latent_train_parquet_files/my_data.parquet
  columns_to_load:
    - example_id
    - path
  filters:
    - "quality_score > 0.8"
  loader:
    parser_args:
      add_missing_atoms: True
```

**Step 2:** Create experiment config in `configs/experiment/my_experiment.yaml`

```yaml
# @package _global_

defaults:
  - /datasets/train/distill/my_dataset@datasets.train
  - _self_

run_name: my_experiment_run
```

**Step 3:** Train

```bash
python proteinfoundation/train.py experiment=my_experiment
```

---

## Design Tasks (Conditional Training)

### What are Design Tasks?

Design tasks control **what information is provided to the model** during training, enabling conditional generation:

| Task | Input | Output | Use Case |
|------|-------|--------|----------|
| `unconditional` | Noise | Structure + Sequence | General generation |
| `folding` | Sequence | Structure | Predict structure from sequence |
| `inverse_folding` | Structure | Sequence | Design sequence for structure |
| `motif_scaffold` | Partial structure (motif) | Complete protein | Scaffold around functional motif |
| `ppi_hotspot_with_3d_target` | Target protein + hotspots | Binder protein | Protein binder design |
| `tip_atom` | Ligand binding residues | Protein | Ligand binding site design |

### Using Design Tasks

Tasks are configured in dataset configs via relative-weight frequencies:

```yaml
# configs/design_tasks/presets/mgnify_unconditional.yaml
unconditional:
  frequency: 90.0
  transform:
    _target_: proteinfoundation.transforms.design_task.UnconditionalTask
    global_design_task_kwargs: ${datasets.global_design_task_kwargs}

folding:
  frequency: 5.0
  transform:
    _target_: proteinfoundation.transforms.design_task.StructurePredictionTask
    global_design_task_kwargs: ${datasets.global_design_task_kwargs}

inverse_folding:
  frequency: 5.0
  transform:
    _target_: proteinfoundation.transforms.design_task.InverseFoldingTask
    global_design_task_kwargs: ${datasets.global_design_task_kwargs}
```

### Design Task Presets

Pre-configured task combinations in `configs/design_tasks/presets/`:

| Preset | Tasks | Use Case |
|--------|-------|----------|
| `unconditional_only` | 100% unconditional | Simple generation |
| `mgnify_unconditional` | 90% uncond, 5% fold, 5% inv-fold | General training |
| `folding_only` | 100% folding | Structure prediction only |
| `motif_pretraining` | 80% motif (AME), 10% uncond, 5% fold, 5% inv-fold | Motif scaffolding |
| `afdb_dimer_with_ppi` | 80% PPI, 10% uncond, 5% fold, 5% inv-fold | Binder design |

### Adding a New Design Task Preset

**Step 1:** Create preset in `configs/design_tasks/presets/my_tasks.yaml`

```yaml
unconditional:
  frequency: 50.0
  transform:
    _target_: proteinfoundation.transforms.design_task.UnconditionalTask
    global_design_task_kwargs: ${datasets.global_design_task_kwargs}

folding:
  frequency: 50.0
  transform:
    _target_: proteinfoundation.transforms.design_task.StructurePredictionTask
    global_design_task_kwargs: ${datasets.global_design_task_kwargs}
```

**Step 2:** Reference in dataset config

```yaml
defaults:
  - ../base@dataset
  - /design_tasks/presets/my_tasks@dataset.transform.design_tasks
  - _self_
```

### Adjusting Task Frequencies

**Option 1: Edit preset file directly**

```bash
vim configs/design_tasks/presets/mgnify_unconditional.yaml
```

**Option 2: Hydra override** (temporary)

```bash
python proteinfoundation/train.py \
    experiment=mgnify_training \
    datasets.train.dataset.transform.design_tasks.unconditional.frequency=80.0 \
    datasets.train.dataset.transform.design_tasks.folding.frequency=10.0
```

---

## Model Training

### Autoencoder vs Diffusion Model

The codebase includes **two separate models**:

| Model | Purpose | Training Script | Checkpoint Used By |
|-------|---------|----------------|-------------------|
| **Autoencoder** | Encode sidechains to latents (atom37 -> latent_dim) | `partial_autoencoder/train.py` | Diffusion model |
| **Diffusion Model** | Generate proteins (main model) | `train.py` | Inference/generation |

**Typical workflow:**
1. Train autoencoder (or use pretrained)
2. Train diffusion model using frozen autoencoder checkpoint
3. Generate with diffusion model (autoencoder used for decoding)

### Local Training (Interactive)

**Diffusion Model:**

```bash
# Single GPU, no logging (quick test)
python proteinfoundation/train.py experiment=mgnify_training +single=true +nolog=true

# Multi-GPU on local machine
python proteinfoundation/train.py experiment=mgnify_training

# Override parameters
python proteinfoundation/train.py \
    experiment=mgnify_training \
    opt.lr=0.0002 \
    training.self_cond=False
```

**Autoencoder:**

```bash
python proteinfoundation/partial_autoencoder/train.py --config-name training_ae
```

### Training Configuration

**Hardware** (`configs/training_proteina.yaml`):

```yaml
hardware:
  accelerator: gpu
  gpus_per_node: 8
  num_nodes: 4
```

**Experiment configs** (`configs/experiment/`):

```yaml
# configs/experiment/mgnify_training.yaml
# @package _global_

defaults:
  - /datasets/train/distill/mgnify@datasets.train
  - _self_

run_name: mgnify_training_run
```

**Available experiments:**
- `mgnify_training` -- MGnify protein dataset
- `omol_training` -- Small molecules
- `afdb_dimer_training` -- Protein dimers for binder design
- `afdb_monomer_training` -- Single-chain proteins
- `pdb_training` -- PDB interfaces

### Key Training Parameters

```yaml
# Optimization
opt:
  lr: 0.0001
  max_epochs: 10000000
  val_check_interval: 10000

# Checkpointing
log:
  checkpoint_every_n_steps: 10000
  last_ckpt_every_n_steps: 1500

# Flow matching
loss:
  t_distribution:
    bb_ca:
      name: mix_unif_beta
      p1: 1.9
      p2: 1.0
      p3: 0.02
    local_latents:
      name: mix_unif_beta
      p1: 1.0
      p2: 1.5
      p3: 0.2

# Autoencoder checkpoint
autoencoder_ckpt_path: ${oc.env:DATA_PATH}/latent_checkpoints/ae_checkpoints/AE_ucond_512.ckpt
```

---

## SLURM Job Launching

The `scripts/` directory provides a unified system for launching jobs across multiple clusters with different containerization backends. See the [main README Cluster Launching section](../README.md#cluster-launching-slurm) for user-facing examples.

### Multi-Cluster Architecture

| Cluster Type | Container Backend | Container Image | Mounts |
|--------------|-------------------|-----------------|--------|
| **DIGS** | Apptainer (Singularity) | `.sif` file | Auto-mounted (Lustre) |
| **Other Clusters** | Docker via enroot | `.sqsh` squashfs file | Explicit `--container-mounts` |

### Available Launch Scripts

| Workflow | Script | Description |
|----------|--------|-------------|
| Train latent diffusion | `launch_train_latents.sh` | Multi-node distributed training |
| Train autoencoder | `launch_train_ae.sh` | Autoencoder training |
| Inference / validation | `launch_infer.sh` | Parallel generation with job arrays; also used for offline validation |
| Parameter sweeps | `launch_sweep.sh` | Generates one config per parameter combo and submits via `launch_infer.sh` |
| Autoencoder inference | `launch_infer_ae.sh` | Autoencoder latent extraction |

**`launch_infer.sh` options:**

| Option | Description |
|--------|-------------|
| `--njobs N` | Number of parallel generation jobs (overrides `gen_njobs` from config) |
| `--gpus-per-job N` | GPUs per SLURM array task (default: 1) |
| `--config-name NAME` | Hydra config name (default: `inference`); same meaning as `generate.py`'s `--config-name` |
| `--config-path DIR` | Hydra search-path root (same as `generate.py`'s `--config-path`); set it to load a config that lives outside the repo `configs/` tree (see [Configs outside the repo tree](#configs-outside-the-repo-tree)) |
| `--copy` | Copy code to run directory via rsync (default: use repo in-place) |
| `--follow` / `-f` | Tail the log file after submission |
| `--no-concatenate` | Skip the automatic post-generation CSV concatenation job |
| `--no-foldseek` | Disable Foldseek structural clustering in the concatenation job (on by default) |
| `--overwrite-rundir` | Allow overwriting an existing code directory |

Any remaining arguments are passed directly to `generate.py` as Hydra overrides.

#### Configs outside the repo tree

To run a config stored anywhere on the filesystem, pass both Hydra flags, exactly
as you would to `generate.py`:

```bash
./scripts/launch_infer.sh digs \
    --config-path /abs/path/to/config/dir --config-name my_experiment \
    gen_njobs=4
```

The config must inherit the base `inference` config and add a `hydra.searchpath`
entry so Hydra can find the in-tree base configs. See
`configs/inference_experiment/TEMPLATE.yaml` (Pattern C) for the two lines to add.

### Cluster Configuration

Each cluster needs a `.user_info_<cluster>` file in `scripts/`:

```bash
# .user_info_digs (Apptainer-based cluster)
CLUSTER_TYPE="digs"
SKIP_ACCOUNT="true"
SIF_PATH="/path/to/container.sif"
PYTHON="/usr/bin/python"
ROOT_REMOTE="/path/to/shared/storage"
DATA_PATH="/path/to/shared/storage"
CACHE_DIR="/path/to/shared/storage"
PARTITION="batch"
RUNTIME="04:00:00"
GPUS_PER_NODE_CLUSTER=8
```

```bash
# .user_info_<other_cluster> (Docker-based cluster)
CLUSTER_TYPE="ord"
ACCOUNT="your_account"
SQSH_IMAGE="/path/to/container.sqsh"
PYTHON="python"
ROOT_REMOTE="/path/to/shared/storage"
DATA_PATH="/path/to/shared/storage"
CACHE_DIR="/path/to/shared/storage"
PARTITION="batch"
RUNTIME="04:00:00"
GPUS_PER_NODE_CLUSTER=8
EXTRA_CONTAINER_MOUNTS="/data:/data:ro"
```

### Inference

Run from `projects/proteinfoundation/` on the cluster:

```bash
# 4 replicates over benchmark inputs, split across 2 parallel jobs
./scripts/launch_infer.sh digs --njobs 2 \
    inference_experiment=inference_like_rfd3 \
    inputs=/path/to/benchmarks/monomer.json \
    ckpt_path=/path/to/checkpoints ckpt_name=last.ckpt \
    out_dir=/path/to/outputs run_name=my_run num_replicates=4 batch_size=2
```

After all generation jobs finish, a dependent SLURM job automatically concatenates per-job CSV metrics into a single file under `<run_dir>/inference_metrics/`. Foldseek structural clustering runs by default; pass `--no-foldseek` to disable it.

### Offline Validation

Offline validation uses the same `launch_infer.sh` script with `inference_experiment=inference_on_val_set` and a `+datasets/val@generation=<dataset>` override that selects the validation dataset:

```bash
# Validate on the MCSA subset (Foldseek clustering on by default)
./scripts/launch_infer.sh digs --njobs 8 \
    ckpt_path=/path/to/checkpoints ckpt_name=last.ckpt \
    inference_experiment=inference_on_val_set \
    +datasets/val@generation=mcsa_subset \
    run_name=my_run_mcsa_val out_dir=/path/to/outputs
```

Available val dataset configs live in `configs/datasets/val/`. The `inference_on_val_set` experiment config drives the generation from the val dataset (rather than an `inputs` file) and loads all metrics.

To sweep for offline validation, use a sweep config as detailed in the main README.
```

### Parameter Sweeps

`launch_sweep.sh` generates one Hydra config per parameter combination and submits each via `launch_infer.sh`:

```bash
# Sweep nsteps x noise scale (Foldseek clustering on by default)
./scripts/launch_sweep.sh digs \
    --sweep-config configs/sweeps/sweep_example.yaml \
    --base-out-dir /path/to/out_dir --njobs 2 \
    ckpt_path=/path/to/checkpoints ckpt_name=last.ckpt

# Preview configs without submitting
./scripts/launch_sweep.sh digs \
    --sweep-config configs/sweeps/sweep_example.yaml --dry-run
```

See `configs/sweeps/` for available sweep configs.

### Training

```bash
# With code sync and auto-requeue
bash scripts/launch_train_latents.sh digs --submissions 2

# Without code sync
bash scripts/launch_train_latents.sh digs --no-copy
```

### What the Launch Scripts Do

1. **Copy code** (opt-in with `--copy`): `rsync` selective directories to `$ROOT_REMOTE/<run_name>/`
2. **Resolve Hydra config**: Reads `run_name`, `num_nodes`, `gpus_per_node` from config
3. **Build and submit sbatch script**: Generates appropriate container commands
4. **Concatenation job** (inference/val only): Dependent job merges per-task CSVs into `inference_metrics/`

### Output Structure

```
$ROOT_REMOTE/<run_name>/
├── slurm_run_outputs/
│   ├── slurm_12345_0.out
│   ├── slurm_12345_0.err
│   └── ...
├── slurm_batch_script.sh
├── .env
├── inference/
│   ├── sample_0_job_0.cif
│   └── ...
└── inference_metrics/
    └── results_concatenated.csv
```

### Adding a New Cluster

1. Determine cluster type (Apptainer vs Docker/enroot)
2. Copy the matching template:
    ```bash
    cp scripts/.user_info_digs scripts/.user_info_mycloud   # Apptainer
    cp scripts/.user_info_ord scripts/.user_info_mycloud    # Docker
    ```
3. Edit with your cluster settings, then launch:
    ```bash
    bash scripts/launch_train_latents.sh mycloud
    ```

---

## Evaluation and Metrics

### How Metrics Work

Metrics inherit from `foundry.metrics.metric.Metric` and compute scores on generated structures.

**Location:** `proteinfoundation/metrics/`

| Category | Metrics | Description |
|----------|---------|-------------|
| **Refolding** | `RF3Metric`, `AF2Metric`, `TMOLMetric` | Refold and evaluate (slow, accurate) |
| **Structure Quality** | `SidechainMetrics`, `SSEMetrics`, `RASAMetrics` | Fast, no refolding required |
| **Validation** | `MolprobityMetrics`, `HbondMetrics` | Stereochemistry, clashes, H-bonds |
| **Task-Specific** | `UnindexingMetrics`, `HotspotMetrics` | Motif RMSD, PPI interface quality |

### Metrics in Training

Configured in subconfigs under `configs/metrics`:

```yaml
metrics:
  sidechain:
    _target_: proteinfoundation.metrics.sidechain_metrics.SidechainMetrics
  sse:
    _target_: proteinfoundation.metrics.sse_metrics.SSEMetrics
  rf3:
    _target_: proteinfoundation.metrics.rf3_metric.RF3Metric
    ckpt_path: /path/to/rf3_latest.pt
    num_steps: 50
    metrics_tags: [ppi, binder]
```

Metrics run during validation and log to WandB.

### Metrics in Inference

```bash
python proteinfoundation/evaluate.py \
    --samples_dir ./outputs/generated_structures/ \
    --metrics designability,novelty \
    --output_csv results.csv
```

Automatically runs after generation when using `launch_infer.sh`.

---

## Adding New Rewards and Metrics

Metrics can be used for **evaluation and search**.

### Adding a New Metric

Create a metric class in `proteinfoundation/metrics/my_metric.py`:

```python
from foundry.metrics.metric import Metric
from biotite.structure import AtomArray

class MyCustomMetric(Metric):
    """Custom metric for protein evaluation."""

    def __init__(self, threshold: float = 10.0):
        super().__init__()
        self.threshold = threshold

    def __call__(
        self,
        predicted_atom_array: AtomArray,
        extra_info: dict | None = None
    ) -> dict[str, float]:
        quality_score = self.compute_quality(predicted_atom_array)
        return {
            "quality_score": quality_score,
            "passes_threshold": float(quality_score > self.threshold),
        }

    def compute_quality(self, atom_array: AtomArray) -> float:
        return atom_array.b_factor.mean()
```

### Adding a Reward Config

Create a reward config in `configs/rewards/my_reward.yaml`:

```yaml
metrics:
  my_metric:
    _target_: proteinfoundation.metrics.my_metric.MyCustomMetric
    threshold: 15.0
  rf3:
    _target_: proteinfoundation.metrics.rf3_metric.RF3Metric
    ckpt_path: /path/to/rf3_latest.pt
    num_steps: 50

weights:
  my_metric.quality_score: 2.0
  rf3.plddt: 1.0
  rf3.iptm: 3.0
```

### Using Custom Rewards

```bash
# With search
python proteinfoundation/generate.py \
    --config-name inference_test_search \
    inputs=design.json ckpt_path=./checkpoints ckpt_name=model.ckpt \
    reward=my_reward
```

### Programmatic Usage

```python
from proteinfoundation.rewards import RewardModel
from proteinfoundation.metrics.my_metric import MyCustomMetric
from proteinfoundation.metrics.rf3_metric import RF3Metric

reward_model = RewardModel(
    metrics={
        "my_metric": MyCustomMetric(threshold=15.0),
        "rf3": RF3Metric(ckpt_path="/path/to/rf3_latest.pt", num_steps=50),
    },
    weights={
        "my_metric.quality_score": 2.0,
        "rf3.plddt": 1.0,
        "rf3.iptm": 3.0,
    },
)

from proteinfoundation.search import BestOfNStrategy
strategy = BestOfNStrategy(n_samples=50)
results = strategy.generate(model, batch, reward_model)
```

---

## Advanced Topics

### LoRA Fine-Tuning

Enable low-rank adaptation for efficient fine-tuning:

```yaml
lora:
  r: 8                 # LoRA rank (null to disable)
  lora_alpha: 32.0
  lora_dropout: 0.0
```

Requires: `pip install loralib`

### Flow Matching Parameters

Control diffusion schedule via time distributions:

```yaml
loss:
  t_distribution:
    bb_ca:
      name: mix_unif_beta
      p1: 1.9
      p2: 1.0
      p3: 0.02
    local_latents:
      name: mix_unif_beta
      p1: 1.0
      p2: 1.5
      p3: 0.2
```

Different parameters for backbone vs sidechains optimize generation quality.

### EMA Checkpointing

Exponential moving average improves generation:

```yaml
callbacks:
  ema:
    _target_: proteinfoundation.utils.ema_callback.EMACallback
    decay: 0.999

log:
  save_ema_separately: false
```

---

## File Structure Reference

```
proteinfoundation/
├── configs/
│   ├── training_proteina.yaml           # Main training config
│   ├── training_ae.yaml                 # Autoencoder training config
│   ├── inference.yaml                   # Inference sweep config
│   ├── search.yaml                      # Search sweep config
│   ├── experiment/                      # Experiment presets
│   ├── datasets/train/distill/          # Dataset configs
│   ├── design_tasks/presets/            # Design task presets
│   ├── rewards/                         # Reward model configs
│   └── nn/                              # Model architecture configs
│
├── proteinfoundation/
│   ├── train.py                         # Diffusion model training
│   ├── generate.py                      # Generation/inference
│   ├── evaluate.py                      # Metric evaluation
│   ├── partial_autoencoder/
│   │   ├── train.py                     # Autoencoder training
│   │   └── inference.py                 # Autoencoder inference
│   ├── result_analysis/
│   │   └── analysis.py                  # Result analysis and clustering
│   ├── metrics/                         # Evaluation metrics
│   ├── search/                          # Search strategies
│   │   ├── best_of_n.py
│   │   ├── beam_search.py
│   │   └── vanilla.py
│   ├── rewards/                         # Reward model wrapper
│   └── transforms/                      # Design tasks
│
├── scripts/                         # SLURM launch scripts
│   ├── common.sh
│   ├── .user_info_digs
│   ├── .user_info_ord
│   ├── launch_train_latents.sh
│   ├── launch_train_ae.sh
│   ├── launch_infer.sh
│   ├── launch_search.sh
│   └── launch_offline_val.sh
│
├── docs/                                # Documentation
│   ├── DEVELOPMENT.md                   # This file
│   ├── input.md                         # Input specification
│   └── input.md                         # Input specification
│
├── benchmarks/                          # Benchmark inputs and targets
├── script_utils/                        # Helper scripts
└── setup.sh                             # First-time setup script
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `DATA_PATH not set` | Ensure `.ipd/.env.<cluster>` is present and your `scripts/.user_info_<cluster>` sources it |
| `Autoencoder checkpoint not found` | Update `autoencoder_ckpt_path` in `training_proteina.yaml` |
| `Parquet file not found` | Verify `DATA_PATH` points to correct directory with parquet files |
| `Training not resuming` | Check `./store/<run_name>/checkpoints/last.ckpt` exists |
| `Out of memory` | Reduce batch size or decrease number of GPUs in config |
| `SLURM job fails immediately` | Check `slurm_run_outputs/slurm.err` on remote cluster |
| `Import errors` | Ensure parent foundry installed: `cd ../.. && pip install -e .` |
| `Hydra config errors` | Use `=` not `--` for arguments: `inputs=file.cif` not `--inputs file.cif` |
| `Squashfs mount fails` | Set `USE_SQUASH_MOUNT="false"` in cluster config |
| `Search needs reward model` | Create reward config in `configs/rewards/` and specify with `reward=<name>` |
| `CLUSTER_TYPE not set` | Missing or invalid `.user_info_<cluster>` file |
| `Code sync failed` | Check `rsync` permissions and network |

---

**For coding standards and contribution guidelines, see [CLAUDE.md](../CLAUDE.md).**

