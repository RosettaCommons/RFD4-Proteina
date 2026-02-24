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

```bash
cd projects/proteinfoundation
pip install -e .
```

### Environment Configuration

Create `.env` file from template:

```bash
cp env.template .env
# Edit .env to set required paths
```

**Required Environment Variables:**

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

All datasets use `${oc.env:DATA_PATH}` and `${oc.env:CACHE_DIR}` from your `.env` file -- no hardcoded paths.

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

run_name_: my_experiment_run
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
  ngpus_per_node_: 8
  nnodes_: 4
```

**Experiment configs** (`configs/experiment/`):

```yaml
# configs/experiment/mgnify_training.yaml
# @package _global_

defaults:
  - /datasets/train/distill/mgnify@datasets.train
  - _self_

run_name_: mgnify_training_run
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

The `slurm_utils/` directory provides a unified system for launching jobs across multiple clusters with different containerization backends.

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
| Inference (generation) | `launch_infer.sh` | Parallel generation with job arrays |
| Autoencoder inference | `launch_infer_ae.sh` | Autoencoder latent extraction |
| Search (beam/best-of-N) | `launch_search.sh` | Parallel search with job arrays |
| Offline validation | `launch_offline_val.sh` | Parallel validation with job arrays |

**Common options:**

| Option | Description |
|--------|-------------|
| `--no-copy` | Skip code copying (use existing code on cluster) |
| `--follow` | Tail the log file after submission |
| `--submissions N` | Submit N jobs with `--dependency=singleton` for auto-requeue |
| `--gpus-per-job N` | (Inference/search only) GPUs per array task |

### Basic Usage

Run scripts from the `projects/proteinfoundation/` directory on cluster:

```bash
# Training with code sync and auto-requeue
bash slurm_utils/launch_train_latents.sh digs --submissions 2

# Training without code sync
bash slurm_utils/launch_train_latents.sh digs --no-copy

# Inference with extra Hydra args
bash slurm_utils/launch_infer.sh digs "nsamples=100"

# Search with extra args
bash slurm_utils/launch_search.sh digs "search.strategy.beam_width=8"
```

### Cluster Configuration

Each cluster needs a `.user_info_<cluster>` file in `slurm_utils/`:

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

### Parallel Jobs with SLURM Arrays

For large inference/evaluation runs, the scripts automatically use SLURM job arrays:

```yaml
# configs/inference.yaml
gen_njobs: 10   # 10 parallel generation jobs
eval_njobs: 4   # 4 parallel evaluation jobs
nsamples: 1000  # Total samples (split across gen_njobs)
```

### Parameter Sweeps

```bash
# Step 1: Create sweep configuration in configs/inference.yaml
# Step 2: Generate individual configs
python script_utils/generate_inference_configs.py
# Step 3: Launch sweep
bash slurm_utils/launch_infer.sh digs
```

### Dataset Squashfs Mounting

```bash
# In .user_info_<cluster>
DATASET_SQUASH_MOUNTS="/data/pdb.sqsh:/mnt/pdb;/data/afdb.sqsh:/mnt/afdb"
```

### What the Launch Scripts Do

1. **Copy code** (default, skip with `--no-copy`): `rsync` selective directories to `$ROOT_REMOTE/<run_name>/`
2. **Resolve Hydra config**: Reads `run_name_`, `nnodes_`, `ngpus_per_node_` from config
3. **Build and submit sbatch script**: Generates appropriate container commands
4. **Follow logs** (optional with `--follow`): Tails the slurm.err log file

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
└── results/
    ├── results_inf_0_0.csv
    └── ...
```

### Result Analysis

```bash
python proteinfoundation/result_analysis/analysis.py \
    --results_dir /path/to/run/results \
    --success_dir ./successes \
    --output ./summary.csv
```

### Adding a New Cluster

1. Determine cluster type (Apptainer vs Docker/enroot)
2. Copy the matching template:
    ```bash
    cp slurm_utils/.user_info_digs slurm_utils/.user_info_mycloud   # Apptainer
    cp slurm_utils/.user_info_ord slurm_utils/.user_info_mycloud    # Docker
    ```
3. Edit with your cluster settings, then launch:
    ```bash
    bash slurm_utils/launch_train_latents.sh mycloud
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

Configured in `configs/metrics/design_metrics.yaml`:

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

Metrics can be used for **evaluation, search, AND refinement** (if differentiable).

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

For differentiable metrics (needed for refinement), implement `compute_with_gradients`. See `AF2Metric` for reference.

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

differentiable: false
```

### Using Custom Rewards

```bash
# With search
python proteinfoundation/generate.py \
    --config-name inference_test_search \
    inputs=design.json ckpt_path=./checkpoints ckpt_name=model.ckpt \
    reward=my_reward

# With refinement (must have differentiable=true)
python proteinfoundation/refine.py \
    --config-name refine_gradient_ascent \
    inputs=structure.pdb \
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
    differentiable=False
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
│   ├── refine.py                        # Standalone refinement
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
│   ├── refinement/                      # Refinement strategies
│   │   ├── bindcraft.py
│   │   └── gradient_ascent.py
│   ├── rewards/                         # Reward model wrapper
│   └── transforms/                      # Design tasks
│
├── slurm_utils/                         # SLURM launch scripts
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
│   ├── symmetry.md                      # Symmetry documentation
│   ├── protein_binder_design.md         # PPI binder examples
│   ├── enzyme_design.md                 # Enzyme design examples
│   ├── small_molecule_binder_design.md  # SM binder examples
│   ├── dna_binder_design.md             # DNA binder examples
│   ├── motif_scaffolding.md             # Motif scaffolding examples
│   ├── unconditional_generation.md      # Unconditional generation
│   └── search_and_refinement.md         # Search and refinement guide
│
├── benchmarks/                          # Benchmark inputs and targets
├── script_utils/                        # Helper scripts
├── env.template                         # Environment variable template
└── .env                                 # Your environment config
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `DATA_PATH not set` | Create `.env` from `env.template` and set required paths |
| `Autoencoder checkpoint not found` | Update `autoencoder_ckpt_path` in `training_proteina.yaml` |
| `Parquet file not found` | Verify `DATA_PATH` points to correct directory with parquet files |
| `Training not resuming` | Check `./store/<run_name>/checkpoints/last.ckpt` exists |
| `Out of memory` | Reduce batch size or decrease number of GPUs in config |
| `SLURM job fails immediately` | Check `slurm_run_outputs/slurm.err` on remote cluster |
| `Import errors` | Ensure parent foundry installed: `cd ../.. && pip install -e .` |
| `Hydra config errors` | Use `=` not `--` for arguments: `inputs=file.cif` not `--inputs file.cif` |
| `Squashfs mount fails` | Set `USE_SQUASH_MOUNT="false"` in cluster config |
| `Search/refinement needs reward model` | Create reward config in `configs/rewards/` and specify with `reward=<name>` |
| `Refinement fails with "not differentiable"` | Use AF2-based reward with `differentiable=true` |
| `CLUSTER_TYPE not set` | Missing or invalid `.user_info_<cluster>` file |
| `Code sync failed` | Check `rsync` permissions and network |

---

**For coding standards and contribution guidelines, see [CLAUDE.md](../CLAUDE.md).**

