# CLI options reference

Every command line option exposed by RFD4-Proteina's installed commands, organized by entry
point. This page is a lookup table; for task-oriented walkthroughs see
[Running the model](../how-to/run-the-model.md), [Partial diffusion and inverse
folding](../how-to/partial-diffusion.md), [Reward guided search](../how-to/search.md), and
[Metric guidance](../how-to/advanced/metric-guidance.md).

**Contents**

- [Conventions](#conventions)
- [CLI options for `rfd4-generate`](#cli-options-for-rfd4-generate)
- [CLI options for `rfd4-evaluate`](#cli-options-for-rfd4-evaluate)
- [CLI options for `rfd4-train`](#cli-options-for-rfd4-train)
- [CLI options for `rfd4-train-ae`](#cli-options-for-rfd4-train-ae)
- [Aggregating multi-job metrics](#aggregating-multi-job-metrics)
- [Cluster launcher scripts](#cluster-launcher-scripts)

## Conventions

- `rfd4-generate`, `rfd4-evaluate`, `rfd4-train`, and `rfd4-train-ae` are [Hydra](https://hydra.cc)
  applications. Every option is set as `key=value`, never `--key value`. Use dotted paths for
  nested keys (`generation.args.nsteps=80`) and `+key=value` to add a key that has no default in
  the composed config (for example `+guidance=contract_ca`).
- **Required** means the underlying Hydra config sets that key to `???`; omitting it raises a
  Hydra `MissingMandatoryValue` error. **Optional** keys all ship a default, shown in the Default
  column.
- A **Type** of *group selector* means the value is the name of a Hydra config group (a YAML file
  under the matching `configs/` subdirectory), not an arbitrary scalar. Pass the group name as the
  value, e.g. `inference_experiment=sm_binder_hbond_search`.
- List-valued options take a bracketed, quoted Hydra list on the shell: `metrics_tags='[design,unconditional_refolding]'`.
- `${oc.env:DATA_PATH}` and similar are OmegaConf resolvers; they read the named environment
  variable at resolution time and are not something you type literally on the command line.
- **Entry point** and **base config**, given at the top of each section below, describe where that
  command's CLI surface comes from rather than something you supply: entry point is the Python
  function the installed command actually runs, and base config is the root Hydra YAML it composes
  against by default (setting some keys directly, pulling in others through its own `defaults:`
  list). Your `key=value` overrides are applied on top of that composed config, not in place of it.
- Run any command with `--help` for Hydra's own generated help, or `--cfg job` to print the fully
  resolved configuration without running anything.

## CLI options for `rfd4-generate`

Generates new designs (both structure and sequence) by running the model's flow-matching
sampling trajectory from a conditioning input, optionally under reward guided search, and can score
each design with inline metrics as it finishes.

- **Entry point:** `script_utils.entrypoints:generate` → `rfproteina.generate.main_hydra`.
- **Base config:** [`configs/inference.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/inference.yaml).

```bash
rfd4-generate inputs=data/benchmarks/monomer num_replicates=80 batch_size=8 out_dir=./runs/monomers
```

### Essential

The handful of options nearly every run sets explicitly: what to design, how many replicates to
generate, how large a batch to run at once, and which metrics to score inline. Their shipped
defaults suit a quick smoke test, not a full-scale run.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `inputs` | path, or group selector for a condition-spec file | **Required** | Annotated CIF/BCIF, JSON/YAML [condition spec](../how-to/input_prep/yaml-json.md), or a directory of them. See [Preparing inputs](../how-to/input_prep/index.md). | `inputs=data/benchmarks/monomer` |
| `out_dir` | string path or `null` | Optional — `null` (defaults to `./runs/<run_name_>`) | Output directory for samples and metrics. Set an external path for anything but a smoke test. | `out_dir=./runs/monomers` |
| `num_replicates` | int | Optional — `1` | Designs generated per input example. | `num_replicates=80` |
| `batch_size` | int | Optional — `1` | Designs processed in parallel on GPU. Confirm your longest input fits at `1` before raising it. | `batch_size=8` |
| `metrics_tags` | list of strings | Optional — `[]` | Tags selecting which metrics run inline after generation. See the [metrics reference](metrics.md) and [Running metrics inline](../how-to/evaluate-with-metrics.md#running-metrics-inline). | `metrics_tags='[unconditional_refolding]'` |

### Inputs and checkpoints

Before any sampling happens, the model needs to know what to condition on and which trained
weights to load. These options select the input examples for the run, control how reproducible
each run is, and point at the denoiser, autoencoder, and optional adapter checkpoints to load. See
the [checkpoints reference](checkpoints.md) for information about the available checkpoints.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `subset_to_examples` | comma-separated keys, or `null` | Optional — `null` | Restrict the run to specific input example keys. | `subset_to_examples=example_3,example_7` |
| `skip_existing` | bool | Optional — `true` | Skip examples whose output structure files already exist in `out_dir`. | `skip_existing=false` |
| `prevalidate_inputs` | bool | Optional — `false` | Sanity-check each unique input once before generation, failing fast on bad inputs. Also skips [sampling transforms](../how-to/run-the-model.md#sampling-transforms). | `prevalidate_inputs=true` |
| `seed` | int or `null` | Optional — `null` (random seed each run) | Random seed. | `seed=1234` |
| `torch_deterministic` | bool | Optional — `false` | Lock cuDNN/cuBLAS to deterministic algorithms. Significantly reduces throughput. | `torch_deterministic=true` |
| `hf_ckpt_path` | Hub repository, local bundle directory, or `null` | Optional — `nvidia/RFD4-Proteina` | Model bundle (denoiser and autoencoder) to load. When set, it overrides `ckpt_path`, `ckpt_name` and `autoencoder_ckpt_path`. Set `null` to load native checkpoints from `DATA_PATH`. | `hf_ckpt_path=null` |
| `hf_ckpt_revision` | Hub commit hash, or `null` | Optional — pinned release commit in `configs/inference.yaml` | Hub commit of the bundle, for reproducible loading. | `hf_ckpt_revision=<commit>` |
| `ckpt_path` | directory path | Optional — `${oc.env:DATA_PATH}/flow_checkpoints` | Directory holding the native denoiser checkpoint; used when `hf_ckpt_path=null`. Override only for a one-off model. | `ckpt_path=/shared/ckpts` |
| `ckpt_name` | filename | Optional — pinned release filename in `configs/inference.yaml` | Denoiser checkpoint filename. | `ckpt_name=my_model-ema.ckpt` |
| `autoencoder_ckpt_path` | file path | Optional — pinned release autoencoder path | Autoencoder checkpoint; fixes the latent space the denoiser was trained in. Must match the denoiser. | `autoencoder_ckpt_path=/shared/ae.ckpt` |
| `adapter_path` | directory path or `null` | Optional — `null` | Local PEFT adapter directory, loaded onto the exact base denoiser/autoencoder pair it was fine-tuned against. | `adapter_path=/shared/adapters/binder_v1` |
| `delete_failing_pre_oracle` | bool | Optional — `true` | When a [pre-oracle filter](metrics.md#pre-oracle-filters) is active and a design fails it, delete that design's structure files. | `delete_failing_pre_oracle=false` |

### Sampling

Sampling is the generative step itself: starting from noise (or, for partial diffusion, a noised
copy of an existing structure), the model integrates its flow-matching trajectory over a fixed
number of steps to produce a finished structure and sequence together. These options control that
trajectory: how many steps it takes, how much stochasticity and trunk recycling it uses, and
which high-level experiment preset composes generation, search, and metrics for the run.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `inference_experiment` | group selector | Optional — `inference_on_user_inputs` | Top-level experiment config composing generation, search, and metrics. Ships `inference_on_user_inputs`, `inference_on_val_set`, `inference_beam_example`, `inference_beam_regression`, `sm_binder_hbond_search`, `ppi_novelty_of3_search`, and the documented [`TEMPLATE`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/inference_experiment/TEMPLATE.yaml). | `inference_experiment=sm_binder_hbond_search` |
| `recycle_strategy` | group selector or `null` | Optional — `thinking_medium` | Trunk recycle-depth schedule: `<low|medium|high>/<flat|midpeak|front_loaded>` (60 steps; 120, 180 or 300 total trunk passes), or the `thinking_low`, `thinking_medium` and `thinking_high` aliases (front-loaded). See [Recycling schedules](../how-to/run-the-model.md#recycling-schedules). | `recycle_strategy=high/midpeak` |
| `generation` | group selector | Optional — `inference_base` | Base generation/sampling config. Ships `inference_base` and `inverse_folding`. | `generation=inverse_folding` |
| `generation.args.nsteps` | int | Optional — `60` | Number of model evaluations (integration steps) in the sampling schedule. | `generation.args.nsteps=80` |
| `generation.args.guidance_w` | float | Optional — `1.0` | Classifier-free guidance weight; `1.0` is no CFG, `0.0` excludes the main model. | `generation.args.guidance_w=1.0` |
| `generation.args.save_trajectory_every` | int | Optional — `0` | Save a trajectory snapshot every N steps; `0` disables it. | `generation.args.save_trajectory_every=10` |
| `generation.args.bb_ca_only_trajectories` | bool | Optional — `true` | Save CA-only XTC trajectories when `true`; save all-atom multi-model CIF trajectories when `false`. | `generation.args.bb_ca_only_trajectories=false` |
| `generation.args.dump_noisy_trajectories` | bool | Optional — `false` | Also save the noisy intermediate state at each saved trajectory step, alongside the clean prediction. | `generation.args.dump_noisy_trajectories=true` |
| `generation.args.condition_decoder_on_sequence_prediction` | bool | Optional — `true` | Condition the autoencoder decoder on the flow model's own sequence readout. Disable for checkpoints without a sequence readout. | `generation.args.condition_decoder_on_sequence_prediction=false` |
| `generation.model.bb_ca.simulation_step_params.sc_scale_noise` | float | Optional — `1.0` | Injected-noise variance multiplier for the backbone. See [Inference temperature parameters](../how-to/run-the-model.md#inference-temperature-parameters). | `generation.model.bb_ca.simulation_step_params.sc_scale_noise=0.5` |
| `generation.model.local_latents.simulation_step_params.sc_scale_noise` | float | Optional — `1.0` | Injected-noise variance multiplier for the sidechain and sequence latents. | `generation.model.local_latents.simulation_step_params.sc_scale_noise=0.5` |
| `generation.model.bb_ca.gt.mode` / `generation.model.local_latents.gt.mode` | string | Optional — `"1/t"` / `"tan"` | Set both to `ode` for deterministic sampling with no mid-trajectory noise injection. | `generation.model.bb_ca.gt.mode=ode generation.model.local_latents.gt.mode=ode` |
| `generation.model.bb_ca.simulation_step_params.tsr_k` / `tsr_sigma` | float | Optional — `16.0` / `3.0` | TSR temperature-rescaling controls for the backbone modality; see [Sampling with TSR](../how-to/run-the-model.md#sampling-with-tsr). | `generation.model.bb_ca.simulation_step_params.tsr_k=20.0` |
| `generation.model.local_latents.simulation_step_params.tsr_k` / `tsr_sigma` | float | Optional — `4.0` / `12.0` | TSR controls for the latent modality. | `generation.model.local_latents.simulation_step_params.tsr_sigma=10.0` |
| `n_recycle` | int | Optional — `0` | Fixed trunk recycle depth per step; only applies when `n_recycle_schedule` is `null`. | `n_recycle=2` |
| `n_recycle_schedule` | list of `[num_steps, recycle_depth]` bands, or `null` | Optional — `null` | Per-step-range recycle depth; overrides `n_recycle` when set. Set by the `recycle_strategy` group. | `n_recycle_schedule='[[20,1],[40,2]]'` |

### Partial diffusion

Partial diffusion starts the sampling trajectory from a noised copy of a structure you already
have, instead of pure noise, so the model refines that structure rather than building one from
scratch. See [Partial diffusion](../how-to/partial-diffusion.md#partial-diffusion) for the commands.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `generation.args.partial_trajectory_fraction_steps` | float in `(0, 1]`, or `null` | Optional — `null` | Simple interface: run the final fraction of `nsteps` (rounded to the nearest integer, at least one step). Mutually exclusive with the advanced `partial_t_*` keys. | `generation.args.partial_trajectory_fraction_steps=0.3` |
| `generation.args.partial_t_backbone` | float in `(0, 1)`, or `null` | Optional — `null` | Advanced interface: starting corruption level for the backbone modality. Near `1` stays close to the input; near `0` is closer to full generation. | `generation.args.partial_t_backbone=0.8` |
| `generation.args.partial_t_latent` | float in `(0, 1)`, or `null` | Optional — `null` | Advanced interface: starting corruption level for the latent modality. | `generation.args.partial_t_latent=0.8` |
| `generation.args.couple_partial_t` | bool | Optional — `false` | Derive one modality's `partial_t` from the other via the default schedule coupling, when only one is set. | `generation.args.couple_partial_t=true` |

### Inverse folding

Inverse folding is partial diffusion with the backbone held fixed throughout the trajectory: the
model resamples only the sequence and side-chain latents, so you get back new sequences that fold
to the same shape as your input backbone. See [Inverse folding](../how-to/partial-diffusion.md#inverse-folding) for the commands.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `generation` | group selector | Optional — `inference_base` | Set to `inverse_folding` to fix the backbone and resample sequence/side chains in one flag. | `generation=inverse_folding` |
| `generation.args.inverse_folding` | bool | Optional — `false` | Hold the input backbone fixed and resample only sequence and side chains. Set automatically by `generation=inverse_folding`. | `generation.args.inverse_folding=true` |
| `generation.args.inverse_folding_method` | string | Optional — `resampled_forward_marginal` | Sampling method; currently the only supported value. | — |

### Reward guided search

Reward guided search replaces one uninterrupted sampling trajectory with beam search: at
configured checkpoints along the trajectory, each surviving candidate branches into several
children, a reward model scores the branches, and only the strongest survive to continue. See
[Reward guided search](../how-to/search.md) for how the pieces fit together, and the
[search parameters reference](search-parameters.md) for every `search.*` field, including the
per-checkpoint schema.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `search.beam_width` | int | Optional — `4` | Number of candidates retained per input after each scored checkpoint. | `search.beam_width=3` |
| `search.n_branch` | int | Optional — `4` | Default number of children per surviving beam, for any checkpoint interval that doesn't override it. | `search.n_branch=2` |
| `search.save_rollouts` | bool | Optional — `false` | Also return completed scoring rollouts that lost selection, in addition to the final beam. | `search.save_rollouts=true` |
| `search.checkpoints` | mapping of step → checkpoint config | Optional — `{}` | Flat mapping of positive integer integration steps to per-checkpoint fields (`metrics`, `rewards`, `n_branch`, `use_clean_prediction`, `continuation_criteria`); see the [search parameters reference](search-parameters.md#per-checkpoint-parameters). Quote YAML keys to support `++search.checkpoints.20.use_clean_prediction=true` style dotted overrides. | `++search.checkpoints.40.rewards.oracle.plddt_coi.weight=1.0` |

### Metric guidance

Metric guidance nudges the sampling trajectory toward a better score on a differentiable metric: at
each active step it differentiates the metric with respect to the model's predicted clean state and
adds a weighted gradient step to the sampler, rather than changing training data or architecture.
See [Metric guidance](../concepts/metric-guidance.md) for the concept and
[Metric guidance](../how-to/advanced/metric-guidance.md) for the YAML. Currently wired to standard full
generation only.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `+guidance` | group selector (newly added key) | Optional — no `guidance` group by default | Adds a metric-guidance config. The shipped `contract_ca` preset is a mechanics test, not a design objective. | `+guidance=contract_ca` |
| `guidance.metrics.<name>._target_` | Hydra target (Python class path) | Required within a `guidance` config | Differentiable `Metric` class constructed for guidance. | `guidance.metrics.contract_ca._target_=rfproteina.metrics.contract_ca_metric.ContractCAMetric` |
| `guidance.terms.<metric>.<output_key>.<modality>` | Hydra target (schedule callable) | Required within a `guidance` config | Signed schedule mapping one `metric_name.output_key` and one modality (`bb_ca` or `local_latents`) to a weight over flow-matching time. | see [Build a real objective](../how-to/advanced/metric-guidance.md#build-a-real-objective) |

### Output and compute

These options control what gets written to disk and how the run uses compute, independent of what
is actually generated: output compression, numeric precision, trajectory and chain-of-interest
exports, and dataloader worker counts all live here.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `verbose` | bool | Optional — `false` | Routine warnings and diagnostic dumps. Defaults `true` for training and standalone evaluation. | `verbose=true` |
| `compress_outputs` | bool | Optional — `false` | Compress persistent structure outputs (samples, refolds, trajectories) with Zstandard level 3 instead of writing plain CIF. See [Output format](output-format.md). | `compress_outputs=true` |
| `save_chain_of_interest` | bool | Optional — `false` | Write per-design chain-of-interest CIFs. Needed only for [FoldSeek diversity clustering](../how-to/run-the-model.md#foldseek-diversity-clustering). | `save_chain_of_interest=true` |
| `compile_lengths` | list of ints, or `null` | Optional — 15 buckets from 128 to 1024 | Token counts (multiples of 16) that inputs are padded to so the compiled trunk sees static shapes. `null` runs the trunk eagerly. The `nn.*` rows below belong to the saved network but may change after training without touching the weights; `null` keeps the checkpoint's value, and any other `nn.*` key is rejected. | `compile_lengths=null` |
| `nn.torch_compile_mode` | string, or `null` | Optional — null (checkpoint: default mode) | `torch.compile` mode of the trunk blocks. `reduce-overhead` replays CUDA graphs: the sampler restarts the graph tree before every trunk call, so the varying recycle depth replays recorded graphs instead of re-recording them. It pays off where a step is host-bound (short sequences, small batches). | `nn.torch_compile_mode=reduce-overhead` |
| `nn.torch_compile_fullgraph` | bool, or `null` | Optional — null (released checkpoint: single graph per block) | Whether each trunk block must compile as a single graph; `false` lets a block split at graph breaks. | `nn.torch_compile_fullgraph=false` |
| `nn.tri_mult_backend` | string, or `null` | Optional — null (keeps the checkpoint's value; `cuequivariance` when it saved none) | `torch` or `cuequivariance` triangle multiplication. The cuEquivariance kernel runs inside the compiled trunk for the shapes it supports; unsupported shapes warn once and run the `torch` formula, as does an eager trunk (`compile_lengths=null`). | `nn.tri_mult_backend=cuequivariance` |
| `nn.attention_type` | string, or `null` | Optional — null (keeps the checkpoint's value; `cuequivariance` when it saved none) | `torch` (SDPA) or `cuequivariance` pair-biased attention. | `nn.attention_type=cuequivariance` |
| `precision` | string | Optional — `bf16-mixed` | Numerical precision. See the analogous [training precision option](#training-precision-and-optimization) for the full set of values. | `precision=bf16-mixed` |
| `num_workers` | int | Optional — `4` | Dataloader worker processes. | `num_workers=8` |

### Hardware and cluster job settings

Set automatically by `scripts/launch_generate.sh`; override directly only when running `rfd4-generate` by hand.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `hardware.accelerator` | string | Optional — `gpu` | Lightning accelerator. | `hardware.accelerator=cpu` |
| `hardware.ngpus_per_node_` | int | Optional — `1` | GPUs per node. | `hardware.ngpus_per_node_=4` |
| `hardware.nnodes_` | int | Optional — `1` | Number of nodes. | `hardware.nnodes_=2` |
| `ncpus_` | int | Optional — `24` | CPUs per task, used by `launch_infer_from_local.sh`. | `ncpus_=16` |
| `gen_njobs` | int | Optional — `1` | Number of parallel generation jobs per config. | `gen_njobs=8` |
| `eval_njobs` | int | Optional — `1` | Number of parallel evaluation jobs per config. | `eval_njobs=8` |
| `job_id` | int or `null` | Optional — `null` | Set by `launch_generate.sh` to disambiguate per-job output files. | — |
| `run_name_` | string | Optional — `inference` | Run name; used to derive `out_dir` when it is `null`. | `run_name_=binder_sweep_01` |
| `hydra.run.dir` | path | Optional — `./hydra_outputs` | Where Hydra writes its own run logs/config dump. Standard Hydra key, not RFD4-Proteina specific. | `hydra.run.dir=./hydra_outputs/my_run` |

## CLI options for `rfd4-evaluate`

Computes metrics, pre-oracle filters, and refolding-oracle scores against structures that already
exist on disk — already generated designs, or any other CIF-like files — without generating
anything itself.

- **Entry point:** `rfproteina.evaluate:main_hydra`.
- **Base config:** [`configs/evaluate.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/evaluate.yaml). Runs the same
  [metrics](metrics.md), [pre-oracle filters](metrics.md#pre-oracle-filters), and
  [refolding oracles](metrics.md#refolding-oracles) as `rfd4-generate`.

```bash
rfd4-evaluate inputs=./runs/monomers/inference_samples metrics_tags='[unconditional_refolding]' out_dir=./runs/monomers_eval
```

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `inputs` | directory path | **Required** | Directory of CIF-like files for already generated/designed structures. | `inputs=./runs/monomers/inference_samples` |
| `metrics_tags` | list of strings | **Required** | Tags controlling which metrics activate. See the [metrics reference](metrics.md). | `metrics_tags='[unconditional_refolding]'` |
| `out_dir` | directory path | **Required** | Output directory for metrics results. | `out_dir=./runs/monomers_eval` |
| `task_csv` | CSV path, or `null` | Optional — `null` | Optional CSV with `designed_structure_path` and `task` columns mapping each input file to a task label, for per-task FoldSeek clustering. A `per_example_metrics.csv` with a `task` column also works directly. | `task_csv=./tasks.csv` |
| `job_id` | int or `null` | Optional — `null` | Chunk index for parallel evaluation. | `job_id=0` |
| `gen_njobs` | int | Optional — `1` | Total number of parallel jobs, used when `chunk_size` is `null`. | `gen_njobs=4` |
| `compress_outputs` | bool | Optional — `false` | Applies to newly written refolds and chain exports; input structures are read unchanged. | `compress_outputs=true` |
| `delete_failing_pre_oracle` | bool | Optional — `false` (differs from `rfd4-generate`'s default, to avoid deleting input structures) | Delete structure files for designs that fail an active pre-oracle filter. | `delete_failing_pre_oracle=true` |
| `verbose` | bool | Optional — `true` | Routine warnings and diagnostic dumps. | `verbose=false` |
| `run_name_` | string | Optional — `evaluate` | Run name. | `run_name_=monomer_eval_01` |
| `hydra.run.dir` | path | Optional — `./hydra_outputs` | Where Hydra writes its own run logs/config dump. | `hydra.run.dir=./hydra_outputs/eval_01` |

## CLI options for `rfd4-train`

Trains the latent flow-matching denoiser — the generative model `rfd4-generate` samples from —
against a frozen autoencoder checkpoint that fixes the latent space it trains in.

- **Entry point:** `rfproteina.train:main`.
- **Base config:** [`configs/training_proteina.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/training_proteina.yaml). A
  composition root, not a runnable experiment snapshot by itself — most runs select an
  `experiment` and set `datasets.train`.

```bash
rfd4-train datasets.train=<dataset-group> out_dir=/shared/runs/my_training_run
```

### Datasets and run identity

Every training run composes a dataset mixture, a network architecture, and a named experiment
recipe before any hyperparameter below takes effect. These options are that composition layer,
plus the run name and output directory the resulting checkpoints and samples land under.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `datasets.train` | group selector / mapping | **Required** | Training dataset composition. Validation datasets are optional. | `datasets.train=pdb_interfaces` |
| `experiment` | group selector | Optional — `sota` | Version-controlled hyperparameter recipe layered over the composition root. | `experiment=sota` |
| `debug` | group selector or `null` | Optional — `null` | Optional overlay for quick local testing/debugging; ships `cpu` and `gpu`. | `+debug=cpu` |
| `nn` | group selector | Optional — `local_latents_score_nn_160M_atom37` | Denoiser network architecture config. | `nn=local_latents_score_nn_160M_atom37` |
| `dataloaders` | group selector | Optional — `default` | Dataloader configuration (batch sizes, workers, samplers). | `dataloaders.train.dataloader_params.batch_size=8` |
| `generation` | group selector | Optional — `inference_base` | Sampling config used for validation-time generation. Same keys as [`rfd4-generate`'s sampling options](#sampling). | `generation.args.nsteps=30` |
| `metrics` | group selector | Optional — `all_metrics` | Metric set evaluated during validation. See the [metrics reference](metrics.md). | — |
| `callbacks` | group selector | Optional — `all_tasks` | Lightning callback set. | — |
| `run_name_` | string | Optional — `test_no_val_datasets` | Run name; used to derive `out_dir` when it is `null`. | `run_name_=sota_run_04` |
| `out_dir` | path or `null` | Optional — `null` (defaults to `./runs/<run_name_>/`) | Output directory for checkpoints, samples, and metrics. | `out_dir=/shared/runs/sota_run_04` |

### Checkpoint loading

See [Training-time checkpoint roles](checkpoints.md#training-time-checkpoint-roles) for the full
semantics; the two paths are mutually exclusive.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `pretrain_ckpt_path` | file path or `null` | Optional — `null` | Start a new run from saved weights only; resets optimizer, scheduler, step, epoch, and RNG state. | `pretrain_ckpt_path=/shared/ckpts/base.ckpt` |
| `resume_ckpt_path` | file path or `null` | Optional — `null` | Resume an interrupted run, restoring full Lightning trainer state. | `resume_ckpt_path=/shared/runs/sota_run_04/last.ckpt` |
| `autoencoder_ckpt_path` | file path | Optional — pinned release autoencoder path | Frozen autoencoder the denoiser trains against; fixes the latent space. | `autoencoder_ckpt_path=/shared/ae.ckpt` |
| `weight_loading` | Hydra target (`rfproteina.utils.weights.WeightLoadingConfig`) or `null` | Optional — `null` (strict load) | Honored only on the `pretrain_ckpt_path` path. When `null`, any key/shape mismatch against the configured architecture is a hard error. Set a `default_policy`, `fallback_policy`, and `param_policies` to deliberately load across an architecture change. | see `configs/training_proteina.yaml` for a worked example |
| `parameter_freezing` | Hydra target (`rfproteina.utils.weights.ParameterFreezingConfig`) or `null` | Optional — `null` (nothing frozen) | Honored only on the `pretrain_ckpt_path` path. Freezes selected parameters by name pattern after loading. | `parameter_freezing.freeze_by_default=false` |
| `delete_failing_pre_oracle` | bool | Optional — `true` | Delete structure files for validation designs that fail an active pre-oracle filter. | `delete_failing_pre_oracle=false` |

### LoRA fine-tuning

Low-Rank Adaptation (LoRA) fine-tunes a pretrained denoiser by training a small set of additional
low-rank weights instead of the full model, so a fine-tuning run is cheaper and its result composes
back onto the base checkpoint as an adapter at inference time (see `adapter_path` in the
[`rfd4-generate` inputs and checkpoints table](#inputs-and-checkpoints)). Leave `lora.r` unset to
train the full model instead.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `lora.r` | int or `null` | Optional — `null` (disabled, full model trained) | LoRA rank. Typical values are `8`, `16`, `32`, `64`. | `lora.r=16` |
| `lora.lora_alpha` | float | Optional — `64.0` | LoRA scaling factor. | `lora.lora_alpha=32.0` |
| `lora.lora_dropout` | float | Optional — `0.0` | Dropout for LoRA layers. | `lora.lora_dropout=0.05` |
| `lora.train_bias` | string | Optional — `none` | Bias training policy; keeps base biases frozen by default. | — |
| `lora.target_modules` | list of strings | Optional — `[to_qkv, to_out_node]` | Denoiser attention projections to adapt, relative to `model.nn`. | `lora.target_modules='[to_qkv]'` |

### Training precision and optimization

These options control the training loop itself — numeric precision, learning rate and warmup,
gradient clipping and accumulation, validation cadence, and the trunk recycling schedule — rather
than the dataset or architecture being trained.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `precision` | string | Optional — `bf16-mixed` | One of `"32"` (full precision), `"bf16-mixed"` (default), or `"fp8"` (requires Transformer Engine). | `precision="32"` |
| `force_precision_f32` | bool | Optional — `false` | **Deprecated**; use `precision: "32"` instead. | — |
| `seed` | int | Optional — `42` | Random seed. | `seed=0` |
| `torch_deterministic` | bool | Optional — `false` | Lock cuDNN/cuBLAS to deterministic algorithms; significantly reduces throughput. | `torch_deterministic=true` |
| `opt.lr` | float | Optional — `0.0001` | Learning rate. | `opt.lr=5e-5` |
| `opt.muon_warmup_steps` | int | Optional — `0` | Linear warmup steps (optimizer updates) to `opt.lr` for Muon-optimized parameters; `0` disables warmup. | `opt.muon_warmup_steps=500` |
| `opt.adam_warmup_steps` | int | Optional — `0` | Linear warmup steps to `opt.lr` for Adam-optimized parameters. | `opt.adam_warmup_steps=500` |
| `opt.gradient_clip_val` | float | Optional — `1.0` | Max gradient norm. | `opt.gradient_clip_val=0.5` |
| `opt.max_epochs` | int | Optional — `10000000` | Maximum training epochs. | `opt.max_epochs=1000` |
| `opt.accumulate_grad_batches` | int | Optional — `1` | Gradient accumulation steps. | `opt.accumulate_grad_batches=4` |
| `opt.dist_strategy` | string | Optional — `ddp` | Lightning distributed strategy. | `opt.dist_strategy=ddp` |
| `opt.steps_per_epoch` | int | Optional — `2000` | Defines epoch length in optimizer steps, not full dataset passes. | `opt.steps_per_epoch=1000` |
| `opt.val_every_n_epochs` | int | Optional — `10` | Validate at the end of every N epochs. | `opt.val_every_n_epochs=5` |
| `opt.val_start_epoch` | int | Optional — `0` | Delay validation until this epoch; `0` starts immediately. | `opt.val_start_epoch=20` |
| `opt.compute_val_loss` | bool | Optional — `false` | Global switch for per-dataset validation-loss logging. | `opt.compute_val_loss=true` |
| `opt.foldseek_timeout_minutes` | int | Optional — `25` | Wall-clock budget for FoldSeek diversity clustering per validation epoch. | `opt.foldseek_timeout_minutes=15` |
| `opt.muon_exclude_param_patterns` | list of regex strings | Optional — ships two default patterns excluding embedding/output-head params | Parameters kept on Adam instead of Muon, matched by name pattern. Muon also never sees non-2D params (norms, biases). | — |
| `opt.muon_adjust_lr_fn` | string | Optional — `match_rms_adamw` | Either `original` or `match_rms_adamw`. | `opt.muon_adjust_lr_fn=original` |
| `opt.muon_grouped` | bool | Optional — `true` | Batch equal-shaped FP32 Muon updates; `false` selects native Muon. | `opt.muon_grouped=false` |
| `training.n_recycle_max` | int | Optional — `1` | Trunk recycle depth is sampled per step and clamped to `[1, n_recycle_max]`. | `training.n_recycle_max=3` |
| `training.n_recycle_sampler` | string | Optional — `uniform` | `uniform` (U{1..n_recycle_max}) or `poisson` (Poisson(`n_recycle_lambda`), clamped). | `training.n_recycle_sampler=poisson` |
| `training.n_recycle_lambda` | float | Optional — `3.0` | Poisson mean when `n_recycle_sampler=poisson`. | `training.n_recycle_lambda=2.0` |
| `training.n_recycle_sync` | bool | Optional — `false` | Derive the recycle-depth draw from `global_step` so all DDP ranks pick the same depth. | `training.n_recycle_sync=true` |
| `training.n_recycle_val_loss` | int | Optional — `1` | Fixed recycle depth for validation-loss forward passes; generation uses its own recycle schedule. | `training.n_recycle_val_loss=2` |
| `training.motif_conditioning` | bool | Optional — `false` | Enable motif-conditioning training behavior. | `training.motif_conditioning=true` |

### Hardware and logging

These options control where training runs — accelerator type, GPU and node counts — and how its
progress gets recorded: Weights & Biases logging and checkpoint-saving cadence.

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `hardware.accelerator` | string | Optional — `gpu` | Lightning accelerator. | `hardware.accelerator=gpu` |
| `hardware.ngpus_per_node_` | int | Optional — `1` | GPUs per node. | `hardware.ngpus_per_node_=8` |
| `hardware.nnodes_` | int | Optional — `1` | Number of nodes. | `hardware.nnodes_=4` |
| `log.wandb_project` | string | Optional — `latent_training` | Weights & Biases project name. | `log.wandb_project=my_project` |
| `log.wandb_entity` | string or `null` | Optional — `${oc.env:WANDB_ENTITY,null}` | Weights & Biases entity; read from the `WANDB_ENTITY` environment variable by default. | `log.wandb_entity=my_team` |
| `log.tags` | list of strings | Optional — `[]` | Weights & Biases run tags. | `log.tags='[ablation]'` |
| `log.log_wandb` | bool | Optional — `true` | Enable Weights & Biases logging. | `log.log_wandb=false` |
| `log.log_every_n_steps` | int | Optional — `10` | Logging frequency, in steps. | `log.log_every_n_steps=50` |
| `log.checkpoint` | bool | Optional — `true` | Whether to save checkpoints at all. | `log.checkpoint=false` |
| `log.checkpoint_every_n_epochs` | int | Optional — `10` | Save a top-k checkpoint every N epochs; matches validation frequency. | `log.checkpoint_every_n_epochs=5` |
| `log.last_ckpt_every_n_epochs` | int | Optional — `1` | Save `last.ckpt` every N epochs, for resume. | `log.last_ckpt_every_n_epochs=1` |
| `compress_outputs` | bool | Optional — `true` | Compress validation structures, refolds, and CIF trajectories. | `compress_outputs=false` |
| `verbose` | bool | Optional — `true` | Routine warnings and diagnostic dumps. | `verbose=false` |
| `hydra.run.dir` | path | Optional — `./hydra_outputs` | Where Hydra writes its own run logs/config dump. | `hydra.run.dir=./hydra_outputs/train_01` |

## CLI options for `rfd4-train-ae`

Trains the partial autoencoder that encodes and decodes the local-latents representation the
denoiser trains and samples against — independently of any denoiser training run.

- **Entry point:** `rfproteina.partial_autoencoder.train:main`.
- **Base config:** [`configs/training_ae.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/training_ae.yaml).

```bash
rfd4-train-ae out_dir=/shared/runs/my_ae_run
```

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `run_name_` | string | Optional — `ae_z16` | Run name; used to derive `out_dir` when it is `null`. | `run_name_=ae_z32_run` |
| `out_dir` | path or `null` | Optional — `null` (defaults to `./runs/<run_name_>/`) | Output directory for checkpoints, samples, and metrics. | `out_dir=/shared/runs/ae_z32_run` |
| `nn_ae` | group selector | Optional — `nn_130m_unified` | Autoencoder network architecture config. | `nn_ae=nn_130m_unified` |
| `datasets` | group selector | Optional — `ae_pdb_datasets` | Training/validation dataset composition. | — |
| `dataloaders` | group selector | Optional — `default` | Dataloader configuration. | `dataloaders.train.dataloader_params.batch_size=12` |
| `callbacks` | group selector | Optional — `autoencoder` | Lightning callback set. | — |
| `experiment_ae` | group selector | Optional — `ae_sota` | Version-controlled hyperparameter recipe. | `experiment_ae=ae_sota` |
| `debug` | group selector or `null` | Optional — `null` | Optional overlay for quick local testing/debugging; ships `cpu` and `gpu`. | `+debug=cpu` |
| `pretrain_ckpt_path` | file path or `null` | Optional — `null` | Optional checkpoint to start training from. | `pretrain_ckpt_path=/shared/ckpts/ae_base.ckpt` |
| `hardware.ncpus_per_task_train_` | int | Optional — `16` | CPUs per task during training. | `hardware.ncpus_per_task_train_=32` |
| `hardware.accelerator` | string | Optional — `gpu` | Lightning accelerator. | `hardware.accelerator=cpu` |
| `hardware.ngpus_per_node_` | int | Optional — `1` | GPUs per node. | `hardware.ngpus_per_node_=4` |
| `hardware.nnodes_` | int | Optional — `1` | Number of nodes. | `hardware.nnodes_=2` |
| `loss.kl.weight` | float | Optional — `1e-4` | KL divergence loss weight. | `loss.kl.weight=1e-3` |
| `loss.kl.anneal` | bool | Optional — `false` | Anneal the KL weight over training. | `loss.kl.anneal=true` |
| `loss.kl.patience` | int | Optional — `10000` | Annealing patience, in steps. | `loss.kl.patience=5000` |
| `loss.struct.type` | string | Optional — `l2` | Structure loss type, e.g. `l2` or `l2cut_<threshold>`. | `loss.struct.type=l2cut_0.01` |
| `loss.struct.weight` | float | Optional — `1.0` | Structure loss weight. | `loss.struct.weight=2.0` |
| `loss.struct.atom_permutation` | bool | Optional — `false` | Permute chemically equivalent ground-truth atoms onto the prediction before scoring. Requires `dataset.transform.precompute_atom_permutation`. | `loss.struct.atom_permutation=true` |
| `loss.struct.atom_permutation_align` | bool | Optional — `false` | Decide the atom-permutation labelling in a superimposed frame rather than the raw one. | `loss.struct.atom_permutation_align=true` |
| `loss.seq.weight` | float | Optional — `2.0` | Sequence loss weight. | `loss.seq.weight=1.0` |
| `force_precision_f32` | bool | Optional — `false` | Use full `fp32` precision instead of `bf16-mixed` when `true`. | `force_precision_f32=true` |
| `opt.lr` | float | Optional — `0.0001` | Learning rate. | `opt.lr=5e-5` |
| `opt.gradient_clip_val` | float | Optional — `1.0` | Max gradient norm. | `opt.gradient_clip_val=0.5` |
| `opt.muon_exclude_param_patterns` | list of regex strings | Optional — ships three default patterns excluding embedding/output-head params | Parameters kept on Adam instead of Muon. | — |
| `opt.muon_adjust_lr_fn` | string | Optional — `match_rms_adamw` | Same semantics as [`rfd4-train`'s option](#training-precision-and-optimization). | `opt.muon_adjust_lr_fn=original` |
| `opt.muon_grouped` | bool | Optional — `true` | Batch equal-shaped FP32 Muon updates. | `opt.muon_grouped=false` |
| `opt.max_epochs` | int | Optional — `10000000` | Maximum training epochs. | `opt.max_epochs=1000` |
| `opt.accumulate_grad_batches` | int | Optional — `1` | Gradient accumulation steps. | `opt.accumulate_grad_batches=4` |
| `opt.dist_strategy` | string | Optional — `ddp` | Lightning distributed strategy. | `opt.dist_strategy=ddp` |
| `opt.steps_per_epoch` | int | Optional — `2000` | Defines epoch length in steps. | `opt.steps_per_epoch=1000` |
| `opt.val_every_n_epochs` | int | Optional — `10` | Validate at the end of every N epochs. | `opt.val_every_n_epochs=5` |
| `opt.val_start_epoch` | int | Optional — `0` | Delay validation until this epoch. | `opt.val_start_epoch=20` |
| `log.wandb_project` | string | Optional — `ae_training` | Weights & Biases project name. | `log.wandb_project=my_ae_project` |
| `log.wandb_entity` | string or `null` | Optional — `${oc.env:WANDB_ENTITY,null}` | Weights & Biases entity. | `log.wandb_entity=my_team` |
| `log.tags` | list of strings | Optional — `["ae"]` | Weights & Biases run tags. | `log.tags='[ablation]'` |
| `log.log_wandb` | bool | Optional — `true` | Enable Weights & Biases logging. | `log.log_wandb=false` |
| `log.log_every_n_steps` | int | Optional — `10` | Logging frequency, in steps. | `log.log_every_n_steps=50` |
| `log.checkpoint` | bool | Optional — `true` | Whether to save checkpoints. | `log.checkpoint=false` |
| `log.checkpoint_every_n_epochs` | int | Optional — `10` | Save a top-k checkpoint every N epochs. | `log.checkpoint_every_n_epochs=5` |
| `log.last_ckpt_every_n_epochs` | int | Optional — `1` | Save `last.ckpt` every N epochs. | `log.last_ckpt_every_n_epochs=2` |
| `log.projection_method` | string or `null` | Optional — `tsne` | Latent-space projection method for training-time visualization: `tsne`, `pca`, `lda`, or `null` to disable. | `log.projection_method=pca` |
| `log.projection_every_n` | int | Optional — `1000` | Log a projection plot every N steps. | `log.projection_every_n=500` |
| `log.projection_max_samples` | int | Optional — `5000` | Max tokens projected per plot; subsampled if needed. | `log.projection_max_samples=2000` |
| `log.save_projections_locally` | bool | Optional — `false` | Save projection plots to local disk. | `log.save_projections_locally=true` |
| `log.local_projection_dir` | path | Optional — `./latent_projections` | Directory for saved projection plots. | `log.local_projection_dir=./my_projections` |
| `seed` | int | Optional — `42` | Random seed. | `seed=0` |
| `torch_deterministic` | bool | Optional — `false` | Lock cuDNN/cuBLAS to deterministic algorithms. | `torch_deterministic=true` |
| `compress_outputs` | bool | Optional — `true` | Compress validation structure samples. | `compress_outputs=false` |
| `verbose` | bool | Optional — `true` | Routine warnings and diagnostic dumps. | `verbose=false` |

## Aggregating multi-job metrics

`rfproteina.inference_utils.aggregate_metrics` is a standalone [`fire`](https://github.com/google/python-fire)
CLI, not a Hydra app: pass `--flag value`, not `key=value`. `launch_generate.sh` runs it
automatically as a dependent SLURM job; invoke it directly to concatenate manually. See
[Multi-job runs](../how-to/advanced/multi-job-runs.md).

```bash
python -m rfproteina.inference_utils.aggregate_metrics ./inference/my_run/metrics/ --cluster_with_foldseek
```

| Option | Type | Required/Default | Description | Example |
| --- | --- | --- | --- | --- |
| `input` (positional) | directory or file path | **Required** | Directory containing per-job result files, or an already-concatenated `per_example_metrics.csv`. | `./inference/my_run/metrics/` |
| `--cluster_with_foldseek` | bool | Optional — `false` | Also run [FoldSeek structural diversity clustering](../how-to/run-the-model.md#foldseek-diversity-clustering). | `--cluster_with_foldseek` |
| `--cleanup` | bool | Optional — `true` | Move individual job result files into `individual_job_results/` after concatenation. | `--cleanup=false` |
| `--cleanup_chains` | bool | Optional — `true` | Delete the `foldseek_input_chains/` directory after clustering completes; the chain CIFs are regenerable. | `--cleanup_chains=false` |
| `--output_dir` | directory path, or `null` | Optional — `null` (input's directory, or its parent when input is a file) | Directory where aggregated results are saved. | `--output_dir=./inference/my_run/metrics_agg` |
| `--foldseek_threads` | int, or `null` | Optional — `null` (all available CPUs) | CPU threads passed to FoldSeek. Set explicitly when running several clustering jobs on one node. | `--foldseek_threads=8` |
| `--sequence_probability_workers` | int, or `null` | Optional — `null` (up to 8 CPUs from the SLURM allocation or process affinity) | Worker processes used to reconstruct sequence-probability observations from output CIFs. | `--sequence_probability_workers=4` |

## Cluster launcher scripts

`scripts/launch_generate.sh`, `scripts/launch_evaluate.sh`, `scripts/launch_train_latents.sh`,
`scripts/launch_train_ae.sh`, `scripts/launch_search.sh`, `scripts/launch_infer.sh`,
`scripts/launch_infer_ae.sh`, and `scripts/launch_sweep.sh` wrap the commands above for SLURM or
local submission. They accept the same Hydra `key=value` overrides (passed straight through to
the underlying entry point) plus their own scheduler flags (`--njobs`, resource overrides, and
similar), which differ per launcher and are controlled in part by your
`scripts/.user_info_<cluster>` profile.

This page does not duplicate those flags, since preserving the exact scheduler intent (partition,
account, GPU type, time limit, `SLURM_EXTRA_OPTIONS`, …) matters more than a generic list — read
the header of the specific launcher you're using before changing or invoking it, as documented in
`AGENTS.md`.
