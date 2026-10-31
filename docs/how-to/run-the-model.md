# Running the model

One command turns an annotated CIF into designs. This page covers that command, the knobs worth
touching, and what to do when a run misbehaves.

Just getting started? [Quickstart](../quickstart.rst) installs the model and generates a first
design. [Preparing inputs](input_prep/index.md) covers where the CIF comes from.

With rewards or metrics enabled, a run also writes a metrics CSV next to the designs:

```{image} /_static/diagrams/design-input-pipeline-metrics-light.svg
:alt: The same input routes write an annotated CIF that goes into RFD4 together with Rewards (Metrics). RFD4 outputs an annotated CIF and a Metrics CSV.
:class: only-light
:width: 100%
:align: center
```

```{image} /_static/diagrams/design-input-pipeline-metrics-dark.svg
:alt: The same input routes write an annotated CIF that goes into RFD4 together with Rewards (Metrics). RFD4 outputs an annotated CIF and a Metrics CSV.
:class: only-dark
:width: 100%
:align: center
```

Choose the sampling workflow that matches what supplies the extra information:

- [Standard generation](#running-inference) starts from noise.
- [Partial diffusion and inverse folding](../concepts/partial-diffusion.md) start from a supplied structure.
- [Metric guidance](../concepts/metric-guidance.md) differentiates a scalar objective during sampling.

## Running inference

The general form:

```bash
rfd4-generate inputs=<file> num_replicates=<N> out_dir=<dir> [overrides...]
```

Activate the environment where you installed the project, for example with
`source .venv/bin/activate`. `rfd4-generate` uses that environment's Python;
without activation, use `.venv/bin/rfd4-generate`.

For SLURM arrays, GPU workers, and metric aggregation, use
`scripts/launch_generate.sh …` without a cluster argument. Refolding requires the
optional oracle setup described in
[installation](../installation.rst).

- **Inputs.** Use an **annotated CIF** ([guide](input_prep/index.md) ·
  [tutorial](../examples/python_api_tutorial.ipynb)) or a
  [condition-spec JSON/YAML file](input_prep/yaml-json.md). `data/benchmarks/` has ready made examples.
- **Every model argument is `key=value`**, Hydra style, never `--key value`.
- **Checkpoints** load from the pinned Hugging Face bundle in `configs/inference.yaml` by default.
  For native files, set `hf_ckpt_path=null` and supply `ckpt_path=<dir>`,
  `ckpt_name=<file.ckpt>` and the matching `autoencoder_ckpt_path=<file.ckpt>`.
- **`batch_size`** of 8 is a good place to start. Bigger batches run faster, but confirm your
  longest input fits at `batch_size=1` first.

```bash
# 80 designs of each example in a JSON, 8 at a time
rfd4-generate \
    inputs=data/benchmarks/monomer \
    num_replicates=80 batch_size=8 out_dir=./runs/monomers
```

:::{tip}
Want a reusable config instead of one off CLI overrides? Start from
[`configs/inference_experiment/TEMPLATE.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/inference_experiment/TEMPLATE.yaml).
It documents all three ways to run inference, which are an in tree experiment config, CLI
overrides, and a config file anywhere on the filesystem.
:::

## Recycling schedules

Select a task-specific `recycle_strategy` below. All presets use 60 sampling steps;
parentheses give total trunk passes, including the first pass at each step.

| Task | Low | Medium | High |
| --- | --- | --- | --- |
| PPI | `low/front_loaded` (120) | `medium/front_loaded` (180) | `high/front_loaded` (300) |
| Fixed/unfixed DNA | `low/front_loaded` (120) | `medium/front_loaded` (180) | `high/flat` (300) |
| MCSA, small-molecule binders (including DISCO) | `low/midpeak` (120) | `medium/midpeak` (150) | `high/midpeak` (180) |
| Inverse folding | `low/flat` (120) | `medium/flat` (180) | `high/flat` (300) |

For example, use `recycle_strategy=medium/midpeak` for small-molecule design.
`thinking_low/medium/high` remain aliases for front-loaded presets, not task-aware
selectors. The CLI defaults to `thinking_medium`; training validation uses each
task's medium preset. These task recommendations were selected using benchmark results.

## Verbose output

`verbose` controls many info/warning dumps. It defaults to `false` for generation
and `true` everywhere else. Override it in the hydra configs or on the command line.

## Evaluating results

Generation can score designs as it makes them, or you can score a folder of structures you already
have. See [Evaluate with metrics](evaluate-with-metrics.md), which also covers the AF3 oracle setup
that refolding metrics and reward guided search need.

### Multi-job runs

See [Multi-job runs](advanced/multi-job-runs.md) for concatenating `per_example_metrics_job{N}.csv`
files from parallel SLURM jobs.

Direct multi-GPU generation uses distinct random streams for each global rank,
derived reproducibly from `seed + job_id` and the rank. This applies to both native
and BioIR inference. The effective rank seed is logged and written to the metrics
CSV's `seed` column. Single-GPU runs, including each worker launched by
`scripts/launch_generate.sh`, retain the existing `seed + job_id` behavior.
Changing the number of ranks, batch size, or dataloader workers can change the
generated designs; a rank seed alone does not identify a sample independently of
those settings. Distinct streams do not guarantee distinct protein sequences.

### Adding custom metrics

You can add your own metrics without touching a tracked file. Two **git ignored** directories hold
them, so your local experiments stay private.

- `src/rfproteina/metrics/custom_metrics/` holds Python metric classes, split across as many
  modules as you like.
- `configs/metrics/custom_metrics/` holds the Hydra config that registers them.
  `all_custom_metrics.yaml` is the entry point, and `setup.sh` scaffolds it on first run.

The config gets picked up automatically, with no extra flags, and results land in
`per_example_metrics.csv` prefixed by the config name, like `residue_count.n_residues`.

The [custom metrics guide](advanced/custom-metrics.md) walks through writing a metric class, refolding
metrics, splitting across files, and contributing one back.

### FoldSeek diversity clustering

See [FoldSeek diversity clustering](advanced/foldseek.md) for clustering structures by structural
similarity, the chain-of-interest extraction it needs, and `FOLDSEEK_EXEC` setup.

## Sampling transforms

By default every replicate of an input file conditions on exactly the same annotated CIF. A
**sampling transform** runs once per replicate, so each design can start from a different
condition. That helps when a conditioning value is a guess rather than a measurement, like the CA
position of a residue whose structure you do not know.

Write one as an AtomWorks `Transform` whose `forward` edits `data["atom_array"]` in place. Drop it
in `src/rfproteina/transforms/sampling/`, which is git ignored and created by `setup.sh`, so your
code stays local. Same arrangement as [custom metrics](advanced/custom-metrics.md).

```python
# src/rfproteina/transforms/sampling/jitter_chain.py
import numpy as np
from atomworks.ml.transforms.base import Transform


class JitterChainCoordinates(Transform):
    """
    Perturb one chain's conditioning coordinates by Gaussian noise.
    NOTE: This can influence chirality featurization as chirality is determined by input coordinates
    """

    def __init__(self, chain_id: str, scale: float = 1.0):
        super().__init__()
        self.chain_id = chain_id
        self.scale = scale

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        mask = atom_array.chain_id == self.chain_id
        atom_array.coord[mask] += np.random.normal(scale=self.scale, size=(mask.sum(), 3))
        return data
```

Then list it in your experiment config. Hydra instantiates the `_target_` args and passes
the transforms to the main pipeline’s `sampling_transforms` argument.

```yaml
generation:
  dataset:
    sampling_transforms:
      - _target_: rfproteina.transforms.sampling.jitter_chain.JitterChainCoordinates
        chain_id: A
        scale: 1.0
```

Four things to know.

- **Input support.** Sampling transforms work with annotated CIF and
  [condition-spec](input_prep/yaml-json.md) inputs.
- Transforms run **after** expandable segments are realized, so residue positions are final, and
  **before** featurization, so any change you make to the atom array reaches the model input.
- `prevalidate_inputs=true` validates the input as you authored it and skips these transforms.
- Is yours broadly useful? Move the class to a tracked module under `src/rfproteina/transforms/`
  and open a PR.

## Inference temperature parameters

Use TSR's `tsr_k` as the main temperature control, separately for backbone Cα coordinates
and local sidechain/sequence latents. Higher `k` sharpens the distribution more strongly;
lower `k` raises temperature. At `k=1`, TSR score sharpening is disabled, while the time grid
and endpoint rules are unchanged. `tsr_sigma` is a secondary control: larger values make
sharpening take effect earlier along the trajectory.

| Parameter | Config key | Default |
| --- | --- | --- |
| Backbone temperature | `generation.model.bb_ca.simulation_step_params.tsr_k` | 16 |
| Latent temperature | `generation.model.local_latents.simulation_step_params.tsr_k` | 4 |
| Backbone sharpening schedule | `generation.model.bb_ca.simulation_step_params.tsr_sigma` | 3 |
| Latent sharpening schedule | `generation.model.local_latents.simulation_step_params.tsr_sigma` | 12 |

`sc_scale_noise` remains configurable and defaults to 1.0 for both modalities.

```bash
rfd4-generate \
    inputs=data/benchmarks/monomer out_dir=./runs/higher_temperature \
    generation.model.bb_ca.simulation_step_params.tsr_k=8 \
    generation.model.local_latents.simulation_step_params.tsr_k=2
```

## Sampling with TSR

**`vf_tsr`**, velocity field with temporal score rescaling, is the default sampler with the released
60-model-evaluation schedule. The default `thinking_medium` recycle strategy uses 180 trunk passes and
places twice the integration-step density in the final 20% of each modality's clock. TSR applies
time-dependent score scaling that drives samples toward lower temperature near the end of the
trajectory. `tsr_k` controls how low the temperature drops, where higher means lower. `tsr_sigma`
controls how fast it gets there. `inference_base` ships tuned defaults for both.

The default sequence-probability v4 checkpoint accepts generalized sequence-probability conditions
(see [condition specification](input_prep/yaml-json.md)). Its flow sequence readout
conditions the autoencoder by default during inference. Set
`generation.args.condition_decoder_on_sequence_prediction=false` to disable it
or use a checkpoint without a sequence readout. Exact user sequence conditions
are preserved in either mode.

Because `inference_base` sets `sc_scale_noise` and `gt` defaults above zero, default `vf_tsr`
sampling is **stochastic**. For deterministic ODE sampling with no noise injected mid trajectory:

```bash
generation.model.bb_ca.gt.mode=ode generation.model.local_latents.gt.mode=ode
```

## Every parameter

Every `rfd4-generate` option — essentials, inputs and checkpoints, sampling, reward guided search,
metric guidance, partial diffusion, and inverse folding — with its type, required/default, a
description, and an example, lives in the [CLI options
reference](../reference/cli-options.md#cli-options-for-rfd4-generate). Every argument is `key=value`, Hydra style,
never `--key value`.

### Inputs and checkpoints

See [Inputs and checkpoints](../reference/cli-options.md#inputs-and-checkpoints) in the CLI options
reference.

## Troubleshooting

| Issue | Solution |
| --- | --- |
| `DATA_PATH` or checkpoint not found | `export DATA_PATH=...` then run `./scripts/download_assets.sh "$DATA_PATH"` |
| `atomworks not found` | Activate your Python environment and reinstall with `python -m pip install -e .` |
| `hydra not found` or import errors | Activate the venv with `source .venv/bin/activate` |
| Hydra config errors | Use `=` not `--`. Write `inputs=file.cif`, not `--inputs file.cif` |
| Out of memory | Lower `batch_size`, and test your longest input at `batch_size=1` first |
| Segfault in rdkit or posebusters | uv's Python needs `LD_PRELOAD` pointed at the bundled `libstdc++.so.6`, as in the `Makefile`. `make test` sets it for you |
| AF3 or refolding errors | Check the configured oracle runtime and weights. Search needs an oracle only for refolding-based rewards or criteria. See [Running metrics inline](evaluate-with-metrics.md#running-metrics-inline) |
| Every design silently fails a pre-oracle filter, with no error | The filter's tag gate is looser than the metric it reads, so it sees a missing key and counts that as a failure. Keep a filter's tag gate at least as strict as its metric's — see [Pre-oracle filters](../reference/metrics.md#pre-oracle-filters) |

```{toctree}
:hidden:
:maxdepth: 1

advanced/multi-job-runs
advanced/foldseek
```
