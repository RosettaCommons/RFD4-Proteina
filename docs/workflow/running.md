# Running the model

One command turns an annotated CIF into designs. This page covers that command, the knobs worth
touching, and what to do when a run misbehaves.

Just getting started? [Quickstart](../quickstart.rst) installs the model and generates a first
design. [Preparing inputs](inputs.md) covers where the CIF comes from.

Choose the sampling workflow that matches what supplies the extra information:

- [Standard generation](#running-inference) starts from noise.
- [Partial diffusion and inverse folding](partial-diffusion.md) start from a supplied structure.
- [Metric guidance](metric-guidance.md) differentiates a scalar objective during sampling.

## What you can design

The released checkpoint supports all of these design tasks.

| Task | Status |
| --- | --- |
| Unconditional generation | Supported |
| Motif scaffolding | Supported |
| Enzyme design | Supported, with a [worked example](../examples/enzyme_design_tutorial.ipynb) |
| Small molecule binder design | Supported |
| Protein binder design | Supported |
| DNA binder design | Supported |

## Running inference

The general form:

```bash
./src/rfproteina/generate.py inputs=<file> num_replicates=<N> out_dir=<dir> [overrides...]
```

**Three entry points.** Submitting a job means `launch_generate.sh`. Running one thing right now
means `generate.py`.

| Entry point | Use it for |
| --- | --- |
| `scripts/launch_generate.sh <cluster> …` | SLURM job arrays, multi GPU fan out, sweeps, metric aggregation, AF2 refolding |
| `python src/rfproteina/generate.py …` | One process in your venv, on a node you already hold or a machine with no container |
| `./src/rfproteina/generate.py …` | One process inside the cluster container, where the shebang runs it in `$LATENT_APPTAINER_SIF` |

All three start AF3's refolding daemon on the host for you.

:::{admonition} On a configured compute cluster?
:class: ipd-cluster

Call the script **directly**. Its shebang executes it inside the apptainer image named by
`LATENT_APPTAINER_SIF` in `.ipd/.env.digs`, so you need no virtualenv.

```bash
./src/rfproteina/generate.py inputs=… num_replicates=… out_dir=…
```

Prefix it with `python` and it runs in your current environment instead. Use that form only where
no container is configured. Run the direct form on an interactive GPU. For batch work, the SLURM
launchers such as `scripts/launch_generate.sh` enter the container for you.
:::

- **Inputs.** Use an **annotated CIF** ([guide](inputs.md) ·
  [tutorial](../examples/annotated_cif_tutorial.ipynb)) or a
  [condition-spec JSON/YAML file](condition-spec.md). `data/benchmarks/` has ready made examples.
- **Every model argument is `key=value`**, Hydra style, never `--key value`.
- **Checkpoints** come from `$DATA_PATH` by default. Override them on the command line only for a
  one off model, with `ckpt_path=<dir> ckpt_name=<file.ckpt>`.
- **`batch_size`** of 8 is a good place to start. Bigger batches run faster, but confirm your
  longest input fits at `batch_size=1` first.

```bash
# 80 designs of each example in a JSON, 8 at a time
./src/rfproteina/generate.py \
    inputs=data/benchmarks/monomer \
    num_replicates=80 batch_size=8 out_dir=./runs/monomers
```

:::{tip}
Want a reusable config instead of one off CLI overrides? Start from
[`configs/inference_experiment/TEMPLATE.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/inference_experiment/TEMPLATE.yaml).
It documents all three ways to run inference, which are an in tree experiment config, CLI
overrides, and a config file anywhere on the filesystem.
:::

## Evaluating results

:::{note}
Refolding metrics and reward guided search both use the **AF3 oracle**, which runs only inside an
apptainer container or a native install, configured through the cluster env files. See the
[Developer Guide](../development.md). Plain generation needs none of it.
:::

### Running metrics inline

Pass `metrics_tags` during inference and metrics get computed right after each design. Results land
in `per_example_metrics.csv` next to the CIF files.

The [metrics guide](metrics.md) covers every metric that ships enabled, what it outputs, which tag
turns it on, and how to layer your own config on top of an existing metric.

`metrics_tags` takes a Hydra style list, so pass it bracketed and quoted.

```bash
# Single tag
./src/rfproteina/generate.py \
    inputs=design.json num_replicates=50 metrics_tags='[unconditional_refolding]'

# Multiple tags
./src/rfproteina/generate.py \
    inputs=design.json num_replicates=50 metrics_tags='[ppi_refolding,hotspot_conditioning]'
```

| Tag | Activates |
| --- | --- |
| `unconditional_refolding` | AF3 refolding plus pLDDT and scRMSD for unconditional generation |
| `ppi_refolding` | AF3 refolding, target templating and interface metrics for protein binder design |
| `motif_scaffold_refolding` | AF3 refolding plus motif RMSD for motif scaffolding and enzyme design |
| `sm_binder_refolding` | AF3 refolding plus binding metrics for small molecule binder design |
| `ame_refolding` | AF3 refolding for enzyme (MCSA) active site benchmarks |
| `hotspot_conditioning` | Hotspot obedience metrics, needs hotspot input |
| `3d_coordinate_conditioning` | Motif coordinate RMSD metrics |
| `rasa_conditioning` | RASA adherence metrics |
| `interchain_interactions` | Interchain interaction adherence and counts, covering H-bonds, π-stacking, π-cation, salt bridges, hydrophobic contacts and metal coordination |

A metric that needs a tag gets skipped silently when the tag is absent. The table above is a
shortlist. [Metrics](metrics.md) covers everything that ships enabled plus every metric class that
is ready made but opt in.

### Concatenating multi job runs

When you run large evaluations across several SLURM jobs, each job writes its own
`per_example_metrics_job{N}.csv`. Once they all finish, concatenate them by passing the metrics
directory as the first positional argument.

```bash
python -m rfproteina.inference_utils.aggregate_metrics ./inference/my_run/metrics/
```

That concatenates the per example job CSVs, writes `aggregated_metrics.csv` and
`per_task_metrics.csv`, and moves the individual job files into `individual_job_results/`. Pass
`--output_dir <dir>` to write elsewhere. `launch_generate.sh` runs this for you as a dependent SLURM
job.

Pass `--cluster_with_foldseek` to also run structural diversity clustering, described below.

### Adding custom metrics

You can add your own metrics without touching a tracked file. Two **git ignored** directories hold
them, so your local experiments stay private.

- `src/rfproteina/metrics/custom_metrics/` holds Python metric classes, split across as many
  modules as you like.
- `configs/metrics/custom_metrics/` holds the Hydra config that registers them.
  `all_custom_metrics.yaml` is the entry point, and `setup.sh` scaffolds it on first run.

The config gets picked up automatically, with no extra flags, and results land in
`per_example_metrics.csv` prefixed by the config name, like `residue_count.n_residues`.

The [custom metrics guide](custom-metrics.md) walks through writing a metric class, refolding
metrics, splitting across files, and contributing one back.

### FoldSeek diversity clustering

Pass `--cluster_with_foldseek` and structures get clustered by structural similarity using
[FoldSeek](https://github.com/steineggerlab/foldseek), which adds a `foldseek_cluster_id` column so
you can count structurally distinct designs per task.

**Clustering only looks at the designed chain.** In a multimer output such as binder plus target,
only the most freely designed chain gets extracted and clustered. The pipeline identifies that
chain from the conditioning annotations and stores it as `chain_of_interest` in the metrics CSV.
Clustering reads the extracted per design chain CIFs, which only get written when you generate with
`save_chain_of_interest=true`, so set that if you plan to cluster.

Point `FOLDSEEK_EXEC` in `.env` at the FoldSeek executable. Clustering runs `foldseek easy-cluster`
on the extracted single chain with a TM-score threshold of 0.5.

## Sampling transforms

By default every replicate of an input file conditions on exactly the same annotated CIF. A
**sampling transform** runs once per replicate, so each design can start from a different
condition. That helps when a conditioning value is a guess rather than a measurement, like the CA
position of a residue whose structure you do not know.

Write one as an AtomWorks `Transform` whose `forward` edits `data["atom_array"]` in place. Drop it
in `src/rfproteina/transforms/sampling/`, which is git ignored and created by `setup.sh`, so your
code stays local. Same arrangement as [custom metrics](custom-metrics.md).

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

Then list it in your experiment config. Hydra instantiates the `_target_` args.

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
  [condition-spec](condition-spec.md) inputs.
- Transforms run **after** expandable segments are realized, so residue positions are final, and
  **before** featurization, so any change you make to the atom array reaches the model input.
- `prevalidate_inputs=true` validates the input as you authored it and skips these transforms.
- Is yours broadly useful? Move the class to a tracked module under `src/rfproteina/transforms/`
  and open a PR.

## Inference noise parameters

The model uses **product space flow matching** with two flows, one for backbone Cα and one for
sidechain latents. Stochasticity temperatures control diversity.

| Parameter | Config key | Controls |
| --- | --- | --- |
| Backbone temperature | `generation.model.bb_ca.simulation_step_params.sc_scale_noise` | Structural diversity |
| Latent temperature | `generation.model.local_latents.simulation_step_params.sc_scale_noise` | Sidechain and sequence diversity |

Raising these buys diversity and can cost designability. The shipped defaults are `0.3` for
backbone and `0.1` for latent. Roughly speaking, `(0.1, 0.1)` stays close to the MPNN baseline with
high designability, `(0.5, 0.5)` shifts sequences toward the natural UniProt distribution, and
`(0.9, 0.9)` maximizes diversity.

```bash
./src/rfproteina/generate.py \
    inputs=data/benchmarks/monomer out_dir=./runs/diverse \
    generation.model.bb_ca.simulation_step_params.sc_scale_noise=0.5 \
    generation.model.local_latents.simulation_step_params.sc_scale_noise=0.5
```

## Sampling with TSR

**`vf_tsr`**, velocity field with temporal score rescaling, is the default sampler at `nsteps=100`.
It applies time dependent score scaling that drives samples toward lower temperature near the end
of the trajectory. `tsr_k` controls how low the temperature drops, where higher means lower.
`tsr_sigma` controls how fast it gets there. `inference_base` ships tuned defaults for both.

Because `inference_base` sets `sc_scale_noise` and `gt` defaults above zero, default `vf_tsr`
sampling is **stochastic**. For deterministic ODE sampling with no noise injected mid trajectory:

```bash
generation.model.bb_ca.gt.mode=ode generation.model.local_latents.gt.mode=ode
```

## Every parameter

Every generation parameter, grouped by purpose. Every argument is `key=value`, Hydra style,
never `--key value`.

### Essential

The four you almost always set. Their defaults are rarely what you want for a real run.

| Parameter | Description | Default |
| --- | --- | --- |
| `out_dir` | Output directory. Set a real path for anything but a quick test. | `./runs/<run_name>` |
| `num_replicates` | Designs generated per input example. `1` is only good for a smoke test. | 1 |
| `batch_size` | Designs processed in parallel on GPU. Start at 8, but confirm your longest input fits at 1 first. | 1 |
| `metrics_tags` | Metrics to compute inline, as a Hydra list. See [Evaluating results](#evaluating-results). Empty means none. | `[]` |

### Inputs and checkpoints

| Parameter | Description | Default |
| --- | --- | --- |
| `inputs` | Input file, an annotated CIF, JSON, YAML, CIF or PDB, or a directory of them | Required |
| `subset_to_examples` | Run only specific input examples by key, comma separated | null |
| `skip_existing` | Skip examples whose output already exists | true |
| `seed` | Random seed. `null` gives you a different one each run | null |
| `ckpt_path` and `ckpt_name` | Override the default checkpoint directory and filename. Read from `$DATA_PATH` otherwise. | from `$DATA_PATH` |

### Sampling

| Parameter | Description | Default |
| --- | --- | --- |
| `inference_experiment` | Config group controlling generation, covering nsteps, noise, search and metrics | `inference_on_user_inputs` |
| `generation.args.nsteps` | Number of model evaluations. The default schedule places twice the step density in the final 20% of each modality's clock. | 60 |
| `generation.args.save_trajectory_every` | Save a trajectory snapshot every N steps, 0 to turn it off | 0 |

### Metric guidance

[Metric guidance](metric-guidance.md) backpropagates a differentiable metric and is currently wired
to standard full generation.

| Parameter | Description | Default |
| --- | --- | --- |
| `+guidance=<preset>` | Add a metric-guidance Hydra group. The shipped `contract_ca` preset is a mechanics test, not a useful design objective. | none |
| `guidance.metrics.<name>._target_` | Differentiable `Metric` class constructed for metric guidance | preset |
| `guidance.terms.<metric>.<modality>` | Signed schedule for one `metric_name.output_key` and one modality (`bb_ca` or `local_latents`) | preset |

### Diversity and stochasticity

See [Inference noise parameters](#inference-noise-parameters)
above.

### Partial diffusion

Refine an existing structure instead of generating from scratch. See [partial
diffusion](partial-diffusion.md#partial-diffusion).

| Parameter | Description | Default |
| --- | --- | --- |
| `generation.args.partial_trajectory_fraction_steps` | Simple interface. Run this fraction of `nsteps`, so 0.3 runs 30% | null |
| `generation.args.partial_t_backbone` and `partial_t_latent` | Advanced. Per modality starting corruption level, a float in (0, 1) | null |
| `generation.args.couple_partial_t` | Derive one modality's `partial_t` from the other through the schedule | false |

### Inverse folding

Hold the input backbone fixed and resample only the sequence and side chains. See [inverse
folding](partial-diffusion.md#inverse-folding). The `generation=inverse_folding` preset
sets both keys at once.

| Parameter | Description | Default |
| --- | --- | --- |
| `generation.args.inverse_folding` | Fix the backbone and resample sequence and side chains only | false |
| `generation.args.inverse_folding_method` | Sampling method. `resampled_forward_marginal` is the only supported value | `resampled_forward_marginal` |

## Troubleshooting

| Issue | Solution |
| --- | --- |
| `No .env file found for cluster 'digs'` | Set `export CLUSTER=local` and create `.ipd/.env.local`. See [Setup](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/README.md#setup) |
| `DATA_PATH` or checkpoint not found | `export DATA_PATH=...` then run `./scripts/download_example_data.sh "$DATA_PATH"` |
| `atomworks not found` | `git submodule update --init lib/atomworks` |
| `hydra not found` or import errors | Activate the venv with `source .venv/bin/activate` |
| Hydra config errors | Use `=` not `--`. Write `inputs=file.cif`, not `--inputs file.cif` |
| Out of memory | Lower `batch_size`, and test your longest input at `batch_size=1` first |
| Segfault in rdkit or posebusters | uv's Python needs `LD_PRELOAD` pointed at the bundled `libstdc++.so.6`, as in the `Makefile`. `make test` sets it for you |
| AF3 or refolding errors | Refolding and search need an AF3 apptainer. See [Evaluating results](#evaluating-results) |
