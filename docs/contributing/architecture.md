# Code structure

A map of the codebase for anyone about to modify it: what lives where, how the pieces call into
each other, and which doc to read next for the reasoning behind a given piece. For how the pipeline
*behaves* end to end, start at {doc}`../concepts/workflow` instead; this page is about where that
behavior lives in the repository.

## Top-level layout

```text
src/rfproteina/   Python package (the implementation; "rfd4_proteina" is its public import alias)
configs/          Hydra configuration groups composed by the entry points below
scripts/          SLURM launchers and setup/data-download shell scripts
tests/            Test suite, laid out to mirror src/rfproteina/
docs/             This site
```

Four Hydra entry points drive almost everything. Each composes a base config, then an
`experiment`/`inference_experiment` override:

| Workflow | Entry point | Base config |
| --- | --- | --- |
| Latent model training | `src/rfproteina/train.py` | `configs/training_proteina.yaml` |
| Autoencoder training | `src/rfproteina/partial_autoencoder/train.py` | `configs/training_ae.yaml` |
| Generation / search | `src/rfproteina/generate.py` | `configs/inference.yaml` |
| Standalone evaluation | `src/rfproteina/evaluate.py` | `configs/evaluate.yaml` |

`src/rfproteina/conditions.py` and `model_masks.py` sit outside any subpackage because they define
shared vocabulary — conditioning enums and emit masks — that every layer below reads.

## Data and conditioning

```text
datapipes/    Assembles AtomWorks transforms into training/inference pipelines (pipeline.py)
transforms/   RFD4-Proteina-specific transforms (cropping, atomization, design tasks, geometry)
conditions.py Conditioning constants and enums, shared across the whole codebase
```

AtomWorks (the `lib/atomworks` submodule) parses structures into Biotite `AtomArray` objects.
`datapipes/pipeline.py` chains `transforms/` modules over that array to annotate, crop, atomize,
condition, encode, and finally tensorize a training or inference example. `transforms/design_task.py`
and `design_task_modifiers.py` implement the design-task vocabulary (unconditional, folding,
motif scaffolding, PPI hotspots, …) that `configs/design_tasks/` selects and weights.

A token can represent a polymer residue or one atomized atom, and the atom/token axes are not
interchangeable — this is the invariant most transform bugs violate. See
{doc}`../concepts/annotated-cif` and {doc}`../concepts/conditioning` for why the format and the
conditioning model are shaped this way, and `utils/condition_resolution.py` plus `inference_utils/`
for how a user-facing condition request resolves into these internal annotations.

## Models and training

```text
partial_autoencoder/   Structure <-> latent autoencoder: its own model, training, and inference
flow_matching/         ProductSpaceFlowMatcher and the per-modality flow matchers it composes
nn/                    LocalLatentsTransformer and the feature factory that builds its inputs
proteina.py            The Lightning module gluing the three together, plus metrics/search/guidance
train.py               Datasets, loaders, model, callbacks, weight-loading, and the Lightning trainer
```

`proteina.py` is the hub: it loads a frozen or trainable `partial_autoencoder`, wraps
`ProductSpaceFlowMatcher` from `flow_matching/` around the latent and backbone modalities, runs
`LocalLatentsTransformer` from `nn/` as the denoiser, and during validation/inference pulls in
metrics, search, and guidance. `train.py` builds everything `Proteina` needs from Hydra config and
owns the Lightning `Trainer`; optimization itself (gradient accumulation, clipping, the Muon/Adam
split, EMA, checkpoint resume) is manual inside `Proteina`, so changes there need focused tests
rather than relying on Lightning's defaults.

The autoencoder under `partial_autoencoder/` has its own training (`train.py`) and inference
(`inference.py`) entry points and checkpoint. The latent model derives its dimensionality from
whichever autoencoder checkpoint it loads, and `utils/weights.py` implements the supported
weight-loading and parameter-freezing policies for mismatched checkpoints. The
{doc}`developer guide <developer-guide>` walks through an actual finetuning run end to end.

## Generation, search, and guidance

```text
generate.py        Composes inputs, loads checkpoints, drives inference, writes samples/metadata
inference_utils/    Input/condition parsing, insertion mapping, symmetry, datasets, aggregation
search/             Search strategies (vanilla single-pass, beam search, checkpoint scheduling)
rewards/            Aggregates metric outputs into the scalar rewards search selects on
guidance/           Differentiable-metric gradient guidance applied during sampling
```

`generate.py` is the thin orchestration layer; `inference_utils/` does the actual work of turning a
user's inputs (an annotated CIF, a condition-spec YAML, a contig string) into the tensors `Proteina`
expects. `search/` implements the search strategies selected by `configs/inference_experiment/`:
`search/base_strategy.py` defines the interface, `vanilla.py` is an unguided single pass, and
`beam_search.py` branches, scores with `rewards/reward_model.py`, and prunes at the checkpoints
defined under `search.checkpoints` in config. `guidance/guidance_model.py` is the differentiable
twin of the reward model — it backpropagates a weighted metric objective into the sampler state
instead of only scoring after the fact. See {doc}`../concepts/search` and
{doc}`../concepts/metric-guidance` for the mechanics, and {doc}`../how-to/search` for driving them.

## Metrics and evaluation

```text
metrics/            Metric and MetricManager base classes, plus every built-in metric
af2/, af3/          Runtime/daemon integration for the AF2, AF3 and OpenFold3 refolding oracles
evaluate.py         Standalone Hydra CLI: scores structures you already generated
result_analysis/    Post-hoc analysis helpers over aggregated metrics output
```

Every metric subclasses `rfproteina.metrics.metric.Metric` and is composed through
`configs/metrics/`; `evaluate.py` and inline generation metrics share the same `MetricManager`.
Refolding metrics depend on the configured AF3, OpenFold3 or AF2 oracle, which runs in a
separate runtime through `af3/` or `af2/`. Metrics that need a refold declare that dependency
explicitly. See {doc}`../how-to/refolding` for setup and supported inputs.
{doc}`../reference/metrics` lists every built-in metric and its gating tags; {doc}`../how-to/advanced/custom-metrics` and {doc}`contribute-a-metric` cover
writing and upstreaming your own.

## Supporting code

```text
utils/        Cross-cutting helpers: structure IO, masks, weights, logging, SLURM/Hydra glue, …
callbacks/    Lightning callbacks (EMA-adjacent bookkeeping, timing, GC coordination, logging)
```

`utils/` is a flat grab-bag by design — look here before adding a new cross-cutting helper, since
one with a similar name (structure file discovery, mask resolution, dtype/device helpers) likely
already exists. `utils/structure_files.py` is the one place that should know about codec-specific
file discovery and naming; new IO code should go through it rather than reimplementing path
resolution.

## Configs

```text
configs/training_proteina.yaml   composition root for latent-model training
configs/training_ae.yaml         composition root for autoencoder training
configs/inference.yaml           composition root for generation/search
configs/evaluate.yaml            composition root for standalone evaluation
configs/nn*, datasets, dataloaders, design_tasks, generation, guidance, metrics, callbacks
                                  reusable groups referenced from the roots above
configs/experiment/, configs/experiment_ae/
                                  release aliases and maintained tutorials
configs/releases/flow/, configs/releases/ae/
                                  dated release recipes (not maintained as living docs)
configs/inference_experiment/    search/inference recipes selected by generate.py
```

Each composition root is a starting point, not a snapshot of one run — defaults-list ordering and
`_self_` placement are load-bearing, and `???` marks values a caller must supply rather than a gap
to fill with a machine-specific default. A config change that affects precedence or compatibility
needs a config-composition test alongside it.

## Scripts and tests

`scripts/launch_*.sh` wrap the four entry points for SLURM: they resolve Hydra config on the
launching host, then submit a container-aware batch job. `scripts/common.sh` and
`scripts/resolve_config.py` hold the shared resolution logic. `tests/` mirrors `src/rfproteina/`
directory for directory, plus `tests/e2e/` for entry-point and launcher-level tests and `tests/data/`
for small fixtures; CPU, GPU, regression, and cluster tests are separated by pytest markers rather
than by directory.

## Where to go next

- {doc}`index` — contributor workflow: branching, commits, pull requests, and docs builds
- {doc}`developer-guide` — setting up a training environment and a worked finetuning run
- {doc}`../concepts/workflow` — the pipeline as an interactive, stage-by-stage diagram
- [AGENTS.md](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/AGENTS.md) — the fuller,
  agent-oriented version of this map, including invariants and validation commands
