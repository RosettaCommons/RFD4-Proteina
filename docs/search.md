# Reward-Guided Search

Instead of sampling once, search explores multiple generation paths and keeps the ones a **reward**
scores highest. **Beam search** ships ready to run; it can score candidates with arbitrary rewards; for example, we provide a default **AF3 oracle** implementation (which
needs an apptainer — see [Evaluating Results](INFERENCE.md#evaluating-results)).

| Strategy | How it works | When to use |
| --- | --- | --- |
| **Vanilla** | Standard sampling, no search | Baseline, fastest |
| **Beam search** | Keep top-k partial trajectories, branch and prune at checkpoints | Best quality, slower |

## Running the example

One example config targets the monomer benchmark, using AF3 refolding confidence as its reward:

```bash
# Beam search (configs/inference_experiment/inference_beam_example.yaml)
python src/rfproteina/generate.py inference_experiment=inference_beam_example
```

Another example does not refold structures during search, instead using interface
protein↔ligand hydrogen bonds as the reward:

```bash
# Beam search, no refolding (configs/inference_experiment/sm_binder_hbond_search.yaml)
python src/rfproteina/generate.py inference_experiment=sm_binder_hbond_search
```

Copy the config to build your own. A search config is an `inference_experiment` with one added
`search:` block: a `strategy` and a `rewards` model.

## The reward

The reward is defined **inline** under `search.rewards` — there is no separate reward file:

```yaml
search:
  rewards:
    reward_metrics_tags: [unconditional_refolding]   # which oracle/metrics run during scoring
    weights:
      oracle.plddt_coi: 1.0                           # metric key → coefficient (negative to minimize)
    clip_values:                                      # optional: bound a key before weighting
      oracle.plddt_coi:
        min: 50.0
```

- **`reward_metrics_tags`** — tags selecting which metrics fire during scoring, independent of the
  top-level `metrics_tags`. `unconditional_refolding` (and the other `*_refolding` tags) run the AF3
  oracle; its outputs are prefixed **`oracle.`**.
- **`weights`** — maps each metric output key to a weight; the reward is their weighted sum. Use a
  negative weight to minimize a metric. Any key produced under the active tags is available, e.g.
  `oracle.plddt_coi`, `oracle.iptm_coi`, `oracle.ipae_coi`, `oracle.min_ipae_coi`.
- **`clip_values`** *(optional)* — maps a weighted key to `{min, max}` bounds (either optional). The
  value is clipped to those bounds **before** its weight is applied, so the reward stops responding
  once the metric passes the bound; the metric is still reported raw. Each clipped key must also
  appear in `weights`.

## Beam search parameters (`rfproteina.search.beam_search.BeamSearchStrategy`)

```yaml
search:
  strategy:
    _target_: rfproteina.search.beam_search.BeamSearchStrategy
    step_checkpoints: [0, 100, 200, 300, 400]   # steps to score + prune at
    n_branch: 2                                  # candidates spawned per surviving beam
    beam_width: 3                                # candidates kept after each prune
    # Optional: also keep rolled-out candidates that lost the beam but clear a bar.
    # Omit to return only the final beam.
    intermediate_sample_save_criterion:
      _target_: rfproteina.metrics.oracles.pass_criterion_metric.PassCriterionMetric
      metric_thresholds:
        search.reward:                           # threshold on the reward (or any metric key)
          min: 80.0
generation:
  args:
    nsteps: 400                                  # must equal the last step_checkpoint
```

| Parameter | Meaning | Default |
| --- | --- | --- |
| `step_checkpoints` | Diffusion steps at which to score and prune. **Must start at 0, be strictly increasing, and end at `nsteps`.** | required |
| `n_branch` | Candidates generated from each surviving beam at every checkpoint | 4 |
| `beam_width` | Candidates retained after each prune (the beam size) | 4 |
| `intermediate_sample_save_criterion` | A `PassCriterionMetric` deciding which lost-beam candidates to also save as outputs. It sees each candidate's metrics plus its reward under `search.reward`. Omit to return only the final beam. | none |

Cost scales roughly with `beam_width × n_branch × (number of checkpoints)` oracle calls — raise
`beam_width`/`n_branch` only as your compute budget allows.
