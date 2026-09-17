# Reward guided search

Instead of sampling once, search explores several generation paths and keeps the ones a **reward**
scores highest. **Beam search** ships ready to run. It can score candidates on any reward you like,
and the default implementation uses an **AF3 oracle**, which needs an apptainer. See [Evaluating
results](running.md#evaluating-results).

:::{admonition} On a configured compute cluster?
:class: ipd-cluster
Drop the `python` and call `./src/rfproteina/generate.py` directly, so its shebang runs it inside
the apptainer image. Do this on an interactive GPU. Full form in
[Running the model](running.md#running-inference).
:::

| Strategy | How it works | When to use |
| --- | --- | --- |
| **Vanilla** | Standard sampling, no search | Baseline, fastest |
| **Beam search** | Keep the top k partial trajectories, branch and prune at checkpoints | Best quality, slower |

## Running the example

One example config targets the monomer benchmark and uses AF3 refolding confidence as its reward.

```bash
# Beam search (configs/inference_experiment/inference_beam_example.yaml)
python src/rfproteina/generate.py inference_experiment=inference_beam_example
```

Another skips refolding during search and rewards interface hydrogen bonds between protein and
ligand instead, then swaps the reward at the final checkpoint. See [Changing the reward mid
trajectory](#changing-the-reward-mid-trajectory).

```bash
# Beam search, no refolding (configs/inference_experiment/sm_binder_hbond_search.yaml)
python src/rfproteina/generate.py inference_experiment=sm_binder_hbond_search
```

Copy either config to build your own. A search config is an `inference_experiment` with one extra
`search:` block holding a `strategy` and a `rewards` model.

## The reward

You define the reward **inline** under `search.rewards`. There is no separate reward file.

```yaml
search:
  rewards:
    reward_metrics_tags: [unconditional_refolding]   # which oracle/metrics run during scoring
    weights:
      oracle.plddt_coi: 1.0                           # metric key to coefficient, negative to minimize
    clip_values:                                      # optional, bound a key before weighting
      oracle.plddt_coi:
        min: 50.0
```

**`reward_metrics_tags`** picks which metrics fire during scoring, independently of the top level
`metrics_tags`. `unconditional_refolding` and the other `*_refolding` tags run the AF3 oracle, and
its outputs carry an **`oracle.`** prefix.

**`weights`** maps each metric output key to a weight, and the reward is their weighted sum. Use a
negative weight to minimize a metric. Any key the active tags produce is fair game, such as
`oracle.plddt_coi`, `oracle.iptm_coi`, `oracle.ipae_coi`, and `oracle.min_ipae_coi`.

**`clip_values`** is optional. It maps a weighted key to `{min, max}` bounds, either of which you can
omit. The value gets clipped **before** its weight applies, so the reward stops responding once the
metric passes the bound, while the metric is still reported raw. Every clipped key must also appear
in `weights`.

## Changing the reward mid trajectory

The block above defines one reward used at every checkpoint. `checkpoint_overrides` lets the reward
change as the trajectory progresses. Each entry is a sub version of the base reward, keyed by the
step it takes effect at, and anything it leaves out it inherits from the base.

```yaml
search:
  strategy:
    step_checkpoints: [0, 20, 40, 50, 60]
  rewards:
    reward_metrics_tags: [interchain_interactions]   # cheap, no refolding
    weights:
      interchain_interactions.hbond_atom_pair_count_coi: 1.0

    checkpoint_overrides:
      60:                                             # only at the final checkpoint
        reward_metrics_tags: [unconditional_refolding]  # replaces the base list, so AF3 now runs
        weights:
          oracle.plddt_coi: 1.0                       # replaces the base weights entirely
```

A few rules govern this.

- **Keys are steps.** Each must be one of `step_checkpoints` past the first, because those are the
  steps where candidates get scored and pruned. A key that never gets scored is an error.
- **Every field an override names replaces the base one wholesale**, whether that is `weights`,
  `clip_values` or `reward_metrics_tags`. List every term that should still count at that
  checkpoint. There is no key by key merge.
- This matters most when an override changes `reward_metrics_tags`. A weight whose metric no longer
  fires scores `-inf`, which would drown every candidate at that checkpoint. Replacing rather than
  merging keeps the weights and the metrics that produce them in sync. If every candidate at a
  checkpoint does score `-inf`, or its `intermediate_sample_save_criterion` cannot be evaluated for
  any of them, the run raises there instead of pruning on nothing.
- Replacing `weights` while inheriting `clip_values` is an error if a clip no longer matches a
  weighted key. Set `clip_values: {}` to clear the inherited clips.
- Checkpoints with no entry use the base reward unchanged, so your existing configs keep working.

**Why stage a reward.** A checkpoint prunes on a rollout of a state that is still noisy. Early in
the trajectory only broad features are settled, things like fold, topology, and whether an interface
forms at all. Fine details are not decided yet, so scoring on them early ranks noise instead of
promise. Match each checkpoint's reward to what the trajectory has actually committed to by then,
broad structure early and fine detail late. Pass rates go up, usually by more than the compute you
save. The same logic applies to refolding. Put the oracle where it changes which candidates survive,
which is not always the final checkpoint. Mid trajectory or even early can beat it. Test it on your
own problem.

> Rewards from different checkpoints are **different objectives**, so their `search.reward` values
> do not compare. If `intermediate_sample_save_criterion` thresholds on `search.reward`, make sure
> the bar means something for every checkpoint's reward, or threshold on a raw metric key instead.

## Beam search parameters (`rfproteina.search.beam_search.BeamSearchStrategy`)

```yaml
search:
  strategy:
    _target_: rfproteina.search.beam_search.BeamSearchStrategy
    step_checkpoints: [0, 20, 40, 50, 60]       # steps to score and prune at
    n_branch: 2                                  # candidates spawned per surviving beam
    beam_width: 3                                # candidates kept after each prune
    # Optional. Also keep rolled-out candidates that lost the beam but clear a bar.
    # Omit to return only the final beam.
    intermediate_sample_save_criterion:
      _target_: rfproteina.metrics.oracles.pass_criterion_metric.PassCriterionMetric
      metric_thresholds:
        search.reward:                           # threshold on the reward, or any metric key
          min: 80.0
generation:
  args:
    nsteps: 60                                   # released default; must equal the last step_checkpoint
```

| Parameter | Meaning | Default |
| --- | --- | --- |
| `step_checkpoints` | Diffusion steps to score and prune at. **Must start at 0, increase strictly, and end at `nsteps`.** | required |
| `n_branch` | Candidates generated from each surviving beam at every checkpoint | 4 |
| `beam_width` | Candidates kept after each prune, the beam size | 4 |
| `intermediate_sample_save_criterion` | A `PassCriterionMetric` that decides which lost beam candidates to save anyway. It sees each candidate's metrics plus its reward under `search.reward`. Omit to return only the final beam. | none |

Cost scales roughly with `beam_width × n_branch × number of checkpoints` oracle calls, so raise
`beam_width` and `n_branch` only as far as your compute budget allows.
