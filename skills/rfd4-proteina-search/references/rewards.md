# Reward reference

The reward tells search which candidates to keep. In RFProteína it is defined
**inline** under `search.rewards` in a search `inference_experiment` config —
there is no separate reward file. This mirrors `docs/workflow/search.md`.

A reward has three parts: `reward_metrics_tags` (which metrics fire during
scoring), `weights` (how to combine them into a scalar), and optional
`clip_values` (per-key bounds applied before weighting). An optional fourth,
`checkpoint_overrides`, lets any of those three change per checkpoint.

## `reward_metrics_tags`

Tags selecting which metrics run during **beam scoring**, independent of the
top-level `metrics_tags` (which controls the epoch-end metrics loop). Anything a
tag produces becomes available as a key in `weights`.

Refolding tags activate the oracle selected by `metrics/refolding_oracles` when
that backend supports the design case. AF3 is the default; AF2 and OpenFold3 are
selected with `metrics/refolding_oracles=af2` and
`metrics/refolding_oracles=of3`. Standard outputs are prefixed `oracle.`. Common
tags are:

| Tag | Activates | Supported backends |
| --- | --- | --- |
| `unconditional_refolding` | Refolding + pLDDT/scRMSD for unconditional generation | AF2, AF3, OpenFold3 |
| `ppi_refolding` | Refolding + interface metrics for protein binder design | AF2, AF3, OpenFold3 |
| `sm_binder_refolding` | Refolding + binding metrics for small-molecule binders | AF3, OpenFold3 |
| `motif_scaffold_refolding` | Refolding + motif RMSD for scaffolding / enzymes | AF2, AF3, OpenFold3 |
| `interchain_interactions` | Interchain H-bond / salt-bridge / etc. counts and adherence | No oracle |

A reward that uses only non-refolding tags (e.g. `interchain_interactions`)
needs **no container** — this is how `sm_binder_hbond_search.yaml` avoids the
oracle entirely.

## `weights`

Maps each metric output key to a coefficient; the reward is the **weighted sum**
of those metric values. Use a **negative weight to minimize** a metric (e.g.
minimize interface PAE).

Common oracle keys (from `src/rfproteina/metrics/oracles/oracle_metric.py`,
"coi" = chain of interest):

| Key | Meaning | Direction to prefer |
| --- | --- | --- |
| `oracle.plddt_coi` | Refolding pLDDT of the chain of interest (0-100) | higher = better -> positive weight |
| `oracle.iptm_coi` | Interface pTM for the chain of interest | higher = better -> positive weight |
| `oracle.ipae_coi` | Mean interface PAE | lower = better -> negative weight |
| `oracle.min_ipae_coi` | Minimum interface PAE | lower = better -> negative weight |

Any key produced under the active tags is usable, not just oracle keys — e.g.
`interchain_interactions.hbond_atom_pair_count_coi` in the H-bond example.

## `clip_values` (optional)

Maps a weighted key to `{min, max}` bounds (either bound optional). The value is
**clipped to those bounds before its weight is applied**, so the reward stops
responding once the metric passes the bound; the raw metric is still reported.
**Each clipped key must also appear in `weights`.**

This is most useful for multi-objective search: cap a metric that is already
"good enough" so the search stops over-optimizing it and spends effort on the
others. For example, clipping `oracle.plddt_coi` at `min: 50.0` treats everything
below pLDDT 50 as equally bad (garbage), so the reward gradient there is flat.

> If your reward seems to plateau or stop improving, check whether a `clip_values`
> bound is being hit — past the bound the metric no longer contributes.

## `checkpoint_overrides` (optional)

By default one reward scores every checkpoint. `checkpoint_overrides` maps a
**scored step** to a *sub-version* of the base reward; anything it does not name
is inherited. Semantics:

| Field in an override | Behaviour |
| --- | --- |
| `weights` | replaces the base weights wholesale |
| `clip_values` | replaces the base clips wholesale (`{}` clears them) |
| `reward_metrics_tags` | replaces the base list wholesale |

Nothing merges key-by-key: a field an override names is the complete value for
that checkpoint, and a field it omits is inherited untouched. This is deliberate
— an override that switches `reward_metrics_tags` changes *which metrics exist*,
and an inherited weight whose metric no longer fires makes the whole reward
`-inf`, so every candidate at that checkpoint ties and the prune ranks nothing.
Restating the weights alongside the tags keeps the two in sync; if it is missed,
the run raises at that checkpoint rather than pruning on nothing (as it does when
the `intermediate_sample_save_criterion` is unevaluable for every candidate).

Keys must be one of `step_checkpoints` **past the first** — step 0 is the beam
initialization and is never scored, so an override there (or on any unscored
step) is rejected up front rather than silently ignored. Malformed overrides
(unknown field, non-integer step key, clip on an unweighted key) also fail at
config load.

```yaml
search:
  strategy:
    step_checkpoints: [0, 20, 40, 50, 60]
  rewards:
    reward_metrics_tags: [interchain_interactions]      # cheap: no refolding
    weights:
      interchain_interactions.hbond_atom_pair_count_coi: 1.0
    checkpoint_overrides:
      50:
        weights:                                        # restate the H-bond term, add packing
          interchain_interactions.hbond_atom_pair_count_coi: 1.0
          interchain_interactions.hydrophobic_atom_pair_count_coi: 0.5
        clip_values:
          interchain_interactions.hbond_atom_pair_count_coi:
            max: 6.0                                    # stop paying past 6 H-bonds
      60:
        reward_metrics_tags: [unconditional_refolding]  # refolding only at the last prune
        weights:                                        # oracle keys only: the interchain
          oracle.plddt_coi: 1.0                         # metrics no longer fire here
```

Every entry is defined against the **base** reward, never against the previous
checkpoint's — the H-bond clip added at 50 does not carry into 60.

### Why stage a reward

A checkpoint prunes on a rollout of a **still-noisy** state. Early in the
trajectory only broad features are settled — fold, topology, whether an interface
forms at all — while fine details are not yet determined by the state being
ranked. Scoring those details early therefore ranks noise, and a candidate killed
at an early checkpoint for a detail it had not yet committed to is a pass lost for
no reason.

Matching each checkpoint's reward to what the trajectory has actually committed
to by then is the point of this feature, and it moves **pass rate**, usually more
than it moves cost:

- **Objective annealing** — broad structural features early, finer details late.
  The primary use: each prune ranks on something the state has actually decided.
- **Targeted oracle placement** — put refolding where it changes *which
  candidates survive*. Oracle calls then scale with the overridden checkpoints
  rather than all of them, so this cuts cost as well. The best placement is not
  automatically the last checkpoint — mid-trajectory, or even early (steering the
  whole run toward foldable topologies from the start), can win. Test it.
- **Gating over time** — clip a term once it is good enough at a later
  checkpoint and let the remaining terms drive the last prunes.

Two cautions: `search.reward` values from different checkpoints are **different
objectives** and are not comparable (so an `intermediate_sample_save_criterion`
threshold on `search.reward` must make sense for each), and metric managers are
set up once for *every* checkpoint's tag set, so a tag used only in an override
still works.

## Worked example

Maximize refolding confidence while keeping interface PAE low, with a floor on
pLDDT so sub-50 candidates are all treated as equally bad:

```yaml
# @package _global_
defaults:
  - inference_on_user_inputs
  - _self_

inputs: ${oc.env:PROJECT_ROOT}/data/benchmarks/monomer
subset_to_examples: monomer-short
num_replicates: 2

out_dir: outputs/my_search
run_name_: my_search

metrics_tags: [design, unconditional_refolding]   # epoch-end loop reuses cached oracle results

search:
  strategy:
    _target_: rfproteina.search.beam_search.BeamSearchStrategy
    step_checkpoints: [0, 20, 40, 50, 60]
    n_branch: 2
    beam_width: 3
  rewards:
    reward_metrics_tags: [unconditional_refolding]
    weights:
      oracle.plddt_coi: 1.0
      oracle.ipae_coi: -1.0        # negative -> minimize interface PAE
    clip_values:
      oracle.plddt_coi:
        min: 50.0
generation:
  args:
    nsteps: 60                      # released default; must equal the last step_checkpoint
```

See [af2-oracle.md](af2-oracle.md), [af3-oracle.md](af3-oracle.md), and
[cost.md](cost.md) for backend requirements and how
`beam_width`/`n_branch`/`step_checkpoints` drive cost.
