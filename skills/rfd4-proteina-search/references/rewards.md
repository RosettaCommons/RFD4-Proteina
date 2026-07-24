# Reward reference

The reward tells search which candidates to keep. In RFProteína it is defined
**inline** under `search.rewards` in a search `inference_experiment` config —
there is no separate reward file. This mirrors `docs/guides/search.md`.

A reward has three parts: `reward_metrics_tags` (which metrics fire during
scoring), `weights` (how to combine them into a scalar), and optional
`clip_values` (per-key bounds applied before weighting).

## `reward_metrics_tags`

Tags selecting which metrics run during **beam scoring**, independent of the
top-level `metrics_tags` (which controls the epoch-end metrics loop). Anything a
tag produces becomes available as a key in `weights`.

The `*_refolding` tags run the **AF3 oracle** and their outputs are prefixed
`oracle.`. Common tags (full list in `docs/guides/inference.md`, "Evaluating
Results"):

| Tag | Activates | Uses AF3 oracle? |
| --- | --- | --- |
| `unconditional_refolding` | AF3 refolding + pLDDT/scRMSD for unconditional generation | Yes |
| `ppi_refolding` | AF3 refolding + interface metrics for protein binder design | Yes |
| `sm_binder_refolding` | AF3 refolding + binding metrics for small-molecule binders | Yes |
| `motif_scaffold_refolding` | AF3 refolding + motif RMSD for scaffolding / enzymes | Yes |
| `interchain_interactions` | Interchain H-bond / salt-bridge / etc. counts and adherence | No |

A reward that uses only non-refolding tags (e.g. `interchain_interactions`)
needs **no container** — this is how `sm_binder_hbond_search.yaml` avoids the
oracle entirely.

## `weights`

Maps each metric output key to a coefficient; the reward is the **weighted sum**
of those metric values. Use a **negative weight to minimize** a metric (e.g.
minimize interface PAE).

Available AF3 oracle keys (from `src/rfproteina/metrics/oracles/oracle_metric.py`,
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

## Worked example

Maximize refolding confidence while keeping interface PAE low, with a floor on
pLDDT so sub-50 candidates are all treated as equally bad:

```yaml
# @package _global_
defaults:
  - inference_on_user_inputs
  - _self_

inputs: ${oc.env:PROJECT_ROOT}/data/benchmarks/monomer.json
subset_to_examples: monomer-short
num_replicates: 2

out_dir: outputs/my_search
run_name_: my_search

metrics_tags: [design, unconditional_refolding]   # epoch-end loop reuses cached oracle results

search:
  strategy:
    _target_: rfproteina.search.beam_search.BeamSearchStrategy
    step_checkpoints: [0, 100, 200, 300, 400]
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
    nsteps: 400                     # must equal the last step_checkpoint
```

See [af3-oracle.md](af3-oracle.md) for what the oracle needs, and
[cost.md](cost.md) for how `beam_width`/`n_branch`/`step_checkpoints` drive cost.
