# Sweep axes (RFProteína-accurate)

Which knobs to vary across waves, and how. Change **one** axis per child branch so
the effect is attributable. These are the axes RFProteína actually exposes — see
`docs/guides/inference.md`, `docs/guides/parameters.md`, and `docs/guides/search.md`.

> Not available in RFProteína: MCTS, best-of-n, and FK-steering are
> **Proteina-Complexa** search algorithms. RFProteína's search is **beam search
> only** (vanilla vs beam). Do not sweep an "algorithm" axis expecting those.

## Search axes (rfd4-proteina-search)

| Axis | Config key | Effect / caution |
| --- | --- | --- |
| Beam width | `search.strategy.beam_width` | Candidates kept after each prune (default 4). More = better designs, more oracle calls. |
| Branching | `search.strategy.n_branch` | Candidates spawned per surviving beam (default 4). |
| Checkpoints | `search.strategy.step_checkpoints` | Steps to score+prune (e.g. `[0,100,200,300,400]`). Must start at 0, strictly increase, and end at `nsteps`. More checkpoints = more oracle calls. |
| Steps | `generation.args.nsteps` | Must equal the last `step_checkpoint`. |
| Vanilla vs beam | (presence of `search:` block) | No search = fastest baseline; beam = best quality, slower. |

Cost scales roughly with `beam_width x n_branch x (#checkpoints)` oracle calls —
raise these only as the budget allows (see rfd4-proteina-search `references/cost.md`).

## Reward axes (see reward-engineering.md)

| Axis | Where | Effect |
| --- | --- | --- |
| Term weights | `search.rewards.weights` | Rebalance which objective dominates |
| Gates | `search.rewards.clip_values` | Turn a "must-clear" term into a bar |
| Which metrics score | `search.rewards.reward_metrics_tags` | Change the objective family entirely |

## Diversity axes (generation)

| Axis | Config key | Default | Effect |
| --- | --- | --- | --- |
| Backbone noise | `generation.model.bb_ca.simulation_step_params.sc_scale_noise` | 0.3 | Structural diversity (higher = more diverse, often less designable) |
| Sidechain/seq noise | `generation.model.local_latents.simulation_step_params.sc_scale_noise` | 0.1 | Sequence diversity |

Rough guide: `(0.1,0.1)` high designability, `(0.5,0.5)` natural UniProt-like,
`(0.9,0.9)` maximum diversity.

## Input axes (rfd4-proteina-inputs)

These change the design *problem*, so they define a new branch input rather than a
CLI override:

| Axis | How | Notes |
| --- | --- | --- |
| Binder / scaffold length | length range or expandable segment in the input | A range is resampled per design; a single value is fixed |
| Epitope / hotspots | `C_HOT` / `select_hotspots` on the target | Which target-interface atoms the binder should engage |
| Ligand pose / conditioning | `C_CRD` on ligand atoms | All atoms vs a catalytically relevant subset |
| Motif placement | indexed vs unindexed, index groups | Independent fragments vs rigidly tied |

Building or editing these inputs is the **rfd4-proteina-inputs** skill's job; treat
each distinct input as a branch seed.

## Diversity accounting

Track structural diversity with `rfd4-proteina-evaluate --cluster_with_foldseek`
(requires `save_chain_of_interest=true` at generation and `FOLDSEEK_EXEC` in the
env) so a campaign does not silently collapse into one structural family while the
reward keeps climbing.
