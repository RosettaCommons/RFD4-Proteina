# Sweep axes (RFD4-Proteína-accurate)

Which knobs to vary across waves, and how. Change **one** axis per child branch so
the effect is attributable. These are the axes RFD4-Proteína actually exposes — see
`docs/workflow/running.md`, `docs/workflow/running.md`, and `docs/workflow/search.md`.

> Not available in RFD4-Proteína: MCTS, best-of-n, and FK-steering are
> **Proteina-Complexa** search algorithms. RFD4-Proteína's search is **beam search
> only** (vanilla vs beam). Do not sweep an "algorithm" axis expecting those.

## Search axes (rfd4-proteina-search)

| Axis | Config key | Effect / caution |
| --- | --- | --- |
| Beam width | `search.strategy.beam_width` | Candidates kept after each prune (default 4). More = better designs, more oracle calls. |
| Branching | `search.strategy.n_branch` | Candidates spawned per surviving beam (default 4). |
| Checkpoints | `search.strategy.step_checkpoints` | Steps to score+prune (e.g. `[0,20,40,50,60]`). Must start at 0, strictly increase, and end at `nsteps`. More checkpoints = more oracle calls. |
| Steps | `generation.args.nsteps` | The default is 60 evaluations, with twice the step density in the final 20% of each modality's clock. |
| Vanilla vs beam | (presence of `search:` block) | No search = fastest baseline; beam = best quality, slower. |

Cost scales roughly with `beam_width x n_branch x (#checkpoints)` oracle calls —
raise these only as the budget allows (see rfd4-proteina-search `references/cost.md`).

## Reward axes (see reward-engineering.md)

| Axis | Where | Effect |
| --- | --- | --- |
| Term weights | `search.rewards.weights` | Rebalance which objective dominates |
| Gates | `search.rewards.clip_values` | Turn a "must-clear" term into a bar |
| Which metrics score | `search.rewards.reward_metrics_tags` | Change the objective family entirely |
| **Reward schedule** | `search.rewards.checkpoint_overrides` | Change the reward *mid-trajectory*, per checkpoint |

### Reward-schedule axis (`checkpoint_overrides`)

A distinct axis from the reward itself: hold the final objective fixed and vary
**when** each term starts counting. Each entry is a sub-version of the base reward
keyed by a scored step (one of `step_checkpoints` past the first); any field it
names (`weights`, `clip_values`, `reward_metrics_tags`) replaces the base one
wholesale, so an arm that changes tags must restate its weights. Branches worth
comparing:

| Branch | Override shape | Hypothesis |
| --- | --- | --- |
| Flat (control) | none — base reward everywhere | Baseline for every schedule arm |
| Annealed | broad structural term early, add the fine terms at a later checkpoint | Early prunes rank noise when scored on details the state has not committed to |
| Oracle placement | `reward_metrics_tags: [*_refolding]` at one chosen checkpoint (early / middle / last) | There is a step where refolding actually changes which candidates survive; find it |
| Gated over time | add `clip_values` on a satisfied term at a later checkpoint | Stops over-optimizing one term and redirects the last prunes |

The reason the schedule matters is that every prune ranks a **noisy** state: early
on, only broad features (fold, topology, whether an interface forms) are settled,
so a candidate killed early on a detail it had not yet committed to is a pass lost
for nothing. Correctly staged rewards can therefore move **pass rate**
substantially, not just cost. Oracle placement is worth its own arm for both
reasons — calls scale with the overridden checkpoints rather than all of them —
and do not assume the last checkpoint is the right place for it; sweep the
placement rather than fixing it.

Note that `search.reward` totals are not comparable across checkpoints or across
schedule arms; compare arms on final-design metrics and pass rate (plus
oracle-call count for the cost story), never on reward.

## Diversity axes (generation)

| Axis | Config key | Default | Effect |
| --- | --- | --- | --- |
| Backbone noise | `generation.model.bb_ca.simulation_step_params.sc_scale_noise` | 0.3 | Structural diversity (higher = more diverse, often less designable) |
| Sidechain/seq noise | `generation.model.local_latents.simulation_step_params.sc_scale_noise` | 0.1 | Sequence diversity |

Rough guide: `(0.1,0.1)` high designability, `(0.5,0.5)` natural UniProt-like,
`(0.9,0.9)` maximum diversity.

## Input axes (rfd4-proteina-annotated-cif)

These change the design *problem*, so they define a new branch input rather than a
CLI override:

| Axis | How | Notes |
| --- | --- | --- |
| Binder / scaffold length | length range or expandable segment in the input | A range is resampled per design; a single value is fixed |
| Epitope / hotspots | `C_HOT` / `select_hotspots` on the target | Which target-interface atoms the binder should engage |
| Ligand pose / conditioning | `C_CRD` on ligand atoms | All atoms vs a catalytically relevant subset |
| Motif placement | indexed vs unindexed, index groups | Independent fragments vs rigidly tied |

Building or editing these inputs is the **rfd4-proteina-annotated-cif** skill's
job; treat each distinct input as a branch seed. Use **rfd4-proteina-inputs** only
when a branch starts from a legacy RFD3 JSON/YAML config.

## Diversity accounting

Track structural diversity with `rfd4-proteina-evaluate --cluster_with_foldseek`
(requires `save_chain_of_interest=true` at generation and `FOLDSEEK_EXEC` in the
env) so a campaign does not silently collapse into one structural family while the
reward keeps climbing.
