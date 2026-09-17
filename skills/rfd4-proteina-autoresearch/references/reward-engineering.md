# Composite reward engineering

The reward is what turns a vague goal into something an autoresearch loop can
optimize. In RFProteína it is defined inline under `search.rewards` (mechanics in
the **rfd4-proteina-search** skill and `docs/workflow/search.md`); this reference is
about *how to shape it* for a campaign.

## RFProteína reward keys

Reward keys come from the metrics that run under `reward_metrics_tags` during
scoring. The AF3 oracle prefixes its outputs with `oracle.`; commonly used:

| Key | Meaning | Typical direction |
| --- | --- | --- |
| `oracle.plddt_coi` | Refold pLDDT of the chain-of-interest | maximize |
| `oracle.iptm_coi` | Interface pTM | maximize |
| `oracle.ipae_coi` | Interface pAE | minimize (negative weight) |
| `oracle.min_ipae_coi` | Min interface pAE | minimize (negative weight) |

Any key produced under the active tags is available. `reward_metrics_tags` is
independent of the top-level `metrics_tags`.

## The shape of a reward block

```yaml
search:
  rewards:
    reward_metrics_tags: [unconditional_refolding]   # which oracle/metrics score
    weights:
      oracle.plddt_coi: 1.0        # metric key -> coefficient
      oracle.ipae_coi: -0.5        # negative to MINIMIZE
    clip_values:                   # clip a key BEFORE weighting (a gate)
      oracle.plddt_coi:
        min: 50.0
```

## Four rules that keep a composite reward honest

Adapted from the muni technical note (the reasoning transfers directly; the exact
term names there are Complexa's — translate to the RFProteína keys above):

1. **Normalize term scales before weighting.** Confidence terms are ~0-1,
   interface-area terms can be hundreds-thousands (Å²), H-bond counts are small
   integers. If you sum raw values, the largest-magnitude term dominates for purely
   numerical reasons. Rescale (or choose weights that put terms on a comparable
   footing) before combining.
2. **Gate instead of paying forever.** For "must clear a bar" quantities (a minimum
   interface size, a confidence floor), use `clip_values` so the term stops
   contributing once satisfied — the agent has to clear the bar but is not rewarded
   for runaway values. (muni gated `interface_dSASA >= 1000 Å²` rather than
   rewarding area continuously.)
3. **Down-weight wide-range terms.** A term reported on 0-100 will swamp a 0-1 term
   at equal weight; scale its coefficient down accordingly. (muni down-weighted a
   0-100 hydrophobicity term to 0.025.)
4. **Use negative weights to minimize.** Error-like metrics (`*_ipae_*`) should
   carry negative coefficients.

## Scheduling the reward across the trajectory

A reward does not have to be constant. `search.rewards.checkpoint_overrides` maps
a scored step to a sub-version of the base reward. Any field an entry names —
`weights`, `clip_values`, `reward_metrics_tags` — **replaces** the base one
wholesale; fields it omits are inherited. So an entry that switches tags must
restate the weights that go with the new metrics, since a weight whose metric no
longer fires scores `-inf` (and a checkpoint where that drowns every candidate
raises rather than pruning on nothing):

```yaml
search:
  rewards:
    reward_metrics_tags: [interchain_interactions]      # cheap prunes early
    weights:
      interchain_interactions.hbond_atom_pair_count_coi: 1.0
    checkpoint_overrides:
      60:                                               # last checkpoint only (step_checkpoints: [0, 20, 40, 50, 60])
        reward_metrics_tags: [unconditional_refolding]
        weights:                                        # oracle keys only — the interchain
          oracle.plddt_coi: 1.0                         # metric does not run under these tags
```

Every prune ranks a **noisy** state. Early in the trajectory only broad features
are settled — fold, topology, whether an interface forms at all — so a candidate
killed at an early checkpoint on a fine detail it had not yet committed to is a
pass lost for nothing. Staging the reward to match what the trajectory has
actually decided by each checkpoint (broad structural features early, finer
details late) is therefore a **pass-rate** lever first and a cost lever second.

The cost lever is still real: restricting the refolding tags to selected
checkpoints makes oracle calls scale with those checkpoints rather than all of
them, freeing budget for a wider beam. But do not assume the oracle belongs at
the end — the useful placement is wherever it changes *which candidates survive*,
which for some problems is mid-trajectory or even early. Treat both "flat vs
staged reward" and "where the oracle sits" as their own branches.

Two things to check when staging: a term added late is only worth it if it is
still *reachable* then (an objective the trajectory can no longer satisfy just
wastes the last prune), and a term dropped early must not be one the search will
irreversibly destroy before anything scores it again.

## Reward as a hypothesis

The score you optimize is not ground truth — it is a hypothesis about what matters,
made explicit so it can steer search and be replaced when better evidence arrives
(a predicted confidence today; a measured binding/expression/stability signal
later). Different composite rewards produce **different search behaviors**, not
just different numbers — so pick the reward that encodes the objective you actually
care about, and expect to iterate on the reward itself, not only the designs.

Do not compare `search.reward` totals across different reward functions — they are
different objectives (different units), only comparable within one campaign.
