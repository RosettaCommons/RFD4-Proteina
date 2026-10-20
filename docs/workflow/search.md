# Reward guided search

Beam search branches partial generation trajectories and keeps the highest scoring candidates
at each checkpoint. Run `rfd4-generate inference_experiment=inference_beam_example` for an AF3
example, or `inference_experiment=sm_binder_hbond_search` for structure-only scoring.

## Checkpoint configuration

Search uses a flat `search` block. Each checkpoint explicitly lists its metrics, rewards, and scoring source:

```yaml
search:
  beam_width: 2
  n_branch: 2
  save_rollouts: false
  checkpoints:
    "20":
      n_branch: 4
      metrics: [interchain_interactions]
      use_clean_prediction: true
      rewards:
        interchain_interactions.hbond_atom_pair_count_coi: {weight: 1.0}
    "40":
      metrics: [interchain_interactions]
      rewards:
        interchain_interactions.hbond_atom_pair_count_coi:
          weight: 1.0
          clip: {max: 6.0}
          normalize_by: 6.0
        interchain_interactions.hydrophobic_atom_pair_count_coi: {weight: 0.5}
    "60":
      metrics: [oracle]
      rewards:
        oracle.plddt_coi: {weight: 1.0}
generation:
  args:
    nsteps: 60
```

- Checkpoint keys name positive integer integration steps, executed in sorted order. Quote YAML
  keys to support dotted Hydra CLI overrides such as `++search.checkpoints.20.use_clean_prediction=true`. Step zero is
  implicit. Keys cannot exceed the actual schedule length, including partial diffusion.
- A checkpoint at the schedule end (normally `60`) is optional. If omitted, every survivor of
  the last configured checkpoint finishes once, without additional branching or selection.
  These completed designs run ordinary final metrics; they have no final `search.reward`.
  If present, the end checkpoint branches and selects using its configured metrics and rewards.
- Each checkpoint is self-contained. Omitted `metrics` means `[]`, omitted `rewards` means `{}`, and omitted
  `use_clean_prediction` means `false`; nothing carries forward from a previous checkpoint.
- Reward terms use `metric_name.output_key` from definitions under `metrics`. Each term clips
  the raw value, divides by `normalize_by` (default 1), then multiplies by `weight` (default 1).
  Terms are summed; negative weights minimize. Clipping accepts `min` and/or `max`.
- An empty reward gives every candidate zero, preserving deterministic input order at selection.
  A zero-weight term does not contribute to ranking or select any metrics.
- `n_branch` (default 4) is the number of children per surviving beam for every incoming
  checkpoint interval. A checkpoint may override it with its own positive integer `n_branch`.
  Omitted values use the global default, not the preceding checkpoint's count. In the example,
  steps 0–20 generate four children per state, while later intervals generate two.
  `beam_width` (default 4) is the number retained per input.

## Metric selection

Each checkpoint's `metrics` list selects named providers defined under `metrics.inline_metrics`,
`metrics.refolding_oracles`, or `metrics.refolding_metrics`. Providers run regardless of dataset
selection tags. Selecting a provider computes all of its outputs; `rewards` chooses which outputs
contribute to ranking. Reward terms, save criteria, and continuation criteria never add metrics implicitly.

Each checkpoint has its own `RewardModel` to aggregate outputs. Missing or ambiguous provider
names fail before sampling. Pass/fail metrics are not supported as search rewards; save
criteria apply thresholds to the computed outputs instead.

For a refolding comparison, explicitly list both the oracle and the comparison metric, e.g.
`metrics: [oracle, refolded_backbone_rmsd]`. Search does not choose oracles or infer dependencies.
Refold routing constraints (`oracle:`, `is_refold:`, and `exists_refold:`) still apply so metrics
compare the intended structures. Refolding requires the normal backend and runtime environment;
see [Evaluating results](running.md#evaluating-results).

The top-level `metrics_tags` continues to control ordinary output evaluation. It does not
select search metrics. Clipping and normalization affect rewards only; metric values remain raw.

## Saving completed rollouts

The final beam is always returned. Set `save_rollouts: true` to also return completed scoring
rollouts, including candidates that lose selection. An optional `save_criteria` mapping filters
these extra outputs using raw metric thresholds:

```yaml
search:
  save_rollouts: true
  save_criteria:
    search.reward: {min: 80.0}
```

The threshold can reference explicitly selected metric outputs or `search.reward`. Every rollout
checkpoint must list all providers needed by its save criteria; missing outputs produce an error.
All thresholds must pass. A saved final-beam member appears only once. Reward scales can differ across checkpoints, so a shared threshold on `search.reward`
should be chosen accordingly. Clean predictions are never saved as completed designs.

### Scoring clean predictions without completing rollouts

Set `use_clean_prediction: true` on a nonfinal checkpoint to score the clean prediction from
its preceding integration step. A checkpoint at 19 uses integration step 18 (zero-based).
Selected trajectories continue from their actual intermediate state. With the default `false`,
search completes a separate scoring rollout and likewise continues from the intermediate state.
A checkpoint at the schedule end must score completed structures. Earlier checkpoints can use
clean predictions even when they are the last configured checkpoint.

Clean scoring decodes the prediction and uses the same design metrics as rollout scoring.
It rejects oracle, refolding, and pass-criteria providers before sampling. Predictions are
used only for ranking: they are neither saved as completed designs nor sent to refolding.

## Continuation criteria

A checkpoint before the schedule end stops an input only when every reward-selected candidate
fails its raw-metric thresholds. If any selected candidate passes, the entire selected beam
continues, including candidates that failed. Passing candidates that lost selection do not count.

```yaml
search:
  beam_width: 3
  checkpoints:
    "40":
      metrics: [interchain_interactions]
      rewards:
        interchain_interactions.hbond_atom_pair_count_coi: {weight: 1.0}
      continuation_criteria:
        interchain_interactions.hbond_atom_pair_count_coi: {min: 4}
```

List every provider needed by the criteria in that checkpoint's `metrics`, including providers
used only for continuation. All `min`/`max` thresholds must pass. Reductions `min_of`, `max_of`,
and `mean_of` can name several raw metric keys, with the outer key serving as a label.
Criteria can also read `search.reward`, which includes clipping and normalization.
Missing or nonfinite values fail that candidate; a metric absent from every candidate is a
configuration error. An omitted or empty mapping imposes no threshold.

Criteria work with both clean predictions and completed scoring rollouts. Stopped inputs do
no further sampling; their saved completed rollouts remain outputs. Other inputs continue with
their original identity and conditioning. In this example, every survivor of a continuing input finishes once
from step 40 to the schedule end, with no additional branching or selection.

## Novelty against passing designs

Use `FoldseekNoveltyMetric` as an ordinary checkpoint metric and threshold its `is_novel`
output with continuation criteria. For example, add these settings to
`inference_experiment=sm_binder_hbond_search`:

```yaml
metrics:
  inline_metrics:
    novelty:
      _target_: rfproteina.metrics.foldseek_novelty.FoldseekNoveltyMetric
      archive_dir: ./passer_archive/SAM
      tm_threshold: 0.5
      min_coverage: 0.8
      threads: 1
      required_tags_any: [search_novelty]
  pass_criteria:
    sm_binder:
      archive_dir: ${metrics.inline_metrics.novelty.archive_dir}
metrics_tags: [interchain_interactions, sm_binder_refolding]
search:
  checkpoints:
    "40":
      metrics: [interchain_interactions, novelty]
      use_clean_prediction: true
      continuation_criteria:
        novelty.is_novel: {min: 1}
```

Explicit checkpoint selection runs the novelty metric regardless of its tags. The
`search_novelty` tag keeps it out of ordinary final evaluation in this example.
Reward selection runs first; any novel selected candidate keeps the whole selected beam.
Novelty can also contribute an ordinary reward term if it should influence ranking.

The metric exhaustively searches the candidate's designed chain against a cached Foldseek
passer database. A match requires both query- and target-normalized TM scores to meet
`tm_threshold`, and both alignment coverages to meet `min_coverage`. Any match gives
`is_novel: 0`; no match, or an empty archive, gives `1`.

Set `FOLDSEEK_EXEC` to the binary. Each metric instance keeps a temporary database and
rebuilds it only when the published archive files change. Searches compare the candidate
against passers without clustering or a prefilter index. `timeout` defaults to 300 seconds
per Foldseek invocation; failures raise rather than report novelty.

Setting `archive_dir` on an existing final `PassCriterionMetric` publishes the designed chain
only when that criterion passes. Its thresholds and tags still define acceptance. Files are
published atomically, so concurrent jobs can share the directory. Use a separate append-only
archive per target and pass definition. Queries never publish clean predictions or scoring
rollouts. Final evaluation runs after generation, so new passers become available to later runs
or concurrently running jobs; an external evaluation run can populate the same archive.

## Continue a saved batch

Save selected candidates, then continue a saved batch with more branches:

```bash
rfd4-generate inference_experiment=sm_binder_hbond_search \
  seed=7 num_workers=0 batch_size=1 num_replicates=1 \
  out_dir=runs/initial search.save_states=true

state=$(find runs/initial/sampling_states -name '*_step40_beam0.pt')
rfd4-generate inference_experiment=sm_binder_hbond_search \
  seed=7 num_workers=0 batch_size=1 num_replicates=1 \
  out_dir=runs/continued search.resume_from="$state" \
  search.resume_seed=11 search.n_branch=8 search.save_states=true
```

Each file holds the batch's survivors at one beam rank (`beam0` is best), named after
its first input. Keep the same inputs, batch size, ordering, weights, generation settings
and `seed`; larger batches work the same way. Resume only the inputs in that saved batch.
Change `resume_seed` and use a fresh output directory for another continuation.
Later intermediate states are saved again; per-checkpoint branch counts override `search.n_branch`.
The saved step, modalities, tensor dimensions and finite values are checked. Input identity,
conditioning, weights and schedule compatibility remain the caller's responsibility.
This starts new seeded branches; it does not restore an interrupted run's RNG state.
Adapters, gradient guidance and inverse folding are unsupported.
