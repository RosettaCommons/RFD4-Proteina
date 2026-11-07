# Reward guided search

How to configure beam search: checkpoints, rewards, continuation criteria, and novelty. See
[Reward guided search](../concepts/search.md) first if you haven't: it explains what search does,
the kinds of reward it can use, and when to reach for it instead of plain generation.

Run `rfd4-generate inference_experiment=inference_beam_example` for an AF3 example, or
`inference_experiment=sm_binder_hbond_search` for structure-only scoring.

Every `search` parameter — the top-level block, per-checkpoint fields, reward terms, and threshold
schemas — is listed in the [search parameters reference](../reference/search-parameters.md). This
page covers how the pieces fit together and when to reach for each one.

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

- Checkpoint keys name positive integer integration steps, executed in sorted order. Keys cannot
  exceed the actual schedule length, including partial diffusion.
- A checkpoint at the schedule end (normally `60`) is optional. If omitted, every survivor of
  the last configured checkpoint finishes once, without additional branching or selection.
  These completed designs run ordinary final metrics; they have no final `search.reward`.
  If present, the end checkpoint branches and selects using its configured metrics and rewards.
- Each checkpoint is self-contained; nothing carries forward from a previous checkpoint. See the
  [per-checkpoint parameters](../reference/search-parameters.md#per-checkpoint-parameters) for
  every field and its default.
- Reward terms use `metric_name.output_key` from definitions under `metrics`, and are clipped,
  normalized, and weighted into a sum; negative weights minimize. See
  [reward terms](../reference/search-parameters.md#reward-terms) for the exact fields.
- An empty reward gives every candidate zero, preserving deterministic input order at selection.
  A zero-weight term does not contribute to ranking.
- `n_branch` is the number of children per surviving beam for every incoming checkpoint interval; a
  checkpoint may override it for its own interval. In the example, steps 0–20 generate four
  children per state, while later intervals generate two. `beam_width` is the number retained per
  input.

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
see [Running metrics inline](evaluate-with-metrics.md#running-metrics-inline).

The top-level `metrics_tags` continues to control ordinary output evaluation. It does not
select search metrics.

## ESM-2 sequence rewards

`inference_esm_beam` ranks decoded protein sequences with a frozen ESM-2 encoder and a trained
linear head. Install the optional `esm` extra in the inference environment with
`pip install -e '.[esm]'` from the checkout.

Fit directly from a completed generation or evaluation run by specifying a numeric target column.

```bash
rfd4-fit-esm-reward --run-dir /path/to/scored_run \
    --target oracle.plddt_coi --output /path/to/esm_head.json
```

The command reads `inference_metrics/per_example_metrics.csv` or
`evaluation_metrics/per_example_metrics.csv` and embeds the original designs from
`designed_structure_path`. Use `--scores /path/to/metrics.csv` to pick a table explicitly,
including a custom score table with one row per design. For a moved run, keep the
`inference_samples` or `evaluation_samples` subtree and keep any evaluation symlink targets
accessible. Consolidate per-job score tables before fitting on a multi-job run. Missing target
values are skipped. A labeled row whose structure or chain cannot be resolved fails with its
path in the error. The chain comes from the table or RFD4's chain-of-interest detection.
Use `--chain-id` if you need to select it explicitly. The default encoder is ESM-2 650M on
CUDA. `--model`, `--layer`, and `--device` override it.

Use `--rewards` to fit to a composite score using the same [reward terms](../reference/search-parameters.md#reward-terms)
as beam search. For example, save the following as `rewards.yaml`:

```yaml
rewards:
  oracle.plddt_coi: {normalize_by: 100}
  refolded_backbone_rmsd.rmsd:
    weight: -1.0
    clip: {max: 5.0}
    normalize_by: 5.0
```

```bash
rfd4-fit-esm-reward --run-dir /path/to/scored_run \
    --rewards rewards.yaml --output /path/to/esm_head.json
```

The fitter applies the same clipping, normalization, and weights as search to the metric columns
in the table. It skips rows with missing or nonfinite values in any nonzero term. The head
predicts the combined reward, so a positive weight on `esm_reward.score` maximizes it. You can
also pass an existing search config with `--rewards search_config.yaml --checkpoint 60` to use
that checkpoint's reward terms. If the config inherits its reward terms through Hydra defaults,
pass the composed config. The head records the resolved reward terms.

You can also fit from raw [FAIR ESM-2 residue means](https://github.com/facebookresearch/esm#compute-embeddings-in-bulk-from-fasta)
and candidate-level scores. The NPZ needs `embeddings` (one row per candidate), `candidate_ids`
(unique Unicode IDs), `embedding_model` (such as `esm2_t33_650M_UR50D`) and `embedding_layer`
(an integer). Pool the entire scored chain and exclude special tokens and padding. Join the
scores through a CSV with a `candidate_id` column and a numeric target column. Aggregate
repeated refolds per candidate before fitting.

```bash
rfd4-fit-esm-reward \
    --embeddings embeddings.npz --scores candidate_scores.csv \
    --target mean_graft_plddt --alpha 1.0 --output esm_head.json
```

This exports a JSON head for the specified encoder and layer. The fitter skips candidates
without finite target values. Choose the target and regularization using held-out designs
before fitting a head for search. A head can predict any numeric score. For an error target,
set a negative weight on `esm_reward.score` at each checkpoint.

```bash
rfd4-generate inference_experiment=inference_esm_beam \
    inputs=design.yaml out_dir=./runs/esm_search \
    metrics.inline_metrics.esm_reward.artifact_path=/path/to/esm_head.json
```

The preset scores completed rollouts at step 40 and completed candidates at step 60. It uses
only the ESM reward at those checkpoints. The first use loads the named FAIR ESM-2 checkpoint
from the PyTorch Hub cache or downloads it from FAIR. The encoder runs on the current CUDA
device by default. Set `metrics.inline_metrics.esm_reward.device=cpu` to place it on CPU.
Repeated sequences reuse their scores within a validation epoch.

The metric uses the full unambiguous chain of interest, with support for up to 1,022 canonical
residues. Set `metrics.inline_metrics.esm_reward.chain_id` to select a chain explicitly, using
its `chain_iid` when that annotation is present. Train on sequences extracted with the same
scope. `esm_reward.score` is a prediction of the chosen training target. Add appropriate final
refolding tags if you want to measure the resulting designs with an oracle. For an early
checkpoint using `use_clean_prediction: true`, validate the head on those decoded predictions
before using it to prune branches.

## Saving completed rollouts

The final beam is always returned. Set `search.save_rollouts: true` to also return completed
scoring rollouts, including candidates that lose selection. An optional `search.save_criteria`
mapping filters these extra outputs using raw metric thresholds:

```yaml
search:
  save_rollouts: true
  save_criteria:
    search.reward: {min: 80.0}
```

Every rollout checkpoint must list all providers needed by its save criteria; missing outputs
produce an error. A saved final-beam member appears only once. Reward scales can differ across
checkpoints, so a shared threshold on `search.reward` should be chosen accordingly. Clean
predictions are never saved as completed designs. See the
[threshold schema](../reference/search-parameters.md#continuation_criteria-and-save_criteria-thresholds)
for the full set of fields.

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
used only for continuation. Criteria can also read `search.reward`, which includes clipping and
normalization. See the
[threshold schema](../reference/search-parameters.md#continuation_criteria-and-save_criteria-thresholds)
for the full `min`/`max`/`min_of`/`max_of`/`mean_of` set of fields.

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
passer database, without clustering or a prefilter index, giving `is_novel: 0` on any match and
`1` otherwise. See the
[novelty metric parameters](../reference/search-parameters.md#novelty-metric-parameters) for
`tm_threshold`, `min_coverage`, `threads`, `timeout`, and the required `FOLDSEEK_EXEC` binary.

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
