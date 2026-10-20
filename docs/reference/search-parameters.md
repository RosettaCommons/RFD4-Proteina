# Search parameters reference

Every parameter accepted under the `search` block, for reward guided (beam) search. This page is a
lookup table; see [Reward guided search](../how-to/search.md) for how the pieces fit together,
when to use `use_clean_prediction`, and the novelty-against-passers walkthrough. See the
[CLI options reference](cli-options.md#reward-guided-search) for the handful of `search.*` keys you
can also override directly on the command line.

**Contents**

- [Top-level `search` block](#top-level-search-block)
- [Per-checkpoint parameters](#per-checkpoint-parameters)
- [Reward terms](#reward-terms)
- [`continuation_criteria` and `save_criteria` thresholds](#continuation-criteria-and-save-criteria-thresholds)
- [Novelty metric parameters](#novelty-metric-parameters)

## Top-level `search` block

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `beam_width` | int | `4` | Candidates retained per input after every scored checkpoint. |
| `n_branch` | int | `4` | Default number of children per surviving beam, for any checkpoint interval that doesn't set its own `n_branch`. |
| `save_rollouts` | bool | `false` | Also return completed scoring rollouts that lost selection, alongside the final beam. Required to set `save_criteria`. |
| `save_criteria` | mapping of metric key → threshold | `{}` | Raw-metric thresholds (see [thresholds](#continuation-criteria-and-save-criteria-thresholds)) an extra rollout must pass to be saved. Every checkpoint that can produce a saved rollout must list the providers these keys need. Can reference `search.reward`. |
| `checkpoints` | mapping of step → checkpoint config | **Required**, nonempty | Positive integer integration steps, see [below](#per-checkpoint-parameters). Quote YAML keys to support dotted CLI overrides such as `++search.checkpoints.20.use_clean_prediction=true`. Step zero is implicit and cannot be listed. |

## Per-checkpoint parameters

Each entry in `search.checkpoints` is keyed by a positive integer step and accepts exactly these
fields. Nothing carries forward from one checkpoint to the next — every field is self-contained.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `metrics` | list of metric names | `[]` | Named providers from `metrics.inline_metrics`, `metrics.refolding_oracles`, or `metrics.refolding_metrics` to run at this checkpoint. Runs regardless of dataset selection tags; selecting a provider computes all of its outputs. A checkpoint at the schedule end must list providers that score completed structures (no oracle/refolding/pass-criteria providers if `use_clean_prediction: true`). |
| `rewards` | mapping of `metric.output_key` → [reward term](#reward-terms) | `{}` | Chooses which of the selected metrics' outputs contribute to ranking, and how. An empty mapping gives every candidate zero reward, preserving deterministic input order at selection. |
| `n_branch` | positive int | the global `search.n_branch` | Overrides the branch count for this checkpoint's incoming interval only. |
| `use_clean_prediction` | bool | `false` | Score the clean (decoded) prediction from the preceding integration step instead of completing a separate scoring rollout. Not valid on the schedule-end checkpoint. Rejects oracle, refolding, and pass-criteria providers in `metrics`. Predictions are used only for ranking — never saved as completed designs or sent to refolding. |
| `continuation_criteria` | mapping of metric key → [threshold](#continuation-criteria-and-save-criteria-thresholds) | `{}` | Raw-metric thresholds that decide whether a checkpoint before the schedule end stops an input. The whole selected beam continues if *any* selected candidate passes every threshold; it stops only if all selected candidates fail. Every metric a criterion needs must also appear in this checkpoint's `metrics`. Not valid on the schedule-end checkpoint. |

## Reward terms

Each entry in a checkpoint's `rewards` mapping is keyed `metric_name.output_key` and accepts:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `weight` | finite number | `1.0` | Multiplier applied after clipping and normalizing. Negative weights minimize the term. A zero weight contributes nothing to ranking or selection. |
| `clip` | mapping with `min` and/or `max` | no clipping | Clips the raw metric value before normalizing. At least one of `min`/`max` must be set; `min` must not exceed `max`. |
| `normalize_by` | positive finite number | `1.0` | Divides the (clipped) value before weighting. |

Terms are summed per candidate into `search.reward`. Clipping and normalization affect the reward
only — the underlying metric values stay raw wherever else they're reported.

## `continuation_criteria` and `save_criteria` thresholds

Both fields use the same threshold schema, keyed by a metric output (or `search.reward`) or by an
arbitrary label when reducing several keys:

| Field | Type | Description |
| --- | --- | --- |
| `min` | number | Candidate value must be at least this (inclusive). |
| `max` | number | Candidate value must be at most this (inclusive). |
| `min_of` | list of metric keys | Reduce several keys to their minimum before thresholding with `min`/`max`. Mutually exclusive with `max_of`/`mean_of`. |
| `max_of` | list of metric keys | Reduce several keys to their maximum before thresholding. |
| `mean_of` | list of metric keys | Reduce several keys to their mean before thresholding. |

All listed thresholds must pass. A metric missing or non-finite for a candidate fails that
candidate; a metric absent from every candidate in the pool is a configuration error.

## Novelty metric parameters

`FoldseekNoveltyMetric` (`rfproteina.metrics.foldseek_novelty.FoldseekNoveltyMetric`) is an ordinary
metric you select like any other at a checkpoint, then gate with `continuation_criteria` on its
`is_novel` output. See [Novelty against passing designs](../how-to/search.md#novelty-against-passing-designs).

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `archive_dir` | path | **Required** | Append-only directory of archived passer CIFs, populated by a `PassCriterionMetric` with its own `archive_dir` set. Use one archive per target and pass definition. |
| `tm_threshold` | float in `(0, 1]` | `0.5` | Minimum query- and target-normalized TM score for a candidate to count as a match against an archived passer. |
| `min_coverage` | float in `(0, 1]` | `0.8` | Minimum aligned fraction of both query and target for a match. |
| `threads` | positive int | `1` | CPU threads per Foldseek invocation. |
| `timeout` | positive float (seconds) | `300.0` | Maximum time for each Foldseek invocation; a timeout raises rather than reporting novelty. |

Output: `is_novel` is `0` on any match, `1` if there is no match or the archive is empty. Requires
the `FOLDSEEK_EXEC` environment variable pointing at the Foldseek binary. Each metric instance keeps
a local database copy and rebuilds it only when the archive's published files change.
