# Cost and runtime

Reward-guided search is **much more expensive** than plain generation. Every
candidate rolled out during search is scored by the reward, and when the reward
uses a refolding oracle, each score is a full refolding call through the selected
backend. Budget accordingly.

## The cost model

The dominant cost is oracle calls, which scale roughly as:

```
oracle calls  ~  beam_width  x  n_branch  x  (number of checkpoints)
```

where the number of checkpoints is `len(step_checkpoints)`. Concretely, at each
checkpoint every surviving beam (`beam_width`) spawns `n_branch` candidates, and
each candidate is scored.

Worked numbers (using the shipped example's shape,
`step_checkpoints: [0, 25, 50, 75, 100]` -> 5 checkpoints):

| beam_width | n_branch | checkpoints | ~oracle calls per input |
| --- | --- | --- | --- |
| 3 | 2 | 5 | ~30 |
| 4 | 4 | 5 | ~80 |
| 8 | 4 | 5 | ~160 |
| 4 | 4 | 9 | ~144 |

Multiply by `num_replicates` and by the number of input examples. Each oracle
call is a refold, so wall-clock and GPU cost grow the same way. (Non-refold
rewards like `interchain_interactions` skip the oracle and are far cheaper.)

## Practical guidance

- **Start small.** Validate a config end to end at tiny settings (small
  `beam_width`/`n_branch`, few short checkpoints, small `nsteps`) before scaling.
- **`beam_width` and `n_branch`** are the two biggest cost multipliers. Defaults
  are 4 and 4; the example ships 3 and 2. Raise them only as compute allows —
  higher values buy quality/diversity at linear cost.
- **`step_checkpoints`** — more checkpoints means finer pruning but more scoring
  passes. Remember the constraint: start at 0, strictly increasing, end at
  `nsteps` (`generation.args.nsteps` must equal the last checkpoint). Fewer,
  later checkpoints score fewer, more-formed candidates.
- **`intermediate_sample_save_criterion`** does not add oracle calls (it reuses
  each candidate's already-computed metrics/reward), but it does write more
  outputs.
- **Prefer non-refold rewards when they capture your objective** (e.g. interface
  H-bond counts) — they avoid the oracle entirely.

## Compared to plain generation

`rfd4-proteina-generate` runs **one trajectory per design with no oracle** — a
single forward sampling pass, optionally followed by metrics. Search runs many
trajectories and scores each, so a search run can cost tens to hundreds of times
more than generating the same number of final designs.

Use plain **rfd4-proteina-generate** for a cheap baseline or when you need many
designs quickly; use search when quality-per-design justifies the refolding or
reward-computation budget.

## SLURM

For real runs, submit through `scripts/launch_search.sh <cluster>` (search jobs
default to a longer runtime, `08:00:00`, and run inside the container). See the
main [SKILL.md](../SKILL.md) Step 5 and rfd4-proteina-setup
[references/clusters.md](../../rfd4-proteina-setup/references/clusters.md).
