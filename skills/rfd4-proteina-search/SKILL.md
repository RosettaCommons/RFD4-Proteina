---
name: rfd4-proteina-search
description: >
  Reward-guided search for RFD4-Proteína (RFD4-Proteína) — the Baker Lab x NVIDIA
  flow-matching protein design model. Reach for this skill for "reward-guided
  search", "beam search", "optimize designs", "AF3 oracle reward", "generate the
  best binders/monomers by pLDDT/ipTM/ipAE", "search with a reward", or "branch
  and prune candidates at checkpoints". Instead of sampling once, it explores
  multiple generation paths and keeps the candidates a reward scores highest —
  running through the same `generate.py` entry point, selected by a search
  `inference_experiment` config. Refolding rewards require the selected oracle
  backend and are much more expensive than plain generation; configured
  structure-only rewards such as interchain interactions do not require a
  refolding oracle.
  Use rfd4-proteina-generate for unguided sampling and this skill when
  reward-guided pruning justifies the additional compute.
license: See the RFD4-Proteína repository (Baker Lab x NVIDIA collaboration)
---

# RFD4-Proteína Reward-Guided Search

## Requirements

Complete `rfd4-proteina-setup` first. Generation requires a CUDA GPU. Rewards
using refolding tags also require a configured refolding backend; RFD4-Proteína
supports AF2, AF3, and OpenFold3 through `metrics/refolding_oracles`. Rewards such
as `interchain_interactions` do not require a refolding oracle.

Search explores multiple generation trajectories and keeps the candidates a
**reward** scores highest. **Beam search** ships ready to run: it branches and
prunes partial trajectories at step checkpoints, scoring survivors with a reward
model. The shipped `inference_beam_example` uses the default **AF3 oracle** for
refolding confidence; other search configs may select another backend or avoid
refolding entirely.

Search runs through the **same entry point as generation** — `generate.py` —
selected by an `inference_experiment` config that adds one `search:` block. All
arguments are Hydra `key=value` (never `--key value`).

Run everything from the repository root (the directory containing `.project-root`).

## Step 1 — Preflight the selected reward dependencies

Every search needs a green rfd4-proteina-setup:

```bash
bash skills/rfd4-proteina-setup/scripts/preflight.sh
```

If the reward uses refolding, inspect the selected backend before running. AF3 is
the default; choose AF2 or OpenFold3 with
`metrics/refolding_oracles=af2` or `metrics/refolding_oracles=of3`. Backend
capabilities differ by design case, so confirm that the selected config contains
an oracle activated by the reward tag. See
[references/rewards.md](references/rewards.md), the AF2/AF3 references, and
rfd4-proteina-setup and the [RFD4-Proteína installation guide](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/installation.rst).
Use `rfd4-generate` from the Python environment activated during setup.
Refolding oracles use their separately configured runtimes. IPD users retain
their existing container launchers.

If a refolding reward fails, verify the selected backend, its tag support, and its
runtime before changing the reward.

## Step 2 — Run the ready example

Two search configs ship ready to run:

```bash
# Beam search, AF3 refolding pLDDT as reward (configs/inference_experiment/inference_beam_example.yaml)
rfd4-generate inference_experiment=inference_beam_example

# Beam search, NO refolding — interface protein<->ligand H-bonds as reward
# (configs/inference_experiment/sm_binder_hbond_search.yaml)
rfd4-generate inference_experiment=sm_binder_hbond_search
```

The first uses the AF3 oracle (needs the container); the second scores with
interface H-bond counts and never refolds, so it needs no oracle.

## Step 3 — Build your own search config

Copy a shipped config and edit it. A search config is a normal
`inference_experiment` with **one added `search:` block** — a `strategy` and a
`rewards` model. Full detail: [references/rewards.md](references/rewards.md).

### The reward (`search.rewards`)

The reward is defined **inline** — there is no separate reward file:

```yaml
search:
  rewards:
    reward_metrics_tags: [unconditional_refolding]   # which oracle/metrics fire during scoring
    weights:
      oracle.plddt_coi: 1.0                           # metric key -> coefficient (negative to minimize)
    clip_values:                                      # optional: bound a key before weighting
      oracle.plddt_coi:
        min: 50.0
```

- **`reward_metrics_tags`** — tags selecting which metrics run during scoring,
  independent of the top-level `metrics_tags`. Refolding tags such as
  `unconditional_refolding` activate the selected oracle backend when that
  backend supports the design case. Standard oracle outputs are prefixed
  **`oracle.`**.
- **`weights`** — maps each metric output key to a coefficient; the reward is
  their weighted sum. Use a **negative weight to minimize** a metric. Available
  oracle keys include `oracle.plddt_coi`, `oracle.iptm_coi`, `oracle.ipae_coi`,
  `oracle.min_ipae_coi`.
- **`clip_values`** *(optional)* — per-key `{min, max}` bounds; the value is
  clipped **before** its weight is applied, so the reward stops responding once
  the metric passes the bound (the raw metric is still reported). Each clipped
  key must also appear in `weights`.

### Changing the reward mid-trajectory (`checkpoint_overrides`, optional)

`search.rewards.checkpoint_overrides` makes the reward change as the trajectory
progresses. Each entry is a **sub-version of the base reward**, keyed by the step
it takes effect at; anything unspecified is inherited:

```yaml
search:
  strategy:
    step_checkpoints: [0, 20, 40, 50, 60]
  rewards:
    reward_metrics_tags: [interchain_interactions]     # cheap, no refolding
    weights:
      interchain_interactions.hbond_atom_pair_count_coi: 1.0
    checkpoint_overrides:
      60:                                               # final checkpoint only
        reward_metrics_tags: [unconditional_refolding]  # selected refolding oracle runs now
        weights:
          oracle.plddt_coi: 1.0                         # replaces the base weights
```

- Keys must be one of `step_checkpoints` **past the first** (those are the scored
  steps); a step never scored at is an error.
- Every field an override names — `weights`, `clip_values`, `reward_metrics_tags`
  — **replaces** the base one wholesale; there is no key-by-key merge. Restate
  every term that should still count at that checkpoint.
- That is what keeps weights and metrics in sync: a weighted key whose metric no
  longer fires under the override's tags scores `-inf`; if that drowns every
  candidate at a checkpoint, the run raises there instead of pruning on nothing.
- Replacing `weights` while inheriting `clip_values` errors if a clip no longer
  matches a weighted key — set `clip_values: {}` to clear them.
- Unlisted checkpoints use the base reward, so existing configs are unaffected.

**Why stage a reward.** Each checkpoint prunes on a rollout of a still-noisy
state. Early on, only broad features are settled — fold, topology, whether an
interface forms at all — so ranking on fine details there ranks noise, not
promise. Matching the reward to what the trajectory has actually committed to by
each checkpoint (**broad structural features early, finer details late**) can
raise pass rates substantially, typically a larger effect than the compute saved.
Placing the oracle deliberately — only where it changes which candidates survive
— also cuts cost; that is not necessarily the last checkpoint, and mid-trajectory
or even early placement is worth testing on your problem.

Rewards from different checkpoints are different objectives, so their
`search.reward` values are not comparable (mind any
`intermediate_sample_save_criterion` bar on `search.reward`).
`sm_binder_hbond_search.yaml` ships a worked example.

### The beam strategy (`search.strategy`)

```yaml
search:
  strategy:
    _target_: rfproteina.search.beam_search.BeamSearchStrategy
    step_checkpoints: [0, 20, 40, 50, 60]       # steps to score + prune at
    n_branch: 2                                  # candidates spawned per surviving beam (default 4)
    beam_width: 3                                # candidates kept after each prune (default 4)
    intermediate_sample_save_criterion:          # optional; omit to return only the final beam
      _target_: rfproteina.metrics.oracles.pass_criterion_metric.PassCriterionMetric
      metric_thresholds:
        search.reward:
          min: 80.0
generation:
  args:
    nsteps: 60                                   # released default; MUST equal the last step_checkpoint
```

- **`step_checkpoints`** — the diffusion steps to score and prune at. **Must
  start at 0, be strictly increasing, and end at `nsteps`.**
- **`n_branch`** — candidates spawned per surviving beam at each checkpoint
  (default 4).
- **`beam_width`** — candidates kept after each prune, i.e. the beam size
  (default 4).
- **`intermediate_sample_save_criterion`** *(optional)* — a
  `PassCriterionMetric` that also saves lost-beam candidates clearing a bar. It
  sees each candidate's metrics plus its reward under `search.reward`. Omit to
  return only the final beam.
- **`generation.args.nsteps` MUST equal the last `step_checkpoint`** — a
  mismatch errors.

## Step 4 — Cost and runtime

Search is **much more expensive** than plain generation. Cost scales roughly as:

```
oracle calls  ~  beam_width  x  n_branch  x  (number of checkpoints)
```

Raise `beam_width`/`n_branch` only as compute allows; add checkpoints only when
finer pruning pays off. For a cheap, unguided baseline use **rfd4-proteina-generate**
(one trajectory per design, no oracle). Full guidance and worked numbers:
[references/cost.md](references/cost.md).

## Step 5 — Submit on SLURM

Search is an inference configuration, so the standard generation launcher is the
preferred batch path:

```bash
./scripts/launch_generate.sh \
    inference_experiment=inference_beam_example
```

Before submission, read `scripts/.user_info_<cluster>` and the launcher. Preserve
the user's partition, account, QoS/priority options, GPU request, runtime, memory,
and CPU settings. Add a launcher resource override only when the user explicitly
requested it. State the resolved scheduler values and inspect the generated
`#SBATCH` directives when available.

Search uses the standard `launch_generate.sh` path so scheduler settings and
`SLURM_EXTRA_OPTIONS` follow the same profile contract as ordinary generation.
Never submit a job as part of skill validation.
## Outputs

Search writes the **final beam** (the surviving top candidates) as annotated
`.cif` files, plus any **intermediate candidates** saved by
`intermediate_sample_save_criterion`. Each output is scored by the reward, and
its metrics land in `per_example_metrics.csv` (the epoch-end metrics loop reuses
the oracle results cached during scoring, so it is fast). CIF annotations follow
the same schema as plain generation (see `docs/workflow/running.md` Output Files).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Refolding errors or oracle not found | Confirm `metrics/refolding_oracles` selects a backend that supports the reward tag, then verify that backend's runtime. See [references/rewards.md](references/rewards.md), the AF2/AF3 references, rfd4-proteina-setup, and the [RFD4-Proteína installation guide](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/installation.rst). |
| Error about `step_checkpoints` / `nsteps` | `step_checkpoints` must start at 0, be strictly increasing, and end at `nsteps`; set `generation.args.nsteps` to the last checkpoint |
| Reward not changing / plateaus early | A `clip_values` bound was hit — the metric is clipped before weighting, so it stops contributing past the bound; raise/remove the bound or reweight |
| Run far slower / more GPU than expected | Cost is `beam_width x n_branch x checkpoints` oracle calls — lower them, or use rfd4-proteina-generate for an unguided baseline. See [references/cost.md](references/cost.md) |
| Error that an override targets a step the search never scores at | `search.rewards.checkpoint_overrides` keys must be one of `step_checkpoints` *past the first* (step 0 is never scored) |
| Only final designs returned, no intermediates | Add `intermediate_sample_save_criterion` (a `PassCriterionMetric`) to also keep lost-beam candidates that clear a bar |
| Hydra `--key value` not applied | All args are `key=value` (Hydra), never `--key value` |

For deeper detail: [rewards](references/rewards.md) ·
[AF2 oracle](references/af2-oracle.md) · [AF3 oracle](references/af3-oracle.md) ·
[cost](references/cost.md). Source of
truth: `docs/workflow/search.md`, `docs/workflow/running.md`, and the current
search implementation/configs.
