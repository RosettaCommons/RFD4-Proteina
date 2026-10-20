---
name: rfd4-proteina-autoresearch
description: >
  Run an iterative, inspectable design campaign with RFD4-Proteina — a
  design → score → iterate loop that composes rfd4-proteina-generate,
  rfd4-proteina-search, and rfd4-proteina-evaluate over multiple waves. Reach for this
  skill for "autoresearch", "run a design campaign", "optimize binders over
  several rounds", "sweep search parameters", "what should I try to improve my
  designs", "tune the reward function", "which knobs matter", "beam width / branch
  / checkpoints sweep", "explore vs exploit design branches", or "make my design
  loop reproducible/inspectable". It provides the methodology: how to define a
  composite reward, which RFD4-Proteina knobs to vary, wave-based freeze/promote/
  allocate rules, and how to keep every wave's decisions and results on disk so
  failed branches stay useful. It does not add new model capability — it
  orchestrates the existing skills.
license: See the RFD4-Proteina repository (Baker Lab x NVIDIA collaboration)
---

# RFD4-Proteina Autoresearch

An autoresearch campaign is a **design → score → iterate** loop: generate
candidates, score them against an explicit reward, keep what improves the metric,
and let the results steer the next round. This skill is the *strategy layer* over
the mechanics that already exist:

- **rfd4-proteina-generate** — produce designs for a task.
- **rfd4-proteina-annotated-cif** — build or edit the annotated CIF that defines
  the design problem (length, epitope/hotspots, ligand).
- **rfd4-proteina-search** — reward-guided beam search (AF3 oracle) for a single run.
- **rfd4-proteina-evaluate** — metrics + aggregation + diversity clustering.

The value is not any single run — it is the **trajectory**: which hypotheses you
tested, where you reset, and what evidence spawned the next wave. Make the reward
and rules explicit so the loop is steerable and replaceable when you get better
evidence.

> Methodology note and provenance. The wave/branch loop and composite-reward
> engineering below are adapted from muni's autoresearch experiments
> ([scoring-function-dependent strategies](https://muni.bio/research/scoring-function-dependent-strategies),
> [autoresearch + CLI](https://www.muni.bio/research/muni-makeover-cli-autoresearch)),
> themselves inspired by Karpathy's autoresearch framing. **Those experiments ran
> Proteina-Complexa on TREM2, a different model and target** — their specific
> findings (e.g. a `cdr2_bias` epitope, `beam_width 4->12`, `MCTS n=8`, an ipSAE of
> ~0.95) are **illustrative, not RFD4-Proteina defaults**, and RFD4-Proteina's search is
> **beam search only** (no MCTS / best-of-n / FK-steering). Port the *method*, not
> the numbers. See [references/loop.md](references/loop.md).

## Step 1 — Make the objective explicit

Before generating anything, write down (on disk, so it is inspectable):

- **Design problem** — task + input (monomer, motif, small-molecule/enzyme
  active site; for binders the target + epitope/hotspots). Build it with
  **rfd4-proteina-annotated-cif**.
- **Reward** — the composite metric you optimize, as RFD4-Proteina reward keys with
  weights and gates (Step 3). A provisional reward is still useful — it is a
  hypothesis about what matters, meant to steer search and be replaced.
- **Rules** — freeze/promote/allocate thresholds and the per-wave job budget
  (Step 4).

Keep these in a small campaign directory, e.g. `campaign/program.md`,
`campaign/wave1.json`, `campaign/decisions.md`, `campaign/results.tsv`.

When a wave is running inside `rfd4-proteina-campaign`, `campaign.sqlite` is
the canonical record. Register each branch input/config and ingest every design,
oracle prediction, and metric observation into that ledger. The Markdown/JSON
wave files remain decision artifacts and `results.tsv` becomes a derived view.

## Scheduler intent

For any batch run, read the selected `scripts/.user_info_<cluster>` and the
launcher before constructing the command. Preserve the user's partition,
account, QoS/priority options, GPU request, runtime, memory, and CPU settings.
Use a resource override only when the user explicitly requested it. State the
resolved scheduler values before submission; use dry-run or non-submitting tests
when validating the workflow.

## Step 2 — Seed the first wave of branches

A **branch** is one (input, search-config) combination. Seed a handful that span
the hypotheses you actually want to test — for a binder that might be
{epitope A, epitope B} x {length 60-100, 80-130}; for a monomer it might be
{diversity noise low/high} x {nsteps}. Each branch is a `rfd4-proteina-search` (or
plain `rfd4-proteina-generate`) run.

## Step 3 — Engineer the composite reward

RFD4-Proteina's reward is defined per step under `search.checkpoints.<step>.rewards` (see
**rfd4-proteina-search**). The autoresearch-specific discipline — from the muni
technical note, and directly applicable here:

- **Normalize term scales before weighting.** Terms live on wildly different
  scales (check each metric's units; interface areas can be hundreds/thousands
  of Å² and H-bond counts small integers). Set each term's `normalize_by` to
  bring values onto comparable scales before weighting.
- **Gate, don't pay forever.** Use each reward term's `clip` to turn a "must clear a bar" term
  into a gate — e.g. clip an interface-area/`min_ipae` term at a threshold so the
  reward stops responding once it is satisfied, instead of rewarding ever-larger
  values.
- **Down-weight wide-range terms** so a 0-100 style term does not swamp a 0-1 one.
- Use **negative weights to minimize** (e.g. minimize `oracle.ipae_coi`).
- **The reward may vary over the trajectory.** Each checkpoint explicitly lists
  its `metrics`, `rewards`, and `use_clean_prediction` setting. Nothing is
  inherited between checkpoints; list any oracle or comparison provider needed.

Full worked patterns and the RFD4-Proteina reward keys
(`oracle.plddt_coi`, `oracle.iptm_coi`, `oracle.ipae_coi`, `oracle.min_ipae_coi`):
[references/reward-engineering.md](references/reward-engineering.md).

## Step 4 — Run the wave, then freeze / promote / allocate

Run the wave's branches and collect common final metrics (during generation or
with **rfd4-proteina-evaluate**), then update branches by **branch health**
(the median of that branch's top-3 scores):

- **Freeze** a branch whose health is in the bottom quartile (stop funding it).
- **Promote** a branch that produced a current global top-k design — spawn one
  child by mutating **exactly one** knob (Step 5).
- **Allocate** the next wave's budget ~70% to the strongest branches
  (promoted + top-half by health) and ~30% to unexplored combinations
  (explore/exploit).

Record every branch outcome and the promote/freeze reasoning. Losing branches are
signal: capture *why* they failed. See [references/loop.md](references/loop.md).

## Step 5 — Choose which knob to mutate (RFD4-Proteina-accurate axes)

Change **one** axis per child so effects are attributable:

- **Search**: `search.beam_width`, `search.n_branch`, and `search.checkpoints`
  alongside `generation.args.nsteps`. Oracle cost depends on which checkpoints
  list oracles. Beam vs vanilla is the only algorithm axis.
- **Reward**: each term's `weight`, `clip`, and `normalize_by`, plus the
  checkpoint's explicit `metrics` list.
- **Reward schedule**: vary the complete configuration at each checkpoint.
  Compare clean-prediction design scoring with completed-rollout scoring, or
  move oracle scoring to a different step. The schedule-end checkpoint is
  optional: without it, each retained beam finishes once without another selection.
  Compare arms using common final metrics, not differently scaled search rewards.
- **Temperature / diversity**: `generation.model.bb_ca.simulation_step_params.tsr_k`
  and `...local_latents...tsr_k` (lower `k` raises temperature). Use `tsr_sigma`
  as a secondary control over when sharpening takes effect.
- **Input**: binder length window, epitope/hotspot set, ligand pose/conditioning
  — via **rfd4-proteina-annotated-cif** (a new branch input).

Full axis catalog with defaults and cautions:
[references/sweep-axes.md](references/sweep-axes.md).

## Step 6 — Track diversity and stop conditions

Use `rfd4-proteina-evaluate` with `--cluster_with_foldseek` (needs
`save_chain_of_interest=true` at generation) to avoid optimizing into a single
structural family. Stop when the running best plateaus across waves, or the budget
is spent — and record that the search space was (or was not) exhausted.

## Outputs

For standalone autoresearch, produce a replayable directory: the objective + rules
(`program.md`), per-wave branch payloads (`wave*.json`), decision rationale
(`decisions.md`), and per-branch outcomes (`results.tsv`), alongside the
per-branch design CIFs and metrics CSVs produced by the underlying skills. The
running-best trajectory — including failed branches and why — is the deliverable,
not just the top design.

Inside an end-to-end campaign, preserve those artifacts while using the campaign
ledger as the source of truth for design lineage, oracle outputs, actual metric
values, and passer evidence.

## Caveats

- Rewards optimize a **proxy** (refolding confidence / interface metrics), not
  experimental success — validate designs downstream.
- Refolding-guided waves are **compute-intensive**; size
  `beam_width`/`n_branch`/waves to budget. See **rfd4-proteina-search** `references/cost.md`.
- The muni TREM2/Complexa numbers are illustrations; derive RFD4-Proteina settings
  empirically from your own early waves.
