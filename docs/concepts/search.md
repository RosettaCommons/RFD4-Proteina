# Reward guided search

Standard generation runs one uninterrupted sampling trajectory per output: you get back whatever
that trajectory produces, good or bad. Reward guided search replaces that with beam search: at a
handful of checkpoints along the trajectory, every surviving candidate branches into several
children, a reward scores the branches, and only the strongest continue. The result concentrates
compute on trajectories that are already looking promising, instead of spending it uniformly across
samples you will mostly throw away.

This page covers what search does and why you'd reach for each piece. For the YAML and exact
mechanics, see [Reward guided search](../how-to/search.md); for every field search accepts, see the
[search parameters reference](../reference/search-parameters.md).

## Why search instead of plain generation

| | Plain generation | Reward guided search |
| --- | --- | --- |
| What runs | One trajectory per output, start to finish | Many partial trajectories, branched and pruned along the way |
| When a bad candidate is caught | After the full trajectory finishes, during evaluation | As early as the first checkpoint, before most of the trajectory runs |
| Where compute goes | Spread evenly across every sample, including ones that fail | Concentrated on the branches a reward already favors |

Search doesn't change what the model can produce; it changes which partial trajectories get to keep
running. You still need a way to tell good candidates from bad ones early enough in the trajectory
for that signal to be useful, that's what a reward is for.

## Checkpoints: where the trajectory branches

A checkpoint is an integration step at which every surviving candidate branches into several
children, gets scored, and is pruned back down to a fixed number of survivors. Checkpoints are
self-contained: nothing about how a candidate was scored or selected at one checkpoint carries
forward to the next, so each one lists its own metrics and reward from scratch.

The last checkpoint, at the end of the schedule, is optional. Configure one and it branches and
selects like any other, over finished designs. Leave it out and every survivor of the last
configured checkpoint simply finishes its trajectory once, with no further branching. This is useful when
you want search to shape the trajectory early on but don't need a final competitive selection.

See [Checkpoint configuration](../how-to/search.md#checkpoint-configuration) for the YAML.

## Rewards: what search optimizes

A reward turns one or more metric outputs into a single number used to rank candidates at a
checkpoint: each term reads one named metric output, is optionally clipped and normalized, and is
combined into a weighted sum (a negative weight minimizes instead of maximizing.) Combining several
terms lets a reward trade off properties that pull in different directions, for example rewarding
hydrogen bonds while also rewarding hydrophobic packing.

Each checkpoint has its own reward, so a run doesn't have to score the same thing at every branch
point. An early checkpoint can prune on something cheap and coarse, while a later one makes the
final call on something expensive and closer to what you actually care about.

See [Metric selection](../how-to/search.md#metric-selection) and
[reward terms](../reference/search-parameters.md#reward-terms) for the fields.

## Kinds of reward: structure-only vs. oracle-based

Where a reward's signal comes from determines both its cost and how trustworthy it is:

- **Structure-only rewards** read metrics computed directly from the generated structure, such as
  interchain hydrogen bond or hydrophobic contact counts. They are cheap enough to run at every
  branch of every checkpoint. `inference_experiment=sm_binder_hbond_search` searches on this kind of
  reward alone.
- **Oracle-based rewards** read the output of an independent structure prediction model (AF3/OF3)
  folding the candidate's sequence, such as pLDDT. They are far more expensive — refolding is the
  most costly step in the pipeline — but measure something closer to what a wet-lab result would
  depend on. `inference_experiment=inference_beam_example` searches with an oracle reward at its
  final checkpoint.

A single run can use both: cheap structure-only rewards at early checkpoints to prune broadly, and
an oracle reward only at the checkpoints where its cost is worth paying.

A related choice is *what gets scored* at a non-final checkpoint: the model's clean prediction at
that step, which is fast but approximate, or a separate rollout completed from that point, which
costs more sampling but scores an actual finished structure. See
[Scoring clean predictions without completing rollouts](../how-to/search.md#scoring-clean-predictions-without-completing-rollouts).

## Continuation criteria: giving up on an input early

Reward ranking picks the best of whatever candidates exist, but it won't tell you an input is a lost
cause — it always keeps the top scorers, even if all of them are bad. Continuation criteria is a
separate check: raw-metric thresholds applied after selection. If every reward-selected candidate
for an input fails its thresholds, that input stops sampling entirely instead of continuing to spend
compute on trajectories unlikely to pay off; if any one of them passes, the whole selected beam
continues, including the ones that failed.

This is also how [novelty-guided search](#novelty-against-already-accepted-designs) decides when to
stop: not "is this the best candidate available" but "is this candidate actually worth continuing."

See [Continuation criteria](../how-to/search.md#continuation-criteria) for the YAML and the full
threshold schema.

## Novelty against already-accepted designs

Reward and continuation criteria don't have to read conventional structural or confidence metrics,
they can read anything registered as a metric, including a structural novelty check against designs
a previous pass has already accepted. Gating continuation on "this candidate doesn't match anything
already in the passer archive" steers search away from re-finding hits you already have and toward
unexplored parts of design space, using the same checkpoint machinery as any other reward.

See [Novelty against passing designs](../how-to/search.md#novelty-against-passing-designs) for the
full walkthrough and [novelty metric parameters](../reference/search-parameters.md#novelty-metric-parameters)
for its configuration.

## What you get back

Search always returns the survivors of the last checkpoint. It can also keep the
scoring rollouts and candidates that lost selection along the way, which is useful for inspecting
why search chose what it did rather than only seeing the winners. See
[Saving completed rollouts](../how-to/search.md#saving-completed-rollouts) for how to turn that on
and filter what gets kept.
