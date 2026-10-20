# Metric guidance

Conditioning tells the model what to build; metric guidance nudges *how* it gets there. At each
active sampling step, guidance differentiates a scalar metric with respect to the model's predicted
clean state and adds a weighted gradient step to the sampler. Use it when you have a differentiable
objective that is better expressed as a score than as an input condition.

Metric guidance uses the `guidance` Hydra group and does not require any separate sampling mode.

This page covers what guidance does and why its pieces behave the way they do. For the YAML, the
smoke test, and building a real objective, see [Metric guidance](../how-to/advanced/metric-guidance.md).

## How it works

At each active sampling step, the guidance path:

1. takes the model's predicted clean state, currently `bb_ca` and `local_latents`;
2. evaluates the configured differentiable metric;
3. selects a scalar output by its `metric_name.output_key`;
4. differentiates that scalar with respect to each configured modality;
5. multiplies each gradient by its signed schedule weight; and
6. adds the resulting delta to the integrated sampler state.

The normal padding mask is applied to both modalities, so guidance never pushes on padding tokens.
Latent deltas are further limited to tokens that can be encoded, since not every token carries a
latent representation to nudge.

Because the update is a gradient step added directly to the sampler state, guidance can only reshape
a trajectory the model is already capable of producing — it cannot introduce content the model
couldn't otherwise generate, and a metric whose gradient is uninformative near the current state
(flat, noisy, or discontinuous) will not steer the trajectory even while it's active.

See [Metric and output keys](../how-to/advanced/metric-guidance.md#metric-and-output-keys) for how a metric
and its output are wired into a `guidance` config.

## Schedules and signs

A schedule decides how much a term's gradient matters at a given point in the trajectory: it's a
callable from flow-matching time to a signed weight. Flow-matching time runs from `0` (noise) to `1`
(clean output), independently for each modality, so a schedule can concentrate its influence in a
narrow window or apply uniformly throughout.

Guidance always **maximizes** the weighted objective. There's no separate sign or magnitude field —
both live in the weight itself, so a positive weight increases a score and a negative weight
decreases a cost. Because the effect compounds over every active step, it's easy to overshoot: start
with a narrow window and a small magnitude, and expect the useful scale to depend on the metric and
its reduction.

A term can also weight several modalities independently. Omitting a modality simply gives it no
direct gradient update from that term, which is useful when an objective should act on, say,
backbone coordinates without touching the latent representation at all. A listed modality still has
no effect if the metric's output doesn't actually depend differentiably on that state leaf.

See [Schedules and signs](../how-to/advanced/metric-guidance.md#schedules-and-signs) and
[Modality weights](../how-to/advanced/metric-guidance.md#modality-weights) for the schedule options and YAML.

## Cost and limitations

- Metric guidance keeps an autograd graph for every active step and backpropagates the metric, so
  runtime and memory depend on the objective and how wide its active window is.
- A metric that requests decoded atom37 coordinates also runs the differentiable decoder on active
  steps, adding further cost. Metrics that call external oracles can't be used at all — guidance
  needs gradients, and nothing outside PyTorch can supply them.
- Guidance doesn't inherently require a second network evaluation per step beyond that decoder cost.
- Guidance currently applies to standard full generation only. Partial diffusion, inverse folding,
  and beam search don't pass a guidance model into their partial-simulation path.
- Guidance optimizes exactly the metric you wrote, including its shortcuts and coordinate
  conventions — it does not guarantee foldability, novelty, or wet-lab success. Treat it as a way to
  bias sampling toward a proxy you trust, not a substitute for evaluating the result afterward.

If a run is slow, unstable, or not visibly changing, see
[Troubleshooting](../how-to/advanced/metric-guidance.md#troubleshooting).
