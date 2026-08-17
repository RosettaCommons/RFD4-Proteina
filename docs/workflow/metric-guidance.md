# Metric guidance

Metric guidance changes the sampling trajectory by differentiating a scalar metric with respect to
the model's clean prediction. Use it when you have a differentiable objective that is better
expressed as a score than as an input condition.

Metric guidance uses the `guidance` Hydra group and does not require any separate sampling mode.

## Smoke-test the mechanics

The repository ships `+guidance=contract_ca` as a dependency-free mechanics test:

```bash
python src/rfproteina/generate.py \
    inputs=data/benchmarks/monomer.json \
    out_dir=./runs/metric-guidance-smoke \
    num_replicates=1 \
    +guidance=contract_ca
```

On an interactive DIGS GPU, use the container entry point:

```bash
./src/rfproteina/generate.py \
    inputs=data/benchmarks/monomer.json \
    out_dir=./runs/metric-guidance-smoke \
    num_replicates=1 \
    +guidance=contract_ca
```

For a submitted DIGS job:

```bash
./scripts/launch_generate.sh digs --njobs 1 \
    inputs=data/benchmarks/monomer.json \
    out_dir=/path/to/runs/metric-guidance-smoke \
    num_replicates=1 \
    +guidance=contract_ca
```

`contract_ca` is **not a useful protein-design objective**. It minimizes the summed squared
distance of Cα coordinates from the origin, so it contracts and translates a structure toward an
arbitrary coordinate-system origin. Its purpose is to prove that Hydra composition, metric
evaluation, autograd, scheduling, and sampler updates are connected.

## How it works

At each active sampling step, the guidance path:

1. takes the model's predicted clean state, currently `bb_ca` and `local_latents`;
2. evaluates the configured differentiable metric;
3. selects a scalar output by its `metric_name.output_key`;
4. differentiates that scalar with respect to each configured modality;
5. multiplies each gradient by its signed schedule weight; and
6. adds the resulting delta to the integrated sampler state.

The normal padding mask is applied to both modalities. Latent deltas are also limited to tokens
that can be encoded.

### Metric and output keys

`metrics` names and constructs the metric:

```yaml
metrics:
  contract_ca:
    _target_: rfproteina.metrics.contract_ca_metric.ContractCAMetric
```

`terms` chooses one scalar output from that metric. The key must have the form
`metric_name.output_key`; `contract_ca.sq_norm` matches the metric's `{"sq_norm": scalar}` return
value:

```yaml
terms:
  contract_ca.sq_norm:
    bb_ca:
      _target_: rfproteina.guidance.schedules.window
      _partial_: true
      w: -0.1
      t_start: 0.0
      t_end: 0.1
```

A missing output key fails when that term becomes active. Multiple terms are allowed, and their
per-modality gradient contributions are added.

### Schedules and signs

A schedule is a callable `t -> signed weight`. Flow-matching time runs from `0` (noise) to `1`
(clean output), independently for each modality.

The current schedule helpers are:

| Schedule | Meaning |
| --- | --- |
| `rfproteina.guidance.schedules.constant` | Return `w` for the whole trajectory |
| `rfproteina.guidance.schedules.window` | Return `w` only for `t_start <= t <= t_end` |

Guidance always **maximizes** the weighted objective. Use a positive `w` to increase a score and a
negative `w` to decrease a cost. There is no separate sign or magnitude field: both live in `w`.
Start with a narrow window and a small magnitude; the useful scale depends on the metric and its
reduction.

### Modality weights

Each term maps modalities to separate schedules:

```yaml
terms:
  objective.score:
    bb_ca:
      _target_: rfproteina.guidance.schedules.window
      _partial_: true
      w: 0.02
      t_start: 0.5
      t_end: 0.9
    local_latents:
      _target_: rfproteina.guidance.schedules.constant
      _partial_: true
      w: 0.005
```

Omitting a modality gives it no direct gradient update. A listed modality also has no effect if the
metric output does not depend differentiably on that state leaf.

## Build a real objective

The shipped source currently contains only one metric explicitly marked `differentiable=True`:
`ContractCAMetric`. A real objective therefore usually requires a metric class of your own.

1. Implement a `Metric` whose computation stays in PyTorch and returns a scalar tensor with its
   gradient intact. Set `differentiable = True`.
2. Map its arguments to differentiable inputs through `kwargs_to_compute_args`. Read
   `("state", "bb_ca")` or `("state", "local_latents")` for raw sampler leaves. A metric that needs
   atom37 coordinates can read the lazily provided `decoded` namespace, at extra decoder cost.
3. Copy
   [`configs/guidance/contract_ca.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/guidance/contract_ca.yaml),
   replace the metric target and `metric_name.output_key`, then choose schedules for only the
   modalities your output actually depends on.
4. Save the config as `configs/guidance/<name>.yaml` and select it with `+guidance=<name>`.
5. First run one replicate with a short active window. Compare against the same seed with no
   guidance, inspect the metric trajectory, and increase magnitude gradually.

Do not convert the objective to Python floats, NumPy arrays, external programs, or detached
tensors. Such operations break the gradient path. Ordinary reporting, refolding, and oracle metrics
are generally not differentiable guidance objectives.

## Cost and limitations

- Metric guidance keeps an autograd graph for every active step and backpropagates the metric.
  Runtime and memory depend on the objective and active window.
- A metric that requests decoded atom37 coordinates also runs the differentiable decoder on active
  steps. Metrics that call external oracles cannot be used.
- Metric guidance does not inherently require a second network evaluation per step.
- Current sampling code passes metric guidance to standard full generation only. Partial diffusion,
  inverse folding, and beam search do not pass a guidance model into their partial-simulation path.
- Guidance optimizes the metric you wrote, including its shortcuts and coordinate conventions. It
  does not guarantee foldability, novelty, or wet-lab success.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Hydra cannot find `guidance` | Add a new group with `+guidance=<name>`; do not write `guidance=<name>` when the base config has no `guidance` key |
| `uses metric ... which is not differentiable` | Set `differentiable=True` only after keeping the full computation in PyTorch with an intact gradient |
| `Guidance term ... not found in metric outputs` | Match `terms` to the exact `metric_name.output_key` returned by the metric |
| No visible change | Confirm the term's window is active, its weight is nonzero, and its output depends on the listed modality |
| Samples collapse or become non-finite | Reduce `abs(w)`, narrow the active window, and check the metric's scale and numerical stability |
| Out of memory or severe slowdown | Shorten the window, reduce `batch_size`, or avoid decoded/external-style objectives |
