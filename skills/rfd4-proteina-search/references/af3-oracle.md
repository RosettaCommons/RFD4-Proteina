# AF3 oracle runtime

## Current interface

The AF3 oracle refolds a design and reports confidence metrics through the metric
keys configured in `configs/metrics/refolding_oracles/af3.yaml`. Common reward
keys include `oracle.plddt_coi`, `oracle.iptm_coi`, `oracle.ipae_coi`, and
`oracle.min_ipae_coi`.

It is used when generation, evaluation, or search activates a `*_refolding` tag.
Plain generation and non-refolding rewards such as `interchain_interactions` do
not use AF3.

## Runtime configuration

Configure the selected oracle's image, weights, and cache paths in `.env.local`,
as described in the [installation guide](../../../docs/installation.rst).
Verify that the required paths are readable in the execution environment.

Run generation in the Python environment activated during setup:

```bash
rfd4-generate inference_experiment=inference_beam_example
```

The model runs in that Python environment; refolding uses the separately configured
oracle runtime and host launcher.

## Search and evaluation behavior

Search passes `search.rewards.reward_metrics_tags` to the metric managers at
each scoring checkpoint. The top-level `metrics_tags` controls final reporting.
When both activate the same oracle result, the final metrics path can reuse the
search result cache.

If expected oracle columns are absent, verify the active metric tags and runtime
configuration before interpreting the output. A non-refolding tag intentionally
produces no oracle columns.

Source of truth: `configs/metrics/refolding_oracles/af3.yaml`,
`src/rfproteina/metrics/oracles/af3_oracle.py`, `.env.local`, and
`docs/workflow/metrics.md`.
