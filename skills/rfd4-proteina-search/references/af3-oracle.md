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

AF3 requires either a native compatible installation or the runtime configured
by the selected cluster profile. Inspect `.ipd/.env.<cluster>` for the current
`AF3_IMAGE`, model directory, database directory, and cache settings. Do not
infer support from the cluster name alone; verify that each required path is set
and readable in the execution environment.

On configured Apptainer clusters, invoke the entry point directly so its shebang
selects the repository runtime:

```bash
./src/rfproteina/generate.py inference_experiment=inference_beam_example
```

Using `python src/rfproteina/generate.py ...` stays in the active Python
environment. The standard SLURM generation launcher supplies the configured
container command.

## Search and evaluation behavior

Search passes `search.rewards.reward_metrics_tags` to the metric managers at
each scoring checkpoint. The top-level `metrics_tags` controls final reporting.
When both activate the same oracle result, the final metrics path can reuse the
search result cache.

If expected oracle columns are absent, verify the active metric tags and runtime
configuration before interpreting the output. A non-refolding tag intentionally
produces no oracle columns.

Source of truth: `configs/metrics/refolding_oracles/af3.yaml`,
`src/rfproteina/metrics/oracles/af3_oracle.py`, `.ipd/.env.<cluster>`, and
`docs/workflow/metrics.md`.
