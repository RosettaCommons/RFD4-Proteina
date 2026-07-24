# The AF3 oracle

## What it is

The **AF3 oracle** is RFProteína's refolding-confidence scorer. Given a design,
it refolds the structure with AlphaFold3 and reports confidence metrics —
per-chain pLDDT, interface pTM, and interface PAE — prefixed `oracle.` in the
metrics output. It is defined in
`src/rfproteina/metrics/oracles/oracle_metric.py`; the keys it produces include
`oracle.plddt_coi`, `oracle.iptm_coi`, `oracle.ipae_coi`, and
`oracle.min_ipae_coi` ("coi" = chain of interest).

## Where it is used

The oracle backs two things, and **both need the container**:

1. **Refolding metrics** — the `*_refolding` metric tags
   (`unconditional_refolding`, `ppi_refolding`, `sm_binder_refolding`,
   `motif_scaffold_refolding`, `ame_refolding`) during evaluation.
2. **Reward-guided search** — when `search.rewards.reward_metrics_tags` includes
   a `*_refolding` tag, the oracle scores every rolled-out candidate at each
   step checkpoint. This is what makes beam search costly.

Plain generation and non-refolding rewards (e.g. `interchain_interactions`) do
**not** use the oracle and need no container.

## It runs only inside a container

The AF3 oracle runs **only inside an apptainer/docker container** (or a native
AF3 install), configured via the per-cluster env file `.ipd/.env.$CLUSTER`. That
file provides the AF3 SIF path and related oracle paths. This is set up by
**rfd4-proteina-setup** — see
[rfd4-proteina-setup/references/clusters.md](../../rfd4-proteina-setup/references/clusters.md)
for cluster/container details and the `CONTAINER_TYPE` per cluster
(`apptainer` on `digs`/`coreweave`, `docker` on `ord`/`gb200`).

## Running in-container (shebang form)

On a cluster with a container configured, run the entry point **directly** so its
shebang re-execs inside the container:

```bash
./src/rfproteina/generate.py inference_experiment=inference_beam_example   # re-execs in the container
python src/rfproteina/generate.py inference_experiment=inference_beam_example  # runs in the current env only
```

Use `python src/rfproteina/generate.py ...` only when you are already inside the
container (or have a native AF3 install). The SLURM launcher
`scripts/launch_search.sh` wraps the command in the container automatically.

## Caching

Oracle results are cached to disk during beam scoring. The epoch-end metrics
loop (driven by the top-level `metrics_tags`) then reuses those cached results
instead of recomputing, so populating `per_example_metrics.csv` after a search is
fast. This is why the example configs list `unconditional_refolding` in both
`search.rewards.reward_metrics_tags` and the outer `metrics_tags`.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `AF3 / refolding errors`, oracle/SIF not found | Container not configured; check `.ipd/.env.$CLUSTER` has the AF3 SIF path, and use the shebang form. See rfd4-proteina-setup [references/clusters.md](../../rfd4-proteina-setup/references/clusters.md) |
| Oracle keys missing from metrics | The active tag is not a `*_refolding` tag, so the oracle never ran |
| Works with `./generate.py` but not `python generate.py` | `python ...` runs in the current env, not the container — use the shebang form on clusters |
