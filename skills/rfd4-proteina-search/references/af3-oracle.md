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

## Rebuilding the AF3/OpenFold3 runtime

The canonical build definitions live with the AF3/OpenFold3 source in
[`kierandidi/alphafold3`](https://github.com/kierandidi/alphafold3):

- [Dockerfile on `main`](https://github.com/kierandidi/alphafold3/blob/main/docker/Dockerfile)
- [Apptainer definition on `main`](https://github.com/kierandidi/alphafold3/blob/main/apptainer/af3-open.def)

Use the validated revision for reproducible builds. `36fad16` is current; each earlier pin
fixed something the one before got wrong, so building older is a downgrade:

| revision | brings |
| --- | --- |
| `cf0fade` | correct weight conversion. Earlier revisions embed RNA residues as their neighbour and transpose the template aatype term; `setup_of3_weights.py` refuses them. |
| `4b38b7b` | addresses a bond to the real atom of an atomized residue, not its `CA`. |
| `36fad16` | keeps covalent bonds between atomized residues, which both cleaning and the layout split were discarding, and ungates the `CA` fix. |

```bash
git clone https://github.com/kierandidi/alphafold3.git
cd alphafold3
git checkout --detach 36fad169902c
```

Build Docker:

```bash
DOCKER_BUILDKIT=1 docker build \
  --file docker/Dockerfile \
  --tag af3-open:36fad16 \
  .
```

Build Apptainer:

```bash
apptainer build af3-open-36fad16.sif apptainer/af3-open.def
```

Both images contain JAX/CUDA inference dependencies and CPU-only PyTorch for
the one-time OpenFold3 checkpoint conversion. They contain no model parameters.
Mount Google AF3 parameters for standard inference, or converted OpenFold3 parameters:
the runtime reads the provenance file beside them and enables the OF3 branches itself.

Validate CUDA visibility and the OpenFold3 parameter reader on a GPU node:

```bash
apptainer exec --nv af3-open-36fad16.sif \
  /alphafold3_venv/bin/python -c \
  'import jax; from alphafold3.model import of3_weight_converter; print(jax.devices())'
```

For Pyxis, publish the Docker image and use it with `--container-image`, or
export it with `enroot import`. Do not directly squash an unpacked SIF: that
drops Docker environment metadata and can cause JAX to fall back to CPU.

## Which artifacts are current

`s3://YOUR_DATASET_STORE/af3/artifacts/af3-open.manifest.json` records every artifact's sha256,
and flags any built before `36fad16` with `stale: true`.

| cluster | what to do |
|---|---|
| coreweave (cw-east, cw-west) | nothing; image and OF3 weights are staged on both |
| digs | re-pull `af3/apptainer_images/af3-open.sif` if you converted weights with an older copy |
| ord, gb200 | no `AF3_*` is set in their env files, so nothing runs AF3 there today. The pyxis `.sqsh` and OCI images are still built from the pre-`cf0fade` commit, so rebuild them from `docker/Dockerfile` before enabling it |

`cf0fade` changed only the converter, so an image older than it still *infers* correctly with
correctly-converted weights -- hence that guard lives in `setup_of3_weights.py`. `4b38b7b` and
`36fad16` changed inference itself, so an image older than those silently drops bonds.

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
