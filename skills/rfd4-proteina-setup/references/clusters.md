# Clusters and SLURM reference

RFProteína reads a **per-cluster env file** at startup: `.ipd/.env.$CLUSTER`.
Set `CLUSTER` (and, for containerized runs, `CONTAINER_TYPE`) in your shell rc —
`setup.sh` offers to do this.

| Cluster | `CONTAINER_TYPE` | Notes |
| --- | --- | --- |
| `digs` | `apptainer` | IPD / DIGS — shared data + checkpoints + SIF prewired in `.ipd/.env.digs` |
| `ord` | `docker` | OCI / ORD |
| `gb200` | `docker` | GB200 |
| `coreweave` | `apptainer` | CoreWeave |
| `local` | (none) | Any single host; create `.ipd/.env.local` with `DATA_PATH=...` |

## Env files

Shipped files: `.ipd/.env.digs`, `.ipd/.env.ord`, `.ipd/.env.gb200`,
`.ipd/.env.coreweave` (each provides at least `DATA_PATH` and any
container/oracle paths, e.g. AF3 SIF, `FOLDSEEK_EXEC`). For a non-cluster host
you create `.ipd/.env.local` yourself (`printf 'DATA_PATH=%s\n' "$DATA_PATH" >
.ipd/.env.local`).

## SLURM launchers (`scripts/`)

| Script | Purpose |
| --- | --- |
| `scripts/launch_generate.sh` | Submit `generate.py` (+ auto metric aggregation as a dependent job) |
| `scripts/launch_generate.sh` with a search `inference_experiment` | Preferred reward-guided search submission |
| `scripts/launch_search.sh` | Specialized search launcher requiring `configs/search.yaml`; inspect its narrower SBATCH support before use |
| `scripts/launch_evaluate.sh` | Standalone post-hoc evaluation |
| `scripts/launch_train_latents.sh` / `launch_train_ae.sh` | Training |
| `scripts/launch_sweep.sh` | Hyperparameter sweeps |

Example:

```bash
./scripts/launch_generate.sh digs --njobs 8 \
    inputs=data/benchmarks/monomer \
    run_name_=my_run \
    out_dir=/path/to/shared/storage
```

Each cluster needs `scripts/.user_info_<cluster>` (copied from
`scripts/.user_info_<cluster>_example` by `setup.sh`). Treat its scheduler values
as user intent. Preserve `PARTITION`, `CPU_PARTITION`, `ACCOUNT`, `GRES`,
`RUNTIME`, `MEM`, `CPUS_PER_TASK`, and `SLURM_EXTRA_OPTIONS`. Explicit launcher
flags override the profile only when the user supplied them. Before submission,
state the resolved scheduler values and inspect generated SBATCH directives when
available.

## Containers and the AF3 oracle

Plain generation needs no container. **Refolding metrics** (`*_refolding` tags)
and **reward-guided search** use the AF3 oracle, which runs only inside
apptainer/docker configured via the cluster env files. On a cluster with a
container configured, run the entry point directly so its shebang re-execs
inside the container:

```bash
./src/rfproteina/generate.py inputs=... out_dir=...     # runs in the container
python src/rfproteina/generate.py inputs=... out_dir=... # runs in current env
```

## OpenFold3: a free, ungated alternative to AF3

AlphaFold3's own weights are gated by Google DeepMind's license. **OpenFold3** runs through the
exact same AF3 runtime and container on Apache-licensed weights — no license request needed —
so it's the default refolding oracle on non-configured clusters: `scripts/download_example_data.sh`
fetches it automatically (skip with `--skip-of3` if you'd rather configure AlphaFold3's licensed
weights instead). configured clusters already have AlphaFold3 pre-configured and skip this.

To fetch it manually (e.g. to redo it, or on a cluster the script doesn't auto-detect as
non-IPD), the public checkpoint bucket hosts a prebuilt container image and pre-converted
weights, so no building or conversion step is required:

```bash
BASE=https://YOUR_DATASET_STORE
mkdir -p "$DATA_PATH/af3/apptainer_images" "$DATA_PATH/af3/af3_weights/of3" "$DATA_PATH/af3/jax_cache"
curl -fL -o "$DATA_PATH/af3/apptainer_images/af3-open-36fad16.sif" \
    "$BASE/af3/apptainer_images/af3-open-36fad16.sif"
curl -fL -o "$DATA_PATH/af3/af3_weights/of3/of3_ported_weights.bin.zst" \
    "$BASE/af3/af3_weights/of3/of3_ported_weights.bin.zst"
curl -fL -o "$DATA_PATH/af3/af3_weights/of3/of3_ported_weights.provenance.json" \
    "$BASE/af3/af3_weights/of3/of3_ported_weights.provenance.json"

export AF3_IMAGE="$DATA_PATH/af3/apptainer_images/af3-open-36fad16.sif"
export AF3_MODEL_DIR="$DATA_PATH/af3/af3_weights/of3"
export AF3_JAX_CACHE_DIR="$DATA_PATH/af3/jax_cache"
```

Then pass `metrics/refolding_oracles=of3` as a Hydra override (instead of leaving the AF3
default) — the same `*_refolding` metric tags work unchanged; only the artifact namespace
changes (`of3`/`of3_ppi`/`of3_dna` instead of `af3`/`af3_ppi`/`af3_dna`). Requires `apptainer`
to be installed (`apt`/release `.deb` from the [apptainer releases
page](https://github.com/apptainer/apptainer/releases) — not preinstalled on most cloud GPU
images). See `docs/workflow/metrics.md` for the full oracle reference.
