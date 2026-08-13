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
| `scripts/launch_search.sh` | Reward-guided search jobs |
| `scripts/launch_evaluate.sh` | Standalone post-hoc evaluation |
| `scripts/launch_train_latents.sh` / `launch_train_ae.sh` | Training |
| `scripts/launch_sweep.sh` | Hyperparameter sweeps |

Example:

```bash
./scripts/launch_generate.sh digs --njobs 8 \
    inputs=data/benchmarks/monomer.json \
    run_name_=my_run \
    out_dir=/path/to/shared/storage
```

Each cluster needs `scripts/.user_info_<cluster>` (copied from
`scripts/.user_info_<cluster>_example` by `setup.sh`). For most inference
workflows no edits are needed.

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
