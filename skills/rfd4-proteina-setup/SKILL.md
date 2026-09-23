---
name: rfd4-proteina-setup
description: >
  Prepare a machine to run RFD4-Proteína (RFD4-Proteína) — the Baker Lab x NVIDIA
  flow-matching protein design model. Reach for this skill for "set up
  rfproteina", "install rfproteina", "environment for rfproteina", "download the
  rfproteina checkpoints", "DATA_PATH not found", "atomworks not found", "which
  checkpoint do I use", "run rfproteina on a cluster (DIGS/ORD/GB200/CoreWeave)",
  or before any generate/evaluate/search run. It installs the uv venv + atomworks
  submodule, wires DATA_PATH and the per-cluster env file, downloads the model
  weights, and preflights GPU + checkpoints so downstream runs do not fail
  mid-job. Hand off to rfd4-proteina-generate once preflight is green.
license: See the RFD4-Proteína repository (Baker Lab x NVIDIA collaboration)
---

# RFD4-Proteína Setup

Get a host ready to run RFD4-Proteína inference: environment, model weights, and a
green preflight. The two failure modes this skill exists to prevent are (1)
importing the model before the `atomworks` submodule / venv exists, and (2)
launching an hours-long run only to have it die because `DATA_PATH` or a
checkpoint is missing. Preflight both up front.

Run everything from the repository root (the directory containing `.project-root`).

## Step 1 — Install the environment

The supported install is `uv` + a Python 3.12 venv, driven by `setup.sh`:

```bash
bash setup.sh
source .venv/bin/activate
```

`setup.sh` (see [references/environment.md](references/environment.md)):
- installs `uv` if missing,
- initializes the `lib/atomworks` submodule (`git submodule update --init lib/atomworks`) —
  this clones over SSH, so you need a GitHub SSH key registered (no key? see
  `docs/installation.rst`'s "Local installation with uv" section for a workaround),
- creates `.venv` with Python `>=3.12,<=3.12.10`,
- `uv pip install -e "lib/atomworks[dev,ml,openbabel,docs]" -e ".[dev]"`,
- scaffolds `configs/metrics/custom_metrics/all_custom_metrics.yaml`,
- **interactively** offers to write `CLUSTER` / `CONTAINER_TYPE` to your shell rc.

On an **IPD/DIGS** cluster choose `1) digs` at the prompt: the shared data dir,
checkpoints, and apptainer SIF are already wired in `.ipd/.env.digs`, so you can
skip Step 2 entirely. On any other host, continue to Step 2.

## Step 2 — Point at a data directory and download weights

RFD4-Proteína reads checkpoints from `$DATA_PATH` and reads a **per-cluster env
file** at startup. For a local (non-IPD) machine:

```bash
export CLUSTER=local
export DATA_PATH="$PWD/data_path"
mkdir -p "$DATA_PATH"
printf 'DATA_PATH=%s\n' "$DATA_PATH" > .ipd/.env.local   # startup reads this
./scripts/download_example_data.sh "$DATA_PATH"
```

This downloads the current release checkpoint pair over public HTTPS — no credentials. See
`configs/inference.yaml` for the active defaults.

The denoiser name matches `ckpt_name` in `configs/inference.yaml`, so a default
`generate.py` run resolves it with no CLI override.

On this non-IPD path it also fetches **OpenFold3** (~5 GB, free, Apache-licensed) if
`apptainer` is installed, so refolding metrics work with no further setup — pass `--skip-of3`
to configure AlphaFold3's licensed weights instead. See
[references/clusters.md](references/clusters.md).

> **Public assets live in one bucket: `YOUR_DATASET_STORE`**
> (`https://YOUR_DATASET_STORE`) — checkpoints, the
> `test_dataset` tarball, and the CCD mirror, all served anonymously. The
> internal `YOUR_DATASET_STORE` / `YOUR_DATASET_STORE` buckets require IPD credentials
> and 403 for external users. See [references/checkpoints.md](references/checkpoints.md).

## Step 3 — Preflight (do this before every fresh clone / new host)

```bash
bash skills/rfd4-proteina-setup/scripts/preflight.sh
```

It checks: venv active + Python 3.12, `atomworks` importable, GPU visible
(`nvidia-smi`), `DATA_PATH` set, the per-cluster `.ipd/.env.$CLUSTER` present,
and that the denoiser + autoencoder checkpoints exist. It prints a `PASS`/`FAIL`
report and exits non-zero on any failure. Fix every `FAIL` before generating.

## Step 4 — Smoke test (optional but recommended)

Confirm the full path works with the smallest possible run:

```bash
python src/rfproteina/generate.py \
    inputs=data/benchmarks/monomer \
    subset_to_examples=monomer-short \
    num_replicates=1 out_dir=./runs/smoke
```

A single annotated `.cif` under `./runs/smoke/` means the install is good. Then
hand off to **rfd4-proteina-generate** for real designs.

## Containers (AF3-backed metrics and search)

Plain generation needs no container. **Refolding metrics and reward-guided
search** use the AF3 oracle, which runs only inside apptainer/docker configured
via the cluster env files. If you plan to use those, verify the container now —
see [references/clusters.md](references/clusters.md). On clusters with a
container configured, the shebang form `./src/rfproteina/generate.py ...`
re-execs inside the container automatically; `python src/rfproteina/generate.py
...` runs in the current environment.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `atomworks not found` | `git submodule update --init lib/atomworks`, then re-run `setup.sh` |
| `hydra not found` / import errors | `source .venv/bin/activate` |
| `CLUSTER environment variable is not set` | `export CLUSTER=local` (or your cluster name) before importing `rfproteina` — including before running a notebook cell. |
| `No .env file found for cluster '<name>'` | `export CLUSTER=local` and create `.ipd/.env.local` |
| `DATA_PATH` / checkpoint not found | `export DATA_PATH=...` and `./scripts/download_example_data.sh "$DATA_PATH"` |
| checkpoint name mismatch | override `ckpt_name=` to the file you actually downloaded (see Step 2 note) |
| Segfault in rdkit/posebusters | uv's Python needs `LD_PRELOAD` set to the bundled `libstdc++.so.6` (see `Makefile`; `make test` sets it) |
| AF3 / refolding errors | refolding & search need an AF3 container. `download_example_data.sh` fetches the gate-free OpenFold3 alternative by default on non-configured clusters (`metrics/refolding_oracles=of3`) if `apptainer` is installed — re-run it if you skipped that or `apptainer` wasn't installed yet. See references/clusters.md. |
| `Environment variable CCD_MIRROR_PATH/PDB_MIRROR_PATH not set` (printed, not raised) | Harmless unless you need CCD/PDB lookups. Plain generation and the tutorial notebooks don't need either. |

For deeper detail: [environment](references/environment.md) ·
[checkpoints](references/checkpoints.md) · [clusters](references/clusters.md).
