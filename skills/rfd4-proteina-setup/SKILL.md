---
name: rfd4-proteina-setup
description: >
  Prepare a machine to run RFProteína (RFD4-Proteína) — the Baker Lab x NVIDIA
  flow-matching protein design model. Reach for this skill for "set up
  rfproteina", "install rfproteina", "environment for rfproteina", "download the
  rfproteina checkpoints", "DATA_PATH not found", "atomworks not found", "which
  checkpoint do I use", "run rfproteina on a cluster (DIGS/ORD/GB200/CoreWeave)",
  or before any generate/evaluate/search run. It installs the uv venv + atomworks
  submodule, wires DATA_PATH and the per-cluster env file, downloads the model
  weights, and preflights GPU + checkpoints so downstream runs do not fail
  mid-job. Hand off to rfd4-proteina-generate once preflight is green.
compatibility: "Linux x86_64/arm64 with an NVIDIA GPU (>=40GB VRAM recommended), git, curl; Python 3.12; ~10GB disk for checkpoints; optional apptainer/docker for AF3-backed metrics + search"
allowed-tools: Bash, Read, Write, AskUserQuestion
license: See the RFProteína repository (Baker Lab x NVIDIA collaboration)
---

# RFProteína Setup

Get a host ready to run RFProteína inference: environment, model weights, and a
green preflight. The two failure modes this skill exists to prevent are (1)
importing the model before the `atomworks` submodule / venv exists, and (2)
launching an hours-long run only to have it die because `DATA_PATH` or a
checkpoint is missing. Preflight both up front.

Run everything from the **repo root** (`RFProteína-dev/`).

## Step 1 — Install the environment

The supported install is `uv` + a Python 3.12 venv, driven by `setup.sh`:

```bash
bash setup.sh
source .venv/bin/activate
```

`setup.sh` (see [references/environment.md](references/environment.md)):
- installs `uv` if missing,
- initializes the `lib/atomworks` submodule (`git submodule update --init lib/atomworks`),
- creates `.venv` with Python `>=3.12,<=3.12.10`,
- `uv pip install -e "lib/atomworks[dev,ml,openbabel,docs]" -e ".[dev]"`,
- scaffolds `configs/metrics/custom_metrics/all_custom_metrics.yaml`,
- **interactively** offers to write `CLUSTER` / `CONTAINER_TYPE` to your shell rc.

On an **IPD/DIGS** cluster choose `1) digs` at the prompt: the shared data dir,
checkpoints, and apptainer SIF are already wired in `.ipd/.env.digs`, so you can
skip Step 2 entirely. On any other host, continue to Step 2.

## Step 2 — Point at a data directory and download weights

RFProteína reads checkpoints from `$DATA_PATH` and reads a **per-cluster env
file** at startup. For a local (non-IPD) machine:

```bash
export CLUSTER=local
export DATA_PATH="$PWD/data_path"
mkdir -p "$DATA_PATH"
printf 'DATA_PATH=%s\n' "$DATA_PATH" > .ipd/.env.local   # startup reads this
./scripts/download_example_data.sh "$DATA_PATH"
```

This downloads two checkpoints (~4 GB each) over public HTTPS — no credentials:
- denoiser (LDM): `latent_checkpoints/ldm_checkpoints/arch_sweep_2026_07_03_pair_trimult_every2_720k-ema.ckpt`
- autoencoder: `latent_checkpoints/ae_checkpoints/AE_ucond_512.ckpt`

> IMPORTANT checkpoint mismatch: `configs/inference.yaml` defaults `ckpt_name`
> to the **release** checkpoint (`2026_07_22_release_280k-ema.ckpt`), which is
> *not* what `download_example_data.sh` fetches. If you only downloaded the
> example data, either obtain the release checkpoint or override on the CLI:
> `ckpt_name=arch_sweep_2026_07_03_pair_trimult_every2_720k-ema.ckpt`.
> See [references/checkpoints.md](references/checkpoints.md).

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
    inputs=data/benchmarks/monomer.json \
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
| `No .env file found for cluster 'digs'` | `export CLUSTER=local` and create `.ipd/.env.local` |
| `DATA_PATH` / checkpoint not found | `export DATA_PATH=...` and `./scripts/download_example_data.sh "$DATA_PATH"` |
| checkpoint name mismatch | override `ckpt_name=` to the file you actually downloaded (see Step 2 note) |
| Segfault in rdkit/posebusters | uv's Python needs `LD_PRELOAD` set to the bundled `libstdc++.so.6` (see `Makefile`; `make test` sets it) |
| AF3 / refolding errors | refolding & search need an AF3 container — see references/clusters.md |

For deeper detail: [environment](references/environment.md) ·
[checkpoints](references/checkpoints.md) · [clusters](references/clusters.md).
