---
name: rfd4-proteina-setup
description: >
  Prepare a machine to run RFD4-Proteína (RFD4-Proteína) — the Baker Lab x NVIDIA
  flow-matching protein design model. Reach for this skill for "set up
  rfproteina", "install rfproteina", "environment for rfproteina", "download the
  rfproteina checkpoints", "DATA_PATH not found", "atomworks not found", "which
  checkpoint do I use", "run rfproteina on a cluster",
  or before any generate/evaluate/search run. It installs the native Python environment and AtomWorks, wires DATA_PATH and runtime settings, downloads the model
  weights, and preflights GPU + checkpoints so downstream runs do not fail
  mid-job. Hand off to rfd4-proteina-generate once preflight is green.
license: See the RFD4-Proteína repository (Baker Lab x NVIDIA collaboration)
---

# RFD4-Proteína Setup

Get a host ready to run RFD4-Proteína inference: environment, model weights, and a
green preflight. The two failure modes this skill exists to prevent are (1)
importing the model before its environment and AtomWorks are installed, and (2)
launching an hours-long run only to have it die because `DATA_PATH` or a
checkpoint is missing. Preflight both up front.

Run everything from the repository root (the directory containing `.project-root`).

## Step 1 — Install the runtime

From the repository root:

```bash
make setup
source .venv/bin/activate
```

Native setup creates `.venv` with the CUDA 13 model dependencies and installs
AtomWorks from the Git revision pinned in `pyproject.toml`. It creates `.env.local`
only if neither `.env.local` nor `.env` exists, preserves existing files, and does
not edit shell startup files.

## Step 2 — Check data paths and optional refolding

Setup downloads the released checkpoint pair automatically into `DATA_PATH`,
defaulting to `./data_path`, and records the default in the active environment
file when no path is configured. Existing files are reused. Exported shell
settings take precedence over `.env.local` (or `.env` when `.env.local` is absent).
Set a custom data path before setup to download there. For an environment-only
installation, use `bash setup.sh --skip-weights`.

`configs/inference.yaml` defines the active checkpoint pair. Downloads use
`https://YOUR_DATASET_STORE` without credentials.
To add OpenFold3 on a host with Apptainer:

```bash
./scripts/download_example_data.sh ./data_path
```

Pass your configured `DATA_PATH` instead if it differs from `./data_path`.
Set the printed oracle paths in `.env.local` and use `metrics/refolding_oracles=of3`.
See `docs/installation.rst` for the separate host oracle.

## Step 3 — Preflight

```bash
bash skills/rfd4-proteina-setup/scripts/preflight.sh
```

This uses the same runtime dispatcher as model entry points and checks imports,
CUDA visibility, and released checkpoints. Resolve failures before sampling.

## Step 4 — Smoke test

```bash
rfd4-generate inputs=data/benchmarks/monomer \
    subset_to_examples=monomer-short num_replicates=1 out_dir=./runs/smoke
```

A generated CIF under `runs/smoke/inference_samples/` confirms the inference path.
Hand off to **rfd4-proteina-generate** for real designs.

## Optional host tools

Setup always creates `.venv` for SLURM configuration resolution.
`bash setup.sh --venv` adds notebooks and development dependencies. Use the installed
`rfd4-*` commands from the activated environment.

## Troubleshooting

- Import failures: rerun setup or the pip installation in your selected environment.
- Missing data/checkpoints: check `DATA_PATH` and repeat the download for that directory.
- Missing GPU: verify the NVIDIA driver and the selected runtime's GPU support.
- Refolding failures: check the separate oracle image, weights, cache, and host Apptainer.
- SLURM host tooling missing: rerun setup, then configure `scripts/.user_info_local`.

For checkpoint details, see [checkpoints](references/checkpoints.md).
