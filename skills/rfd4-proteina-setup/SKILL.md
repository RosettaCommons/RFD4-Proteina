---
name: rfd4-proteina-setup
description: Install the public RFD4-Proteina release, download checkpoints, and preflight Python, AtomWorks, and GPU availability before generation or evaluation.
license: See the repository LICENSE.
---

# RFD4-Proteína setup

Run from the repository root. Follow [installation](../../../docs/installation.rst)
for prerequisites and [environment](references/environment.md) for dependency details.

## Install and download

```bash
bash setup.sh
source .venv/bin/activate
export DATA_PATH="$PWD/data_path"
bash scripts/download_example_data.sh "$DATA_PATH" --skip-of3
```

Install uv first if absent. Setup uses Python 3.12 and the public AtomWorks
Git revision pinned in pyproject.toml; do not initialize a submodule or install
a private AtomWorks checkout. No Rosetta dependency is required.

Leave CLUSTER unset for local use. The optional root .env file may provide
DATA_PATH. An explicitly selected CLUSTER requires a user-created runtime
profile; do not assume private shared storage or container images exist.

## Preflight and smoke test

```bash
bash skills/rfd4-proteina-setup/scripts/preflight.sh
python src/rfproteina/generate.py inputs=data/benchmarks/monomer subset_to_examples=monomer-short num_replicates=1 out_dir=./runs/smoke
```

Resolve failed checks before generating. Use rfd4-proteina-generate for
task-specific settings. Read [checkpoints](references/checkpoints.md) for
checkpoint selection. Plain generation needs no oracle or container; optional
refolding requirements are covered by [metrics](../../../docs/workflow/metrics.md).
