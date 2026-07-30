# Environment reference

Source of truth: `setup.sh`, `pyproject.toml`, `.ipd/apptainer/`, `.ipd/docker/`.

## What `setup.sh` does

1. Installs [`uv`](https://github.com/astral-sh/uv) if missing.
2. Initializes the `lib/atomworks` submodule (`git submodule update --init lib/atomworks`).
   AtomWorks is intentionally omitted from `pyproject.toml` and installed from the
   local submodule instead.
3. Creates `.venv` with Python `>=3.12,<=3.12.10`.
4. `uv pip install -e "lib/atomworks[dev,ml,openbabel,docs]" -e ".[dev]"`.
5. Scaffolds a git-ignored custom-metrics config at
   `configs/metrics/custom_metrics/all_custom_metrics.yaml` and the package dir
   `src/rfproteina/metrics/custom_metrics/`.
6. Optionally appends `CLUSTER` / `CONTAINER_TYPE` to `~/.bashrc` or `~/.zshrc`
   and copies `scripts/.user_info_<cluster>_example` -> `scripts/.user_info_<cluster>`.

## Dependencies (from `pyproject.toml`)

- Core: `torch>=2.9`, `lightning`, `hydra-core`, `biotite==1.6.0`, `posebusters`,
  `mace-torch`, `wandb`, `typer`, `fire`, and the editable `atomworks` submodule.
- Optional extras: `tmol-cuda12` / `tmol-cuda13`, `metrics-cuda12` /
  `metrics-cuda13`, `dev`, `docs`.
- Build backend: `hatchling`. Lock file: `uv.lock`.

## Containers

Image definitions ship in the repo; on configured clusters prebuilt SIFs are symlinked
in place.

- **Apptainer** — `.ipd/apptainer/latent-dev.def` / `latent-dev.sif`.
- **Docker** — `.ipd/docker/latent-dev.Dockerfile` (NGC PyTorch base, CUDA 12/13 stacks).

Run inside apptainer explicitly:

```bash
apptainer exec --nv --bind "$PWD" .ipd/apptainer/latent-dev.sif \
    python src/rfproteina/generate.py inputs=data/benchmarks/monomer.json \
    num_replicates=1 out_dir=./runs/smoke
```

Or rely on the entry-point shebang (`./src/rfproteina/generate.py ...`) which
re-execs inside the configured container on clusters where `CONTAINER_TYPE` is set.

## Known gotcha — `LD_PRELOAD`

uv's Python may segfault in rdkit/posebusters unless `LD_PRELOAD` points at the
bundled `libstdc++.so.6`. The `Makefile` sets this automatically for `make test`;
mirror it if you hit a segfault outside `make`.

## PyMOL (optional, for the annotated-CIF plugin)

```bash
make setup-pymol      # builds a dedicated conda env (open-source PyMOL isn't pip-installable)
```

See `docs/guides/annotated_cif_inputs.md` (Option A) for the plugin workflow.
