<p align="center">
  <img src="assets/wordmark.svg" width="587" alt="RFD4-Proteína">
</p>

<p align="center"><em>A flow-matching foundation model for protein design — Baker Lab × NVIDIA GenAI</em></p>

<p align="center">
  <a href="https://pypi.org/project/rfproteina/"><img src="https://img.shields.io/pypi/v/rfproteina.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/rfproteina/"><img src="https://img.shields.io/pypi/pyversions/rfproteina.svg" alt="Python versions"></a>
  <a href="https://pypi.org/project/rfproteina/"><img src="https://img.shields.io/pypi/dm/rfproteina.svg" alt="PyPI downloads"></a>
  <a href="https://github.com/RosettaCommons/RFD4-Proteina/actions/workflows/lint_and_test.yaml"><img src="https://github.com/RosettaCommons/RFD4-Proteina/actions/workflows/lint_and_test.yaml/badge.svg" alt="CI status"></a>
  <a href="https://rosettacommons.github.io/RFD4-Proteina/"><img src="https://github.com/RosettaCommons/RFD4-Proteina/actions/workflows/docs.yaml/badge.svg" alt="Documentation"></a>
  <a href="https://codecov.io/gh/RosettaCommons/RFD4-Proteina"><img src="https://codecov.io/gh/RosettaCommons/RFD4-Proteina/graph/badge.svg?token=21O2N8apcG" alt="Code coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache License 2.0"></a>
</p>

<p align="center">
  <img src="assets/hero.gif" width="300" alt="A DISCO-designed BODIPY fluorophore binder, spinning">
</p>

RFProteína generates proteins — and RNA, DNA, and small-molecule complexes — under rich conditioning.

Check out the **[RFProteína website](https://rosettacommons.github.io/RFD4-Proteina/)** for complete benchmarks and documentation.

Two ways in:

- 🎯 **Run designs** → [inference quickstart](#inference-quickstart) below, then the full **[Inference guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/inference.html)**.
- 🛠️ **Train or develop** → the **[Training & Development guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/development.html)**.

---

## Setup

> 🖥️ **On a configured compute cluster?** Just run `bash setup.sh` below and choose `1) digs` at the
> cluster prompt — the shared data dir, checkpoints, and apptainer SIF are already wired in
> `.ipd/.env.digs`, so you can skip the data-configuration step entirely and run inference directly.
> Call the entry scripts **directly** rather than prefixing them with `python`, so the shebang runs
> them inside the container — no virtualenv needed:
>
> ```bash
> ./src/rfproteina/generate.py inputs=data/benchmarks/monomer out_dir=./runs/smoke
> ```
>
> Do this on an interactive GPU; the SLURM launchers (e.g. `scripts/launch_generate.sh`) enter the
> container for you.

These instructions use [`uv`](https://github.com/astral-sh/uv). Prefer apptainer or docker? See
[Running in a container](#running-in-a-container).

You need a Linux machine with an NVIDIA GPU, `git`, and `curl`.

```bash
bash setup.sh                 # installs uv, the atomworks submodule, a py3.12 venv, and deps
source .venv/bin/activate
```

Point RFProteína at a local data directory and create a tiny env file for it:

```bash
export CLUSTER=local
export DATA_PATH="$PWD/data_path"
mkdir -p "$DATA_PATH"
printf 'DATA_PATH=%s\n' "$DATA_PATH" > .ipd/.env.local   # startup reads a per-cluster env file
```

Download the model (two checkpoints, ~4 GB each) over public HTTPS — no credentials:

```bash
./scripts/download_example_data.sh "$DATA_PATH"
```

> Everything published for external use — both checkpoints, the tutorial `test_dataset`, and the
> CCD mirror — lives in the public bucket **`YOUR_DATASET_STORE`**, served anonymously at
> `https://YOUR_DATASET_STORE` (override with `RFPROTEINA_DATA_URL`). It is the
> only bucket reachable without private credentials.

---

## Running in a container

Prefer apptainer or docker over `uv`? Image definitions ship in the repo (and on configured clusters the
prebuilt SIFs are symlinked in place):

- **Apptainer** — `.ipd/apptainer/` holds the definitions and SIFs (`latent-dev.def` /
  `latent-dev.sif`). Run any command inside the image with a repo bind mount and GPU:
  ```bash
  apptainer exec --nv --bind "$PWD" "$LATENT_APPTAINER_SIF" \
      python src/rfproteina/generate.py inputs=data/benchmarks/monomer out_dir=./runs/smoke
  ```
  Where no prebuilt SIF is staged, build one: `apptainer build latent-dev.sif .ipd/apptainer/latent-dev.def`.
  `LATENT_APPTAINER_SIF` comes from `.ipd/.env.<cluster>`, so it follows image upgrades;
  `.ipd/apptainer/latent-dev.sif` symlinks to the same image.
- **Docker** — Dockerfiles are in `.ipd/docker/` (e.g. `latent-dev.Dockerfile`).
- **Auto** — the entry scripts re-exec themselves inside the SIF when `CONTAINER_TYPE=apptainer`, so
  running `./src/rfproteina/generate.py …` directly uses the container, while
  `python src/rfproteina/generate.py …` uses your local environment.

You still set `DATA_PATH` and download the model as in [Setup](#setup) — only the Python environment differs.

---

## Inference quickstart

Generate one design to confirm the setup (checkpoints load from `$DATA_PATH`, so no checkpoint args):

```bash
python src/rfproteina/generate.py \
    inputs=data/benchmarks/monomer \
    num_replicates=1 out_dir=./runs/smoke
```

A CIF appears under `./runs/smoke/`. From here:

- **[Full inference guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/inference.html)** — design use cases (binders, enzymes, motif
  scaffolding), conditioning, partial diffusion, and relevant parameters.
- **[Annotated CIF inputs](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/annotated_cif_inputs.html)** — the recommended input format (motifs, hotspots, ligands, bonds, H-bonds, RASA, PTMs, read natively; legacy JSON: [input spec](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/input.html)).
- **[Metrics reference](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/metrics.html)** — every metric that ships enabled, what it outputs, which tag turns it on, and how to add your own.

### Reward-guided search

Generate several candidates with beam search and keep the ones an AF3 reward scores highest. The
example config runs as-is on the monomer benchmark:

```bash
python src/rfproteina/generate.py inference_experiment=inference_beam_example
```

See **[Reward-Guided Search](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/search.html)** for the strategy and reward parameters and how to build your own search config.

---

## Training & development

Continue training the released model on your own data, add datasets or design tasks, or work on the
architecture — all in the same `uv` environment. See the
**[Training & Development guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/development.html)**, which includes a fresh-cluster,
no-container walkthrough that finetunes the released checkpoint on a tiny example dataset.

---

## Links

- 🌐 **[Benchmarks &amp; gallery](https://rosettacommons.github.io/RFD4-Proteina/)** — how RFProteína compares (design quality, speed, compute).
- 🎯 **[Inference guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/inference.html)** · 🔎 **[Reward-guided search](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/search.html)** · 🛠️ **[Training guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/development.html)** · 📄 **[Annotated CIF inputs](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/annotated_cif_inputs.html)** · 📊 **[Metrics reference](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/metrics.html)** · 🤝 **[Contributing](https://rosettacommons.github.io/RFD4-Proteina/latest/contributor_guide.html)**

---

## License

RFProteína is licensed under the [Apache License 2.0](LICENSE).
