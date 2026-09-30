<p align="center">
  <img src="assets/wordmark.svg" width="587" alt="RFD4-Proteína">
</p>

<p align="center"><em>A flow-matching foundation model for protein design — Baker Lab × NVIDIA GenAI</em></p>

<p align="center">
  <a href="https://pypi.org/project/rfproteina/"><img src="https://img.shields.io/pypi/v/rfproteina.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/rfproteina/"><img src="https://img.shields.io/pypi/pyversions/rfproteina.svg" alt="Python versions"></a>
  <a href="https://pypi.org/project/rfproteina/"><img src="https://img.shields.io/pypi/dm/rfproteina.svg" alt="PyPI downloads"></a>
  <a href="https://github.com/RosettaCommons/RFD4-Proteina/actions/workflows/manual_tests.yaml"><img src="https://github.com/RosettaCommons/RFD4-Proteina/actions/workflows/manual_tests.yaml/badge.svg" alt="CI status"></a>
  <a href="https://rosettacommons.github.io/RFD4-Proteina/"><img src="https://github.com/RosettaCommons/RFD4-Proteina/actions/workflows/docs.yaml/badge.svg" alt="Documentation"></a>
  <a href="https://codecov.io/gh/RosettaCommons/RFD4-Proteina"><img src="https://codecov.io/gh/RosettaCommons/RFD4-Proteina/graph/badge.svg?token=21O2N8apcG" alt="Code coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="BSD 3-Clause License"></a>
</p>

<p align="center">
  <img src="assets/hero.gif" width="300" alt="A DISCO-designed BODIPY fluorophore binder, spinning">
</p>

RFD4-Proteína generates proteins — and RNA, DNA, and small-molecule complexes — under rich conditioning.

Check out the **[RFD4-Proteína website](https://rosettacommons.github.io/RFD4-Proteina/)** for complete benchmarks and documentation.

Two ways in:

- 🎯 **Run designs** → [inference quickstart](#inference-quickstart) below, then the full **[Inference guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/inference.html)**.
- 🛠️ **Train or develop** → the **[Training & Development guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/development.html)**.

---

## Setup

You need Linux, an NVIDIA GPU with a CUDA-13-compatible driver, Git, curl, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/RosettaCommons/RFD4-Proteina.git
cd RFD4-Proteina
bash setup.sh
source .venv/bin/activate
export DATA_PATH="$PWD/data_path"
bash scripts/download_example_data.sh "$DATA_PATH" --skip-of3
```

Setup installs Python 3.12 and AtomWorks directly from the public
`release/atomworks-3-0` branch revision pinned in `pyproject.toml`.
No submodule, GitHub credentials, Rosetta, or private cluster profile is required.
The default release uses the 340k EMA flow checkpoint and the z16 autoencoder.

Keep `DATA_PATH` exported in your shell, or save it in an optional project-root
`.env` file. Do not set `CLUSTER` unless you have created a matching runtime profile.

Checkpoints are downloaded anonymously from the public release bucket.
Plain generation needs no oracle or container. For optional refolding, see the
[metrics documentation](docs/workflow/metrics.md).

## Running in a container

Run the same installation commands in a GPU-enabled container with Python 3.12
and Git. Mount the repository, weights, and output directory, and invoke
`python src/rfproteina/generate.py` inside it. Private cluster images and runtime
profiles are not included in this release.

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
- **[Annotated CIF inputs](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/annotated_cif_inputs.html)** — the recommended input format for motifs, hotspots, ligands, bonds, H-bonds, RASA, and PTMs. For text-based inputs, use the [condition-spec format](https://rosettacommons.github.io/RFD4-Proteina/latest/condition_spec_yaml.html).
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

- 🌐 **[Benchmarks &amp; gallery](https://rosettacommons.github.io/RFD4-Proteina/)** — how RFD4-Proteína compares (design quality, speed, compute).
- 🎯 **[Inference guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/inference.html)** · 🔎 **[Reward-guided search](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/search.html)** · 🛠️ **[Training guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/development.html)** · 📄 **[Annotated CIF inputs](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/annotated_cif_inputs.html)** · 📊 **[Metrics reference](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/metrics.html)** · 🤝 **[Contributing](https://rosettacommons.github.io/RFD4-Proteina/latest/contributor_guide.html)**

---

## License

RFD4-Proteína is licensed under the [BSD 3-Clause License](LICENSE).
Third-party code retains the license notices included with those components.
