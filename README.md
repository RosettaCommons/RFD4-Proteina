<p align="center">
  <img src="assets/wordmark.svg" width="587" alt="RFD4-Proteína">
</p>

<p align="center"><em>A flow-matching foundation model for protein design — Baker Lab × NVIDIA GenAI</em></p>

<p align="center">
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

Start with the **[quickstart](docs/quickstart.rst)** to install and generate your first design.
See the [installation guide](docs/installation.rst#slurm-clusters) for SLURM setup.
The checkout installs into `.venv` and runs locally, without containers.
You need Linux, an NVIDIA GPU with a compatible driver, `git`, and `curl`.

```bash
git clone https://github.com/RosettaCommons/RFD4-Proteina.git
cd RFD4-Proteina
make setup                         # installs .venv and downloads released weights (~8 GB)
source .venv/bin/activate           # exposes rfd4-generate and the other commands
```

Setup installs dependencies, including AtomWorks, from `pyproject.toml`.
It creates `.env.local` from [`.env.local.example`](.env.local.example)
only if neither `.env.local` nor `.env` exists, and preserves existing settings.
Use `.env.local` for new configurations. An existing `.env` remains supported when
`.env.local` is absent; the files are not merged. Edit whichever file is active.
Exported shell variables take precedence.

Setup also downloads the two released checkpoints (about 8 GB) into `DATA_PATH`,
defaulting to `./data_path`. Existing checkpoints are reused. Set `DATA_PATH` in
your environment or configuration before setup to use another directory.
For an environment-only install, use `bash setup.sh --skip-weights`.

## Local pip installation

To install into an existing Python 3.12 virtual environment, run `python -m pip install -e .`
from the clone. AtomWorks is installed automatically. Use `rfd4-generate`,
`rfd4-evaluate`, or the training commands from that environment.
For the CUDA 13 environment, including optional metrics and TMol, use
`make setup` or the [equivalent pip command](docs/installation.rst#native-python-installation).
RFD4-Proteina is installed from this repository; no PyPI release is required or published by setup.

Optional refolding oracles use separate containers; see the [installation guide](docs/installation.rst).
Native installation and CLI configuration have been checked; GPU numerical parity remains to be validated.

---

## Inference quickstart

Generate one design to confirm the setup (checkpoints load from `$DATA_PATH`, so no checkpoint args):

```bash
rfd4-generate \
    inputs=data/benchmarks/monomer \
    subset_to_examples=monomer-short num_replicates=1 out_dir=./runs/smoke
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
rfd4-generate inference_experiment=inference_beam_example
```

See **[Reward-Guided Search](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/search.html)** for the strategy and reward parameters and how to build your own search config.

---

## Training & development

Continue training the released model on your own data, add datasets or design tasks, or work on the
architecture — using the installed Python environment. See the
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
