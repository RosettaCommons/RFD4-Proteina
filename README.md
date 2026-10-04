<p align="center">
  <img src="assets/wordmark.svg" width="587" alt="RFD4-Proteína">
</p>

<p align="center"><em>A flow-matching foundation model for biomolecular design — Baker Lab × NVIDIA GenAIR</em></p>

<p align="center">
  <a href="https://codecov.io/gh/RosettaCommons/RFD4-Proteina/tree/main"><img src="https://codecov.io/gh/RosettaCommons/RFD4-Proteina/branch/main/graph/badge.svg" alt="Codecov coverage"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.12.0%E2%80%933.12.10-blue?logo=python&amp;logoColor=white" alt="Python 3.12.0–3.12.10"></a>
  <a href="docs/index.rst"><img src="https://img.shields.io/badge/docs-read-blue" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="BSD 3-Clause License"></a>
</p>

RFD4-Proteína generates proteins — and RNA, DNA, and small-molecule complexes — under rich conditioning.
Developed by the **Baker Lab × NVIDIA GenAIR**, it supports sequence–structure co-design,
binding-site and motif conditioning, partial diffusion, and reward-guided search.

<picture>
  <source media="(max-width: 600px)" srcset="assets/readme/design-gallery-mobile.png">
  <img src="assets/readme/design-gallery.png" width="1050" alt="Six molecular examples: nanobody complex, protein–DNA binder, RNA pseudoknot, ligand pocket, glycan complex, and protein assembly.">
</picture>

Native model outputs rendered in Blender: DNA-conditioned protein generation and topology-conditioned
RNA generation, alongside sequence- and backbone-conditioned reconstructions of the other four systems.
[Render provenance](assets/readme/provenance.json).

<details>
<summary>Watch a DNA binder emerge · native denoising trajectory</summary>

<p align="center">
  <a href="assets/readme/dna-binder-generation.mp4"><img src="assets/readme/dna-binder-generation.gif" width="360" alt="Native 60-state generation of a protein binder conditioned on a DNA duplex, followed by reverse playback."></a>
</p>

[Watch or download the full-resolution MP4](assets/readme/dna-binder-generation.mp4).
The protein sequence and coordinates are generated freely, conditioned on the DNA sequence and reference
coordinates. The 18-second loop plays generation forward, holds the endpoint, then reverses for playback;
the reverse segment is not a second model run.

</details>

Start with the **[documentation](docs/index.rst)** and [worked examples](docs/examples.rst).

- 🎯 **Run designs** → [inference quickstart](#inference-quickstart), then the full [workflow guide](docs/workflow.rst).
- 🛠️ **Train or develop** → the [Training & Development guide](docs/development.md).

---

## Setup

Start with the **[quickstart](docs/quickstart.rst)** to install and generate your first design.
See the [installation guide](docs/installation.rst#slurm-clusters) for SLURM setup.
The checkout installs into `.venv` and runs locally, without containers.
You need Linux x86_64, an NVIDIA GPU with a CUDA 13-compatible driver, `git`, and `curl`.
Setup installs Python 3.12 and the matching dependencies.

```bash
git clone https://github.com/RosettaCommons/RFD4-Proteina.git
cd RFD4-Proteina
make setup                         # installs .venv and downloads released weights (~8 GB)
source .venv/bin/activate           # exposes rfd4-generate and the other commands
```

Setup installs dependencies, including the AtomWorks 3.0 release branch, from [pyproject.toml](pyproject.toml).
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

To install into an existing Python 3.12.0–3.12.10 virtual environment, run `python -m pip install -e .`
from the clone. AtomWorks is installed automatically. Use `rfd4-generate`,
`rfd4-evaluate`, or the training commands from that environment.
For the CUDA 13 environment, including optional metrics and TMol, use
`make setup` or the [equivalent pip command](docs/installation.rst#native-python-installation).
RFD4-Proteína is distributed through this repository and installed locally with pip.

Optional refolding oracles use separate containers; see the [installation guide](docs/installation.rst).

---

## Inference quickstart

Generate one design to confirm the setup (checkpoints load from `$DATA_PATH`, so no checkpoint args):

```bash
rfd4-generate \
    inputs=data/benchmarks/monomer \
    subset_to_examples=monomer-short num_replicates=1 out_dir=./runs/smoke
```

A CIF appears under `./runs/smoke/inference_samples/`; enabled metrics are written to
`./runs/smoke/inference_metrics/`. From here:

- **[Running the model](docs/workflow/running.md)** — design use cases (binders, enzymes, motif
  scaffolding), conditioning, partial diffusion, and relevant parameters.
- **[Annotated CIF inputs](docs/workflow/inputs.md)** — the recommended input format for motifs, hotspots, ligands, bonds, H-bonds, RASA, and PTMs. For text-based inputs, use the [condition-spec format](docs/workflow/condition-spec.md).
- **[Metrics reference](docs/workflow/metrics.md)** — available metrics, outputs, and activation tags; see [custom metrics](docs/workflow/custom-metrics.md) to add your own.

### Reward-guided search

Generate several candidates with beam search and keep the ones an AF3 reward scores highest. The
example config targets the monomer benchmark and requires the [refolding oracle setup](docs/installation.rst):

```bash
rfd4-generate inference_experiment=inference_beam_example
```

See **[Reward-Guided Search](docs/workflow/search.md)** for the strategy and reward parameters and how to build your own search config.

---

## Training & development

Continue training the released model on your own data, add datasets or design tasks, or work on the
architecture — using the installed Python environment. See the
**[Training & Development guide](docs/development.md)** for training commands, dataset
configuration, and a small example dataset. Match the model configuration to the checkpoint
when starting a fine-tuning run.

---

## Links

- [Documentation & benchmarks](docs/index.rst) · [Examples](docs/examples.rst) · [Input preparation](docs/workflow/inputs.md)
- [Inference](docs/workflow/running.md) · [Reward-guided search](docs/workflow/search.md) · [Metrics](docs/workflow/metrics.md)
- [Training](docs/development.md) · [Contributing](docs/contributor_guide.rst) · [AtomWorks](https://github.com/RosettaCommons/atomworks)

---

## License

RFD4-Proteína is licensed under the [BSD 3-Clause License](LICENSE).
Third-party code retains the license notices included with those components.
