<p align="center">
  <img src="assets/wordmark.svg" width="587" alt="RFD4-Proteína">
</p>

<p align="center"><em>A flow-matching foundation model for biomolecular design — Baker Lab × NVIDIA GenAIR</em></p>

<p align="center">
  <a href="https://codecov.io/gh/RosettaCommons/RFD4-Proteina/tree/main"><img src="https://codecov.io/gh/RosettaCommons/RFD4-Proteina/branch/main/graph/badge.svg" alt="Codecov coverage"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.12-blue?logo=python&amp;logoColor=white" alt="Python 3.12"></a>
  <a href="https://rosettacommons.github.io/RFD4-Proteina/"><img src="https://img.shields.io/badge/docs-read-blue" alt="Documentation"></a>
  <a href="https://huggingface.co/nvidia/RFD4-Proteina"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model-yellow" alt="Hugging Face model"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="BSD 3-Clause License"></a>
</p>

<p align="center">
  <a href="#system-requirements">Requirements</a> ·
  <a href="#install">Installation</a> ·
  <a href="#run-the-model">Quick start</a> ·
  <a href="#more-workflows">Workflows</a> ·
  <a href="#tutorials">Tutorials</a> ·
  <a href="https://rosettacommons.github.io/RFD4-Proteina/">Documentation</a>
</p>

<picture>
  <source media="(max-width: 600px)" srcset="assets/readme/design-gallery-mobile.png">
  <img src="assets/readme/design-gallery.png" width="1050" alt="Nanobody complex, protein–DNA binder, RNA pseudoknot, ligand pocket, glycan complex, and protein assembly.">
</picture>

RFD4-Proteína designs proteins and biomolecular complexes with sequence–structure co-design,
motif and binding-site conditioning, partial diffusion, and reward-guided search.

Key features:

- **Many design tasks from one model:** monomers, protein and DNA binders, antibodies and nanobodies, enzymes and motif scaffolding, small-molecule pockets.
- **Rich conditioning:** annotated CIF inputs or text-based YAML specifications for fixed motifs, hotspots, burial, hydrogen bonds, and more.
- **Reward-guided search:** beam search over generation paths scored by structure-prediction oracles and custom metrics.
- **Partial diffusion** to refine or diversify existing structures.
- **Fast inference:** `torch.compile` by default, with an optional BioIR backend (`inference_backend=bioir`).
- **Hugging Face release and Python API** alongside the `rfd4-generate`, `rfd4-evaluate` and training CLIs.

## System requirements

- Linux x86_64, Python 3.12, and an NVIDIA GPU with a CUDA 13-compatible driver (CUDA 12 and CPU-only containers are also available).
- Tested GPUs: see the [supported hardware table](https://rosettacommons.github.io/RFD4-Proteina/installation.html#supported-hardware). Check yours with `nvidia-smi`.
- macOS and Windows are not supported.

## Install

**Start with `make setup`.** On a machine that meets the requirements above:

```bash
git clone https://github.com/RosettaCommons/RFD4-Proteina.git
cd RFD4-Proteina
make setup
source .venv/bin/activate
```

The repository history is large because of past notebook outputs. For a fast clone use
`git clone --depth 1` (no history) or `git clone --filter=blob:none` (full history, blobs fetched on demand).
Notebooks are committed without outputs; run `pre-commit install` to enforce this with `nbstripout`.

Setup installs the environment, caches the released Hugging Face model bundle
(about 1.7 GB, including its autoencoder), and configures `DATA_PATH` for optional data.
The cache location follows `HF_HOME`. See the
[installation guide](https://rosettacommons.github.io/RFD4-Proteina/installation.html)
for CUDA build prerequisites, optional metrics, and cluster setup.

Other options (CUDA 12, CPU-only, Docker, Apptainer, pip) are listed in the
[deployment options](https://rosettacommons.github.io/RFD4-Proteina/installation.html#deployment-options).
For example, a container with CUDA 13 (use `RFD4_ACCELERATOR=cu12` or `cpu` for the others):

```bash
docker build -t rfd4-proteina:cu13 .
docker run --rm --gpus all rfd4-proteina:cu13 rfd4-generate --help
```

While the Hugging Face repositories are private, supply `HF_TOKEN` or use `hf auth login`
with an account that has access. For interactive login on a fresh checkout, run
`bash setup.sh --skip-weights`, activate `.venv`, log in, then run `bash setup.sh --native`.
Once public, the same setup works without authentication.

## Run the model

After setup, generate a design from the included [example input](data/benchmarks/monomer/monomer-short.cif):

```bash
rfd4-generate inputs=data/benchmarks/monomer/monomer-short.cif \
    num_replicates=1 out_dir=./runs/first \
    ~metrics compile_lengths=null
```

Designs are written to `runs/first/inference_samples/`. Increase `num_replicates`
for more designs, or replace the example path
with your own [annotated input](https://rosettacommons.github.io/RFD4-Proteina/workflow/inputs.html).

### Hugging Face and Python

The default model is the pinned Hugging Face release. To select a different Hub repo
or a local exported bundle, use:

```bash
rfd4-generate hf_ckpt_path=nvidia/RFD4-Proteina \
    inputs=design.yaml out_dir=./runs/hf ~metrics compile_lengths=null
```

After the package is published, install it with `pip install rfd4-proteina` on a
supported CUDA/Python system. The same API works from an installed development checkout:

```python
from hydra import compose, initialize_config_dir
import rootutils
import rfproteina
from rfd4_proteina import Proteina
from rfproteina.generate import run_generation

model = Proteina.from_pretrained("nvidia/RFD4-Proteina")
config_dir = rootutils.find_root(rfproteina.__file__, indicator=".project-root") / "configs"
with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
    cfg = compose(config_name="inference", overrides=[
        "inputs=design.yaml", "out_dir=runs/hf", "~metrics", "compile_lengths=null",
    ])
run_generation(cfg, model=model)
```

Use an annotated CIF or condition-spec YAML as `inputs`. Add `revision="<commit>"`
to `from_pretrained` (or `hf_ckpt_revision=<commit>` to the CLI) for reproducibility.
HF caches downloaded weights; `run_generation` reuses the loaded model.
For a native local checkpoint, set `hf_ckpt_path=null`,
`ckpt_path=/path/to/checkpoints`, `ckpt_name=model.ckpt`, and
`autoencoder_ckpt_path=/path/to/ae.ckpt`.
Training datasets are separate from model downloads: see the
[training guide](https://rosettacommons.github.io/RFD4-Proteina/development.html).

- [Running the model](https://rosettacommons.github.io/RFD4-Proteina/workflow/running.html): binders, motif scaffolding, partial diffusion, and sampling settings.
- [Input preparation](https://rosettacommons.github.io/RFD4-Proteina/workflow/inputs.html) · [Text-based design specifications](https://rosettacommons.github.io/RFD4-Proteina/workflow/condition-spec.html)
- [Scoring and refolding](https://rosettacommons.github.io/RFD4-Proteina/workflow/metrics.html) · [Reward-guided search](https://rosettacommons.github.io/RFD4-Proteina/workflow/search.html)
- [Training and fine-tuning](https://rosettacommons.github.io/RFD4-Proteina/development.html) · [Worked examples](https://rosettacommons.github.io/RFD4-Proteina/examples.html)

## More workflows

- **Binders and motif scaffolding:** condition on an annotated CIF or a YAML design spec; see [Running the model](https://rosettacommons.github.io/RFD4-Proteina/workflow/running.html).
- **Reward-guided search:** select a search `inference_experiment` config to branch and prune candidates against an oracle; see [Reward-guided search](https://rosettacommons.github.io/RFD4-Proteina/workflow/search.html).
- **Partial diffusion:** re-noise and refine an existing design; see [Partial diffusion](https://rosettacommons.github.io/RFD4-Proteina/workflow/partial-diffusion.html).
- **Scoring and refolding:** run `rfd4-evaluate` on generated structures; see [Scoring and refolding](https://rosettacommons.github.io/RFD4-Proteina/workflow/metrics.html).
- **Inference backends:** `native` (default, compiled or eager) or `bioir` via `inference_backend=`; see [Inference backends](https://rosettacommons.github.io/RFD4-Proteina/workflow/backends.html).

## Tutorials

- [Annotated CIF tutorial](docs/examples/annotated_cif_tutorial.ipynb): build and validate conditioned structure inputs.
- [Enzyme design tutorial](docs/examples/enzyme_design_tutorial.ipynb): motif scaffolding and catalytic-site metrics, end to end.
- [PPI specificity tutorial](docs/examples/ppi_specificity_tutorial.ipynb): protein-binder design and specificity checks.
- [Antibody campaign examples](examples/antibody_design/README.md): cropped PMVK/MZB1 VHH inputs, hotspot sampling, high-early recycling, and partial refinement.
- [Agent skills](skills/README.md): task-specific guides for setup, generation, search, and evaluation.

[Documentation](https://rosettacommons.github.io/RFD4-Proteina/) · [Contributing](https://rosettacommons.github.io/RFD4-Proteina/contributor_guide.html) · [BSD 3-Clause license](LICENSE)

<p align="center">
  <img src="assets/readme/dna-binder-generation.gif" width="360" alt="Protein-binder generation conditioned on a DNA duplex.">
</p>
