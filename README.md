<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RosettaCommons/RFD4-Proteina/main/docs/_static/wordmark-dark.webp">
    <img src="https://raw.githubusercontent.com/RosettaCommons/RFD4-Proteina/main/docs/_static/wordmark.webp" width="520" alt="RFD4-Proteina">
  </picture>
</p>

<p align="center"><em>A flow-matching foundation model for biomolecular design — Baker Lab × NVIDIA GenAIR</em></p>

<p align="center">
  <a href="https://github.com/RosettaCommons/RFD4-Proteina/actions/workflows/ci.yaml"><img src="https://github.com/RosettaCommons/RFD4-Proteina/actions/workflows/ci.yaml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.org/project/rfd4-proteina/"><img src="https://img.shields.io/pypi/v/rfd4-proteina?logo=pypi&amp;logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/RosettaCommons/RFD4-Proteina/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.12-blue?logo=python&amp;logoColor=white" alt="Python 3.12"></a>
  <a href="https://rfd4-proteina.org/latest/"><img src="https://img.shields.io/badge/docs-read-blue" alt="Documentation"></a>
  <a href="https://github.com/RosettaCommons/RFD4-Proteina/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="BSD 3-Clause License"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/nvidia/RFD4-Proteina"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model-yellow" alt="Hugging Face model"></a>
  <a href="https://huggingface.co/datasets/nvidia/RFD4-Proteina"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-dataset-yellow" alt="Hugging Face dataset"></a>
  <a href="https://huggingface.co/nvidia/RFD4-Proteina-Diffusers"><img src="https://img.shields.io/badge/%F0%9F%A7%A8%20Diffusers-pipeline-orange" alt="Diffusers pipeline"></a>
  <a href="https://huggingface.co/spaces/nvidia/rfd4-proteina-zerogpu"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-ZeroGPU-blue" alt="ZeroGPU Space"></a>
  <a href="https://huggingface.co/spaces/nvidia/rfd4-proteina-space"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-dedicated%20GPU-blue" alt="Dedicated GPU Space"></a>
</p>

<p align="center">
  <a href="https://rfd4-proteina.org/">Project page</a> ·
  <a href="#system-requirements">Requirements</a> ·
  <a href="#install">Installation</a> ·
  <a href="#run-the-model">Quick start</a> ·
  <a href="#more-workflows">Workflows</a> ·
  <a href="#tutorials">Tutorials</a> ·
  <a href="https://rfd4-proteina.org/latest/">Documentation</a>
</p>

<picture>
  <source media="(max-width: 600px)" srcset="https://huggingface.co/datasets/nvidia/RFD4-Proteina/resolve/main/media/assets/readme/design-gallery-mobile.png">
  <img src="https://huggingface.co/datasets/nvidia/RFD4-Proteina/resolve/main/media/assets/readme/design-gallery.png" width="1050" alt="Nanobody complex, protein–DNA binder, RNA pseudoknot, ligand pocket, glycan complex, and protein assembly.">
</picture>

RFD4-Proteina designs proteins and biomolecular complexes with sequence–structure co-design,
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
- Tested GPUs: see the [supported hardware table](https://rfd4-proteina.org/latest/installation.html#supported-hardware). Check yours with `nvidia-smi`.
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
[installation guide](https://rfd4-proteina.org/latest/installation.html)
for CUDA build prerequisites, optional metrics, and cluster setup.

Setup also asks whether to add an AlphaFold 2 or AlphaFold 3 / OpenFold3 environment for
reward-guided search and refolding metrics (and, if so, your CPU architecture and CUDA version).
Skip that with `bash setup.sh --skip-search-envs`; run it later with `bash scripts/setup_search_envs.sh`.

Other options (CUDA 12, CPU-only, Docker, Apptainer, pip) are listed in the
[deployment options](https://rfd4-proteina.org/latest/installation.html#deployment-options).
For example, a container with CUDA 13 (use `RFD4_ACCELERATOR=cu12` or `cpu` for the others):

```bash
docker build -t rfd4-proteina:cu13 .
docker run --rm --gpus all rfd4-proteina:cu13 rfd4-generate --help
```

If access to the Hugging Face repositories is restricted, supply `HF_TOKEN` or use `hf auth login`
with an account that has access. For interactive login on a fresh checkout, run
`bash setup.sh --skip-weights`, activate `.venv`, log in, then run `bash setup.sh --native`.
Public repositories need no authentication.

### Quick start with pip

On Linux x86_64 with Python 3.12, an NVIDIA GPU and a CUDA 13-compatible driver:

```bash
pip install uv                      # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "rfd4-proteina[all-cuda13]"   # CUDA 12: all-cuda12; the base package alone is "rfd4-proteina"

EXAMPLE=$(python -c "import rfproteina, pathlib; print(pathlib.Path(rfproteina.__file__).parent / 'data/benchmarks/monomer/monomer-short.cif.zst')")
rfd4-generate hf_ckpt_path=nvidia/RFD4-Proteina inputs=$EXAMPLE num_replicates=1 out_dir=./runs/first ~metrics
```

The environment is about 7 GB, mostly PyTorch and the CUDA libraries. `uv` downloads in parallel and caches wheels, so it installs in
about two minutes on a fresh cluster node where plain `pip` takes about ten; `pip install` works too, only more slowly.

The model can also be run as a Hugging Face Diffusers pipeline. The [quickstart](https://rfd4-proteina.org/latest/quickstart.html) covers all three routes in order: GitHub clone, PyPI and Diffusers.

The CUDA extra also installs the fused cuEquivariance kernels the model loads, so use it even if you skip metrics. Add `inference_backend=bioir` to use the BioIR runtime (included in `all-cuda13`). `all-cuda12` and the plain package
use the native runtime. If `pip` cannot see a very recent release, retry with `--no-cache-dir`.

## Run the model

After setup, generate a design from the included [example input](https://github.com/RosettaCommons/RFD4-Proteina/tree/main/data/benchmarks/monomer):

```bash
rfd4-generate inputs=data/benchmarks/monomer/monomer-short.cif.zst \
    num_replicates=1 out_dir=./runs/first \
    ~metrics compile_lengths=null
```

Designs are written to `runs/first/inference_samples/`. Increase `num_replicates`
for more designs, or replace the example path
with your own [annotated input](https://rfd4-proteina.org/latest/how-to/input_prep/index.html).

### Hugging Face and Python

To load the model from the Hugging Face Hub, with no local checkpoints and no `DATA_PATH`, pass
`hf_ckpt_path`; use a different Hub repo or a local exported bundle the same way:

```bash
rfd4-generate hf_ckpt_path=nvidia/RFD4-Proteina \
    inputs=design.yaml out_dir=./runs/hf ~metrics compile_lengths=null
```

Install it with `uv pip install rfd4-proteina` on a
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
[training guide](https://rfd4-proteina.org/latest/contributing/developer-guide.html).

- [Running the model](https://rfd4-proteina.org/latest/how-to/run-the-model.html): binders, motif scaffolding, partial diffusion, and sampling settings.
- [Input preparation](https://rfd4-proteina.org/latest/how-to/input_prep/index.html) · [Text-based design specifications](https://rfd4-proteina.org/latest/how-to/input_prep/yaml-json.html)
- [Scoring and refolding](https://rfd4-proteina.org/latest/how-to/evaluate-with-metrics.html) · [Reward-guided search](https://rfd4-proteina.org/latest/how-to/search.html)
- [Training and fine-tuning](https://rfd4-proteina.org/latest/contributing/developer-guide.html) · [Worked examples](https://rfd4-proteina.org/latest/examples/index.html)

## More workflows

- **Binders and motif scaffolding:** condition on an annotated CIF or a YAML design spec; see [Running the model](https://rfd4-proteina.org/latest/how-to/run-the-model.html).
- **Reward-guided search:** select a search `inference_experiment` config to branch and prune candidates against an oracle; see [Reward-guided search](https://rfd4-proteina.org/latest/how-to/search.html).
- **Partial diffusion:** re-noise and refine an existing design; see [Partial diffusion](https://rfd4-proteina.org/latest/how-to/partial-diffusion.html).
- **Scoring and refolding:** run `rfd4-evaluate` on generated structures; see [Scoring and refolding](https://rfd4-proteina.org/latest/how-to/evaluate-with-metrics.html).
- **Inference backends:** `native` (default, compiled or eager) or `bioir` via `inference_backend=`; see [Inference backends](https://rfd4-proteina.org/latest/how-to/advanced/backends.html).

## Tutorials

- [Python API tutorial](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/examples/python_api_tutorial.ipynb): build and validate conditioned structure inputs.
- [Enzyme design tutorial](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/examples/enzyme_design_tutorial.ipynb): motif scaffolding and catalytic-site metrics, end to end.
- [PPI specificity tutorial](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/examples/ppi_specificity_tutorial.ipynb): protein-binder design and specificity checks.
- [Antibody campaign examples](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/examples/antibody_design/README.md): cropped PMVK/MZB1 VHH inputs, hotspot sampling, high-early recycling, and partial refinement.
- [Agent skills](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/skills/README.md): task-specific guides for setup, generation, search, and evaluation.

[Documentation](https://rfd4-proteina.org/latest/) · [Contributing](https://rfd4-proteina.org/latest/contributing/index.html) · [BSD 3-Clause license](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/LICENSE)

<p align="center">
  <img src="https://huggingface.co/datasets/nvidia/RFD4-Proteina/resolve/main/media/assets/readme/dna-binder-generation.gif" width="360" alt="Protein-binder generation conditioned on a DNA duplex.">
</p>
