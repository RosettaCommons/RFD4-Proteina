<p align="center">
  <img src="assets/wordmark.svg" width="587" alt="RFD4-Proteína">
</p>

<p align="center"><em>A flow-matching foundation model for biomolecular design — Baker Lab × NVIDIA GenAIR</em></p>

<p align="center">
  <a href="https://pypi.org/project/rfd4-proteina/"><img src="https://img.shields.io/pypi/v/rfd4-proteina" alt="PyPI"></a>
  <a href="https://codecov.io/gh/RosettaCommons/RFD4-Proteina/tree/main"><img src="https://codecov.io/gh/RosettaCommons/RFD4-Proteina/branch/main/graph/badge.svg" alt="Codecov coverage"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.12.0%E2%80%933.12.10-blue?logo=python&amp;logoColor=white" alt="Python 3.12.0–3.12.10"></a>
  <a href="https://rosettacommons.github.io/RFD4-Proteina/"><img src="https://img.shields.io/badge/docs-read-blue" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="BSD 3-Clause License"></a>
</p>

RFD4-Proteína designs proteins and biomolecular complexes with sequence–structure co-design,
motif and binding-site conditioning, partial diffusion, and reward-guided search.

## Install

Use Linux x86_64, Python 3.12.0–3.12.10, and an NVIDIA GPU with a CUDA 13-compatible driver.

```bash
pip install rfd4-proteina
```

See the [installation guide](https://rosettacommons.github.io/RFD4-Proteina/installation.html) for CUDA dependencies, optional metrics,
source installation, and cluster setup.

## Run the model

Download the [flow checkpoint](https://rfd4-proteina-public-e04a.cwobject.com/flow_checkpoints/chk_epoch_00000169_step_000000340000-ema.ckpt)
and [autoencoder checkpoint](https://rfd4-proteina-public-e04a.cwobject.com/ae_checkpoints/2026_09_04_release_ae_z16_400k-ema.ckpt)
(about 8 GB total), saving them as `weights/flow.ckpt` and `weights/ae.ckpt`.
Save the [example input](data/benchmarks/monomer/monomer-short.cif)
as `design.cif`, then generate a design:

```bash
rfd4-generate inputs=design.cif \
    ckpt_path=./weights ckpt_name=flow.ckpt \
    autoencoder_ckpt_path=./weights/ae.ckpt \
    num_replicates=1 out_dir=./runs/first \
    compile_lengths=null '~metrics'
```

Designs are written to `runs/first/inference_samples/`. This first run disables compilation
and optional metrics. Increase `num_replicates` for more designs, or replace `design.cif`
with your own [annotated input](https://rosettacommons.github.io/RFD4-Proteina/workflow/inputs.html).

- [Running the model](https://rosettacommons.github.io/RFD4-Proteina/workflow/running.html): binders, motif scaffolding, partial diffusion, and sampling settings.
- [Input preparation](https://rosettacommons.github.io/RFD4-Proteina/workflow/inputs.html) · [Text-based design specifications](https://rosettacommons.github.io/RFD4-Proteina/workflow/condition-spec.html)
- [Scoring and refolding](https://rosettacommons.github.io/RFD4-Proteina/workflow/metrics.html) · [Reward-guided search](https://rosettacommons.github.io/RFD4-Proteina/workflow/search.html)
- [Training and fine-tuning](https://rosettacommons.github.io/RFD4-Proteina/development.html) · [Worked examples](https://rosettacommons.github.io/RFD4-Proteina/examples.html)

## Examples

<picture>
  <source media="(max-width: 600px)" srcset="assets/readme/design-gallery-mobile.png">
  <img src="assets/readme/design-gallery.png" width="1050" alt="Nanobody complex, protein–DNA binder, RNA pseudoknot, ligand pocket, glycan complex, and protein assembly.">
</picture>

Conditioned generation (DNA binder and RNA) and conditioned reconstruction examples.
[Example details](assets/readme/provenance.json).

<details>
<summary>DNA-binder generation trajectory</summary>

<p align="center">
  <a href="assets/readme/dna-binder-generation.mp4"><img src="assets/readme/dna-binder-generation.gif" width="360" alt="Protein-binder generation conditioned on a DNA duplex."></a>
</p>

[Full-resolution video](assets/readme/dna-binder-generation.mp4). The loop plays generation forward, then reverses for playback.

</details>

[Documentation](https://rosettacommons.github.io/RFD4-Proteina/) · [Contributing](https://rosettacommons.github.io/RFD4-Proteina/contributor_guide.html) · [BSD 3-Clause license](LICENSE)
