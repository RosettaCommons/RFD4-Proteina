# Protein Foundation Model

A flow-matching based generative model for protein design, supporting proteins, RNA, DNA, and small molecules under complex conditioning constraints.

<p align="center">
  <img src="assets/latent_design.png" alt="Latent design overview">
</p>

> [!NOTE]
> Looking for information about input configuration options? See the full [input specification](docs/input.md).

## Getting Started

1. Install the package:
```bash
cd projects/proteinfoundation
pip install -e .
```

2. Configure environment variables:
```bash
cp env.template .env
    # Edit .env to set DATA_PATH, CACHE_DIR, CCD_MIRROR_PATH
    ```

3. Download or point to the model checkpoint and autoencoder checkpoint. Set `autoencoder_ckpt_path` in your config or `.env`.

## Running Inference

```bash
python proteinfoundation/generate.py \
    inputs=design.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    nsamples=10
```

Inputs can be JSON, YAML, CIF, or PDB files. See the [input specification](docs/input.md) for the full schema. Key options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `inputs` | Input file (CIF/JSON/YAML) or directory | Required |
| `ckpt_path` | Directory containing checkpoint | Required |
| `ckpt_name` | Checkpoint filename | Required |
| `nsamples` | Number of samples to generate | 10 |
| `generation.args.nsteps` | Diffusion steps | 400 |
| `generation.args.guidance_w` | Guidance weight (1.0 = no guidance) | 1.0 |
| `generation.args.self_cond` | Enable self-conditioning | True |

---

## Design Use Cases

The model supports a wide range of biomolecular design tasks. Each link below provides example inputs and run commands.

<table>
  <tr>
    <td align="center">
      <h3><a href="docs/protein_binder_design.md">Protein binder design</a></h3>
      Design proteins that bind a target protein at specified hotspot residues.
    </td>
    <td align="center">
      <h3><a href="docs/small_molecule_binder_design.md">Small molecule binder design</a></h3>
      Design proteins that bind small molecule ligands with burial/exposure control.
    </td>
    <td align="center">
      <h3><a href="docs/dna_binder_design.md">DNA binder design</a></h3>
      Design transcription-factor-like proteins that bind DNA.
    </td>
  </tr>
  <tr>
    <td align="center">
      <h3><a href="docs/enzyme_design.md">Enzyme design</a></h3>
      Design enzymes with catalytic residues around a ligand active site.
    </td>
    <td align="center">
      <h3><a href="docs/motif_scaffolding.md">Motif scaffolding</a></h3>
      Build a complete protein around one or more structural motifs.
    </td>
    <td align="center">
      <h3><a href="docs/unconditional_generation.md">Unconditional generation</a></h3>
      Generate novel protein structures from scratch.
    </td>
  </tr>
</table>

---

## Inference Noise Parameters

The model uses **product-space flow matching** with two separate flows: one for **backbone Cα coordinates** and one for **sidechain latent variables**. During inference, the key parameters controlling generation diversity and quality are the noise scales for each modality:

| Parameter | Config Key | Controls | Default |
|-----------|-----------|----------|---------|
| **Backbone noise** (η_x) | `generation.model.bb_ca.simulation_step_params.sc_scale_noise` | Structural diversity of generated backbones | 0.1 |
| **Latent noise** (η_z) | `generation.model.local_latents.simulation_step_params.sc_scale_noise` | Sidechain/sequence diversity | 0.1 |

These noise parameters directly control the **stochasticity of the SDE sampler** during generation. Higher values increase diversity but may reduce designability; lower values produce more deterministic, higher-confidence outputs.

The figure below shows how different noise settings (η_x, η_z) affect the **amino acid distribution** of generated proteins compared to a standard MPNN baseline and the UniProt reference distribution:

<p align="center">
  <img src="assets/aa_distribution_latent_model.png" alt="Impact of noise parameters on amino acid distribution" width="800">
</p>

- **(η_x, η_z) = (0.1, 0.1)**: Low noise — sequences are closest to the MPNN baseline, high designability, narrow distribution.
- **(η_x, η_z) = (0.5, 0.5)**: Moderate noise — sequences shift toward the natural UniProt distribution, balancing diversity and designability.
- **(η_x, η_z) = (0.9, 0.9)**: High noise — sequences most closely match the natural UniProt amino acid frequencies, maximum diversity but potentially lower designability.

```bash
# Example: generate with moderate noise for more natural-like sequences
python proteinfoundation/generate.py \
    inputs=design.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    generation.model.bb_ca.simulation_step_params.sc_scale_noise=0.5 \
    generation.model.local_latents.simulation_step_params.sc_scale_noise=0.5
```

For the full inference configuration and training-time flow matching parameters, see the [Developer Guide](docs/DEVELOPMENT.md#flow-matching-parameters).

---

## Search and Refinement

Beyond standard sampling, the model supports **search strategies** to explore multiple generation paths and **refinement strategies** to optimize structures post-generation. Full details in the [Search and Refinement guide](docs/search_and_refinement.md).

### Search Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Vanilla** | Standard diffusion sampling | Baseline, fastest |
| **Best-of-N** | Generate N samples, pick highest reward | Simple, effective, parallelizable |
| **Beam Search** | Maintain top-k candidates, prune at checkpoints | Best quality, slower |

```bash
# Best-of-N with RF3 reward
python proteinfoundation/generate.py \
    --config-name inference_test_search \
    inputs=design.json \
    ckpt_path=./checkpoints ckpt_name=model.ckpt \
    search.strategy=best_of_n search.strategy.n_samples=50 \
    reward=rf3
```

### Refinement Strategies

| Strategy | What it optimizes | Requires Autoencoder | Speed |
|----------|-------------------|---------------------|-------|
| **BindCraft** | Sequence only (structure fixed) | No | Fast |
| **Gradient Ascent** | Latents (structure + sequence) | Yes | Medium |

```bash
# BindCraft: optimize sequence with AF2 reward
python proteinfoundation/refine.py \
    --config-name refine_bindcraft \
    inputs=my_binder.pdb

# Gradient ascent: optimize in latent space
python proteinfoundation/refine.py \
    --config-name refine_gradient_ascent \
    inputs=my_binder.pdb \
    autoencoder_ckpt_path=/path/to/autoencoder.ckpt
```

### Available Reward Models

Reward models score generated structures to guide search and refinement. Configure via YAML files in `configs/rewards/`. Metrics are organized into three categories:

**Refolding Metrics** — Refold with an independent model (RF3, AF2) to validate designability. Both are differentiable.

**Force Field Metrics** — Evaluate physical plausibility using the Rosetta scoring function (via TMol) or MACE (ML force field). Includes SoftDDG, charge complementarity, TMol interface energy, and MACE energy.

**Structure Quality Metrics** — Fast geometric/chemical checks (PoseBusters, SSE, RASA, MolProbity, sidechains, H-bonds).

| Config | Category | Key Signals | Differentiable |
|--------|----------|-------------|----------------|
| `rf3.yaml` | Refolding | pLDDT, ipTM, ipAE, min_ipAE | Yes |
| `af2_binder.yaml` | Refolding | pLDDT, pTM, PAE, max_PAE | Yes |
| `multi_source.yaml` | Refolding + Force Field + Quality | ipAE, H-bonds, RASA, clashes | No |
| `structure_quality.yaml` | Structure Quality | Sidechains, SSE, RASA, MolProbity | No |

See the [Search and Refinement guide](docs/search_and_refinement.md) for the full metric catalog, custom reward creation, and **parameter sweeps**.

### Inference Sweeps

Run the same pipeline across many parameter combinations (checkpoints, temperatures, strategies) with a single command:

```bash
# Define sweep in YAML → generate + submit all jobs
./slurm_utils/launch_sweep.sh digs --sweep-config configs/sweeps/ppi_benchmark.yaml
```

See [Inference Sweeps](docs/search_and_refinement.md#inference-sweeps) for the full guide and pre-built sweep configs.

---

## For Developers

Training, SLURM launching, dataset configuration, design task implementation, adding new metrics/rewards, and architecture details are documented in the **[Developer Guide](docs/DEVELOPMENT.md)**.

For coding standards and contribution guidelines, see [CLAUDE.md](CLAUDE.md).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `DATA_PATH not set` | Create `.env` from `env.template` and set required paths |
| `Autoencoder checkpoint not found` | Update `autoencoder_ckpt_path` in config or `.env` |
| `Hydra config errors` | Use `=` not `--` for arguments: `inputs=file.cif` not `--inputs file.cif` |
| `Import errors` | Ensure parent foundry installed: `cd ../.. && pip install -e .` |
| `Out of memory` | Reduce batch size or number of GPUs in config |

See the [Developer Guide troubleshooting section](docs/DEVELOPMENT.md#troubleshooting) for the full list.
