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
    num_replicates=10
```

Inputs can be JSON, YAML, CIF, or PDB files. See the [input specification](docs/input.md) for the full schema.

> **Syntax note**: All arguments use `key=value` (Hydra style), not `--key value`.

### Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `inputs` | Input file (JSON/YAML/CIF/PDB) or directory | Required |
| `ckpt_path` | Directory containing checkpoint | Required |
| `ckpt_name` | Checkpoint filename | Required |
| `num_replicates` | Designs generated per input example | 1 |
| `batch_size` | Designs processed in parallel on GPU (increase if memory allows) | 1 |
| `seed` | Random seed (`null` = different seed each run) | null |
| `skip_existing` | Skip examples whose output files already exist on disk | false |
| `subset_to_examples` | Run only specific examples from a JSON, as comma-separated keys | null |
| `global_prefix` | String prefix prepended to all output file paths | null |
| `out_dir` | Output directory (auto-inferred from checkpoint path if not set) | null |
| `generation.args.nsteps` | Diffusion steps — higher is slower but often better quality | 400 |
| `generation.args.guidance_w` | Guidance weight (1.0 = unconditional; increase for stronger conditioning) | 1.0 |
| `generation.args.self_cond` | Self-conditioning — leave enabled for best quality | true |

> **RFD3 users**: `diffusion_batch_size` is accepted as an alias for `batch_size`, and `json_keys_subset` for `subset_to_examples`.

> [!WARNING]
> **RFD3 users — no ORI token.** This model does not use an ORI residue in the input structure. The model has been trained to select a reasonable COM, and its position relative to other chains can be guided using hotspots (proteins) or RASA (ligands).

### Common Patterns

```bash
# Generate many designs per input (use batch_size to fill GPU memory)
python proteinfoundation/generate.py \
    inputs=design.json ckpt_path=./checkpoints ckpt_name=model.ckpt \
    num_replicates=100 batch_size=4

# Resume a run — skip examples that already have output files
python proteinfoundation/generate.py \
    inputs=design.json ckpt_path=./checkpoints ckpt_name=model.ckpt \
    num_replicates=100 skip_existing=true

# Run only selected examples from a multi-example JSON
python proteinfoundation/generate.py \
    inputs=design.json ckpt_path=./checkpoints ckpt_name=model.ckpt \
    subset_to_examples="binder_A,binder_B"

# Reproducible run with a fixed random seed
python proteinfoundation/generate.py \
    inputs=design.json ckpt_path=./checkpoints ckpt_name=model.ckpt \
    num_replicates=10 seed=42
```

### Metrics Tags

`metrics_tags` activates post-generation evaluation metrics (e.g. RF3 refolding and scoring). Pass one or more tags as a Hydra list (or comma-separated string):

```bash
python proteinfoundation/generate.py \
    inputs=design.json ckpt_path=./checkpoints ckpt_name=model.ckpt \
    num_replicates=50 metrics_tags=[unconditional_refolding]
```

Multiple tags: `metrics_tags=[ppi_refolding,hotspot_conditioning]`

| Tag | Activates |
|-----|-----------|
| `unconditional_refolding` | RF3 refolding + pLDDT/scRMSD for unconditional generation |
| `ppi_refolding` | RF3 refolding + target templating + interface metrics for protein binder design |
| `motif_scaffold_refolding` | RF3 refolding + motif RMSD for motif scaffolding / enzyme design |
| `sm_binder_refolding` | RF3 refolding + binding metrics for small molecule binder design |
| `ame_refolding` | RF3 refolding for enzyme (MCSA) active-site benchmarks |
| `hotspot_conditioning` | Hotspot obedience metrics (requires hotspot input) |
| `3d_coordinate_conditioning` | Motif coordinate RMSD metrics |
| `hbond_conditioning` | H-bond satisfaction metrics |
| `rasa_conditioning` | RASA adherence metrics |

Metrics that require a tag will be silently skipped if the tag is not present. Tags have no effect unless a metrics config that references them is loaded (controlled by the `inference_experiment` config group).

---

### Partial Diffusion

Partial diffusion starts the denoising from a noisy (but non-random) initial state, allowing you to refine an existing structure without re-generating it from scratch.

Enable it by setting `partial_t_backbone` and/or `partial_t_latent` in the input spec. This is corruption level of the input, interpolating linearly between the input structure and Gaussian noise. It is **NOT** a noise level in Angstroms.

> **Note:** 1 corresponds to the input structure, so setting `partial_t_backbone=0.8` means the backbone starts closer to the input than to random noise.

An example might look like:

```yaml
# design.yaml
input: my_structure.pdb
contig: A1-100
partial_t_backbone: 0.8   # noise level: 1 = no change, 0 = fully random
```

**Backbone, sequence/rotamer, and step count are completely decoupled.**

This model uses two independent diffusion flows — one for backbone Cα coordinates and one for sidechain latent variables — and these can be noised to completely different levels:

| Input spec field | Controls | Default if omitted |
|-----------------|----------|-------------------|
| `partial_t_backbone` | Backbone Cα noise (Å) only | `0.0` (fully diffused backbone) |
| `partial_t_latent` | Sidechain/sequence latent noise only | `0.0` (fully diffused sequence/rotamer) |

The number of denoising steps (`nsteps`) is **independent of both noise levels**. Setting `partial_t=0.5` does not halve the number of steps — the sampler schedules exactly `generation.args.nsteps` denoising steps over the range `[partial_t, 1.0]`.

> **Note**: `partial_t` values must be set in the input spec, not on the command line. It is a per-example property, so different examples in the same JSON can use different noise levels.

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

## Supported Conditioning

The table below lists all conditioning modalities and their current status. Each conditioning type is specified per-residue or per-atom in the input spec — see [Input Specification](docs/input.md) for syntax.

| Conditioning | Input field | Status |
|---|---|---|
| **Coordinate (indexed motif)** | `contig` + `select_fixed_atoms` | ✅ Supported |
| **Coordinate (unindexed motif)** | `unindex` | ✅ Supported |
| **Internal coordinate** | `contig` + relative geometry | ✅ Supported |
| **Sequence constraint** | `select_unfixed_sequence` | ✅ Supported |
| **Hotspot residues** | `select_hotspots` | ✅ Supported |
| **Partial diffusion** | `partial_t` / `partial_t_backbone` / `partial_t_latent` | ✅ Supported |
| **Symmetry** | `symmetry` | ✅ Supported |
| **Small molecules / ligands** | `ligand` | ✅ Supported |
| **H-bond donors/acceptors** | `select_hbond_donor` / `select_hbond_acceptor` | 🔜 Planned |
| **RASA (buried/exposed)** | `select_buried` / `select_exposed` | 🔜 Planned |
| **Secondary structure** | `select_helix_conditioning` / `select_sheet_conditioning` | 🔜 Planned |
| **CATH fold class** | `add_cath_code` | 🔜 Planned |

> **Planned features** are parsed and stored in the input spec but the model is not yet trained to respond to them. Passing them will not cause errors but will have no effect on generation.

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

## Output Files

Each generated design is saved as a CIF file. Alongside standard structural data, the model writes several custom per-atom annotations that record how each residue was produced.

### CIF Annotations

| Annotation | Level | Type | Description |
|---|---|---|---|
| `src_component` | Residue | string | Source component identifier for each residue. Tracks which part of the input specification a residue originated from. `"."` means no specific source detected (unexpected). |
| `mapped_to_coordinate_conditioned_region` | Atom | bool | `True` for atoms whose final coordinates were mapped from a coordinate-conditioned motif region, whether using absolute coords or internal coords, and whether indexed or unindexed. |
| `mapped_to_coordinate_conditioned_region_token_lvl` | Residue | bool | Token-level (residue-level) aggregation of the above: `True` if any atom in the residue was mapped from an unindexed motif. Can be used to determine what should be held fixed for downstream sequence redesign (e.g. ProteinMPNN). |
| `coords_with_insertion` | Atom | float\[3\] | Coordinates after inserting motif atoms into the generated backbone. For atoms with internal coordinate conditioning, holds the superimposed ground-truth coordinates used as the conditioning reference. |
| `src_coordinate_conditioned_region` | Atom | string | Identifies which input motif region provided the coordinate conditioning for each atom. `"."` means the atom is freely generated; otherwise the value is a string like `"A1"` tracing the source chain/residue. |

These annotations are written by default to all output CIF files and are consumed downstream by the metrics pipeline and sequence redesign steps.

---

## Evaluating Results

### Running Metrics Inline

Pass `metrics_tags` during inference to compute metrics immediately after each design (see [Metrics Tags](#metrics-tags)). Results are written to `per_example_metrics.csv` alongside the CIF files.

### Concatenating Multi-Job Runs

When running large evaluations in parallel across multiple SLURM jobs, each job writes its own `per_example_metrics_job{N}.csv`. After all jobs complete, concatenate them into a single file:

```bash
python -m proteinfoundation.inference_utils.concatenate_external_eval_results \
    --output_dir ./inference/my_run/metrics/
```

Or with optional FoldSeek clustering (see below):

```bash
python -m proteinfoundation.inference_utils.concatenate_external_eval_results \
    --output_dir ./inference/my_run/metrics/ \
    --cluster_with_foldseek
```

This script:
1. Finds all `per_example_metrics_job*.csv` files in the output directory
2. Concatenates them into a single `per_example_metrics.csv`
3. Optionally computes FoldSeek cluster assignments (see below)
4. Aggregates per-task mean metrics into `aggregated_metrics.csv`
5. Moves individual job files into a `job_results/` subdirectory

### FoldSeek Diversity Clustering

When `--cluster_with_foldseek` is passed, structures are clustered by structural similarity using [FoldSeek](https://github.com/steineggerlab/foldseek). The resulting `foldseek_cluster_id` column enables computing the number of structurally distinct designs per task.

**Clustering is scoped to the chain of interest.** In multimer outputs (e.g. binder + target), only the designed chain is extracted and clustered — the fixed target is excluded. This chain is identified automatically from the conditioning annotations using a heuristic that selects the protein chain with the fewest coordinate-conditioned atoms (i.e. the most freely designed chain). The resolved chain ID is stored as `chain_of_interest` in the metrics CSV and must be consistent within a task for FoldSeek to run.

If `chain_of_interest` is `None` or inconsistent across examples in a task, FoldSeek clustering is skipped for that task with a warning.

FoldSeek must be installed and available on `$PATH`. For monomer designs, clustering uses TM-score (default threshold 0.5). For multimers, it additionally uses interface lDDT.

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
