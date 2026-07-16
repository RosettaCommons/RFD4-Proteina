# Inference Guide

Full reference for running inference — examples, conditioning, partial diffusion, search, and every
parameter. **New here? Start with the [main README](../README.md)** for setup (uv install, download
the model, environment); this guide assumes that's done.

> Configuring inputs? **[Annotated CIF inputs](annotated_cif_inputs.md)** is the recommended format (with a [tutorial notebook](tutorials/annotated_cif_tutorial.ipynb)); the JSON [Input Specification](input.md) is legacy.

## Contents

- [Design Use Cases](#design-use-cases)
- [Running Inference](#running-inference)
- [Partial Diffusion](#partial-diffusion)
- [Evaluating Results](#evaluating-results)
- [Search Strategies](#search-strategies)
- [Parameter Reference](#parameter-reference)
- [Supported Conditioning](#supported-conditioning)
- [Inference Noise Parameters](#inference-noise-parameters)
- [Sampling with TSR](#sampling-with-tsr)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

---

## Design Use Cases

Each page below has example inputs and run commands.

<table>
  <tr>
    <td align="center">
      <h3><a href="unconditional_generation.md">Unconditional generation</a></h3>
      Generate novel protein structures from scratch.
    </td>
    <td align="center">
      <h3><a href="motif_scaffolding.md">Motif scaffolding</a></h3>
      Build a complete protein around one or more structural motifs.
    </td>
    <td align="center">
      <h3><a href="enzyme_design.md">Enzyme design</a></h3>
      Design enzymes with catalytic residues around a ligand active site.
    </td>
  </tr>
  <tr>
    <td align="center">
      <h3><a href="small_molecule_binder_design.md">Small molecule binder design</a></h3>
      Design proteins that bind small-molecule ligands.
    </td>
    <td align="center">
      <h3><a href="protein_binder_design.md">Protein binder design</a></h3>
      Design binders to a target protein at hotspot residues.
      <br><em>⚠️ Not fully supported in the current checkpoint — planned.</em>
    </td>
    <td align="center">
      <h3><a href="dna_binder_design.md">DNA binder design</a></h3>
      Design transcription-factor-like proteins that bind DNA.
      <br><em>⚠️ Not fully supported in the current checkpoint — planned.</em>
    </td>
  </tr>
</table>

---

## Running Inference

The general form is:

```bash
python src/rfproteina/generate.py inputs=<file> num_replicates=<N> out_dir=<dir> [overrides...]
```

> [!TIP]
> On a cluster with apptainer/docker configured (e.g. IPD), the **preferred** form is to run the
> script directly — `./src/rfproteina/generate.py inputs=… …` — whose shebang executes it inside the
> container. `python src/rfproteina/generate.py …` (used throughout these docs) runs in your current
> environment instead; use it when no container is set up.

- **Inputs** — **annotated CIF** is the recommended format ([guide](annotated_cif_inputs.md) · [tutorial](tutorials/annotated_cif_tutorial.ipynb)); JSON/YAML/CIF/PDB via the legacy
  [Input Specification](input.md) also work. See `data/benchmarks/` for ready-made examples.
- **All arguments are `key=value`** (Hydra style), never `--key value`.
- **Checkpoints** are read from `$DATA_PATH` by default; override on the CLI only for a one-off
  model with `ckpt_path=<dir> ckpt_name=<file.ckpt>`.
- **`batch_size`** of 8 is a good starting point. Larger batches are faster, but first confirm
  your longest input fits at `batch_size=1`.

```bash
# 100 designs of each example in a JSON, 8 at a time
python src/rfproteina/generate.py \
    inputs=data/benchmarks/monomer.json \
    num_replicates=100 batch_size=8 out_dir=./runs/monomers
```

> [!WARNING]
> **RFD3 users — no ORI token.** This model does not use an ORI residue in the input. It selects a
> reasonable center of mass; its position relative to other chains can be guided with hotspots
> (proteins) or RASA (ligands). RFD3 aliases: `diffusion_batch_size` → `batch_size`,
> `json_keys_subset` → `subset_to_examples`.

---

## Partial Diffusion

Partial diffusion starts denoising from a noisy (but non-random) copy of an existing structure,
refining it instead of generating from scratch. Pass the structure with `inputs=` and set the noise
level with **Hydra overrides on the command line** (under `generation.args`). Two mutually exclusive
interfaces:

- **Simple** — `generation.args.partial_trajectory_fraction_steps=<f>` runs that fraction of `nsteps`
  (e.g. `0.3` = 30% of the steps); the per-modality `partial_t` values are inferred from the schedule.
- **Advanced** — `generation.args.partial_t_backbone=<t>` and/or `generation.args.partial_t_latent=<t>`
  (floats in `(0, 1)`); set `generation.args.couple_partial_t=true` to derive one from the other.

`partial_t` is a corruption level, **not** Angstroms: near `1` runs very little of the trajectory
(output stays close to the input), near `0` runs most of it (closer to full generation). `nsteps` is
independent of the level.

```bash
python src/rfproteina/generate.py \
    inputs=my_structure.cif \
    generation.args.partial_t_backbone=0.8 generation.args.partial_t_latent=0.8 \
    num_replicates=8 out_dir=./runs/partial
```

---

## Evaluating Results

> [!NOTE]
> Refolding metrics and reward-guided search both use the **AF3 oracle**, which runs only inside an
> apptainer container (or a native install), configured via the cluster env files (see the
> [Developer Guide](DEVELOPMENT.md)). Plain generation needs none of it.

### Running Metrics Inline

Pass `metrics_tags` during inference to compute metrics immediately after each design. Results are written to `per_example_metrics.csv` alongside the CIF files.

`metrics_tags` takes a Hydra-style list — pass it as a bracket-quoted list:

```bash
# Single tag
python src/rfproteina/generate.py \
    inputs=design.json num_replicates=50 metrics_tags='[unconditional_refolding]'

# Multiple tags
python src/rfproteina/generate.py \
    inputs=design.json num_replicates=50 metrics_tags='[ppi_refolding,hotspot_conditioning]'
```

| Tag | Activates |
| --- | --- |
| `unconditional_refolding` | AF3 refolding + pLDDT/scRMSD for unconditional generation |
| `ppi_refolding` | AF3 refolding + target templating + interface metrics for protein binder design |
| `motif_scaffold_refolding` | AF3 refolding + motif RMSD for motif scaffolding / enzyme design |
| `sm_binder_refolding` | AF3 refolding + binding metrics for small molecule binder design |
| `ame_refolding` | AF3 refolding for enzyme (MCSA) active-site benchmarks |
| `hotspot_conditioning` | Hotspot obedience metrics (requires hotspot input) |
| `3d_coordinate_conditioning` | Motif coordinate RMSD metrics |
| `rasa_conditioning` | RASA adherence metrics |
| `interchain_interactions` | Interchain interaction (H-bond/π-stack/π-cation/salt-bridge/hydrophobic/metal-coordination) adherence and counts |

Metrics that require a tag will be silently skipped if the tag is not present.

### Concatenating Multi-Job Runs

When running large evaluations in parallel across multiple SLURM jobs, each job writes its own `per_example_metrics_job{N}.csv`. After all jobs complete, concatenate them by passing the metrics directory as the first (positional) argument:

```bash
python -m rfproteina.inference_utils.aggregate_metrics ./inference/my_run/metrics/
```

This concatenates all per-example job CSVs, writes `aggregated_metrics.csv` and `per_task_metrics.csv`, and moves the individual job files into `individual_job_results/`. Pass `--output_dir <dir>` to write results elsewhere. When using `launch_infer.sh`, this runs automatically as a dependent SLURM job.

Pass `--cluster_with_foldseek` to also run structural diversity clustering (see below).

### Adding Custom Metrics

Add your own metrics without touching any tracked files. Two **git-ignored** directories hold them, so local experiments stay private:

- `src/rfproteina/metrics/custom_metrics/` — Python metric classes (split across as many modules as you like).
- `configs/metrics/custom_metrics/` — Hydra config registering them; `all_custom_metrics.yaml` is the entry point (`setup.sh` scaffolds it on first run).

The config is picked up automatically — no extra flags — and results land in `per_example_metrics.csv`, prefixed by the config name (e.g. `residue_count.n_residues`).

**See [docs/custom_metrics.md](custom_metrics.md)** for the full walkthrough: writing a metric class, refolding metrics, splitting across files, and contributing one back.

### FoldSeek Diversity Clustering

When `--cluster_with_foldseek` is passed, structures are clustered by structural similarity using [FoldSeek](https://github.com/steineggerlab/foldseek), adding a `foldseek_cluster_id` column for counting structurally distinct designs per task.

**Clustering is scoped to the designed chain.** In multimer outputs (e.g. binder + target), only the most freely designed chain is extracted and clustered. This chain is identified automatically from the conditioning annotations and stored as `chain_of_interest` in the metrics CSV. Clustering reads the extracted per-design chain CIFs, which are only written when you run generation with `save_chain_of_interest=true` — set it if you plan to cluster.

Point `FOLDSEEK_EXEC` (in `.env`) at the FoldSeek executable. Clustering runs `foldseek easy-cluster` on the extracted single chain using a TM-score threshold of 0.5.

---

## Search Strategies

Beyond standard sampling, the model supports **reward-guided search** — generate multiple candidates,
keep the ones an AF3 reward scores highest. A ready-to-run example:

```bash
# Beam search — branch and prune at checkpoints.
python src/rfproteina/generate.py inference_experiment=inference_beam_example
```

**See [Reward-Guided Search](search.md)** for the strategy and reward parameters (beam width, branching,
checkpoints, metric weights) and how to build your own search config.

---



## Parameter Reference

**The four you almost always set** — defaults are rarely right for a real run:

| Parameter | Description | Default |
| --- | --- | --- |
| `out_dir` | Output directory (set a real path for anything but a smoke test) | `./runs/<run_name>` |
| `num_replicates` | Designs generated per input example | 1 |
| `batch_size` | Designs processed in parallel on GPU (start at 8) | 1 |
| `metrics_tags` | Evaluation metrics to compute inline — see [Evaluating Results](#evaluating-results) | `[]` |

**[→ Full parameter reference](parameters.md)** — every parameter, grouped by purpose (inputs & checkpoints, sampling, diversity, partial diffusion).

---

## Supported Conditioning

Each modality is specified per-residue or per-atom. **Annotated CIFs are the recommended way to set
them** ([guide](annotated_cif_inputs.md)); the legacy JSON syntax is in the [Input Specification](input.md).

> The status column reflects the legacy JSON path; the **annotated-CIF path additionally exposes
> H-bond and RASA conditioning** — see [Annotated CIF inputs](annotated_cif_inputs.md).

| Conditioning | Input field | Status |
| --- | --- | --- |
| **Coordinate (indexed motif)** | `contig` + `select_fixed_atoms` | ✅ Supported |
| **Coordinate (unindexed motif)** | `unindex` | ✅ Supported |
| **Sequence constraint** | `select_unfixed_sequence` | ✅ Supported |
| **Hotspot residues** | `select_hotspots` | ✅ Supported |
| **Partial diffusion** | `partial_t_backbone` / `partial_t_latent` | ✅ Supported |
| **Small molecules / ligands** | `ligand` | ✅ Supported |
| **Internal coordinate** | `contig` + relative geometry | 🔜 Planned |
| **Symmetry** | `symmetry` | 🔜 Planned |
| **H-bond donors/acceptors** | `select_hbond_donor` / `select_hbond_acceptor` | ✅ via annotated CIF |
| **RASA (buried/exposed)** | `select_buried` / `select_exposed` | ✅ via annotated CIF |
| **Secondary structure** | — | 🔜 Planned |
| **CATH fold class** | `add_cath_code` | 🔜 Planned |
| **π-stacking** | — (annotated CIF) | 🔜 Planned |
| **π-cation** | — (annotated CIF) | 🔜 Planned |
| **Not-cysteine** | — (annotated CIF) | 🔜 Planned |

> **Planned** features are parsed and stored but the model is not yet trained to respond to them —
> passing them causes no error but has no effect.

---

## Inference Noise Parameters

The model uses **product-space flow matching** with two flows — backbone Cα and sidechain latents.
Stochasticity temperatures control diversity:

| Parameter | Config key | Controls |
| --- | --- | --- |
| Backbone temperature | `generation.model.bb_ca.simulation_step_params.sc_scale_noise` | Structural diversity |
| Latent temperature | `generation.model.local_latents.simulation_step_params.sc_scale_noise` | Sidechain/sequence diversity |

Higher values increase diversity but may reduce designability. The shipped defaults are `(0.3, 0.1)`
(backbone, latent). Roughly: `(0.1, 0.1)` stays close to the MPNN baseline with high designability;
`(0.5, 0.5)` shifts sequences toward the natural UniProt distribution; `(0.9, 0.9)` maximizes diversity.

```bash
python src/rfproteina/generate.py \
    inputs=data/benchmarks/monomer.json out_dir=./runs/diverse \
    generation.model.bb_ca.simulation_step_params.sc_scale_noise=0.5 \
    generation.model.local_latents.simulation_step_params.sc_scale_noise=0.5
```

---

## Sampling with TSR

**`vf_tsr`** (velocity field with temporal score rescaling) is the default sampler at `nsteps=100`.
It applies time-dependent score scaling that drives samples toward lower temperature near the end
of the trajectory. `tsr_k` controls how low the temperature drops (higher → lower); `tsr_sigma`
controls how quickly. `inference_base` ships pre-tuned defaults.

Because `inference_base`'s `sc_scale_noise`/`gt` defaults are nonzero, default `vf_tsr` sampling is
**stochastic**. For deterministic (ODE) sampling with no noise injected mid-trajectory:

```bash
generation.model.bb_ca.gt.mode=ode generation.model.local_latents.gt.mode=ode
```

---

## Output Files

Each design is a CIF with per-atom annotations recording how each residue was produced:

| Annotation | Level | Description |
| --- | --- | --- |
| `src_component` | Residue | Which part of the input spec a residue originated from |
| `mapped_to_coordinate_conditioned_region` | Atom | `True` for atoms mapped from a coordinate-conditioned motif |
| `mapped_to_coordinate_conditioned_region_token_lvl` | Residue | Residue-level aggregation — use to hold regions fixed in downstream MPNN redesign |
| `src_coordinate_conditioned_region` | Atom | Which input motif region provided each atom's conditioning |

---

## Troubleshooting

| Issue | Solution |
| --- | --- |
| `No .env file found for cluster 'digs'` | Set `export CLUSTER=local` and create `.ipd/.env.local` (see [Setup](../README.md#setup)) |
| `DATA_PATH` / checkpoint not found | `export DATA_PATH=...` and run `./scripts/download_example_data.sh "$DATA_PATH"` |
| `atomworks not found` | `git submodule update --init lib/atomworks` |
| `hydra not found` / import errors | Activate the venv: `source .venv/bin/activate` |
| Hydra config errors | Use `=` not `--`: `inputs=file.cif`, not `--inputs file.cif` |
| Out of memory | Lower `batch_size`; test your longest input at `batch_size=1` first |
| Segfault in rdkit/posebusters | uv's Python needs `LD_PRELOAD` set to the bundled `libstdc++.so.6` (see `Makefile`); `make test` sets it automatically |
| AF3 / refolding errors | Refolding & search need an AF3 apptainer — see [Evaluating Results](#evaluating-results) |
