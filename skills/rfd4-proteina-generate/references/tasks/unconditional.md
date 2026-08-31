# Unconditional generation

Generate novel protein structures (and multi-chain complexes) from scratch — no
motif, no ligand, no target. Back to the [skill](../../SKILL.md).

## Example inputs (`data/benchmarks/`)

| File | Keys | What it is |
| --- | --- | --- |
| `monomer` | `monomer-short` (100), `monomer-medium` (150), `monomer-long` (250) | Plain single-chain monomers by length |
| `monomer_helical` | `monomer-short/medium/long` | Monomers with a CATH mainly-alpha code (`add_cath_code` — planned, has no effect yet) |
| `monomer_ig` | `monomer-short` | Monomer with an immunoglobulin CATH code (planned) |
| `multipolymer_uncond` | `multipolymer_120p_70r`, `multipolymer_90p_60r_25d_25d`, `rna_50/60/100/150`, `dna_60`, `protein_60` | Multi-polymer / protein+RNA+DNA complexes and nucleic-acid-only chains |

> The `add_cath_code` conditioning in `monomer_helical` / `monomer_ig` is a
> **planned** conditioning — they are parsed but the current checkpoint does not
> respond to them.

## Run

```bash
# Smoke test: one short monomer
python src/rfproteina/generate.py \
    inputs=data/benchmarks/monomer subset_to_examples=monomer-short \
    num_replicates=1 out_dir=./runs/uncond_smoke

# Real run: 80 designs of each monomer example, 8 at a time
python src/rfproteina/generate.py \
    inputs=data/benchmarks/monomer \
    num_replicates=80 batch_size=8 out_dir=./runs/monomers
```

## Task-specific overrides & metrics

- **Diversity.** The main dial for unconditional work. Sweep
  `generation.model.bb_ca.simulation_step_params.sc_scale_noise` and
  `generation.model.local_latents.simulation_step_params.sc_scale_noise`
  together — `(0.1,0.1)` high designability, `(0.5,0.5)` UniProt-like,
  `(0.9,0.9)` max diversity (see [overrides](../overrides.md)).
- **Metrics.** `metrics_tags='[unconditional_refolding]'` for AF3 refolding +
  pLDDT/scRMSD (needs an AF3 container — see **rfd4-proteina-evaluate**).
- **Length** is set inside the input (`length` field), not on the CLI. For a new
  length, build an annotated CIF input via **rfd4-proteina-inputs**.

## Gotchas

- `batch_size=8` is a good start, but the longest example (`monomer-long`, 250
  residues; multipolymer up to 200) should be confirmed at `batch_size=1` first.
- Nucleic-acid-only or mixed complexes are fully supported here (unlike the
  *DNA-binder* task, which is planned).
