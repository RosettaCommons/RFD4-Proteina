# Protein binder design

Design a binder to a target protein at specified hotspot residues. Back to the
[skill](../../SKILL.md).

> [!WARNING]
> **Not fully supported in the current checkpoint (planned).** Protein-binder
> design runs without error but may produce low-quality results. Tell the user
> this before spending compute; treat any output as exploratory.

## What the task is

Condition on a target protein's structure (coordinate-conditioned) and a set of
**hotspot** residues (`select_hotspots` in JSON/YAML, the `C_HOT` annotation in a
CIF), then design a new binder chain against that interface.

## Example inputs (`data/benchmarks/`)

| File | Notes |
| --- | --- |
| `cleaned_ppi.yaml` | One structure per target; keys like `rfd3_insulinr`, `rfd3_pdl1`, `01_PD1`, `30_SC2RBD`. Each has `contig` (binder length + target chain) and `select_hotspots` |
| `complexa_paper_ppi.yaml`, `all_ppi_helical.yaml`, `all_ppi_ig.yaml` | Larger PPI benchmark sets |
| `complexa_targets/` | Target CIFs referenced by the YAMLs |

## Run

```bash
python src/rfproteina/generate.py \
    inputs=data/benchmarks/cleaned_ppi.yaml \
    subset_to_examples=rfd3_pdl1 \
    num_replicates=8 batch_size=8 out_dir=./runs/ppi \
    metrics_tags='[ppi_refolding,hotspot_conditioning]'
```

## Task-specific overrides & metrics

- **`ppi_refolding`** — AF3 refolding + target templating + interface metrics.
- **`hotspot_conditioning`** — hotspot obedience (requires hotspot input).
- Both need the AF3 container (see **rfd4-proteina-evaluate**).
- `save_chain_of_interest=true` if you plan to cluster designed binders by
  structure (foldseek).

## Gotchas

- The planned-support caveat above is the headline gotcha — set expectations.
- Hotspots are set in the input (`select_hotspots`, per-residue atom selections
  like `TIP` or `ALL`), not on the CLI. To build a new target/hotspot input, use
  **rfd4-proteina-inputs**.
- The designed (binder) chain is the "chain of interest" for clustering; it is
  identified automatically from the conditioning annotations.
