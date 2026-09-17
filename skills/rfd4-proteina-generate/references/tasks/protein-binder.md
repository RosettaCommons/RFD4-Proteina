# Protein binder design

Design a binder to a target protein at specified hotspot residues. Back to the
[skill](../../SKILL.md).

## What the task is

Condition on a target protein's structure (coordinate-conditioned) and a set of
**hotspot** residues (the `C_HOT` annotation in a CIF), then design a new binder
chain against that interface.

## Example inputs (`data/benchmarks/`)

| File | Notes |
| --- | --- |
| `ppi/ppi_all_atom/` | One annotated CIF per target, including `rfd3_insulinr`, `rfd3_pdl1`, `01_PD1`, and `30_SC2RBD` |

## Run

```bash
./src/rfproteina/generate.py \
    inputs=data/benchmarks/ppi/ppi_all_atom/rfd3_pdl1.cif.zst \
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
- Hotspots are set by the input's `C_HOT` annotation, not on the CLI. To build a new target/hotspot input, use
  **rfd4-proteina-inputs**.
- The designed (binder) chain is the "chain of interest" for clustering; it is
  identified automatically from the conditioning annotations.
