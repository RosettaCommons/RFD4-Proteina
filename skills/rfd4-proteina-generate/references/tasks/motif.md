# Motif scaffolding

Build a complete protein around one or more structural motifs. A motif is
residues carrying **both** coordinate (`C_CRD`) and sequence (`C_SEQ`)
conditioning; the model designs the surrounding scaffold. Back to the
[skill](../../SKILL.md).

## Example inputs (`data/benchmarks/`)

| File | Notes |
| --- | --- |
| `motif_scaffold_afdb.json` | 6 tasks (e.g. `M0630_1j79_motif`, `M0157_1qh5_motif`); unindexed motifs with `select_fixed_atoms` per residue, `length` 180 |
| `motif_scaffold_aa_tasks_indexed.json` | Indexed-motif variants |
| `motif_scaffold_aa_tasks_unindexed.json` | Unindexed-motif variants (model chooses chain position) |
| `motif_scaffold_aa_tasks_indexed_internal_coords.json` / `..._unindexed_internal_coords.json` | Internal-coordinate groups (motif's internal geometry without fixing where it sits) |
| `motif_scaffold_aa_tasks_indexed_all_resolved.json` / `..._unindexed_all_resolved.json` | Fully-resolved variants |

## Run

```bash
# One AFDB motif task
python src/rfproteina/generate.py \
    inputs=data/benchmarks/motif_scaffold_afdb.json \
    subset_to_examples=M0630_1j79_motif \
    num_replicates=8 batch_size=8 out_dir=./runs/motif \
    metrics_tags='[motif_scaffold_refolding,3d_coordinate_conditioning]'
```

## Task-specific overrides & metrics

- **`motif_scaffold_refolding`** — AF3 refolding + motif RMSD.
- **`3d_coordinate_conditioning`** — coordinate RMSD of the conditioned motif.
- Both refolding tags need the AF3 container (see **rfd4-proteina-evaluate**); plain
  scaffolding (no tags) needs none.
- **Indexed vs unindexed** is a property of the input, not a CLI flag: indexed
  pins the motif to a chain position; unindexed lets the model place it. Choose
  when building the input.

## Gotchas

- **Length ranges are re-rolled per design.** `"180-180"` is degenerate (one
  length); a real range like `"150-350"` resamples length each replicate — the
  motif residues are *not* subtracted from the count.
- Building a new motif input (which atoms to fix, index vs unindex, internal
  coordinate groups) is exactly what **rfd4-proteina-inputs** is for — hand off
  rather than editing JSON by hand. `select_fixed_atoms` semantics are subtle
  (sidechain atoms leak identity via `C_SEQ`).
