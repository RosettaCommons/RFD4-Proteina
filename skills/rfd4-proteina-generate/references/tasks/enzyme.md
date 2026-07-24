# Enzyme design

Design enzymes with catalytic residues arranged around a ligand active site — a
motif-scaffolding problem where the motif is a set of catalytic residues
(usually unindexed, tip-atom or sidechain conditioned) plus a ligand. Back to the
[skill](../../SKILL.md).

## Example inputs (`data/benchmarks/`)

| File | Notes |
| --- | --- |
| `mcsa_40.json` | 40 M-CSA active sites *with ligands* (e.g. `M0040_13pk`, `M0630_1j79`); `ligand`, `unindex`, per-residue `select_fixed_atoms`, `length` 180 |
| `mcsa_40_fixed_ligand.json` | Same sites with the ligand pose fixed |
| `mcsa_40_no_ligand.json` | Same sites without the ligand (catalytic residues only) |
| `enz_design.json` | `lenacap_partial_sc`, `rubisco_ts_tip_atoms` (tip-atom + H-bond + buried RASA), `rubisco_crystal_full_sc`; annotated-CIF inputs under `enz_design_cifs/` |

## Run

```bash
# One M-CSA active site with its ligand
python src/rfproteina/generate.py \
    inputs=data/benchmarks/mcsa_40.json \
    subset_to_examples=M0630_1j79 \
    num_replicates=8 batch_size=8 out_dir=./runs/enzyme \
    metrics_tags='[ame_refolding,motif_scaffold_refolding]'
```

## Task-specific overrides & metrics

- **`ame_refolding`** — AF3 refolding for enzyme (M-CSA) active-site benchmarks.
- **`motif_scaffold_refolding`** — motif RMSD of the catalytic residues.
- **`interchain_interactions`** — if the input specifies H-bonds to the ligand,
  reports adherence (see `rubisco_ts_tip_atoms`, which sets `select_hbond_*`).
- All refolding tags need the AF3 container (see **rfd4-proteina-evaluate**).

## Gotchas

- **Ligand present vs absent changes the problem** — pick the right file
  (`mcsa_40.json` vs `mcsa_40_no_ligand.json`). With a ligand, the model designs
  around a real active-site chemistry; without, it only places catalytic
  geometry.
- **Never split a resname on `;` or `:`** — `T:S`, `H:H`, `K:X` in
  `rubisco_ts_tip_atoms` are *literal* residue names; `ligand` splits on `,` only.
- Tip-atom (`select_fixed_atoms: {"A175": "TIP"}`) vs full-sidechain vs
  backbone-only conditioning are distinct design decisions — build these inputs
  via **rfd4-proteina-inputs**, which interrogates exactly this.
- Charges/bonds on a PDB-derived ligand are often wrong (aromaticity lost);
  validate the input before spending compute.
