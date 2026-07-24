# Small-molecule binder design

Design a protein that binds a small-molecule ligand. The input must contain the
ligand; the model designs a protein pocket around it. Back to the
[skill](../../SKILL.md).

## Example inputs (`data/benchmarks/`)

| File | Keys | Notes |
| --- | --- | --- |
| `sm_binder.json` | `FAD`, `IAI`, `OQO`, `SAM` | Ligand named via `ligand`, `contig` 150, `select_fixed_atoms: false` (model invents pose/conformer) |
| `sm_binder_fixed_ligand.json` | — | Ligand pose fixed |
| `sm_binder_hbonds.json` / `sm_binder_hbonds_fixed_ligand.json` | — | Adds H-bond donor/acceptor conditioning to the ligand |
| `rna_sm_aptamer.json` | — | Small-molecule aptamer (nucleic-acid binder) variant |

## Run

```bash
# One small-molecule binder target
python src/rfproteina/generate.py \
    inputs=data/benchmarks/sm_binder.json \
    subset_to_examples=SAM \
    num_replicates=8 batch_size=8 out_dir=./runs/sm_binder \
    metrics_tags='[sm_binder_refolding,interchain_interactions]'
```

## Task-specific overrides & metrics

- **`sm_binder_refolding`** — AF3 refolding + binding metrics.
- **`interchain_interactions`** — H-bond / pi-stack / salt-bridge etc. adherence
  and counts; pair with the `sm_binder_hbonds*.json` inputs to score H-bond
  adherence to the ligand.
- Refolding tags need the AF3 container (see **rfd4-proteina-evaluate**).

## Gotchas

- **The ligand must be in the input** — this task cannot be run from a bare
  length spec. Use the ligand-carrying benchmarks or build an annotated CIF with
  the ligand via **rfd4-proteina-inputs**.
- `select_fixed_atoms: false` means **nothing is coordinate-conditioned** — the
  model invents both the pose and the conformer. Confirm that is intended; fix
  the pose (`sm_binder_fixed_ligand.json`) if you want the ligand held in place.
- PDB-derived ligands frequently arrive with wrong bond orders / missing charges;
  validate with **rfd4-proteina-inputs** before committing compute.
