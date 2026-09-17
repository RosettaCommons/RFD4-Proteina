# DNA binder design

Design transcription-factor-like proteins that bind DNA. Back to the
[skill](../../SKILL.md).

## What the task is

Condition on a DNA duplex (coordinate-conditioned, often with H-bond acceptor /
donor conditioning to base edges) and design a protein that binds a target site.

## Example inputs (`data/benchmarks/`)

| File | Keys | Notes |
| --- | --- | --- |
| `dna_binder/dna_binder_denovo_val` | `7n5u_structure_context`, `7n5u_sequence_context`, `7m5w_*`, `7rte_*` | Structure- vs sequence-context variants |

## Run

```bash
python src/rfproteina/generate.py \
    inputs=data/benchmarks/dna_binder/dna_binder_denovo_val \
    subset_to_examples=7n5u_structure_context \
    num_replicates=8 batch_size=8 out_dir=./runs/dna_binder \
    metrics_tags='[dna_refolding,interchain_interactions]'
```

## Task-specific overrides & metrics

- **`interchain_interactions`** — protein-DNA H-bond / interaction adherence and
  counts (no AF3 container required; see **rfd4-proteina-evaluate**).
- **`dna_refolding`** — AF3 refolding and DNA-interface quality metrics; requires
  the configured AF3/OpenFold3 container.

## Gotchas

- The retired `paper_dna` benchmark carried `ori_token` and `atomwise_hbond`,
  which are **dead knobs** in the current path (the model does not use an ORI
  token). Use the de novo inputs. See **rfd4-proteina-inputs** for the
  catalog of known-bad legacy fields.
- Base-edge H-bond selections (`select_hbond_acceptor` / `select_hbond_donor`)
  are per-atom and easy to get wrong — build DNA inputs with
  **rfd4-proteina-inputs**.
