# DNA binder design

Design transcription-factor-like proteins that bind DNA. Back to the
[skill](../../SKILL.md).

> [!WARNING]
> **Not fully supported in the current checkpoint (planned).** DNA-binder design
> runs without error but may produce low-quality results. Tell the user this
> before spending compute; treat any output as exploratory.

## What the task is

Condition on a DNA duplex (coordinate-conditioned, often with H-bond acceptor /
donor conditioning to base edges) and design a protein that binds a target site.

## Example inputs (`data/benchmarks/`)

| File | Keys | Notes |
| --- | --- | --- |
| `dna_binder_120_150_campaign.json` | `TATA_120_150`, ... | DNA cif + `contig`, `select_fixed_atoms`, base-edge `select_hbond_acceptor` |
| `dna_binder_denovo_val.json` | `7n5u_structure_context`, `7n5u_sequence_context`, `7m5w_*`, `7rte_*` | Structure- vs sequence-context variants |
| `dna_binder_short_site.json` | `7n5u_short_site_fixed_dna_HB-conditioned`, `..._uncond`, `7m5w_*`, `7rte_*` | Short target site, with/without H-bond conditioning |
| `paper_dna.json` | `7n5u_dimer_ori`, `7rte_dimer_ori`, ... | Paper benchmark (note: contains unsupported `ori_token`/`atomwise_hbond` fields — see rfd4-proteina-inputs) |

## Run

```bash
python src/rfproteina/generate.py \
    inputs=data/benchmarks/dna_binder_denovo_val.json \
    subset_to_examples=7n5u_structure_context \
    num_replicates=8 batch_size=8 out_dir=./runs/dna_binder \
    metrics_tags='[interchain_interactions]'
```

## Task-specific overrides & metrics

- **`interchain_interactions`** — protein-DNA H-bond / interaction adherence and
  counts (no AF3 container required; see **rfd4-proteina-evaluate**).
- There is no DNA-specific `*_refolding` tag; use `interchain_interactions` for
  interface adherence.

## Gotchas

- The planned-support caveat above is the headline gotcha — set expectations.
- `ori_token` and `atomwise_hbond` are unsupported compatibility-format
  fields. Prefer inputs that use supported annotations, and validate them with
  **rfd4-proteina-inputs**.
- Base-edge H-bond selections (`select_hbond_acceptor` / `select_hbond_donor`)
  are per-atom and easy to get wrong — build DNA inputs with
  **rfd4-proteina-inputs**.
