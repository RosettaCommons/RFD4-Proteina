# Antibody and nanobody design

Starting recommendations, especially for VHHs; adapt them to the target and project.

- **Rank by `oracle_ppi.min_ipae_coi`: lower is better, with values < 1.0 preferred.**
  The binding criterion allows values up to 2.0, but passing is a cutoff, not a ranking.
  Among otherwise comparable designs for the same target, prioritize lower values
  for selection and partial diffusion; do not treat all passers as equally good.
- Also aim for target-aligned
  North-CDR backbone RMSD < **1.5 Å** (`antibody_target_aligned_cdr_rmsd.rmsd`) and
  whole-binder backbone RMSD < **2.0 Å** (`antibody_target_aligned_backbone_rmsd.rmsd`).
- Prefer candidates passing **all antibody developability checks**
  (`antibody_developability.pass`); failures carry higher risk of expression difficulties.
- Use backbone-only target conditioning (`C_CRDGRP`) with full target sequence
  (`C_SEQ`). Crop the target around the intended interface to reduce spurious AF3 docks.
- Use the early high-recycle schedule, `recycle_strategy=high/front_loaded`, for
  every trajectory, including partial diffusion.
- Start with **large campaigns without search**, varying frameworks, CDR lengths,
  and hotspots. Search is useful when a particular dock or docking angle is desired.
  [SampleNearbyHotspots](../../src/rfproteina/transforms/hotspot_sampling.py) samples
  nearby exposed target atoms per trajectory; [vhh.yaml](vhh.yaml) enables it.
- Partially diffuse the best designs to increase sequence diversity while largely
  preserving the dock. This may also improve developability; rescore every child.

After [setup](../../docs/installation.rst), including AF3, run from the repository root:

```bash
# Check one design per supplied input, then increase num_replicates for the campaign.
rfd4-generate --config-path "$PWD/examples/antibody_design" --config-name vhh \
  num_replicates=1 batch_size=1 out_dir=./runs/vhh_pilot

# Paired VH/VL, all six CDRs designed against the same target.
rfd4-generate --config-path "$PWD/examples/antibody_design" --config-name vh_vl \
  num_replicates=1 batch_size=1 out_dir=./runs/vh_vl_pilot

# Refine a generated annotated parent; this config preserves its hotspot patch.
rfd4-generate --config-path "$PWD/examples/antibody_design" --config-name partial \
  inputs=/path/to/selected_generated_complex.cif num_replicates=100 out_dir=./runs/vhh_partial
```

All three configs select the high front-loaded schedule, `antibody_refolding`
and design/conditioning checks. Use fresh output directories; `num_replicates`
is per input.

## Supplied inputs

| Config | Structures | Designed |
| --- | --- | --- |
| [vhh.yaml](vhh.yaml) | [structures/vhh](structures/vhh) — four VHH frameworks against two targets (MZB1 `7AAH`, PMVK `3CH4`) | CDR1/2/3 |
| [vh_vl.yaml](vh_vl.yaml) | [structures/vh_vl](structures/vh_vl) — adalimumab Fv against MZB1 `7AAH` | All six CDRs (H1–H3, L1–L3) |

The VH/VL input uses the same MZB1 target crop as the VHH inputs, so the two can
be compared directly on one target.

**Developability applies to VHHs only.** `antibody_developability` scores a
single chain of interest with thresholds calibrated for single-domain VHHs, so
`vh_vl.yaml` leaves that tag off; `vhh.yaml` and `partial.yaml` keep it. Judge
paired-Fv designs on binding and the CDR/backbone RMSDs above.

See [Antibody metrics](../../docs/reference/antibody-metrics.md) for score
definitions, passing thresholds and output columns.

## Choosing the checkpoint

These configs inherit the checkpoint from [configs/inference.yaml](../../configs/inference.yaml),
which selects the general release on the Hugging Face Hub. That release is the
starting point and works for every config here.

An antibody fine-tune is also available: a LoRA fine-tune on a 33% general
interface / 67% antibody PDB mixture, merged into its base and exported as EMA
weights. It is a separate lineage from the general release — it was fine-tuned on
a pretrained base trained without TED dimers — so it is a self-contained
checkpoint rather than an adapter, and it cannot be layered onto the released
base. It is published as a native checkpoint rather than a model bundle, so it is
downloaded and then selected by path — `hf_ckpt_path` does not reach it.

```bash
hf download nvidia/RFD4-Proteina antibody/rfd4-proteina-antibody-2026-10-30-ema.ckpt \
  --revision 621b785abf9d61dd24f4b81ae4dbd1330728f030 --local-dir ./checkpoints
```

```bash
rfd4-generate --config-path "$PWD/examples/antibody_design" --config-name vhh \
  hf_ckpt_path=null ckpt_path=./checkpoints/antibody \
  ckpt_name=rfd4-proteina-antibody-2026-10-30-ema.ckpt \
  num_replicates=1 batch_size=1 out_dir=./runs/vhh_pilot
```

Verify the download against the published
`antibody/rfd4-proteina-antibody-2026-10-30-ema.ckpt.sha256`
(`6abf4ed2…`) before a large campaign.

Both `ckpt_path` and `ckpt_name` are required when `hf_ckpt_path=null`.

**Keep the `-ema.ckpt` suffix.** Inference needs EMA weights, and the loader
picks them by filename: a name ending in `-ema.ckpt` is read as EMA weights
directly, while any other name is searched for the `SimpleEMA` training state.
Released exports carry EMA weights only — that state has been stripped — so
renaming one to drop the suffix makes it fail to load rather than silently
loading the wrong weights.

Keep the matching autoencoder: this checkpoint pairs with the 2026-10-19 release
AE, which `inference.yaml` already selects. The checkpoint carries
`nn.attention_type=torch` from LoRA training; the merged weights do not depend on
it, so override `nn.attention_type` if you want a fused backend.
