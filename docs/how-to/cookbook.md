# Cookbook

Copy-paste recipes, one per task. Each has the YAML [condition spec](input_prep/yaml-json.md) that
says what to design and the `key=value` Hydra overrides that matter for that task. Run everything from
the repository root; save a spec next to where you run it, because `input` paths resolve relative to
the spec file. New here? Start with [Design a binder in 10 lines](binder-in-10-lines.md).

Commands with a refolding tag in `metrics_tags` (`ppi_refolding`, `ame_refolding`,
`motif_scaffold_refolding`, `sm_binder_refolding`, `unconditional_refolding`) need the AlphaFold 3 /
OpenFold3 (or AlphaFold 2) setup from [installation](../installation.rst); add `'~metrics'` instead
to generate without scoring.

Every option below is documented in the [CLI options reference](../reference/cli-options.md). The
common ones:

| Override | Meaning |
| --- | --- |
| `num_replicates=N` | designs per input |
| `batch_size=N` | designs generated in parallel; test your longest input at `1` first |
| `generation.args.nsteps=N` | sampling steps (default 60) |
| `recycle_strategy=<budget>/<shape>` | trunk recycling; use the task's medium preset ([table](run-the-model.md#recycling-schedules)) |
| `metrics_tags='[...]'` | inline metrics; refolding tags need the AlphaFold 3 / OpenFold3 setup |
| `'~metrics'` | no scoring at all (quoted so zsh keeps the `~`) |

## Monomer

A monomer is an unconditioned chain: the annotated CIFs in `data/benchmarks/monomer/` hold one
expandable chain (100, 150 and 250 residues). A spec cannot start from nothing, so point it at the
CIF, or pass the CIF directly.

```yaml
monomer:
  input: data/benchmarks/monomer/monomer-short.cif.zst
  input_as_annotated_cif: true
```

```bash
rfd4-generate inputs=monomer.yaml num_replicates=16 batch_size=8 out_dir=./runs/monomer \
    metrics_tags='[unconditional_refolding]'
# equivalent, no spec:
rfd4-generate inputs=data/benchmarks/monomer/monomer-short.cif.zst num_replicates=16 batch_size=8 out_dir=./runs/monomer
```

For more diverse or more conservative samples, lower or raise
`generation.model.bb_ca.simulation_step_params.tsr_k` ([temperature](run-the-model.md#inference-temperature-parameters)).

## Protein binder

The spec from [Design a binder in 10 lines](binder-in-10-lines.md), saved as `binder.yaml`:

```yaml
pdl1_binder:
  input: docs/examples/inputs/5o45_cropped.cif.zst
  contig: "50-120, /0, A17-131"
  conditions:
    C_CRDGRP: {1: [{select: "sel('A/*/*/[N,CA,C,O]')"}]}
    C_SEQ: {true: [{select: "sel('A')"}]}
    C_HOT: {true: [{select: "sel('A/*/56/[CG,OH]')"}, {select: "sel('A/*/115/[CG,SD]')"}, {select: "sel('A/*/123/[CD2,OH]')"}]}
```

```bash
rfd4-generate inputs=binder.yaml recycle_strategy=medium/front_loaded \
    num_replicates=8 batch_size=8 out_dir=./runs/binder \
    metrics_tags='[ppi_refolding,hotspot_conditioning]'
```

Add `save_chain_of_interest=true` if you will cluster binders structurally. To change the hotspots,
edit the `C_HOT` selections; to change the target, change `input`, the target residue range in
`contig` and the selections.

## Enzyme and motif scaffolding

The [M-CSA benchmark](https://github.com/RosettaCommons/RFD4-Proteina/tree/main/data/benchmarks/mcsa)
inputs are annotated CIFs with the catalytic residues, ligand and scaffold length already set. Wrap one
in a spec with `input_as_annotated_cif: true` so existing conditions are kept:

```yaml
rubisco_site:
  input: data/benchmarks/mcsa/mcsa_37/M0907_1rbl.cif.zst
  input_as_annotated_cif: true
```

```bash
rfd4-generate inputs=enzyme.yaml recycle_strategy=medium/midpeak \
    num_replicates=8 batch_size=8 out_dir=./runs/enzyme \
    metrics_tags='[ame_refolding,motif_scaffold_refolding]'
```

For your own motif, build the structure with `contig` and `C_CRD`/`C_IDX` conditions as in the
[spec guide](input_prep/yaml-json.md#examples). Pure motif scaffolding without a ligand uses
`metrics_tags='[motif_scaffold_refolding,3d_coordinate_conditioning]'`.

## Antibody and nanobody

The supplied VHH inputs have framework, target and CDR conditioning already annotated, and sample new
hotspot patches per design. Loading them as a spec stops at a residue-numbering safeguard, so run them
through the example config in `examples/antibody_design/` instead, which sets the right transforms,
`recycle_strategy=high/front_loaded` and metric tags:

```bash
rfd4-generate --config-path "$PWD/examples/antibody_design" --config-name vhh \
    num_replicates=8 batch_size=8 seed=42 out_dir=./runs/vhh '~metrics'
```

The example config scores designs with the AlphaFold 3 / OpenFold3 oracle, which needs that
environment from setup (`AF3_IMAGE`); without it the run writes the designs and then fails. `'~metrics'`
skips scoring. Remove it once the oracle is set up.

Single input instead of all eight: `inputs=examples/antibody_design/structures/vhh/<file>.cif.zst`
after the config name. The [VHH campaign guide](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/examples/antibody_design/README.md)
covers scaling and refinement.

## Small-molecule binder

```yaml
sam_binder:
  input: data/benchmarks/small_molecule_binder/sm_binder/SAM.cif.zst
  input_as_annotated_cif: true
```

```bash
rfd4-generate inputs=sm_binder.yaml recycle_strategy=medium/midpeak \
    num_replicates=8 batch_size=8 out_dir=./runs/sm_binder \
    metrics_tags='[sm_binder_refolding,interchain_interactions]'
```

`SAM.cif.zst` is a 150-residue expandable chain plus the SAM ligand. To build a spec around your own
ligand, see the pocket example in [`condition_spec_complicated.yaml`](input_prep/examples/condition_spec_complicated.yaml).
For hydrogen-bond conditioning use `data/benchmarks/small_molecule_binder/SAM_hbonds.cif.zst`.

## Partial diffusion

Refine or diversify a structure you already have by re-noising it part-way. Use a generated complex
(`*_generated.cif`, not an oracle refold) as the input:

```yaml
refine:
  input: runs/binder/inference_samples/<example>/<example>_generated.cif
  input_as_annotated_cif: true
```

```bash
rfd4-generate inputs=refine.yaml generation.args.partial_trajectory_fraction_steps=0.3 \
    num_replicates=8 out_dir=./runs/refine
```

`0.3` reruns the last 30% of the trajectory: close to the input. Larger values drift further. To keep
the backbone and resample only sequence, use `generation=inverse_folding recycle_strategy=medium/flat`
instead. See [Partial diffusion and inverse folding](partial-diffusion.md).

## From flags to YAML

Every override is a path into the composed config. `num_replicates=8` sets the top-level
`num_replicates`; `generation.args.nsteps=80` sets `nsteps` under `generation: args:`. Print the
result of any command without running it:

```bash
rfd4-generate inputs=binder.yaml num_replicates=8 generation.args.nsteps=80 --cfg job
```

Group selectors such as `recycle_strategy=medium/front_loaded` choose a config file;
`+guidance=contract_ca` adds a key that has no default (the shipped preset is a mechanics test, see
[Metric guidance](advanced/metric-guidance.md)).

To keep a set of flags, copy
[`configs/inference_experiment/TEMPLATE.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/inference_experiment/TEMPLATE.yaml)
or save a file like this anywhere, for example `pdl1_binder.yaml`:

```yaml
defaults:
  - inference
  - override /recycle_strategy: medium/front_loaded
  - _self_

hydra:
  searchpath:
    - file://${oc.env:PROJECT_ROOT}/configs

inputs: ${oc.env:PROJECT_ROOT}/binder.yaml
run_name_: pdl1_binder
out_dir: ./runs/pdl1_binder

num_replicates: 8
batch_size: 8
metrics_tags: [ppi_refolding, hotspot_conditioning]

generation:
  args:
    nsteps: 80
```

```bash
rfd4-generate --config-path "$PWD" --config-name pdl1_binder
rfd4-generate --config-path "$PWD" --config-name pdl1_binder num_replicates=100   # override on top
```

Command-line overrides win over the file. Keep `_self_` last in `defaults` so the file wins over the
recycle preset, which also sets `nsteps`. A file placed in `configs/inference_experiment/` is selected
with `inference_experiment=<name>` instead, but there the recycle preset is applied after the file,
so pass `recycle_strategy=` and `generation.args.nsteps=` on the command line.
