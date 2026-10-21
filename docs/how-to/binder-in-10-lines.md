# Design a binder in 10 lines

One YAML file, one command. This page designs a de novo protein binder against PD-L1 using the
inputs that ship in the repository. It needs a clone of the repository and a GPU; see the
[Quickstart](../quickstart.rst) if you have not installed the model yet.

## The spec

Save this as `binder.yaml` in the repository root (the `input` path is resolved relative to the
spec file):

```yaml
binder:
  input: docs/examples/inputs/5o45_cropped.cif.zst   # PD-L1, chain A, residues 17-131
  contig: "50-120, /0, A17-131"                      # a new 50-120 residue chain, then the target
  conditions:
    C_CRDGRP: {1: [{select: "sel('A/*/*/[N,CA,C,O]')"}]}   # target backbone geometry, free pose
    C_SEQ: {true: [{select: "sel('A')"}]}                   # target sequence is known
    C_HOT: {true: [{select: "sel('A/*/56/[CG,OH]')"}, {select: "sel('A/*/115/[CG,SD]')"}, {select: "sel('A/*/123/[CD2,OH]')"}]}  # hotspots
```

JSON with the same keys works too; YAML is just easier to read and comment.

## The command

```bash
rfd4-generate inputs=binder.yaml num_replicates=4 batch_size=4 out_dir=./runs/binder '~metrics'
```

That is the whole run. `inputs` takes the spec, `num_replicates` is the number of designs,
`batch_size` is how many are generated at once, and `'~metrics'` skips scoring so no refolding
environment is needed. The quotes stop zsh from reading `~metrics` as a home directory. For a quick
smoke test also add `compile_lengths=null` to skip the `torch.compile` warm-up.

## What you get

- `runs/binder/inference_samples/`: one generated complex per design (`*_generated.cif`), holding the
  new binder chain and the PD-L1 target. The conditioned input is saved next to each design as
  `*_input.cif`.
- Open the CIFs in any structure viewer. The binder is the chain that was not in the input.

This spec is the same one as [`condition_spec_ppi.yaml`](input_prep/examples/condition_spec_ppi.yaml),
written compactly. It follows the repository's default for protein binders: target backbone-only
internal geometry (`C_CRDGRP`), full target sequence (`C_SEQ`) and hotspots (`C_HOT`).

## Next steps

- **Score the designs.** Replace `'~metrics'` with `metrics_tags='[ppi_refolding,hotspot_conditioning]'`
  and set `recycle_strategy=medium/front_loaded`; refolding needs the AlphaFold 3 / OpenFold3
  environment from setup. See [Evaluate with metrics](evaluate-with-metrics.md).
- **Other tasks.** The [cookbook](cookbook.md) has one-liners for monomers, motif scaffolding, enzymes,
  small molecules, nanobodies and partial diffusion.
- **Change the spec.** [YAML/JSON condition spec](input_prep/yaml-json.md) explains each line above.

## Gallery

Every task has an input and generated designs on the model repository:
[`nvidia/RFD4-Proteina`, `examples/`](https://huggingface.co/nvidia/RFD4-Proteina/tree/main/examples).
The table pairs each task with the matching input in this repository and its cookbook recipe.

| Task | Input in this repository | Recipe |
| --- | --- | --- |
| Monomer | `data/benchmarks/monomer/monomer-short.cif.zst` | [cookbook](cookbook.md#monomer) |
| Protein binder | `docs/examples/inputs/5o45_cropped.cif.zst` (this page) | [cookbook](cookbook.md#protein-binder) |
| Enzyme motif | `data/benchmarks/mcsa/mcsa_37/` | [cookbook](cookbook.md#enzyme-and-motif-scaffolding) |
| Nanobody | `examples/antibody_design/structures/vhh/` | [cookbook](cookbook.md#antibody-and-nanobody) |
| Small-molecule binder | `data/benchmarks/small_molecule_binder/sm_binder/` | [cookbook](cookbook.md#small-molecule-binder) |
| Partial diffusion | any generated complex | [cookbook](cookbook.md#partial-diffusion) |

Designs are computational predictions. Binding needs experimental validation.
