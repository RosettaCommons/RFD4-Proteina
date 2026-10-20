# A VHH design campaign

Build a diverse collection of VHHs predicted to bind your target: discover
productive combinations, scale for diversity, then refine within clusters.

Complete [setup](../../docs/installation.rst) first, including AF3 for scoring.
Use `recycle_strategy=high/front_loaded`, already selected by these configs.

## 1. Discover productive combinations with best-of-N

Generate N independent candidates across VHH frameworks, CDR lengths, and target
hotspots, then score them and retain the strongest. CATH conditioning
supplies the immunoglobulin fold context; together with the VHH framework, it
tells the model what kind of binder to build, without guaranteeing success.

The [eight supplied complexes](structures/vhh) pair four frameworks with two
cropped targets, PMVK and MZB1. Inputs fix the target and preserve the framework's
internal geometry while allowing it to move. Frameworks vary across files;
CDR lengths and nearby hotspot patches are sampled anew for each candidate.

First check one candidate per input, from the repository root:

```bash
rfd4-generate --config-path "$PWD/examples/antibody_design" --config-name vhh \
  num_replicates=1 batch_size=1 seed=42 out_dir=./runs/vhh_pilot
```

Inspect structures and confirm scoring works before increasing `num_replicates`.
This count is **per input**: 100 across eight inputs means 800 attempts.
Use a fresh output directory for each screen.

## 2. Compare, then scale for diverse passers

Compare **within each target**: which frameworks, realized CDR lengths, and hotspot
patches yield passers? Review pass rates, attempt counts, and structural clusters;
many near-duplicates are not many distinct solutions. Keep failed evaluations in
the attempt count. A “passer” meets computational criteria; binding still requires
experimental validation.

Launch production-scale runs for promising combinations, retaining several
alternatives. Keep sampling where useful; save a new annotated input to fix a
hotspot patch or narrow CDR lengths (see the [input guide](../../docs/workflow/inputs.md)).
To retain a chosen hotspot patch, set `generation.dataset.sampling_transforms=null`.
Use `inputs=/path/to/selected_inputs` and increase `num_replicates` for production.
Cluster passers and continue until satisfied with diversity.

## 3. Refine and proliferate within clusters

Choose annotated generated parents from selected clusters for partial diffusion.
This explores nearby designs to proliferate passers within a family and potentially
improve their metrics—not replace the earlier search for distinct solutions.

```bash
rfd4-generate --config-path "$PWD/examples/antibody_design" --config-name partial \
  inputs=/path/to/selected_generated_complex.cif \
  num_replicates=100 out_dir=./runs/vhh_refinement
```

Use the generated annotated complex, not an oracle-refolded structure. The partial
config retains its sampled hotspots, target, framework conditioning, and realized
CDR lengths. Track each child's parent, score children independently, and retain
them only if they meet your criteria; improvement is not guaranteed.
