# Enzyme design

Design enzymes with catalytic residues around a ligand active site. The input is a **theozyme** — a
handful of catalytic residues and a substrate, with no scaffold — and the model builds a protein that
holds them in place. For a hands-on walkthrough see the
[enzyme design tutorial](tutorials/enzyme_design_tutorial.ipynb); this page is the quick reference.

## Conditions

A theozyme can be represented as an [annotated CIF](annotated_cif_inputs.md). Condition **only the
atoms that belong to the theozyme** — the catalytic residues and the substrate — with the three
conditions below; every other atom is left unconditioned for the model to design.

| Condition | Granularity | In a theozyme |
| --- | --- | --- |
| `C_CRD` — coordinate | per **atom** | Pin the atoms that define the active-site geometry — a catalytic residue's functional atoms, the substrate's reactive core. Partial by design: leave flexible parts (e.g. a phosphate tail) free. |
| `C_SEQ` — sequence | per **residue** | Fix a residue's (or the substrate's) identity. All-or-nothing within a residue; across a motif you can reveal it on some residues and not others. |
| `C_IDX` — index | per **residue** | On by default. Turn it **off** on the catalytic residues so the model places them freely along the new chain — an *unindexed* motif. |

So a catalytic residue typically fixes its functional atoms (`C_CRD`) and identity (`C_SEQ`) while
being left **unindexed** (`C_IDX` off); the substrate fixes its reactive atoms and identity and simply
keeps its default index. Build a theozyme with the PyMOL plugin or programmatically — see
[Annotated CIF inputs](annotated_cif_inputs.md); the legacy JSON path is the
[input specification](input.md).

## Command line

Generate scaffolds around the theozyme:

```bash
./src/rfproteina/generate.py \
    inputs=theozyme.cif \
    num_replicates=25 batch_size=8 \
    metrics_tags=[design,unindexed,3d_coordinate_conditioning,ame_refolding] \
    out_dir=./runs/enzyme
```

`num_replicates` scaffolds are generated and scored by the **AME** benchmark's pass criteria — AME is the in-silico enzyme
motif-scaffolding benchmark used to evaluate de novo enzyme design and atom-level active-site
placement. A design passes when it clears the refolded backbone RMSD, motif all-atom RMSD, and
clash thresholds computed inline by the `metrics_tags` above. See the
[Inference guide](inference.md) and [parameters](parameters.md) for the general workflow and every knob.

Once you have a scaffold you like, carry its conditions forward (see [Annotations](#annotations)) and
refine or diversify it with either follow-up mode.

### Inverse folding

Redesign the sequence on the fixed backbone, holding the catalytic motif:

```bash
./src/rfproteina/generate.py \
    inputs=best_conditioned.cif \
    generation=inverse_folding \
    num_replicates=8 batch_size=8 \
    metrics_tags=[ame_refolding,design,unindexed] \
    out_dir=./runs/if
```

### Partial diffusion

Re-diffuse from a noised copy of the design for extra structural diversity:

```bash
./src/rfproteina/generate.py \
    inputs=best_conditioned.cif \
    generation.args.partial_trajectory_fraction_steps=0.3 \
    num_replicates=8 batch_size=8 \
    metrics_tags=[ame_refolding,design,unindexed,partial_diffusion] \
    out_dir=./runs/partial
```

See [Inverse Folding](inference.md#inverse-folding) and
[Partial Diffusion](inference.md#partial-diffusion) for the parameters and semantics.

### Search (custom objectives)

The runs above sample freely. To steer toward a **custom objective** instead — during either hit
design or refinement — use **reward-guided search**: generate many candidates and keep the ones a
structure-oracle reward (e.g. AF3 pLDDT, or a metric you define) scores highest. It is turned on with a search
`inference_experiment` config; the shipped example is a good starting point:

```bash
./src/rfproteina/generate.py inference_experiment=inference_beam_example
```

Reward-guided search layers on top of any generation mode: scaffold generation, inverse folding, and
partial diffusion can each run under it. Inverse folding and partial diffusion are the only pair that
cannot be combined.

See [Reward-Guided Search](search.md) for the beam-search parameters and how to define a reward for
your own objective, then point it at your theozyme.

## Annotations

The designed chain is renumbered from scratch, so the catalytic residues no longer sit at their input
indices. Each output CIF records where the conditioned motif landed:

| Annotation | Level | Use |
| --- | --- | --- |
| `mapped_to_coordinate_conditioned_region` | Atom | Atoms mapped from the input's coordinate-conditioned motif |
| `mapped_to_coordinate_conditioned_region_token_lvl` | Residue | Residue-level aggregation |

These are written to the output as `_atom_site.*` columns
(`mapped_to_coordinate_conditioned_region` ⇄ `src-coordinate-conditioned-region`, where `.` marks an
unconditioned atom) and reload with the file. Because they are ordinary annotations, you can pull out
any region with the **[atomworks selection syntax](https://rosettacommons.github.io/atomworks/latest/io/utils/selection_syntax.html)** —
`aa.mask(...)`/`aa.idxs(...)` evaluate a boolean query in which annotation names are variables:

```python
from atomworks.io import parse
from rfproteina.utils.parse_presets import ANNOTATED_CIF

aa = parse("design.cif", config=ANNOTATED_CIF)["assemblies"]["1"][0]

# Annotation names are variables in the query; index the array with the mask to subset it.
motif  = aa[aa.mask("mapped_to_coordinate_conditioned_region & ~hetero")]  # the catalytic residues
ligand = aa[aa.mask("hetero")]                                             # the substrate
ca     = aa[aa.mask("(chain_id == 'A') & (atom_name == 'CA')")]            # the backbone trace
```

This is how the tutorial re-applies `C_CRD`/`C_SEQ` from the mask to carry the conditions onto a
chosen design, and how it highlights the motif in the
[interactive 3D viewer](tutorials/enzyme_design_tutorial.ipynb) — selecting the conditioned atoms and
colouring them apart from the rest of the scaffold.

For a quick visual check without wiring up a viewer yourself, AtomWorks' `view` helper renders an
`AtomArray` — or any masked subset such as `aa[aa.mask("mapped_to_coordinate_conditioned_region & ~hetero")]` —
inline in a notebook: the visualization companion to the selection syntax above (see the
[AtomWorks documentation](https://rosettacommons.github.io/atomworks/latest/)). AtomWorks' own worked examples go further —
[annotating and saving structures](https://rosettacommons.github.io/atomworks/latest/auto_examples/annotate_and_save_structures.html)
and [loading and visualizing structures](https://rosettacommons.github.io/atomworks/latest/auto_examples/load_and_visualize_structures.html).
