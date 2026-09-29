# Outputs

What a run writes to disk, and how to read it.

Each design comes back as a CIF. Per atom annotations record how the model produced each atom, and
they sit in ordinary `_atom_site` columns, so any CIF reader picks them up.

Generation and standalone evaluation write plain `.cif` files by default; use `compress_outputs=true` for Zstandard level 3.
Training validation (flow model and autoencoder) defaults to `.cif.zst`;
use `compress_outputs=false` for plain CIF. Plain, gzip, and Zstd inputs are supported.
Resume preserves existing files; Foldseek and
sharing exports receive plain CIFs. Keep one encoding per design in each directory.

| Accessor | CIF column | Description |
| --- | --- | --- |
| `SA_SRC.annotation` | `annotation_src-component_1_atom` | Which part of the input spec the atom originated from (`.` if unassigned) |
| `SA_CRDSRC.annotation` | `annotation_src-coordinate-conditioned-region_1_atom` | Which input motif region provided each atom's coordinate conditioning (`.` if none); this value also determines `SA_CRDSRC.mask` |
| `SA_IDXGRPSRC.annotation` | `annotation_src-index-group_1_atom` | Original unindexed `IndexGroup` of a mapped motif token (`-1` if none) |
| `SA_UII.annotation` | `is_unindexed_insertion` | `True` for atoms corresponding to an unindexed input motif |

The aliases live in `rfproteina.utils.standard_annotations`. Read them off a parsed structure with
`SA_CRDSRC.mask(atom_array)` or `SA_SRC.annotation(atom_array)`.

Aligned motif coordinates are the one exception. They go in their own
`_annotation_coords-with-insertion_1_atom` loop instead of an `_atom_site` column, under `SA_CWI`.
You read them the same way as any other `StandardAnnotation`.

## Finding a region again

The designed chain gets renumbered from scratch, so a conditioned motif no longer sits at its input
indices. The annotations above are how you find where it landed.

They are ordinary annotations, so you can pull out any region with the **[atomworks selection
syntax](https://rosettacommons.github.io/atomworks/latest/io/utils/selection_syntax.html)**.
`aa.mask(...)` and `aa.idxs(...)` evaluate a boolean query where annotation names act as variables.

```python
from atomworks.io import parse
from rfproteina.utils.parse_presets import ANNOTATED_CIF

aa = parse("design.cif", config=ANNOTATED_CIF)["assemblies"]["1"][0]

# Annotation names are variables in the query; index the array with the mask to subset it.
motif  = aa[aa.mask("sa_mask('SA_CRDSRC') & ~hetero")]  # the conditioned motif
ligand = aa[aa.mask("hetero")]                                             # the ligand
ca     = aa[aa.mask("(chain_id == 'A') & (atom_name == 'CA')")]            # the backbone trace
```

That is how the [enzyme design tutorial](../examples/enzyme_design_tutorial.ipynb) re-applies
`C_CRD` and `C_SEQ` from the mask to carry conditions onto a design it picked, and how it colours
the motif apart from the rest of the scaffold in its interactive viewer.

For a quick look without wiring up a viewer yourself, AtomWorks' `view` helper renders an
`AtomArray` inline in a notebook, including any masked subset. It is the visualization companion to
the selection syntax above. See the [AtomWorks
documentation](https://rosettacommons.github.io/atomworks/latest/), whose own worked examples
cover [annotating and saving
structures](https://rosettacommons.github.io/atomworks/latest/auto_examples/annotate_and_save_structures.html)
and [loading and visualizing
structures](https://rosettacommons.github.io/atomworks/latest/auto_examples/load_and_visualize_structures.html).
