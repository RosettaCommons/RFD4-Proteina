# Reading outputs

The designed chain gets renumbered from scratch, so a conditioned motif no longer sits at its input
indices. The [standard annotations](../reference/output-format.md#standard-annotations-and-why-there-is-no-output-json)
written into every generated CIF are how you find where it landed, without hand-parsing the file or
reconstructing an unindex mapping yourself.

## Selecting a region by annotation

Standard annotations are ordinary annotations, so you can pull out any region with the
**[atomworks selection syntax](https://rosettacommons.github.io/atomworks/latest/io/utils/selection_syntax.html)**:
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

Parse with the `STANDARD_ANNOTATIONS_ONLY` preset from `rfproteina.utils.parse_presets` for minimal
overhead when all you need is this provenance.

## Reading compressed structures

The same `parse` call reads plain CIF, gzip (`.cif.gz`) and Zstandard (`.cif.zst`) inputs.
Generation and standalone evaluation still write plain CIF by default; `compress_outputs=true`
selects Zstandard for newly written outputs. See [Compression](../reference/output-format.md#compression).

To discover outputs in either encoding, use the structure-file helpers rather than a plain
`*.cif` glob:

```python
from rfproteina.utils.structure_files import iter_structure_files, structure_file_stem

for path in iter_structure_files("runs/my_run/inference_samples", stem_suffix="_generated"):
    design_id = structure_file_stem(path)  # strips both the format and compression suffix
    aa = parse(path, config=ANNOTATED_CIF)["assemblies"]["1"][0]
```

Keep one encoding of each structure in a directory: discovery rejects duplicate plain/compressed
copies with the same design ID.

## Worked examples

The [enzyme design tutorial](../examples/enzyme_design_tutorial.ipynb) uses this same selection to
colour the motif separately from the scaffold, and the **[Python API tutorial
notebook](../examples/python_api_tutorial.ipynb)** builds up the annotation system atom by atom.
For a quick look without wiring up a viewer, AtomWorks' `view` helper renders an `AtomArray` inline
in a notebook, including any masked subset — see the [AtomWorks
documentation](https://rosettacommons.github.io/atomworks/latest/).
