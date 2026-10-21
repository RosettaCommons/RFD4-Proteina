# Preparing inputs

An annotated CIF is an ordinary mmCIF with extra columns carrying the conditions: hold this atom
still, keep that residue's identity, treat this one as a hotspot. See [Annotated
CIF](../../concepts/annotated-cif.md) for what the format is, why it keeps structure and intent in one
file, and how coordinate and sequence conditioning attach at different granularities. This page
covers how to build one and run the model against it.

## Three ways to make an annotated CIF

Pick by how much of the design problem you need to express. The Python API can express anything
the model can read.

```{image} /_static/diagrams/design-input-spectrum-light.svg
:alt: Four ways to specify a design, left to right from simple and weakly expressive to complex and maximally expressive: YAML or JSON (preferred, the simplest place to start), Agent Skill, the AtomWorks Studio web GUI, and the Python API.
:class: only-light
:width: 100%
:align: center
```

```{image} /_static/diagrams/design-input-spectrum-dark.svg
:alt: Four ways to specify a design, left to right from simple and weakly expressive to complex and maximally expressive: YAML or JSON (preferred, the simplest place to start), Agent Skill, the AtomWorks Studio web GUI, and the Python API.
:class: only-dark
:width: 100%
:align: center
```

| Route | Reach for it when |
| --- | --- |
| [`rfd4-proteina-annotated-cif`](agent.md) | You want an agent to interview you, inspect existing artifacts, and author or edit the CIF |
| [YAML condition spec](yaml-json.md) | You want the problem as data, in version control |
| [Python API](python-api.md) | You are generating many problems, or need something the others cannot say |

### Python API

Parse a starting CIF into an `AtomArray`, set annotations with the AtomWorks condition API, and
write the result back out as an *annotated* CIF. Use this when you want to script many variants of a design, or express
something the other two routes cannot say directly.

See the [Python API how-to page](python-api.md) for a guided walkthrough of parsing, selecting atoms, and setting
conditions, including the two core methods — `set_annotation_from_ground_truth()` and
`set_annotation()` — and when to reach for each. You can find the full API docs [here](../../reference/index.rst).

### YAML condition spec

To script or template design problems without writing Python, describe them in a
small YAML or JSON file. You subset the input, insert expandable segments, and set conditions on
atom selections. From there you either write annotated CIFs or hand the spec straight to
`generate.py`. Either way you end up with the same annotated CIF representation described above.
This is shorthand for authoring one, not a different format.

See the [YAML/JSON condition spec how-to guide](yaml-json.md) for a guided tour of the file shape, `contig`, and
setting conditions; the [field-by-field reference](../../reference/yaml-json.md) has the full syntax
and examples.

All routes end in the same annotated CIF, which is what the model reads:

```{image} /_static/diagrams/design-input-pipeline-light.svg
:alt: A YAML file (preferred) or JSON feeds the Python API, as do an Agent Skill and the AtomWorks Studio web GUI. The Python API writes an annotated CIF, which goes into RFD4, which outputs an annotated CIF.
:class: only-light
:width: 100%
:align: center
```

```{image} /_static/diagrams/design-input-pipeline-dark.svg
:alt: A YAML file (preferred) or JSON feeds the Python API, as do an Agent Skill and the AtomWorks Studio web GUI. The Python API writes an annotated CIF, which goes into RFD4, which outputs an annotated CIF.
:class: only-dark
:width: 100%
:align: center
```

## Running the model

```bash
rfd4-generate \
    inputs=annotated.cif \
    num_replicates=10 batch_size=8
```

`inputs` takes a single `.cif` or `.bcif`, a **directory** of them, or a list of file names.
Plain, gzip (`.cif.gz`, `.bcif.gz`) and Zstandard (`.cif.zst`, `.bcif.zst`) inputs are read directly;
there is no need to decompress them first. Repository benchmark CIFs use `.cif.zst`. Each file is one
design problem, and compression does not change its example ID. <!--It also takes `.json` and `.yaml`, though we do not recommend those.--> Outputs echo
source component provenance (information on where each residue, inserted segment in the output structure came from) 
as per-residue `SA_SRC` labels like `A3`; expandable segments retain
range labels like `3-10`. This lets you trace each output residue to its input residue or inserted
segment.

## Supported annotations

Every condition the model reads, its exact CIF column, accepted values, and training definition
are in the [annotations reference](../../reference/annotations.md#supported-annotations). That page
also covers coordinate groups and which conditions are actually wired up today. For *why* conditions
behave the way they do, for example why fixing a sidechain coordinate also fixes its sequence, and
why charges and bonds are consumed the way they are, see [Conditioning](../../concepts/conditioning.md).

### Motif scaffolding example

Here we describe how the annotation conditions work for describing a motif scaffolding design task.

Starting from a theozyme (a handful of catalytic residues and a substrate with no sacffold)
you can condition only the atoms in the motif and leave the rest for your model to design using
these three annotation conditions: 

| Condition | Granularity | In a theozyme |
| --- | --- | --- |
| `C_CRD`, coordinate | per **atom** | Pin the atoms that set the active site geometry, so a catalytic residue's functional atoms and the substrate's reactive core. Partial on purpose. Leave flexible parts like a phosphate tail free. |
| `C_SEQ`, sequence | per **residue** | Fix the identity of a residue or the substrate. All or nothing within one residue, though across a motif you can reveal it on some residues and not others. |
| `C_IDX`, index | per **residue** | On by default. Turn it **off** on the catalytic residues so the model places them freely along the new chain, which makes an *unindexed* motif. |

So a catalytic residue usually fixes its functional atoms with `C_CRD` and its identity with
`C_SEQ`, while staying unindexed with `C_IDX` off. The substrate fixes its reactive atoms and
identity and keeps its default index. The [enzyme design
tutorial](../../examples/enzyme_design_tutorial.ipynb) builds one of these and runs it end to end.

Read [Conditioning](../../concepts/conditioning.md#conditions-are-not-independent) before conditioning
anything with a ligand, metal, covalent modification, or non-canonical residue: those couplings are
enforced during input validation and determine what you can fix without unintentionally revealing
sequence.

To insert a variable-length placeholder segment, including how it combines with an unindexed
motif, see [Expandable segments](../../concepts/conditioning.md#expandable-segments).

## FAQ and gotchas

**Do I have to use annotated CIFs?** Not directly. JSON inputs and the [YAML
condition spec](yaml-json.md) are supported. However, these JSON/YAML spec
files will create an annotated CIF (or annotated CIFS) based on the specifications.
The only input type RFD4-Proteina directly uses is an annotated CIF. 
Annotated CIFs are useful
when conditions and explicit component chemistry should be inspected and edited
with the structure. The [input-authoring skill](agent.md) creates or edits
annotated CIFs directly from prose and structures.

**Can I open an annotated CIF in PyMOL or other tools?** Yes. See [It's still a
CIF](../../concepts/annotated-cif.md#its-still-a-cif).

**Why is a fixed sidechain also fixing my sequence?** Because `atom37` reveals identity, as
[Conditions are not independent](../../concepts/conditioning.md#conditions-are-not-independent)
explains. To redesign sequence under a fixed fold, condition on backbone atoms only.

**Can I condition expandable segments?** Yes. For residue based conditions like chain COM or
secondary structure, conditioning a whole expanded region at once often makes sense. Condition an
expandable segment and the condition applies to every token its expansion produces.

**I have a ligand bound to a peptide chain via a PTM. Should it have its own `IndexGroup`?** No. The
model reasons over _molecules_, not _chains_. A PTM is part of the same molecule as the peptide
chain by definition, so a valid bond graph distance exists between atoms on the PTM and any residue
on the main chain. The model sees bond graph distance features whenever the tokens share a molecule
and an `IndexGroup`. Gaps in `res_id` between the PTM and the peptide chain are fine, because we use
bond graph distance for relative positional encoding of atomized segments instead of `res_id` gaps.

```{toctree}
:hidden:
:maxdepth: 1

agent
yaml-json
python-api
```
