# YAML/JSON condition spec

The condition spec builds an [annotated CIF](../../concepts/annotated-cif.md) from a small YAML or 
JSON file. You name a design, point it at a starting structure, say which parts to keep, and set
the conditions you want. For those of you coming from previous RFdiffusion versions, this should
be the most familiar of the [three ways to make an annotated
CIF](index.md#three-ways-to-make-an-annotated-cif). It is best used when the design problem is better expressed 
as data than as code: something you template across many targets, diff in a pull request, or hand 
to a teammate who does not want to touch Python.

This page is a tour of the moving parts. For the exhaustive syntax, every `contig` token, and every
condition annotation, see the
[field-by-field YAML/JSON reference](../../reference/yaml-json.md). 

This page is an overview of 
how to think about building your YAML/JSON condition specification files, the YAML/JSON reference
contains all of the conditions you could possibly use to define your design. 

## The shape of the workflow

Every design (unless generating an unconditioned monomer) in a condition spec goes through the same three stages, in order:

1. **Load** one or more starting structures under a design key you choose.
2. **`contig`** subsets the loaded structure to the residues you want and inserts any variable-length
   placeholder segments, in one pass.
3. **`conditions`** sets the conditions you want, resolved against the structure `contig` produced.

The tool then runs the same dry-run safeguards and prints a summary of what it it thinks you are
trying to design.
If a safeguard fails, it stops rather than writing a CIF it cannot guarantee.

Here is a short example of all three pieces in a YAML file:

```yaml
binder_redesign:
  input: ./inputs/complex.cif
  contig: "A1-120, /0, B1-80"
  conditions:
    C_CRD: {true: [{select: "sel('B') and is_protein_backbone"}]}   # backbone-only, no sequence leak
    C_HOT: {true: [{select: "sel('A/*/[45, 46, 82]')"}]}
```

`binder_redesign` is a name (key) you chose, not a reserved keyword. A single file can define several
designs this way, each under its own key. The one reserved top-level key is `global_args`, for
defaults you want folded into every design in the file.

```{important}
Anything you do not mention is not provided. Setting only `C_HOT` above gets you a hotspot and
nothing else: no fixed coordinates, no known sequence. Trust the printed summary over your intent.
```

Now let's look at what each stage does in more detail.

## Step 1: describe the input

```yaml
my_design:
  input: ./inputs/complex.cif
  input_as_annotated_cif: true   # optional: load existing conditions instead of starting fresh
```

`input` takes one path or a list; a list gets combined into one structure, and chain ids must not
overlap. 
```{important}
Relative paths resolve against the spec file's directory, not your working directory.
```

Set `input_as_annotated_cif` to `false` for a fresh structure you want cleaned up (missing atoms added, `MSE` converted to `MET`,
any existing conditioning ignored). Set it to `true` for a CIF you or someone else already annotated,
so editing it further does not lose what is already set.

## Step 2: subset and insert segments with `contig`

`contig` is one comma-separated string of tokens, such as `"A42-48, E1:2-20, A52, E2:10R"`. It does
two things in one pass:

- **Subsets.** Any input atom not covered by a preserved segment (`A42-48`, a chain letter plus a
  residue range) is dropped. This is how you crop a structure down to a motif or an interface.
- **Inserts expandable segments.** A token like `E1:2-20` adds a variable-length placeholder the
  model fills in at generation time; naming it (`E1`) lets `conditions:` select it later.

```{warning}
`res_id` in `contig` and in `conditions:` selections resolves against your **original** input
numbering, the mmCIF `label_seq_id`, before `contig` renumbers the output. Expandable segments
have no original `res_id`, so select them by `res_name` instead.
```

```{note}
A `contig` string must keep **at least one preserved segment**. One that is entirely expandable
segments is rejected: `input` exists to be referenced by a preserved segment, so a design that
generates everything from scratch has no reason to reference one. For a pure de novo design with no
real anchor at all, see [where this pattern doesn't reach](#where-this-pattern-scales).
```

Omit `contig` entirely and the whole input is kept as is. 

See [`contig`
tokens](../../reference/yaml-json.md#contig-tokens) for chain-assignment rules when a run of tokens
mixes a preserved segment with an expandable one.

## Step 3: select atoms, then set a condition

A `select:` value is one AtomWorks query string that allows you to select specific
subsets of your structure. This is what will allow you to tell RFD4-Proteina if you 
want to unindex certain residues, redesign a seqence for part of a chain, etc.
It takes on two possible formats, shown below:

```yaml
# Path syntax: sel('<chain>/<res_name>/<res_id>/<atom_name>'), trailing fields default to '*'
select: "sel('A/*/[45, 46, 82]')"

# Query syntax: a Python boolean expression over any annotation
select: "is_ligand and not is_protein_backbone"
```

```{note}
The selecion syntax has 4 possible fields: `chain/res_name/res_id/atom_name`, if you omit one of these
specifications, it omits everything that comes after it. For example,
`sel('B/*/N')` is **not** "chain B, atom `N`"; it parses as `chain=B, res_name=*,
res_id=N` and fails, since `res_id` must be numeric. To reach `atom_name`, fill every field before
it: `sel('B/*/*/N')`. See [Selection syntax](../../reference/yaml-json.md#selection-syntax) for the
full grammar of both styles.
```

With a selection in hand, a condition entry maps the value you want to the list of selections that
should get it:

```yaml
conditions:
  C_CRD:                 # reveals ground-truth coordinates (fixes the atomic coordinates of chain B)
    true:
      - select: "sel('B')"
  C_HB:                   # an enum value with no "ground truth", here it is used to say that any nitrogen backbone atom in chain B can be used as an H-bond donor
    DONOR:
      - select: "sel('B/*/*/N')"
  C_SEQ_PROB_CYS:         # explicit zero is different from leaving CYS unspecified, here it means residues B10-20 in the output structure should not be cysteine
    0.0:
      - select: "sel('B/*/[10-20]')"
```

`C_CRD` and `C_SEQ` take a boolean directly, `true`
copies the structure's own coordinate or identity at the selection, `false` leaves it free. Every
other condition is an enum, string, or float value you choose yourself.
Overlapping entries for the same annotation resolve **last write wins**, in
declaration order. The [annotations reference](../../reference/annotations.md#supported-annotations)
and the [condition annotations table](../../reference/yaml-json.md#condition-annotations) list every
value each one accepts, including `C_IDX` and `C_CRDGRP`, whose group labels work a little
differently, see [index and coordinate group
fields](../../reference/yaml-json.md#index-and-coordinate-group-fields).

## Running the model

Point `inputs=` at the spec file exactly the way you would at a CIF:

```bash
rfd4-generate inputs=my_designs.yaml num_replicates=10 batch_size=8
```

Each top-level design key runs as one design problem; `generate.py` builds the annotated CIF in
memory and consumes it directly. You do not need to write anything to disk first.

To inspect what a design actually produces before spending GPU time on it, materialize it to
annotated CIFs instead:

```bash
scripts/write_condition_spec_cifs.py my_designs.yaml --out-dir ./cifs
```

The resulting file is exactly what [`rfd4-generate`](index.md#running-the-model) would have read.

## Examples

Two complete, runnable specs live under `docs/how-to/input_prep/examples/`, one for a simpler [ppi example](examples/condition_spec_ppi.yaml)
and one for a more complicated [binding pocket redesign](examples/condition_spec_complicated.yaml). The [PPI specificity
tutorila notebook](../../examples/ppi_specificity_tutorial.ipynb) also uses the YAML specification format.

- [condition_spec_ppi.yaml](examples/condition_spec_ppi.yaml) builds a de novo PPI binder against
  PD-L1, using an expandable binder chain, backbone-only coordinate conditioning on the target, and
  atom-level hotspots.
- [condition_spec_complicated.yaml](examples/condition_spec_complicated.yaml) redesigns a small
  molecule pocket, exercising a multi-chain contig, an expandable segment carrying secondary
  structure conditioning, index groups, and both selection styles.

**Fix a binder backbone, redesign its sequence, hotspot the target.**

```yaml
binder_redesign:
  input: complex.cif
  conditions:
    # `is_protein_backbone` is what makes this backbone-only -- `sel('B')` alone would
    # condition on every sidechain atom too, which is not "redesign its sequence".
    C_CRD: {true: [{select: "sel('B') and is_protein_backbone"}]}  # backbone-only coords, no sequence leak
    C_HOT: {true: [{select: "sel('A/*/[45, 46, 82]')"}]}
```

**Scaffold a motif with a variable length linker to a new binder chain.**

```yaml
motif_scaffold:
  input: motif.cif
  contig: "A1-50, LINK:10-30, /0, BINDER:60-90"   # BINDER's run has no anchor -> fresh chain id
  conditions:
    C_CRD: {true: [{select: "sel('A/*/[1, 50]')"}]}
    C_SEQ: {true: [{select: "chain_id == 'A' and res_name != 'LINK'"}]}   # query syntax: exclude the placeholder
```

## Where this pattern scales

A condition spec is the right call when the design problem is naturally a handful of selections
over a starting structure (a motif, an interface, a redesign) and you want to template or diff
it as data. The [examples above](#examples) show a de novo PPI binder and a small-molecule pocket
redesign built this way.

However, there are three things the Python API can do that a condition spec cannot express:

- **Two-body conditions**, like base pairing, which have no selection-pair grammar in the spec
  format.
- **Programmatic, data-driven selections** computed over the parsed structure, and combining
  structures beyond a simple concatenation of `input` paths.
- **Pure de novo generation with no real anchor residue.** A spec's `contig` always needs at least
  one preserved segment from `input` (see the note above), so it cannot express a design built
  entirely from scratch. `data/benchmarks/monomer/*.cif.zst` are exactly this case — a single wholly
  expandable chain with no conditions at all — built directly with `insert_expandable_segment()` on
  a placeholder `AtomArray` rather than through a spec.

If you hit either of those, or just want to generate many variants programmatically, switch to the
[Python API](python-api.md). If you get stuck choosing between all three routes, see the
[comparison table](index.md#three-ways-to-make-an-annotated-cif).
