# Python API

The Python API builds an [annotated CIF](../../concepts/annotated-cif.md) by editing an `AtomArray`
directly: parse a structure, set the conditions you want on it, write it back out. It is the most
flexible of the [three ways to make an annotated CIF](index.md#three-ways-to-make-an-annotated-cif) (anything
the model can read, the Python API can express) and the natural choice once you are generating
many variants of a problem instead of authoring one by hand. 

This page is a tour of how the RFD4-Proteina Python API works. For a complete,
runnable walkthrough, see the [Python API tutorial](../../examples/python_api_tutorial.ipynb):
it parses a protein–ligand complex, applies a hard conditioning task with unindexed tip atoms around
the ligand, RASA, hydrogen bond conditioning, an expandable segment and a PTM, writes an annotated
CIF, and finishes with a loop that generates many variants by perturbing a conditioned atom's
coordinate. 

## The shape of the workflow

Every script that builds an annotated CIF in Python follows the same three steps:

1. **Parse** a starting structure into an `AtomArray`: the in-memory representation AtomWorks and
   RFD4-Proteina share for a structure plus all of its annotations.
2. **Set conditions** on that `AtomArray` by selecting which atoms or residues a condition applies
   to, and what value it should take.
3. **Write** the result to an annotated CIF, which is just an ordinary mmCIF with those conditions
   saved as extra columns.

Here is a short example of these three pieces:

```python
from atomworks.io import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.conditions import C_SEQ, C_CRD, C_IDX
from rfproteina.conditions import C_HOT, C_RAS, C_HB, C_IDXGRP, RasaEnum, HBondEnum
from rfproteina.utils.parse_presets import ANNOTATED_CIF, RCSB_WITH_METALS

# 1. Parse
aa = as_atom_array_plus(parse("input.cif", config=RCSB_WITH_METALS)["assemblies"]["1"][0])

# 2. Set conditions
C_CRD.set_annotation_from_ground_truth(aa, fixed_atom_mask)
C_SEQ.set_annotation_from_ground_truth(aa, known_seq_res_mask)
C_HOT.set_annotation(aa, hotspot_atom_mask)  # the boolean annotation is itself the hotspot mask

# 3. Write
to_cif_file(aa, "annotated.cif", save_standard_annotations=True, chain_disambiguation="chain_iid")
```

```{important}
Import `rfproteina.conditions` before parsing an *existing* annotated CIF. 
It registers the RFD4-Proteina-specific condition columns (`C_HOT`,
`C_RAS`, `C_HB`, `C_IDXGRP`, and more); without the import, those columns are silently dropped on
parse.
```

Now that you've seen an example, let's dive into what each step does in more detail:

## Step 1: parse with the right preset

[`parse()`](https://rosettacommons.github.io/atomworks/latest/io/parser.html#atomworks.io.parser.parse)
takes a `config` argument that controls how much the parser is allowed to change about the
structure. `rfproteina.utils.parse_presets` has one for each situation you are likely to be in:

| You have | Use | Why |
| --- | --- | --- |
| A fresh structure from the PDB or AlphaFold, not yet annotated | `RCSB_WITH_METALS` | Adds missing atoms, reads covalent and metal bonds, strips waters and crystallization aids — the cleanup a raw structure usually needs before you can condition on it |
| A CIF you (or someone else) already annotated, that you want to inspect or edit further | `ANNOTATED_CIF` | Makes minimal structural changes and loads the conditions already saved in the file, so you do not lose them |

```{warning}
Pick deliberately: using `RCSB_WITH_METALS` on an already-annotated file can rebuild the structure
and lose the conditions you set earlier.
```

`parse()` returns one result per biological assembly; `["assemblies"]["1"][0]` picks the first model
of assembly `"1"`. Inspect `result["assemblies"].keys()` if you are not sure which assembly you want.
`as_atom_array_plus()` wraps the parsed structure in the richer `AtomArrayPlus` type that the
condition API expects.

## Step 2: select atoms, then set a condition

An `AtomArray` is a structure composed of columns of [NumPy](https://numpy.org/) arrays you can filter
with ordinary boolean indexing — `aa.chain_id`, `aa.res_id`, `aa.atom_name`, `aa.element`, and more. It also contains
a 2-D array for coordinates.
A **selection** is just a boolean mask the same length as the array:

```python
import numpy as np

# Every atom in chain A, residue 41
residue_41 = (aa.chain_id == "A") & (aa.res_id == 41)

# Only the CA atoms in chain A
chain_a_ca = (aa.chain_id == "A") & (aa.atom_name == "CA")
```

If you are not sure what chains or residues exist, inspect them first: `np.unique(aa.chain_id)`.
When an assembly repeats the same source chain more than once, use `aa.chain_iid` instead of
`aa.chain_id` to distinguish the copies.

With a mask in hand, set a condition one of two ways, depending on where the value comes from.

### Copying the parsed structure's own value

Most motif problems want a selected atom or residue fixed to *exactly what it already is* in the
parsed structure — "hold this backbone atom here", "keep this residue's identity." That is what
`set_annotation_from_ground_truth()` is for: pass it a mask, and it copies the structure's own
coordinate or sequence at the selected positions into the condition.

```python
C_CRD.set_annotation_from_ground_truth(aa, fixed_atom_mask)      # fix these atoms' coordinates
C_SEQ.set_annotation_from_ground_truth(aa, known_seq_res_mask)   # reveal these residues' identity
```

Everywhere the mask is `False`, the condition is left free for the model to design.

`set_annotation_from_ground_truth()` **replaces** the whole condition. If you are adding to an
annotated CIF that already has some positions conditioned and want to keep them, start from the
current value instead of a bare mask:

```python
known = C_SEQ.annotation(aa, default="generate").copy()
known[additional_mask] = C_SEQ.ground_truth_value(aa)[additional_mask]
C_SEQ.set_annotation(aa, known)
```

### Setting a value that has no "ground truth"

Some conditions do not describe something the structure already has like a hotspot flag, an H-bond
role, a RASA bin. For these, build the array of values yourself and call `set_annotation()`.

A boolean condition like hotspot takes the mask itself as the value:

```python
C_HOT.set_annotation(aa, hotspot_atom_mask)
```

An enum-valued condition starts from `default_annotation()` (everything "off"), then assigns the
enum value at the atoms you care about:

```python
hbond = C_HB.default_annotation(aa).copy()
hbond[aa.is_ligand & (aa.atom_name == "N")] = HBondEnum.DONOR
hbond[aa.is_ligand & (aa.atom_name == "O1")] = HBondEnum.ACCEPTOR
C_HB.set_annotation(aa, hbond.astype(int))
```

`RasaEnum` and `HBondEnum` are the two enums you will reach for most; the
[annotations reference](../../reference/annotations.md#supported-annotations) lists every condition,
its import name, and the values it accepts including the coordinate, index, and fold-class conditions
not shown here.

## Step 3: write the annotated CIF

```python
to_cif_file(aa, "annotated.cif", save_standard_annotations=True, chain_disambiguation="chain_iid")
```

`save_standard_annotations=True` is what actually embeds the conditions you set as extra columns in
the output file. Without it you get an ordinary, unconditioned CIF. `chain_disambiguation` controls
how chains that share a `chain_id` but differ by `chain_iid` are labeled on write; `"chain_iid"` is
the safe default once an assembly has repeated chains.

The file this produces is exactly what [`rfd4-generate`](index.md#running-the-model) reads.

## Where this pattern scales

Scripting a loop around steps 1–3 is how you generate many variants of one design problem — the
[Python API tutorial](../../examples/python_api_tutorial.ipynb) ends with exactly this, nudging
one conditioned atom's coordinate and writing out a fresh annotated CIF each time. The
[enzyme design tutorial](../../examples/enzyme_design_tutorial.ipynb) builds a full
[motif problem](index.md#motif-scaffolding-example) with coordinates, sequence, and
unindexing together using this technique.

Two things the Python API can do that the other two routes cannot express as directly:

- **Programmatic, data-driven selections.** Anything you can compute over the parsed structure, such as 
  distance to a ligand, residues in a CATH domain, atoms matching a chemical pattern, can become a
  mask, and therefore a selection.
- **[Expandable segments](../../concepts/conditioning.md#expandable-segments) and combining
  structures.** Inserting a variable-length placeholder, or merging two annotated structures into
  one design problem, both edit the `AtomArray` directly.

If you get stuck choosing between the various ways to generate an annotated CIF input file, see the
[comparison table](index.md#three-ways-to-make-an-annotated-cif).
