# Conditioning

A condition is a per-atom or per-residue overlay on a structure: a flag that tells the model
"you're given this value" rather than "design this yourself." The [annotations
reference](../reference/annotations.md) lists every condition's CIF column,
accepted values, and training definition. 

**This page explains the behavior behind that table:** why
conditions are gated by an active mask, why some conditions quietly imply others, and why the model
needs charges and bonds at all. Read [Annotated CIF](annotated-cif.md) first if you haven't seen the
format; come here when something in the reference table needs a "why."

## Every condition is on or off per position

Every condition has an **active mask**: a per-atom or per-residue true/false flag saying whether
that condition is actually set at that position. Where the mask is true, the model is given the
value and must respect it; where it is false, the model designs that value itself. You don't set
the mask directly — it's read off the value column: a sentinel like `MASKED`, NaN, or an enum's
"unspecified" value means the mask is false there, and a boolean-valued condition (like `C_HOT`) is
simply its own mask. This is why an unset coordinate defaults to NaN and an unset identity defaults
to `UNK`, those sentinels *are* "design this," not placeholder zeros.

## Condition the way the model was trained

The **Training Definition** column in the annotations reference matters more than it looks. Each
condition was supervised on a specific distribution: hotspots were sampled as target-chain atoms
within 4.5 Å of the binder with a `geometric(p=0.2)` count, RASA was binned from Shrake-and-Rupley
surface area at fixed edges, secondary structure came from P-SEA assignment. Feeding the model a
condition that technically satisfies the type but doesn't resemble how it was supervised (for
example, marking every exposed atom as a hotspot instead of a sparse interface patch) moves you off
the distribution the model knows how to act on. When a condition's behavior surprises you, check its
training definition before assuming a bug.

## Conditions are not independent

Conditions interact, and those couplings are enforced by the same consistency checks at training
time and at **input validation**, which reports violations before generation with messages like
*"N token(s) with non-backbone coordinate conditioning lack sequence conditioning."*

1. **Conditioning a sidechain coordinate forces sequence conditioning.** The model lays coordinates
   out in a fixed 37-slot `atom37` layout, where each slot is a canonical atom name specific to a
   residue. The pattern of filled **sidechain** slots fingerprints the amino acid, so fixing
   sidechain atoms reveals identity whether you meant to or not. Fixing even one of them reveals
   that residue's full sequence. There is one exception: backbone-only conditioning on `N`, `CA`,
   `C` and `O` shares its slots across every residue and leaks nothing, so it does *not* force
   sequence conditioning. That is exactly how you fix a backbone and redesign its sequence. The rule
   applies to internal coordinate conditioning (`C_INTCRD`) the same way it applies to absolute
   conditioning (`C_CRD`), because the slot-occupancy fingerprint leaks identity either way.

2. **Atomized tokens are always sequence conditioned.** Non-standard residues, PTMs, ligands and
   small molecules (everything with `atomize` set true) are context the model gets told about and
   never generates. The model generates only the 20 standard amino acids; nucleic acid and small
   molecule generation are coming. The inference pipeline sets this sequence conditioning for you
   automatically, so you do not need to set it by hand.

What this means in practice is if you want a fixed fold with a redesigned sequence, condition backbone atoms
only. If you want to fix a catalytic residue's identity, go ahead and fix its sidechain too. The
sequence conditioning that implies is no extra cost, since you wanted that identity fixed anyway.

## Absolute vs. internal coordinates

`C_CRD` gives an atom's ground-truth coordinate in the global frame: where it actually sits in the
design. That works for a single rigid motif, but it can't express "these atoms are rigid relative to
each other, but I don't know or care where that rigid body sits in space" — which is exactly what an
unindexed motif, or a design with several independent motifs, needs.

`C_INTCRD` gives the same kind of geometry, but centered and randomly rotated in its own frame
instead of the global one, so it fixes internal shape without fixing a global dock. `C_CRDGRP` is
what makes several internal-coordinate motifs coexist: it partitions atoms into groups whose
*relative* coordinates are known, with relative geometry known within a group and unknown across
groups. Group `0` is reserved for atoms on absolute (`C_CRD`) conditioning, and `-1` for atoms with
no known relative geometry; everything else is one independently-placed rigid body. This is how one
design can carry several motifs that are each internally rigid but placed independently of each
other — the model was trained for exactly this by converting sampled motif regions from absolute to
internal conditioning.

Because `C_INTCRD` and `C_CRD` both describe an atom's geometry, they're mutually exclusive on the
same atom: an atom is either pinned globally or free to float with its group. The [annotations
reference](../reference/annotations.md#coordinate-groups) covers the mechanics of editing group
membership without silently invalidating saved internal-coordinate targets.

## Expandable segments

An expandable segment is the structural equivalent of a variable length contig like `"10-15"`. It is
a placeholder that tells the model to design a segment and sample its length at runtime. Insert one
with
`atomworks.ml.utils.expandable_segment.insert_expandable_segment`. You give it `seg_min`, `seg_max`,
and where it goes, either inside a chain or as a new chain.

**Keep source templates collapsed when lengths should resample per design.** A collapsed segment
is a sentinel expanded per sample. An intentionally materialized child CIF, such as a fixed-length
benchmark input, records one concrete sampled length and will not resample it. Follow workspace
rules for retaining the source template and recording child provenance. Only polypeptide
expandable segments work for now.

In an **unindexed motif**, expandable segments specify spacing rather than extra conditioning
residues. Mark the segment and its flanking motif residues unindexed and give them the same index
group. The sampled length becomes a residue-number gap: `C–[2–2]–C` gives the cysteines relative
indices `0` and `3`, without creating unindexed spacer tokens. Length ranges are sampled per design,
and indexed expandable segments still produce generated residues as usual.

## Why charges and bonds matter

The model consumes both. It embeds formal charge per atom across 11 classes, and it turns
`bond_mask`, `bond_order` (Kekulé, with coordination as its own class) and `is_aromatic` into
pairwise features. That fidelity survives all the way to the model, it's the concrete reason to use
annotated CIFs for anything with a ligand, metal, covalent modification, or non-canonical residue,
rather than relying on a format that drops this information. Leaving a charge as "unknown" is not
equivalent to neutral, and leaving a bond order unresolved is not equivalent to a single bond: both
are read by the model as the value you wrote, not as "no information."

This is also why aromatic bonds need to be resolved by hand: the model has no `AROMATIC` bond-order
class, so any bond left as `AROMATIC` gets cast to `SINGLE` and the ring's bond-order information is
lost (the `is_aromatic` feature alone survives). Kekulizing to `AROMATIC_SINGLE` / `AROMATIC_DOUBLE`
is what lets both the aromaticity and the bond order reach the model intact.

## Where to go next

| Task | Page |
| --- | --- |
| See how conditioning attaches to a structure at a glance | [Annotated CIF](annotated-cif.md) |
| Look up a condition's CIF column, values, and training definition | [Annotations reference](../reference/annotations.md) |
| Build an annotated CIF, including motif and coordinate-group problems | [Preparing inputs](../how-to/input_prep/index.md) |
| Describe a design problem as YAML/JSON instead of Python | [Condition-spec reference](../reference/yaml-json.md) |
