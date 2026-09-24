---
name: rfd4-proteina-annotated-cif
description: >
  Create, inspect, explain, modify, and validate annotated CIF/mmCIF inputs for
  RFProteína with AtomWorks Python APIs. Use automatically for any question or
  task involving an existing annotated CIF, adding or changing conditioning,
  selecting fixed backbone or motif atoms, indexing/unindexing residues,
  hotspots, RASA, H-bonds, internal-coordinate groups, centroids, expandable
  segments, bonds, charges, combining annotated structures, or preparing a
  structure file as an annotated CIF.
  For interpreting or converting legacy RFD3 JSON/YAML input specifications,
  use rfd4-proteina-inputs instead.
license: See the RFProteína repository (Baker Lab x NVIDIA collaboration)
---

# RFProteína annotated CIF authoring

An annotated CIF is a structure plus AtomWorks `StandardAnnotation` columns that
encode the RFProteína design problem. Prefer direct Python manipulation of an
`AtomArrayPlus`: parse, derive selection annotations, set conditions, validate,
write, and round-trip. Use the AtomWorks Design Studio PyMOL plugin for visual
inspection of chemistry and intent, especially for ligands, metals, PTMs, or PDB
sources.

This skill defines CIF authoring semantics. Workspace instructions and the
active campaign govern shared-input ownership, output locations, length
materialization, generation, and database accounting. Authoring a CIF does not
by itself authorize a generation run or establish a campaign ledger policy.
For legacy RFD3 JSON/YAML interpretation, use
[rfd4-proteina-inputs](../rfd4-proteina-inputs/SKILL.md).

## Workflow

1. Inspect the source, existing conditions, requested assembly, and design intent.
   Ask only about ambiguities that change the design problem.
2. Read [python-authoring.md](references/python-authoring.md) for registry import,
   parser selection, condition edits, expandable segments, and combining arrays.
3. Read [annotators-and-selections.md](references/annotators-and-selections.md)
   when translating structural intent into atom masks or checking installed
   annotation support.
4. Use the canonical [condition reference](../../docs/workflow/inputs.md#supported-annotations-reference)
   and [interaction rules](../../docs/workflow/inputs.md#interaction-rules-read-before-conditioning)
   for condition levels, values, and model support. Do not infer support from an
   annotation's ability to round-trip.
5. Complete [validation.md](references/validation.md), including chemistry and
   indexing review, pre-save checks, and save/reload comparison.

## Output standard

Return the annotated CIF artifacts required by the task, identifying source
templates and any sampled children. Report input provenance and parse preset,
selections and conditioning counts, segment and chemistry edits, validation
results, and any unresolved scientific ambiguity.
