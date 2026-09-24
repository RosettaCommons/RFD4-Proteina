---
name: rfd4-proteina-inputs
description: >
  Adapt legacy RFD3-style contig/length JSON or YAML InputSpecification configs
  into validated RFD4-Proteína annotated CIFs. Use for legacy design2cif conversions,
  migrating old benchmarks, or interpreting legacy fields such as contig,
  length, unindex, ligand, select_fixed_atoms, select_unfixed_sequence,
  select_hotspots, select_buried/select_exposed, select_hbond_donor/acceptor,
  redesign_motif_sidechains, or internal_coord_groups. Resolve the old
  dialect's ambiguous defaults before conversion. For prose, structures, or
  existing annotated CIFs, use rfd4-proteina-annotated-cif instead.
license: See the RFD4-Proteína repository (Baker Lab x NVIDIA collaboration)
---

# RFD4-Proteína legacy input adaptation

Convert a supplied legacy RFD3 JSON/YAML mini-language into RFD4's native input
representation: an annotated CIF. A mechanically valid translation can still
encode the wrong design problem, so restate every material field in plain design
terms and confirm ambiguities before building.

For annotated-CIF semantics, direct Python authoring, condition editing,
unindex/index-group rules, combining arrays, expandable segments, chemistry,
and final invariant checks, use the sibling
[rfd4-proteina-annotated-cif](../rfd4-proteina-annotated-cif/SKILL.md) skill.
This skill adds only legacy-dialect interpretation and the compatibility adapter.

## Boundary

Use this skill only when the source is an existing RFD3-style JSON/YAML spec, a
benchmark entry using `InputSpecification`, or a request to reproduce or migrate
old input behavior. Route prose design requests, raw PDB/mmCIF structures, and
existing annotated CIFs directly to `rfd4-proteina-annotated-cif`; do not create
a JSON/YAML intermediary.

## Required references

- Read [references/json-dialect.md](references/json-dialect.md) for every legacy
  conversion. It defines schema dispatch, field semantics, length conversion,
  and the current wrapper CLI.
- Read [references/legacy-edge-cases.md](references/legacy-edge-cases.md) when
  the config contains unsupported or historically lossy fields, unindexed
  noncanonicals/PTMs, overlapping H-bond roles, or when native output disagrees
  with a legacy oracle.
- Read the canonical
  [condition reference](../../docs/workflow/inputs.md#supported-annotations-reference)
  when checking the resulting `C_IDX`, `C_IDXGRP`, coordinate, sequence, RASA,
  H-bond, hotspot, or internal-coordinate conditioning.
- Read the generic skill's
  [validation reference](../rfd4-proteina-annotated-cif/references/validation.md)
  before accepting the converted CIF.

## Workflow

1. Read the spec and any referenced structure. Inspect chains, components,
   ligands, metals, noncanonicals, `struct_conn`, charges, and bonds rather than
   reasoning from selectors alone.
2. Classify the dialect by `contig` versus `length`/`unindex`. They are distinct
   schemas and must not co-occur.
3. Translate fields into intent: what the model knows about coordinates,
   sequence, index placement, ligand pose, length, and chemistry. Label what the
   file states, what you inferred, and what still needs a decision.
4. Ask only material questions. A fully specified unconditional monomer needs a
   concise restatement, not ceremonial questioning, but never guess when two
   readings change the design task.
5. Build with the checked-in compatibility adapter:

   ```bash
   bash skills/rfd4-proteina-inputs/scripts/build_from_rfd3.sh \
       input.json example_key --output out.cif
   ```

6. Inspect and validate the result:

   ```bash
   bash skills/rfd4-proteina-inputs/scripts/inspect_cif.sh out.cif
   bash skills/rfd4-proteina-inputs/scripts/check_annotated_cif.sh out.cif
   ```

7. Explain failures in design-intent terms. If a failure exposes an ambiguity,
   return to field interpretation rather than weakening the validator.

The adapter builds directly with public Biotite/AtomWorks APIs. Do not route the
conversion through `rfproteina.inference_utils.input_parsing` or `...datasets`,
and do not use the adapter for prose, raw structures, or existing annotated CIFs.

The adapter's output is an `AtomArrayPlus` written with `to_cif_file()`.
Readers, writers, inspection, and validation support plain CIFs and `.cif.gz`/`.cif.zst`.

## Questions that commonly change the conversion

- **Length:** Is a range resampled independently for every design? Does it
  constrain one variable segment or the sum of several? The latter may not be
  expressible as independent expandable segments.
- **Coordinates versus sequence:** `select_fixed_atoms` identifies known
  coordinates, while legacy sequence defaults independently reveal identity.
  Confirm what the author intends the model to know.
- **Unindexing:** Are comma-separated fragments independent islands, or should a
  peptide-bonded range retain relative offsets? No group may mix indexed and
  unindexed residues, and atomized tokens cannot be unindexed.
- **Ligand pose:** `select_fixed_atoms: false` means no ligand coordinates are
  conditioned, so the model invents pose and conformer. Confirm this explicitly.
- **Ligand syntax:** `ligand` splits on commas only. Semicolons and colons may be
  literal residue-name characters.
- **Chemistry:** Unknown charge is not neutral. PDB-derived ligands commonly lose
  bond order, aromaticity, and connectivity. Never infer charges, bonds, or
  metal coordination from proximity.
- **Preprocessing:** Report MSE conversion, removed waters or crystallization
  aids, altloc selection, residue renumbering, and other destructive changes.

## Never paper over these

- `select_fixed_atoms` has no safe implicit default in the legacy dialect.
- Do not delete or add atoms to imitate a selector. Review incomplete known CCD
  components; explicit custom components are valid only when their atoms, bonds,
  bond orders, aromaticity, and charges are intentionally specified.
- Do not unindex atomized residues, mix indexed and unindexed residues in one
  index group, or tie disconnected unindexed islands without confirmed intent.
- Use the generic skill's [segment guidance](../rfd4-proteina-annotated-cif/references/python-authoring.md#expandable-segments)
  to distinguish source templates from intentionally materialized children.
- Do not add coordination bonds unless the user identifies the partners.
- Do not assume a legacy oracle is correct when native output disagrees.
- Do not claim inspection or validation passed unless it was run.

## Metals

Candidate discovery is read-only and adds no bonds:

```bash
bash skills/rfd4-proteina-inputs/scripts/build_from_rfd3.sh \
    input.json example_key --list_metal_candidates
```

Add only author-confirmed bonds. Selectors address the built array as
`CHAIN/RES_NAME/RES_ID/ATOM_NAME`:

```bash
bash skills/rfd4-proteina-inputs/scripts/build_from_rfd3.sh \
    input.json example_key \
    --coordination_bond 'C/ZN/296/ZN,A/CYS/365/SG' \
    --output out.cif
```

## Deliverable

Return one annotated `.cif` per migrated example, a plain-language mapping from
legacy fields to encoded intent, the exact inspection and validation commands
run, and every unresolved chemistry or behavior difference. The CIF must pass
`check_annotated_cif.sh` with zero errors before generation handoff.
