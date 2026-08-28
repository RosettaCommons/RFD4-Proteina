---
name: rfd4-proteina-inputs
description: >
  Interview the user about a protein-design intent, inspect any prose, structure,
  existing contig/length JSON/YAML, or annotated-CIF artifacts they provide, and build or edit a
  validated annotated CIF for RFProteína with the Python AtomWorks API. Reach for
  this skill for "design2cif", "prepare an RFProteína input", "convert this RFD3
  JSON", "edit this annotated CIF", fixed atoms, sequence conditioning,
  unindexing, hotspots, RASA, hydrogen bonds, expandable segments, ligand bonds,
  charges, PTMs, and non-canonical residues. JSON is optional compatibility input,
  never a required intermediate representation.
license: See the RFProteína repository (Baker Lab x NVIDIA collaboration)
---

# RFProteína annotated-CIF authoring

The product is a design problem whose meaning the user has confirmed, encoded in
one annotated CIF. The route is:

**interview -> inspect evidence -> confirm intent -> build or modify with Python -> validate**

Do not turn prose or an existing CIF into JSON on the way. An existing JSON/YAML input
specification is one possible artifact to interpret; it is not the
canonical representation and does not define intent more authoritatively than the
user's explanation or an existing annotated CIF.

## Begin with intent and artifacts

Do not write the output first.

1. Gather the user's intended design and every relevant artifact: prose or notes,
   source structures, an existing annotated CIF, and any existing JSON/YAML input specification.
2. Inspect supplied artifacts before asking detailed follow-ups. Read prose and
   specs as evidence of intent. For every CIF, run:
   ```bash
   bash skills/rfd4-proteina-inputs/scripts/inspect_cif.sh <file.cif>
   ```
   Also inspect raw `atom_site`, `struct_conn`, formal charges, alternate
   locations, ligands, metals, PTMs, and non-canonical residues when material.
3. Present a compact intent plan in ordinary language. Cover only relevant items:
   source/assembly and retained components; designed topology and length;
   coordinate and sequence conditioning; indexing and index groups; other active
   conditions; bonds and charges; preprocessing; output path.
4. Label what the artifacts state, what you inferred, and what remains unresolved.
   Ask a small batch of targeted questions about material ambiguities. If the
   artifacts conflict, show the conflict and let the user decide. Do not ask about
   conditions irrelevant to the design merely because they exist.
5. Build after the material intent is confirmed. A fully specified request needs a
   concise restatement, not ceremonial questioning.

The most important distinction in the interview is independent:

- **Coordinates:** does the model know where selected atoms are?
- **Sequence:** does the model know which residue or component they belong to?
- **Index:** does the model know their chain position, and which unindexed pieces
  retain relative offsets?

For variable lengths, confirm whether the range is sampled independently for each
design and whether it counts only generated residues. For ligands and modified
residues, confirm the exact atoms, pose, formal charges, bond orders, and any
explicit covalent or coordination bonds.

## Choose the authoring route from the evidence

All routes end in an `AtomArrayPlus` written with `to_cif_file()`.

- **Prose or raw structure:** prepare a potentially incomplete source with
  `RCSB_WITH_METALS`, then write a short, task-specific Python recipe using the
  public AtomWorks/Biotite API. Inspect and confirm every material preparation
  choice. Do not create a JSON or YAML spec.
- **Existing annotated CIF:** parse it with `ANNOTATED_CIF`, copy it, and edit the
  requested annotations or topology in place. Preserve every unmentioned
  condition, bond, charge, component, and expandable segment. Write a new file
  unless the user explicitly requests an overwrite.
- **Existing contig/length JSON/YAML:** read
  [references/json-dialect.md](references/json-dialect.md), inspect the referenced
  structures, and translate its fields into user-facing intent. After confirmation,
  use the tested `build_from_rfd3.sh` compatibility adapter when the confirmed
  intent fits the supported dialect. The adapter builds the annotated CIF directly
  with the Python AtomWorks API; it does not emit another JSON file. Use a
  task-specific Python recipe when the confirmed intent exceeds that dialect.

Read [references/python-api.md](references/python-api.md) before authoring or
editing with Python. Read [references/annotated-cif.md](references/annotated-cif.md)
for condition levels and interaction rules, and
[references/validation.md](references/validation.md) before final validation.

## Import before parsing

```python
import rfproteina.conditions  # noqa: F401
```

This import registers RFProteína conditions with AtomWorks. It must execute before
any annotated CIF is parsed; otherwise RFProteína-specific condition columns can
be dropped during deserialization. The inspection and validation helpers enforce
this for their own parsing; custom authoring recipes must import it explicitly.

## Non-negotiable construction rules

- Build and edit with public Biotite/AtomWorks primitives. Do not use
  `rfproteina.inference_utils.input_parsing` or `...datasets` as an authoring API.
- `mask=True` means conditioned. A motif is the combination of coordinate and
  sequence conditioning; there is no separate motif condition.
- `C_IDX` defaults on. Turning it off unindexes a residue. Preserve the requested
  indexing intent and let the checked-out validation pipeline determine whether
  the selected token types support it; never silently force an invalid selection
  back to indexed.
- Sidechain coordinate conditioning reveals residue identity and therefore
  requires sequence conditioning. Backbone-only `N/CA/C/O` conditioning does not.
- Keep condition levels consistent: coordinate/hotspot/RASA/H-bond at atom level;
  sequence/index at residue level; index groups at token level.
- Never split a residue name on `;` or `:`. In the RFD3 `ligand` field, only a
  comma separates residue names.
- Do not delete atoms to express partial conditioning; retain the complete
  component and select conditioned atoms with masks. For raw or incomplete
  structures, choose and disclose the preparation preset explicitly. Treat
  incomplete polymer residues and CCD mismatches as review items, not automatic
  evidence of validity or corruption; custom non-CCD components are supported
  when their explicit atoms, bonds, and charges are complete.
- Keep expandable segments collapsed. Do not turn a genuine length range into a
  sampled fixed length before saving.
- Never infer coordination bonds from distance. Report candidates and add only
  bonds the user identifies. Preserve existing covalent bonds unless removal is
  explicitly requested.
- Comments and reports describe the current input and its current behavior. Do not
  narrate historical bugs or superseded implementations.
- Never claim an inspection or validation passed unless it was run.

## Validate meaning, not only syntax

After writing the CIF:

```bash
bash skills/rfd4-proteina-inputs/scripts/inspect_cif.sh out.cif
bash skills/rfd4-proteina-inputs/scripts/check_annotated_cif.sh out.cif
```

Then reparse the output with `ANNOTATED_CIF`. Confirm that each intended selection
is nonempty and has the intended condition values. For edits, compare the source
and output inventories and verify that unrequested conditions, bonds, charges,
components, and expandable ranges are unchanged. Validation failures that expose
an unresolved design choice return to the interview; do not weaken the validator.

For a supplied RFD3 contig/length JSON/YAML artifact only, run:

```bash
bash skills/rfd4-proteina-inputs/scripts/build_from_rfd3.sh input.json example_key --output out.cif
```

The adapter can report metal neighbors with `--list_metal_candidates`; candidates
are not bonds. Explicit selectors use `CHAIN/RES_NAME/RES_ID/ATOM_NAME` against
the built array. Do not use this adapter for prose, raw structures, or existing
annotated CIFs.

## Deliverable

Return the annotated `.cif`, the confirmed intent summary, and the exact inspection
and validation commands run. Include a small Python recipe when it is useful for
reproducibility or future variants. Do not create a JSON/YAML record unless the
user explicitly asks for one.

Reject intent that the checked-out validation pipeline cannot represent. Explain
the design-level conflict and ask the user which constraint to change; never
rewrite indexing, H-bond roles, or coupled length constraints merely to pass.
