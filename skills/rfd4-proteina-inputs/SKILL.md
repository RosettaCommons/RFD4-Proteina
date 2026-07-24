---
name: rfd4-proteina-inputs
description: >
  Turn a design intent (or a legacy RFD3-style JSON/YAML spec) into a correct
  **annotated CIF** for RFProteína — the recommended, lossless input format. Reach
  for this skill for "convert this JSON to a CIF", "json2cif", "build an annotated
  CIF", "prepare an input for rfproteina", "how do I fix atoms / unindex a motif /
  set hotspots / condition RASA or hydrogen bonds", "my ligand lost its charges /
  bonds", "how do I express this motif", or when rfd4-proteina-generate needs an input
  it does not yet have. The conversion is not the product — the **interrogation of
  intent** is: length, what to condition (coordinates vs sequence), what to
  index/unindex, ligand atoms, charges, and bonds. Builds with native
  biotite/atomworks primitives and validates before handing back to
  rfd4-proteina-generate.
compatibility: "rfd4-proteina-setup complete (venv + atomworks); no GPU required for building CIFs; ALLOW_BIOTITE_CCD=True for CCD lookups"
allowed-tools: Bash, Read, Write, AskUserQuestion
license: See the RFProteína repository (Baker Lab x NVIDIA collaboration)
---

# RFProteína Inputs (JSON -> annotated CIF)

An **annotated CIF** is a standard mmCIF file extended with AtomWorks condition
blocks — all design intent (fixed atoms, known sequence, hotspots, H-bonds, RASA,
unindexing, expandable segments, bonds, charges, PTMs) lives *inside the file*,
per-atom and per-residue. The model reads it natively (CIF -> AtomArray ->
features): no JSON, no contig string, no lossy conversion. This is the
recommended way to run RFProteína.

**The conversion is not the product. The interrogation is.** A legacy JSON is a
*statement of intent* in a mini-language whose defaults are counter-intuitive.
Your job is to work out what the author actually meant and encode *that*.
Transliterating the JSON silently is the failure mode. The one unforgivable move
is guessing silently — ask *"this from the JSON I think means XYZ, is that
right?"* about everything material.

There are three ways to produce an annotated CIF; this skill drives (2)/(3):

1. **PyMOL plugin** (AtomWorks Design Studio) — visual; see
   `docs/guides/annotated_cif_inputs.md` (Option A) and `make setup-pymol`.
2. **`/json2cif` from a legacy JSON** — this skill's primary path.
3. **Programmatic AtomWorks** — for scripting many variants; see the tutorial
   notebook `docs/guides/tutorials/annotated_cif_tutorial.ipynb`.

## The import-order landmine — read before parsing anything

```python
import rfproteina.conditions  # noqa: F401  -- REQUIRED, registry side-effect only
```

The StandardAnnotation registry is populated at **import time**, and the CIF
reader only recognizes columns whose names are registered. Parse an annotated CIF
without this import and **every rfproteina-specific condition is silently
dropped** (index-group, hotspot, RASA, hbond, ChainCentroid, CoordinateGroup) —
deserialization failures are warnings, not errors. The provided scripts already
do this import; if you write your own Python, do it first.

## Never do these

- **Never build with `rfproteina.inference_utils.input_parsing` or `...datasets`.**
  Build from scratch with native biotite/atomworks primitives; the legacy JSON
  pipeline is lossy by design. The `scripts/json2cif/build_cif.py` path does this.
- **Never split a resname on `;` or `:`.** `ligand` splits on `,` **only**.
  `L;G`, `T:S`, `H:H`, `K:X`, `lpr` are *literal residue names*; splitting them
  invents components that do not exist.
- **Never add missing atoms.** Never repair; the author fixes the source. (Polymer
  residues missing sidechain atoms are normal — that is what tip-atom scaffolding
  is. A *ligand* missing heavy atoms vs its CCD entry is a real defect: report it.)
- **Never realize an expandable segment.** Save it *collapsed* — a realized
  segment is frozen to one length and the model will not resample it.
- **Never add coordination bonds by default, and never infer which atoms are
  bonded.** Unbonded metals are the normal, usually-correct state. Surface and ask.
- **Never claim a check passed that you did not run.**

## Semantics you must not re-derive (verified against source)

- **`mask=True` means CONDITIONED.** There is no "diffuse" flag; diffusion is the
  *absence* of conditioning.
- **There is no `motif` condition.** A motif is residues carrying *both* sequence
  (`C_SEQ`) and coordinate (`C_CRD`) conditioning — always decompose into those two.
- **`C_IDX` defaults ALL-ON.** Unindexing is the marked operation.
- **Sidechain `C_CRD` => `C_SEQ`** (atom37 slots fingerprint the residue).
  Backbone-only (`N/CA/C/O`) does **not** — that is exactly how you fix a fold and
  redesign its sequence.
- **Atomized tokens (ligands, PTMs, non-canonicals) are always
  sequence-conditioned**, automatically, and **cannot be unindexed**.
- Conditions live at exactly one level: `C_CRD`/hotspot/RASA/hbond at **atom**
  level, `C_SEQ`/`C_IDX` at **residue** level, index-group/coord-group at **token**
  level. Most bugs are level confusions — e.g. fixing one sidechain atom fixes the
  whole residue's sequence.

Full detail: [references/import-order.md](references/import-order.md),
[references/annotated-cif.md](references/annotated-cif.md),
[references/json-dialect.md](references/json-dialect.md).

## Protocol

1. **Read the JSON and the input CIF.** Do not reason about the structure — look
   at it: how many chains, which ligands, any metals, non-canonicals, does
   `struct_conn` exist, are `pdbx_formal_charge` set?
2. **Classify the schema** (contig vs length/unindex) and **restate every field as
   intent**, then grill (see checklist below).
3. **Build**:
   ```bash
   bash skills/rfd4-proteina-inputs/scripts/build_cif.sh <benchmark.json> <key> --output out.cif
   ```
4. **Validate**:
   ```bash
   bash skills/rfd4-proteina-inputs/scripts/check_annotated_cif.sh out.cif
   ```
5. **Iterate.** Explain each failure in *intent* terms, not stack traces. Loop to
   step 2 if a failure reveals a misread of intent.

Metals: `build_cif.sh <json> <key> --list_metal_candidates` reports neighbors and
adds nothing; add only the bonds the author names with
`--coordination_bond 'C/ZN/296/ZN,A/CYS/365/SG'` (selectors are
`CHAIN/RES_NAME/RES_ID/ATOM_NAME` against the **built** array).

## Interrogation checklist (ask what is material and ambiguous)

- **Length.** The most important question. `"100-155"` is re-rolled *per design*
  (collapsed expandable segment); `"180-180"` is degenerate (realize it). Does the
  range count *only* diffused residues? (It does — unindexed motif residues are not
  subtracted.)
- **What to condition.** Always split in two: *"know where it is (coordinates), or
  what it is (sequence), or both?"* For each ligand: all atoms, or a catalytically
  relevant subset?
- **Index / unindex.** Unindexed = the model chooses chain position. Independent
  fragments (`"A39,A219"`) or rigidly tied (`"A39-41"`)? That is the index-group
  question and changes the design problem.
- **Ligand pose.** `select_fixed_atoms: false` => nothing coordinate-conditioned;
  the model invents pose and conformer. Confirm that is intended.
- **Charges.** `pdbx_formal_charge = ?` means unknown, not neutral. The model
  embeds per-atom formal charge — ask what they should be.
- **Bonds.** PDB-derived ligands often arrive single-bond-only (aromaticity/order
  lost). Kekulize atomized regions; use `AROMATIC_SINGLE`/`AROMATIC_DOUBLE`, never
  bare `AROMATIC` (cast to `SINGLE`).
- **Metals with no bonds.** Confirm, never infer. The model embeds a COORDINATION
  bond class, so it is a design decision.
- **Prep mutations.** Report every destructive preprocessing step (MSE->MET, waters
  dropped, altloc picked) and confirm each.

## Output

A single annotated `.cif` that passes `check_annotated_cif.sh` with 0 errors. Hand
it to **rfd4-proteina-generate** via `inputs=<file>.cif`. Outputs from generation echo
source-component provenance (`SA_SRC`) back to the input regions you defined here.

## Known-bad in the corpus — surface, don't fix

Some benchmark inputs are inexpressible or trip invariants by design (unindexed
*atomized* residues; H-bond `BOTH` on a hydroxyl; `nucleic_ss` not applied by the
legacy path; `ori_token`/`atomwise_hbond` dead knobs). Where your build and the
legacy oracle disagree, that is a question for the user, not an automatic bug. See
[references/import-order.md](references/import-order.md) for the catalog.
