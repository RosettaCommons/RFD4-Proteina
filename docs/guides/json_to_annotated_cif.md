# Converting a legacy JSON to an annotated CIF

Have a legacy RFD3-style JSON input and want the recommended [annotated CIF](annotated_cif_inputs.md)
instead? Use the **`/design2cif`** Claude command. It does not transliterate the JSON field-by-field —
that loses intent — it interrogates you where the JSON is ambiguous, then builds the CIF from scratch.

## Running it

In Claude Code:

```
/design2cif <input.json> [example_key]
```

Pass the JSON path, and (for a multi-example JSON) the key to convert.

You can also skip the JSON entirely and just describe what you want:

```
/design2cif scaffold a 120-residue binder against the SAM pocket of 1abc,
            keeping the four catalytic residues fixed but letting the model place them
```

The command then elicits the same fields by asking, drafts the equivalent JSON, shows it to you for
sign-off, and builds from that. The draft is kept alongside the CIF as the reproducible record of
what was actually built.

## What to expect

1. **Questions first.** Claude restates each field as intent and grills you where the JSON is silent
   or the docs are stale — e.g. which ligand atoms are coordinate-fixed, how a resname resolves,
   whether an `unindex` range should stay index-tied, what a prep mutation (MSE→MET, dropped waters)
   changed. Answer these; they determine the output.
2. **A built CIF.** Claude writes an annotated CIF from native atomworks primitives and validates it.
3. **A report.** Anything worth confirming is surfaced — missing atoms (never silently added),
   unusual or absent formal charges, metal coordination, covalent links. Nothing is auto-repaired.

> ⚠️ **Always double-check the result.** Claude is not perfect and can misread intent. Before
> running the model, open the annotated CIF in the [AtomWorks Design Studio plugin](annotated_cif_inputs.md#option-a-the-pymol-plugin-atomworks-design-studio)
> and confirm the conditioning is what you meant (fixed atoms, sequence, ligands, unindexing).

## What it will not do

- Silently drop covalent bonds, charges, or PTMs (the failure modes of the legacy JSON path).
- Guess past a genuine ambiguity — it asks instead.
- Add missing atoms — it flags them for you to fix at the source.
