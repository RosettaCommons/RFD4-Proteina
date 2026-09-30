# Describe your design to an agent

Tell your coding agent what you want to build. It interviews you about the parts you left open, checks the
structure for the usual traps, and hands back a model ready [annotated CIF](inputs.md).

How you state the intent is up to you. A paragraph, lab note, source structure,
existing annotated CIF, or condition-spec JSON/YAML can all provide evidence. The
agent examines what you already have, asks about unresolved choices, and edits the
annotated CIF directly with the Python API.

**The conversion is not the product. The interview is.** A structured input records fields whose
material meaning still needs confirmation, while prose commonly leaves choices unstated. Either
way the job is to work out what you meant and encode *that*, rather than treating an artifact as a
substitute for intent.

## Running it with the shared skill

For prose, a structure, or an existing annotated CIF, ask Codex or Claude Code
to use `rfd4-proteina-annotated-cif` with the artifact:

```
Use rfd4-proteina-annotated-cif with <artifact>
```

For an existing legacy RFD3 JSON/YAML file, use `rfd4-proteina-inputs` and pass
the example key when the file contains several entries:

```
Use rfd4-proteina-inputs with <legacy.json> [example_key]
```

Or just say what you want:

```
Use rfd4-proteina-annotated-cif to scaffold a 120-residue binder against the SAM pocket of 1abc,
keeping the four catalytic residues fixed but letting the model place them.
```

The `rfd4-proteina-inputs` skill covers legacy RFD3 JSON/YAML adaptation. For
new annotated-CIF creation or editing from a structure, use
`rfd4-proteina-annotated-cif`; `rfd4-proteina-generate` routes to the appropriate
one based on the source format. Claude Code's `/design2cif` compatibility alias
loads the legacy adapter.

## What to expect

1. **Questions first.** The agent opens the structure rather than reasoning about it, then restates
   each thing it plans to do as intent and asks. The pattern is *"I read this as X, is that
   right?"* Your answers determine the output, so take them seriously.
2. **An intent plan.** It separates what you stated, what the artifacts show, what it inferred, and
   what still needs a decision. This is ordinary language, not a generated JSON spec.
3. **A built or edited CIF.** It uses the Python AtomWorks API directly.
4. **A report.** Everything worth a second look, described in terms of your design rather than
   stack traces.

## What it asks about

The agent asks about the choices material to your design. It uses your artifacts to avoid generic
questions and does not make you decide conditions that are irrelevant to the requested task.

**Length.** Is that range meant to be sampled per design, or did you mean one fixed length?

**What to condition.** Split in two, always. Do you want the model to know *where* an atom is, or
*what* it is, or both? For a ligand, all atoms or a catalytically relevant subset?

**What to index.** Unindexed means the model chooses where in the chain a residue goes. Should your
unindexed fragments float independently, or stay rigidly tied to each other?

**Ligand pose, charges and bonds.** Whether the pose is given or invented, what the formal charges
should be, and whether the inferred bond graph is right.

**Everything preprocessing changed.** So you can catch it before it costs you a run.

## Gotchas it looks for

These are the ones that bite people. Most are silent.

| What happens | Why it matters |
| --- | --- |
| Fixing one sidechain atom also reveals that residue's identity | The atom37 slots fingerprint the amino acid. Backbone only conditioning on `N`, `CA`, `C` and `O` leaks nothing, and that is exactly how you fix a fold and redesign its sequence |
| A length written as a range gets resampled on every design | `"100-155"` is a distribution, not a number. `"180-180"` is one length |
| The length counts only what gets designed | 180 plus a four residue motif gives you 184 residues, not 180 |
| Leaving ligand coordinate conditioning unset conditions no pose | The model invents the ligand's pose *and* its conformer. It gets identity, bonds and charge only |
| A `?` in `pdbx_formal_charge` means unknown, not neutral | The model embeds formal charge per atom, so unknown is a question, not a default |
| PDB-derived ligands often arrive single bonded | Aromaticity and bond order are lost, and the model reads both |
| A metal with no bonds is normal, and still a decision | The agent surfaces it and asks. It never picks coordination partners by distance, because a nearby atom might be a real bond or an incidental contact and only you know which |
| Missing atoms require intent review | Keep complete residues and express tip-atom conditioning with masks. Raw unresolved structures may need preparation; custom components are valid when their explicit atoms, bonds and charges are complete |
| Token types can have different indexing capabilities | The agent preserves your requested indexing, runs the checked-out validator, and reports any unsupported combination instead of silently changing it |
| Covalent polymer-to-ligand bonds must remain present | The annotated CIF preserves and validates the bond explicitly |

> **Check the result.** An agent can misread intent. Before you run the model, inspect the
> annotated CIF and its validation report. Confirm that the conditioning says what you meant,
> including fixed atoms, sequence, ligands and unindexing.

## What it will not do

- Guess its way past a real ambiguity. It asks instead.
- Silently choose structural preparation. It tells you which preset it used and
  asks you to confirm material atom, component, and bond changes.
- Silently drop covalent bonds, charges or PTMs.
- Freeze a variable source template unintentionally. For intentionally materialized child CIFs,
  follow the [expandable-segment guidance](inputs.md#expandable-segments) and workspace policy.
- Claim a check passed that it did not run.
- Create a JSON/YAML intermediate unless you explicitly ask for one.
