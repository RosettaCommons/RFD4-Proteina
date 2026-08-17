# Describe your design to Claude

Tell Claude what you want to build. It interviews you about the parts you left open, checks the
structure for the usual traps, and hands back a model ready [annotated CIF](inputs.md).

How you state the intent is up to you. A legacy RFD3 style JSON works. So does a paragraph you
typed, or one you dictated. All three are statements of intent, and all three under specify, so the
route through them is the same.

**The conversion is not the product. The interview is.** A JSON written months ago encodes a
mini-language whose defaults surprise people. A paragraph never mentions the dozen switches that
decide what the model actually sees. Either way the job is to work out what you meant and encode
*that*. Transliterating your input is the failure mode, not the goal.

## Running it

In Claude Code:

```
/design2cif <input.json> [example_key]
```

Pass the path. If the file holds several examples, pass the key you want too.

Or just say what you want:

```
/design2cif scaffold a 120-residue binder against the SAM pocket of 1abc,
            keeping the four catalytic residues fixed but letting the model place them
```

There is also an `rfd4-proteina-inputs` skill covering the same ground, which
`rfd4-proteina-generate` reaches for when it needs an input you do not have yet.

## What to expect

1. **Questions first.** Claude opens the structure rather than reasoning about it, then restates
   each thing it plans to do as intent and asks. The pattern is *"I read this as X, is that
   right?"* Your answers determine the output, so take them seriously.
2. **A JSON draft, if you started from prose.** Claude shows you the spec it inferred, in full,
   marking every value it guessed rather than heard, and waits for your sign off. It never builds
   from an unconfirmed draft. The draft stays next to the CIF as the record of what got built.
3. **A built CIF.** Written from native atomworks primitives, not the legacy parser, then
   validated.
4. **A report.** Everything worth a second look, described in terms of your design rather than
   stack traces.

## What it asks about

The interview covers the same ground whichever way you came in. Claude walks the whole list even
when your paragraph was short, because a description that never mentions hotspots or RASA usually
means you did not think about them, not that you wanted the defaults.

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
| `select_fixed_atoms: false` conditions nothing | The model invents the ligand's pose *and* its conformer. It gets identity, bonds and charge only |
| A `?` in `pdbx_formal_charge` means unknown, not neutral | The model embeds formal charge per atom, so unknown is a question, not a default |
| PDB-derived ligands often arrive single bonded | Aromaticity and bond order are lost, and the model reads both |
| A metal with no bonds is normal, and still a decision | Claude surfaces it and asks. It never picks coordination partners by distance, because a 2.42 Å neighbour might be a real bond or an incidental contact and only you know which |
| A ligand missing heavy atoms is a defect, a residue missing sidechain atoms is not | The second one is what tip atom scaffolding *is*. Gating on the combined count buries the real problem |
| Unindexing a PTM or non-canonical residue is not expressible | Claude says so plainly rather than building something that breaks downstream |
| The legacy JSON path silently drops a covalent polymer to ligand bond | This path keeps it, and tells you the result will differ from what your old JSON runs produced |

> **Check the result.** Claude misreads intent sometimes. Before you run the model, open the
> annotated CIF in the [AtomWorks Design Studio plugin](inputs.md#pymol-plugin) and confirm the
> conditioning says what you meant, including fixed atoms, sequence, ligands and unindexing.

## What it will not do

- Guess its way past a real ambiguity. It asks instead.
- Add missing atoms. It flags them so you fix them at the source.
- Silently drop covalent bonds, charges or PTMs. Those are the failure modes of the legacy JSON
  path.
- Realize an expandable segment. A realized segment is frozen to one length and the model will not
  resample it.
- Claim a check passed that it did not run.
