# Annotated CIF Inputs

> **The recommended way to run RFProteina.** An *annotated CIF* is a standard mmCIF
> file extended with AtomWorks condition blocks. All of your design intent —
> which atoms are fixed, which sequence is known, hotspots, hydrogen bonds, RASA,
> unindexing, expandable segments, bonds, charges, PTMs — lives *inside the file*,
> per-atom and per-residue. The model reads it **natively**: no JSON, no YAML, no
> contig string, no lossy conversion step.

- [Why annotated CIFs](#why-annotated-cifs)
- [Two ways to make one](#two-ways-to-make-one)
  - [Option A — the PyMOL plugin (AtomWorks Design Studio)](#option-a--the-pymol-plugin-atomworks-design-studio)
  - [Option B — programmatically with AtomWorks](#option-b--programmatically-with-atomworks)
- [Running the model](#running-the-model)
- [Supported annotations (reference)](#supported-annotations-reference)
- [Interaction rules (read before conditioning)](#interaction-rules-read-before-conditioning)
- [Charges and bonds](#charges-and-bonds)
- [Expandable segments](#expandable-segments)
- [How annotations are stored in the CIF](#how-annotations-are-stored-in-the-cif)
- [FAQ / gotchas](#faq--gotchas)

---

## Why annotated CIFs

Designers start in PyMOL. The **legacy** path makes you transcribe that mental
model into a JSON/YAML spec, which the pipeline then converts *back* into a
structure — two lossy hops. The **plugin** path saves a CIF straight from your
PyMOL session, and the model ingests CIFs natively (CIF → `AtomArray` →
features), so it is robust and essentially lossless:

```text
Legacy:   PyMOL ──transcribe by hand──▶ JSON/YAML ──build atom array──▶ CIF ──▶ model
                        (lossy)                    (lossy in practice)

Plugin:   PyMOL ─────── Save annotated CIF ───────▶ CIF ──▶ model
                          (robust, 1:1)
```

The lossy hops hurt most where modern design is interesting — **non-canonicals,
PTMs, and small molecules**:

- **Bonds** — PDB files and JSON-rebuilt structures don't reliably preserve the
  bond graph (order, aromaticity, explicit covalent/metal-coordination bonds),
  which the model uses as features.
- **Charges** — PDB has no reliable formal-charge field, and rebuilt inputs
  default missing charges to `0`; the model embeds per-atom formal charge.
- **PTMs / non-canonicals** — awkward to express as a contig, and round-tripping
  tends to strip the atoms, bonds, and charges that define the modification.

An annotated CIF is built (or loaded) once with correct atoms, bonds, and
charges, then carries your design intent as annotations straight into the model.
**What you see is what the model gets.**

> See `docs/input.md` for the **legacy** JSON `InputSpecification` (dialect 2).
> It is still supported for RFD3 compatibility, but annotated CIFs are preferred
> for anything involving newer conditioning types or complex inputs like non-canonicals,
> PTMs, small molecules, etc.

## Two ways to make one

### Option A — the PyMOL plugin (AtomWorks Design Studio)

The plugin turns the whole workflow visual: load a real structure, edit it,
annotate exactly what you want the model to do by clicking on atoms, and **Save
Active As** an annotated CIF that goes straight into the model. No config files
are ever written.

Full setup and usage live in the plugin README:
[`lib/atomworks/src/atomworks/pymol_plugin/README.md`](../lib/atomworks/src/atomworks/pymol_plugin/README.md)
(also symlinked at [`src/rfproteina/pymol/README.md`](../src/rfproteina/pymol/README.md)).

The short version, from the **atomworks repo root**:

```bash
# 1. Build + activate the dedicated conda env (open-source PyMOL isn't pip-installable)
make setup-pymol
micromamba activate atomworks          # or mamba / conda activate atomworks
make install

# 2. Register the plugin with PyMOL (writes a loader line into ~/.pymolrc)
atomworks setup pymol

# 3. Register the RFProteina condition editors (REQUIRED — without this the
#    Annotations tab only shows base AtomWorks annotations)
atomworks pymol add rfproteina.pymol.editors

# 4. Launch PyMOL with the env active, then open Plugin -> AtomWorks
pymol
```

Inside the panel: **Load CIF** (Main tab), annotate on the **Annotations** tab
(each entry is a condition the model reads — see the table below), edit the bond
graph on the **Bonds** tab, formal charges on the **Charges** tab, insert
variable-length **expandable segments** from the Main tab, then **Save Active
As**. Toggling the **Index** condition off is how you mark residues *unindexed*.

### Option B — programmatically with AtomWorks

Parse a starting CIF into an `AtomArray`, set annotations with the AtomWorks
condition API, and write the result back out. This is ideal for scripting many
variants of a design.

See the runnable, end-to-end walkthrough:
[**`docs/tutorials/annotated_cif_tutorial.ipynb`**](tutorials/annotated_cif_tutorial.ipynb).
It parses a protein–ligand complex, applies a complex conditioning task
(unindexed tip atoms around a ligand + RASA + hydrogen-bond conditioning + an
expandable segment + a PTM), writes an annotated CIF, and closes with a loop that
generates many annotated-CIF variants (e.g. perturbing a conditioned atom's
coordinate) — all without ever touching JSON.

The core primitives (see the notebook for full context):

```python
from atomworks.io import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.conditions import C_SEQ, C_CRD, C_IDX
from rfproteina.conditions import C_HOT, C_RAS, C_HB, C_IDXGRP, RasaEnum, HBondEnum
from rfproteina.utils.parse_presets import ANNOTATED_CIF, RCSB_WITH_METALS

aa = as_atom_array_plus(parse("input.cif", config=RCSB_WITH_METALS)["assemblies"]["1"][0])

C_CRD.set_mask(aa, fixed_atom_mask)     # fix coordinates of selected atoms
C_SEQ.set_mask(aa, known_seq_res_mask)  # reveal sequence of selected residues
C_HOT.set_mask(aa, hotspot_atom_mask)   # mark hotspots

to_cif_file(aa, "annotated.cif", save_standard_annotations=True, chain_disambiguation="chain_iid")
```

## Running the model

```bash
./src/rfproteina/generate.py \
    inputs=annotated.cif \
    ckpt_path=./checkpoints ckpt_name=model.ckpt \
    num_replicates=10 batch_size=8
```

`inputs` may be a single `.cif`/`.bcif` (or `.json`/`.yaml` although this is not preferred),
a **directory** of them, or a list — each file is one design problem. Outputs echo
the source-component provenance (`SA_SRC` contig labels like `A3-10`) so you can
trace which input region each output region came from.

## Supported annotations (reference)

Each annotation below is a *condition* the model reads. The **CIF column** is the
exact key written into the file (naming scheme:
`condition_{name}_{n_body}_{level}` for values, `mask_{name}_{n_body}_{level}`
for the on/off mask; boolean "mask" conditions use a single `mask_…` column).

The **"Training Definition"** column is the important part for getting good
results: condition the model the way it was supervised. The **Import** column is
the Python shorthand used in the [notebook](tutorials/annotated_cif_tutorial.ipynb);
`C_CRD`, `C_SEQ`, and `C_IDX` come from `atomworks.ml.conditions`, the rest from
`rfproteina.conditions`.

| Condition | Import | CIF column | What it does | Values | Training Definition (where applicable) |
|---|---|---|---|---|---|
| **Fixed coordinates** (motif / "fixed atoms") | `C_CRD` | `mask_coordinate_1_atom` + `condition_coordinate_1_atom` | Gives the model the ground-truth xyz of the selected atoms; the rest is generated. | per-atom xyz (Å) |  |
| **Known sequence** (folding / fix identity) | `C_SEQ` | `mask_sequence_1_residue` + `condition_sequence_1_residue` | Reveals the residue identity; where the mask is off, sequence is designed. | per-residue `res_name`; `MASKED` where free |  |
| **Index / unindexing** | `C_IDX` | `mask_index_1_residue` | On = the model knows this residue's position in the chain; **off = unindexed** (position is for the model to place). | bool (default **all on**) | Atomized tokens cannot be unindexed. |
| **Index group** | `C_IDXGRP` | `condition_index-group_1_token` | Groups residues whose *relative* offsets are known (so an unindexed motif can still be internally rigid). | int group id |  |
| **Hotspot** | `C_HOT` | `mask_hotspot_1_atom` | Marks target-interface atoms the designed binder should engage. | bool | A hotspot is a **target-chain atom within 4.5 Å of any binder-chain atom** (Euclidean). Count sampled `geometric(p=0.2)`, capped at 10 % of eligible atoms; 5 % of the time none are given. Trained alongside a **coordinate- and sequence-conditioned target** (the binder is generated). |
| **RASA bin** | `C_RAS` | `condition_rasa-binned_1_atom` | Tells the model how buried/exposed an atom should be. Useful for steering ligand pockets and surface design. | `RasaEnum`: `0`=UNSPECIFIED, `1`=BURIED, `2`=INTERMEDIATE, `3`=EXPOSED | Relative SASA (Shrake–Rupley, probe 1.4 Å, ProtOr radii, 100 points). Bins by `rASA` edges `[0.0, 0.1, 0.2]`: `[0,0.1)`→BURIED, `[0.1,0.2)`→INTERMEDIATE, `≥0.2`→EXPOSED. |
| **Hydrogen bond** | `C_HB` | `condition_hbond_1_atom` | Requests a donor/acceptor role for an atom (e.g. specify a ligand-binding H-bond network). | `HBondEnum`: `0`=NONE, `1`=DONOR, `2`=ACCEPTOR, `3`=BOTH | Computed with `hbplus` (H–A ≤ 2.2 Å, D–A ≤ 3.2 Å). **Only H-bonds between the motif and the diffused region** are conditioned on; subsampled when many are present. |
| **Chain centroid** | `C_CENTROID` | `condition_ChainCentroid_1_atom` | Provides a target center-of-mass for a chain (position a binder/ligand without fixing atoms). | per-atom xyz (Å), broadcast per chain | Used in place of an explicit ORI token to guide COM placement. |
| **Coordinate group** | `C_CRDGRP` | `condition_CoordinateGroup_1_atom` | Groups atoms with known *relative* coordinates (internal structure without a fixed global dock). | int: `0`=known-absolute, `-1`=unknown, `1`=internal | Note internal coords are not yet supported, but will be soon |

Additional conditions are *defined* and will round-trip through a CIF but are
**not consumed by the current feature pipeline** (so they have no effect at
inference): π-stacking, π-cation, salt bridge, hydrophobic, metal-coordination,
antibody CDR/heavy/light, pairwise distance, secondary structure, not-cysteine,
CATH label, and nucleic-acid base-pairing. The plugin's Annotations tab only
exposes conditions the model actually uses.

## Interaction rules (read before conditioning)

Conditions are **not independent**. The couplings below are *enforced*: the same
batch-invariant checks that run in training also run in the plugin's **pre-save
validation**, so if you break one the plugin flags it (e.g. *"N token(s) with
non-backbone coordinate conditioning lack sequence conditioning"*) before you
save the CIF.

1. **Sidechain Coordinate conditioning ⇒ sequence conditioning.** The model represents
   coordinates in the fixed 37-slot `atom37` layout (each slot is a canonical,
   residue-specific atom name), so the pattern of populated **sidechain** slots
   fingerprints the amino acid — fixing sidechain atoms inherently reveals
   identity, and fixing even one of them reveals that residue's full sequence.
   **Exception:** backbone-only (`N/CA/C/O`) conditioning shares its slots across
   all residues and leaks nothing, so it does *not* force sequence conditioning —
   this is exactly how you fix a backbone and redesign its sequence.

2. **Atomized tokens are always sequence-conditioned.** Non-standard residues,
   PTMs, ligands, and small molecules (`atomize` = true) are context the model is
   told about, never generated. Only the 20 standard amino acids are generated
   (nucleic-acid and small-molecule generation are coming). **Note this happens
   automatically in our inference pipeline**

## Charges and bonds

The model consumes both: per-atom formal charge is embedded (11-class), and
`bond_mask` / `bond_order` (Kekulé; coordination → its own class) / `is_aromatic`
become pairwise features. This is the concrete reason to prefer annotated CIFs
for anything with a ligand, metal, covalent modification, or non-canonical: the
fidelity survives all the way to the model.

## Expandable segments

An expandable segment is the visual/structural equivalent of a variable-length
contig like `"10-15"`: a placeholder telling the model to design a segment whose
length it samples at runtime. Insert one with the plugin (Main tab → **Insert
Expandable Segment...**) or programmatically
(`atomworks.ml.utils.expandable_segment.insert_expandable_segment`), specifying
`seg_min`/`seg_max` and where it goes (within a chain, or as a new chain).

**Save with segments *collapsed*, not realized.** A collapsed segment is a
sentinel the model expands per sample; a *realized* segment has already been
frozen to one concrete length, so the model will not resample it. Only
polypeptide expandable segments are supported for now.

## FAQ / gotchas

**Do I have to use annotated CIFs?** No — JSON/YAML (`docs/input.md`) still works
for RFD3-style contigs. But for non-canonicals, PTMs, small molecules, bonds,
charges, or new conditioning types not available in RFD3, annotated CIFs are strongly
preferred and lossless.

**Can I open an annotated CIF in PyMOL / other tools?** Yes. The annotations are
extra columns/categories; the coordinates and topology are standard mmCIF. Use
the plugin's **Load CIF** (not File → Open) if you want the annotations loaded and visualized.

**Why is a fixed sidechain also fixing my sequence?** By design — `atom37`
reveals identity (see interaction rules). To redesign sequence under a fixed
fold, condition on backbone atoms only.

**Can I condition expandable segments?** Yes, for residue-based conditions like chain COM or secondary structure conditioning (coming soon), it may make sense to condition entire expanded regions at once. If you condition an expandable segment, the condition is applied to all tokens that result from its expansion.
