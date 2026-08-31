# Preparing inputs

An annotated CIF is an ordinary mmCIF with extra columns. The usual columns hold the structure, so
chains, residues, atoms and coordinates. The extra ones hold what you want the model to do with each
atom. Hold this one still. Keep that residue's identity. Treat this one as a hotspot.

Both live in the same file, so nothing gets converted and nothing gets rebuilt. What you see is what
the model gets.

<div class="cif-figure">
<svg viewBox="0 0 766 228" xmlns="http://www.w3.org/2000/svg" class="wf aa-fig" role="img" aria-labelledby="aa-t aa-d">
  <title id="aa-t">Where each condition attaches</title>
  <desc id="aa-d">An AtomArray drawn as a strip, one cell per atom, with bars above grouping the atoms into residues and the residues into chains. Below the strip, two rows show where conditions are set. Coordinate conditioning is per atom, so it marks single cells: the CA of the first glycine and the OG of the serine. Sequence conditioning is per residue, so it marks a whole residue at a time, the serine. Everything unset defaults to NaN coordinates and UNK identity, which is what the model designs.</desc>
  <text x="68" y="34" class="aa-rowlab">chain</text>
  <rect x="80" y="22" width="488" height="16" rx="3" class="aa-group"/>
  <text x="324.0" y="34" class="aa-grouplab">A</text>
  <rect x="600" y="22" width="128" height="16" rx="3" class="aa-group"/>
  <text x="664.0" y="34" class="aa-grouplab">B</text>
  <text x="68" y="56" class="aa-rowlab">residue</text>
  <rect x="80" y="44" width="128" height="16" rx="3" class="aa-group"/>
  <text x="144.0" y="56" class="aa-grouplab">GLY</text>
  <rect x="212" y="44" width="160" height="16" rx="3" class="aa-group"/>
  <text x="292.0" y="56" class="aa-grouplab">ALA</text>
  <rect x="376" y="44" width="192" height="16" rx="3" class="aa-group"/>
  <text x="472.0" y="56" class="aa-grouplab">SER</text>
  <rect x="600" y="44" width="128" height="16" rx="3" class="aa-group"/>
  <text x="664.0" y="56" class="aa-grouplab">GLY</text>
  <text x="68" y="84" class="aa-rowlab">atom</text>
  <rect x="80" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="96.0" y="84" class="aa-atomlab">N</text>
  <rect x="112" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="128.0" y="84" class="aa-atomlab">CA</text>
  <rect x="144" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="160.0" y="84" class="aa-atomlab">C</text>
  <rect x="176" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="192.0" y="84" class="aa-atomlab">O</text>
  <rect x="212" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="228.0" y="84" class="aa-atomlab">N</text>
  <rect x="244" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="260.0" y="84" class="aa-atomlab">CA</text>
  <rect x="276" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="292.0" y="84" class="aa-atomlab">C</text>
  <rect x="308" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="324.0" y="84" class="aa-atomlab">O</text>
  <rect x="340" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="356.0" y="84" class="aa-atomlab">CB</text>
  <rect x="376" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="392.0" y="84" class="aa-atomlab">N</text>
  <rect x="408" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="424.0" y="84" class="aa-atomlab">CA</text>
  <rect x="440" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="456.0" y="84" class="aa-atomlab">C</text>
  <rect x="472" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="488.0" y="84" class="aa-atomlab">O</text>
  <rect x="504" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="520.0" y="84" class="aa-atomlab">CB</text>
  <rect x="536" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="552.0" y="84" class="aa-atomlab">OG</text>
  <rect x="600" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="616.0" y="84" class="aa-atomlab">N</text>
  <rect x="632" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="648.0" y="84" class="aa-atomlab">CA</text>
  <rect x="664" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="680.0" y="84" class="aa-atomlab">C</text>
  <rect x="696" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="712.0" y="84" class="aa-atomlab">O</text>
  <text x="584.0" y="84" class="aa-ell">…</text>
  <text x="744.0" y="84" class="aa-ell">…</text>
  <path d="M80 108 H728" class="aa-rule"/>
  <text x="68" y="129" class="aa-condlab">C_CRD</text>
  <text x="68" y="142" class="aa-condsub">per atom</text>
  <rect x="80" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="112" y="120" width="32" height="20" rx="3" class="aa-on"/>
  <rect x="144" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="176" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="212" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="244" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="276" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="308" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="340" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="376" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="408" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="440" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="472" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="504" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="536" y="120" width="32" height="20" rx="3" class="aa-on"/>
  <rect x="600" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="632" y="120" width="32" height="20" rx="3" class="aa-on"/>
  <rect x="664" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="696" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <text x="68" y="165" class="aa-condlab">C_SEQ</text>
  <text x="68" y="178" class="aa-condsub">per residue</text>
  <rect x="80" y="156" width="128" height="20" rx="3" class="aa-off"/>
  <rect x="212" y="156" width="160" height="20" rx="3" class="aa-off"/>
  <rect x="376" y="156" width="192" height="20" rx="3" class="aa-on"/>
  <text x="472.0" y="170" class="aa-onlab">SER</text>
  <rect x="600" y="156" width="128" height="20" rx="3" class="aa-off"/>
  <path d="M80 196 H728" class="aa-rule aa-rule-soft"/>
  <text x="80" y="214" class="aa-note"><tspan class="aa-swatch-on">■</tspan> set, so the model is given it &#160;·&#160; <tspan class="aa-swatch-off">■</tspan> unset, so the model designs it</text>
</svg>
</div>

This example fixes two things and designs everything else. `C_CRD` pins the backbone CA of the
first glycine and the OG tip atom of the serine, so those two atoms land where you put them.
`C_SEQ` reveals the serine, so the model knows that residue has to *be* a serine rather than
whatever fits. Everything else is open: the alanine, the second chain, and every atom without a
marked cell. That is a motif of one anchored residue and one catalytic tip, scaffolded by the
model.


Notice the two rows work at different granularities, and that is the whole point. Coordinates are
per atom, which is what lets you pin one tip atom and leave the rest of its residue free. Identity
is per residue, all or nothing. Anything unset falls back to `NaN` coordinates and `UNK` identity,
and that is exactly what the model designs.

**Prefer to learn by doing?** The [annotated CIF tutorial](../examples/annotated_cif_tutorial.ipynb)
builds a pocket-conditioned input step by step and ends with an interactive view of exactly what
each annotation fixed.

## Why annotated CIFs

An annotated CIF is an ordinary structure file that also carries the conditions. They sit next to the
atoms they apply to, so the model reads one file and there is no second config to keep in sync.
Existing JSON/YAML input specifications remain supported for established workflows. Prefer an
annotated CIF when the design depends on explicit component chemistry or conditions that should be
inspected and edited alongside the structure.

Three details deserve particular attention for non-canonicals, PTMs and small molecules.

- **Bonds.** Confirm bond order, aromaticity, and explicit covalent or metal
  coordination bonds. The model uses all of them as features.
- **Charges.** Confirm formal charge rather than treating an unknown value as
  neutral. The model embeds formal charge per atom.
- **PTMs and non-canonicals.** Preserve the atoms, bonds, charges, and component
  identity that define the modification.

A CIF saved out of your PyMOL session keeps all three, and the model reads CIFs natively, so nothing
gets rebuilt on the way in.

> The [input-authoring skill](design2cif.md) can inspect an existing JSON/YAML
> specification and build an annotated CIF without requiring JSON as an
> intermediate for prose or structure-based authoring.

## Four ways to make one

Pick by how much of the design problem you need to express. The plugin is the fastest way to point
at atoms. The Python API can express anything the model can read.

| Route | Reach for it when |
| --- | --- |
| [PyMOL plugin](#pymol-plugin) | You want to select atoms by eye |
| [`rfd4-proteina-inputs` / `/design2cif`](design2cif.md) | You want an agent to interview you, inspect existing artifacts, and author or edit the CIF |
| [YAML condition spec](#yaml-condition-spec) | You want the problem as data, in version control |
| [Python API](#python-api) | You are generating many problems, or need something the others cannot say |

### PyMOL plugin

The plugin makes the whole workflow visual. Load a real structure, edit it, click on atoms to say
what the model should do with them, then **Save Active As** an annotated CIF that goes straight into
the model. It never writes a config file.

The plugin has its own walkthrough, **[AtomWorks Design
Studio](https://github.com/RosettaCommons/atomworks/blob/dev/src/atomworks/pymol_plugin/README.md)**,
covering setup, the panel, expandable segments, annotations, bonds and the console commands, with
screenshots. It lives in the `atomworks` submodule, so once you run
`git submodule update --init` you also have it locally at
`lib/atomworks/src/atomworks/pymol_plugin/README.md`.

The short version, from the **rfproteina repo root**:

```bash
# 0. Install conda or miniconda on your machine (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

# 1. Build and activate the dedicated conda env (open source PyMOL is not pip installable)
make setup-pymol
conda activate rfproteina          # or mamba / conda activate atomworks
uv pip install -e "./lib/atomworks[ml]" -e ./

# 2. Register the plugin with PyMOL (writes a loader line into ~/.pymolrc)
atomworks setup pymol

# 3. Register the RFProteina condition editors. REQUIRED. Without this the
#    Annotations tab only shows base AtomWorks annotations.
atomworks pymol add rfproteina.pymol.editors

# 4. Launch PyMOL with the env active, then open Plugin -> AtomWorks
pymol
```

**On Windows** there is no `make`, so replace `make setup-pymol` with
`conda env create -f src\rfproteina\pymol\pymol_env.yaml` and leave the other steps as they are.
Three things bite only there, all covered in the plugin README's *Windows (PowerShell)* section:
`conda activate` silently no-ops until you run `conda init powershell` and
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`; the submodule clone needs an elevated
`Start-Service ssh-agent` plus an SSH key authorized for `private-organization` SSO; and pip's
`pymol requires numpy<2` warning is stale metadata that you should not act on.

Inside the panel, start with **Load CIF** on the Main tab. Annotate on the **Annotations** tab,
where each entry is a condition the model reads, listed in the table below. Edit the bond graph on
the **Bonds** tab and formal charges on the **Charges** tab. Insert variable length **expandable
segments** from the Main tab. Then **Save Active As**. To mark residues *unindexed*, toggle the
**Index** condition off.

### Python API

Parse a starting CIF into an `AtomArray`, set annotations with the AtomWorks condition API, and
write the result back out. Use this when you want to script many variants of a design.

The [annotated CIF tutorial](../examples/annotated_cif_tutorial.ipynb) runs the whole thing end to
end. It parses a protein and ligand complex, applies a hard conditioning task with unindexed tip
atoms around the ligand, RASA, hydrogen bond conditioning, an expandable segment and a PTM, writes
an annotated CIF, and finishes with a loop that generates many variants by perturbing a conditioned
atom's coordinate. No JSON anywhere.

Here are the core primitives. The notebook gives them full context.

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

### YAML condition spec

To script or template design problems without writing Python or driving PyMOL, describe them in a
small YAML or JSON file. You subset the input, insert expandable segments, and set conditions on
atom selections. From there you either write annotated CIFs or hand the spec straight to
`generate.py`. Either way you end up with the same annotated CIF representation described above.
This is shorthand for authoring one, not a different format.

The [condition spec reference](condition-spec.md) has the full syntax and examples.

## Running the model

```bash
./src/rfproteina/generate.py \
    inputs=annotated.cif \
    num_replicates=10 batch_size=8
```

`inputs` takes a single `.cif` or `.bcif`, a **directory** of them, or a list. Each file is one
design problem. It also takes `.json` and `.yaml`, though we do not recommend those. Outputs echo
source component provenance as per-residue `SA_SRC` labels like `A3`; expandable segments retain
range labels like `3-10`. This lets you trace each output residue to its input residue or inserted
segment.

## Supported annotations (reference)

Each annotation below is a *condition* the model reads. The **CIF column** is the exact key written
into the file. Values follow `condition_{name}_{n_body}_{level}` and the on/off mask follows
`mask_{name}_{n_body}_{level}`. Boolean mask conditions use a single `mask_…` column.

The **Training Definition** column matters most for getting good results. Condition the model the
way it was supervised. The **Import** column gives the Python shorthand the
[notebook](../examples/annotated_cif_tutorial.ipynb) uses. `C_CRD`, `C_SEQ` and `C_IDX` come from
`atomworks.ml.conditions`, and the rest from `rfproteina.conditions`.

| Condition | Import | CIF column | What it does | Values | Training Definition (where applicable) |
|---|---|---|---|---|---|
| **Fixed coordinates** (motif / "fixed atoms") | `C_CRD` | `mask_coordinate_1_atom` + `condition_coordinate_1_atom` | Gives the model the ground truth xyz of the selected atoms. Everything else is generated. | per-atom xyz (Å) |  |
| **Known sequence** (folding / fix identity) | `C_SEQ` | `mask_sequence_1_residue` + `condition_sequence_1_residue` | Reveals the residue identity. Where the mask is off, the sequence is designed. | per residue `res_name`, `MASKED` where free |  |
| **Index / unindexing** | `C_IDX` | `mask_index_1_residue` | On means the model knows this residue's position in the chain. **Off means unindexed**, so the model places it. | bool (default **all on**) | Atomized tokens cannot be unindexed. |
| **Index group** | `C_IDXGRP` | `condition_index-group_1_token` | Groups residues whose *relative* offsets are known, so an unindexed motif can still be internally rigid. | int group id |  |
| **Hotspot** | `C_HOT` | `mask_hotspot_1_atom` | Marks target-interface atoms the designed binder should engage. | bool | A hotspot is a **target-chain atom within 4.5 Å of any binder-chain atom** (Euclidean). Count sampled `geometric(p=0.2)`, capped at 10 % of eligible atoms. 5 % of the time none are given. Trained alongside a **target conditioned on both coordinates and sequence**, with the binder generated. |
| **RASA bin** | `C_RAS` | `condition_rasa-binned_1_atom` | Tells the model how buried or exposed an atom should be. Useful for steering ligand pockets and surface design. | `RasaEnum`: `0`=UNSPECIFIED, `1`=BURIED, `2`=INTERMEDIATE, `3`=EXPOSED | Relative SASA, computed Shrake and Rupley with a 1.4 Å probe, ProtOr radii and 100 points. Bins by `rASA` edges `[0.0, 0.1, 0.2]`: `[0,0.1)`→BURIED, `[0.1,0.2)`→INTERMEDIATE, `≥0.2`→EXPOSED. |
| **Hydrogen bond** | `C_HB` | `condition_hbond_1_atom` | Requests a donor or acceptor role for an atom, which is how you specify a ligand binding H-bond network. | `HBondEnum`: `0`=NONE, `1`=DONOR, `2`=ACCEPTOR. The enum defines `3`=BOTH, but current input validation rejects it on valid atoms. | Computed with `hbplus`, using H to A ≤ 2.2 Å and D to A ≤ 3.2 Å. **Only H-bonds between the motif and the diffused region** are conditioned on, and they get subsampled when there are many. |
| **Chain centroid** | `C_CENTROID` | `condition_ChainCentroid_1_atom` | Provides a target center of mass for a chain, so you can position a binder or ligand without fixing any atoms. | per-atom xyz (Å), broadcast per chain | Used in place of an explicit ORI token to guide COM placement. |
| **Secondary structure** | `C_SEC` | `condition_secondary-structure_1_residue` | Requests helix, sheet or loop at the selected residues. Leave `NONE` where you do not care. | `SSEnum`: `0`=NONE, `1`=HELIX, `2`=SHEET, `3`=LOOP | P-SEA assignment on the ground-truth structure. |
| **Internal coordinate** | `C_INTCRD` | `mask_InternalCoordinate_1_atom` + `condition_InternalCoordinate_1_atom` | Gives the *internal* geometry of a group of atoms, centered and randomly rotated, without fixing where it sits globally. Mutually exclusive with `C_CRD` on the same atom. | per atom xyz (Å) in the group's own frame, NaN where unset | Trained by converting sampled motif regions from absolute to internal conditioning, so one design can carry several rigid motifs placed independently. |
| **Coordinate group** | `C_CRDGRP` | `condition_CoordinateGroup_1_atom` | Groups atoms with known *relative* coordinates, meaning internal structure without a fixed global dock. | int: `-1`=unknown, `0`=known-absolute, `≥1`=one internal-coordinate group per id | Each internal coord region gets its own id. Relative geometry is known within an id and unknown across ids. |

A few more conditions are *defined* and will round trip through a CIF, but the current feature
pipeline **does not consume them**, so they do nothing at inference. Those are π-stacking, π-cation,
salt bridge, hydrophobic, metal coordination, antibody CDR, heavy and light chain, pairwise
distance, not-cysteine, CATH label, and nucleic acid base pairing. The plugin's Annotations tab only
shows you conditions the model actually uses.

### Combining them, in a motif problem

A motif scaffolding problem uses the first three at once. A theozyme is the clearest case, being a
handful of catalytic residues and a substrate with no scaffold, which you want a protein built
around. Condition only the atoms in the motif and leave the rest for the model to design.

| Condition | Granularity | In a theozyme |
| --- | --- | --- |
| `C_CRD`, coordinate | per **atom** | Pin the atoms that set the active site geometry, so a catalytic residue's functional atoms and the substrate's reactive core. Partial on purpose. Leave flexible parts like a phosphate tail free. |
| `C_SEQ`, sequence | per **residue** | Fix the identity of a residue or the substrate. All or nothing within one residue, though across a motif you can reveal it on some residues and not others. |
| `C_IDX`, index | per **residue** | On by default. Turn it **off** on the catalytic residues so the model places them freely along the new chain, which makes an *unindexed* motif. |

So a catalytic residue usually fixes its functional atoms with `C_CRD` and its identity with
`C_SEQ`, while staying unindexed with `C_IDX` off. The substrate fixes its reactive atoms and
identity and keeps its default index. The [enzyme design
tutorial](../examples/enzyme_design_tutorial.ipynb) builds one of these and runs it end to end.

### Status at a glance

Which conditions the model actually acts on today. The **Input field** column names the
equivalent field in the legacy RFD3 JSON, for anyone migrating one.

| Conditioning | Input field | Status |
| --- | --- | --- |
| **Coordinate (indexed motif)** | `contig` + `select_fixed_atoms` | Supported |
| **Coordinate (unindexed motif)** | `unindex` | Supported |
| **Sequence constraint** | `select_unfixed_sequence` | Supported |
| **Hotspot residues** | `select_hotspots` | Supported |
| **Partial diffusion** | `partial_t_backbone` / `partial_t_latent` | Supported |
| **Inverse folding** | `generation=inverse_folding` | Supported |
| **Small molecules and ligands** | `ligand` | Supported |
| **Internal coordinate** | `internal_coord_groups` | Supported |
| **H-bond donors and acceptors** | `select_hbond_donor` / `select_hbond_acceptor` | Annotated CIF only |
| **RASA (buried or exposed)** | `select_buried` / `select_exposed` | Annotated CIF only |
| **Secondary structure** | annotated CIF | Annotated CIF only |
| **Symmetry** | `symmetry` | Planned |
| **CATH fold class** | `add_cath_code` | Planned |
| **π-stacking** | annotated CIF | Planned |
| **π-cation** | annotated CIF | Planned |
| **Not-cysteine** | annotated CIF | Planned |

**Planned** conditions parse and get stored, but the model is not trained on them yet. Passing
one raises no error and changes nothing.

## Interaction rules, read before conditioning

Conditions are **not independent**, and the couplings below get enforced. The same batch invariant
checks that run in training also run in the plugin's **pre-save validation**, so if you break one
the plugin tells you before you save the CIF, with something like *"N token(s) with non-backbone
coordinate conditioning lack sequence conditioning"*.

1. **Conditioning a sidechain coordinate forces sequence conditioning.** The model lays coordinates
   out in a fixed 37 slot `atom37` layout, where each slot is a canonical atom name specific to a
   residue. The pattern of filled **sidechain** slots fingerprints the amino acid, so fixing
   sidechain atoms reveals identity whether you meant to or not. Fixing even one of them reveals
   that residue's full sequence. There is one exception. Backbone only conditioning on `N`, `CA`,
   `C` and `O` shares its slots across every residue and leaks nothing, so it does *not* force
   sequence conditioning. That is exactly how you fix a backbone and redesign its sequence. The rule
   applies to internal coordinate conditioning (`C_INTCRD`) the same way it applies to absolute
   conditioning (`C_CRD`), because the slot occupancy fingerprint leaks identity either way.

2. **Atomized tokens are always sequence conditioned.** Non-standard residues, PTMs, ligands and
   small molecules, everything with `atomize` set true, are context the model gets told about and
   never generates. The model generates only the 20 standard amino acids. Nucleic acid and small
   molecule generation are coming. Our inference pipeline handles this for you automatically.

## Charges and bonds

The model consumes both. It embeds formal charge per atom across 11 classes, and it turns
`bond_mask`, `bond_order` (Kekulé, with coordination as its own class) and `is_aromatic` into
pairwise features. That is the concrete reason to use annotated CIFs for anything with a ligand,
metal, covalent modification or non-canonical. The fidelity survives all the way to the model.

**Kekulize your atomized regions.** Do not leave any bond type as `AROMATIC`, because those all get
cast to `SINGLE`. Edit them by hand to `AROMATIC_SINGLE` or `AROMATIC_DOUBLE` based on a valid
kekulization. Then both the aromatic information and the bond order reach the model intact. Atomized
regions cover any non-standard residue, ligands, post translational modifications including the
canonical residue carrying the PTM, and any other residue marked `S_ATM` true.

## Expandable segments

An expandable segment is the structural equivalent of a variable length contig like `"10-15"`. It is
a placeholder that tells the model to design a segment and sample its length at runtime. Insert one
from the plugin's Main tab under **Insert Expandable Segment...**, or in code with
`atomworks.ml.utils.expandable_segment.insert_expandable_segment`. You give it `seg_min`, `seg_max`,
and where it goes, either inside a chain or as a new chain.

**Save your segments *collapsed*, not realized.** A collapsed segment is a sentinel that the model
expands per sample. A realized segment is already frozen to one concrete length, so the model will
not resample it. Only polypeptide expandable segments work for now.

## FAQ and gotchas

**Do I have to use annotated CIFs?** No. Existing JSON inputs and the [YAML
condition spec](condition-spec.md) remain supported. Annotated CIFs are useful
when conditions and explicit component chemistry should be inspected and edited
with the structure. The input-authoring skill can [interpret an existing
JSON/YAML specification](design2cif.md), but it does not require JSON for prose
or CIF-based workflows.

**Can I open an annotated CIF in PyMOL or other tools?** Yes. The annotations are extra columns and
categories, and the coordinates and topology are standard mmCIF. To load and visualize the
annotations, use the plugin's **Load CIF** rather than File then Open.

**Why is a fixed sidechain also fixing my sequence?** Because `atom37` reveals identity, as the
interaction rules explain. To redesign sequence under a fixed fold, condition on backbone atoms
only.

**Can I condition expandable segments?** Yes. For residue based conditions like chain COM or
secondary structure, conditioning a whole expanded region at once often makes sense. Condition an
expandable segment and the condition applies to every token its expansion produces.

**I have a ligand bound to a peptide chain via a PTM. Should it have its own `IndexGroup`?** No. The
model reasons over _molecules_, not _chains_. A PTM is part of the same molecule as the peptide
chain by definition, so a valid bond graph distance exists between atoms on the PTM and any residue
on the main chain. The model sees bond graph distance features whenever the tokens share a molecule
and an `IndexGroup`. Gaps in `res_id` between the PTM and the peptide chain are fine, because we use
bond graph distance for relative positional encoding of atomized segments instead of `res_id` gaps.

```{toctree}
:hidden:
:maxdepth: 1

condition-spec
```
