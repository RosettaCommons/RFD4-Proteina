# Annotations reference

Every condition the model reads, the CIF column it lives in, the values it accepts, and how the
model was trained on it. 

For *why* conditions behave the way they do (the active mask, the
couplings between conditions, absolute vs. internal coordinates, and why charges and bonds matter)
see [Conditioning](../concepts/conditioning.md). For how to build an annotated CIF that sets these
conditions in the first place, see the [annotated CIF guide](../how-to/input_prep/index.md); for the
YAML/JSON shorthand for setting them, see the [condition-spec reference](yaml-json.md).

**Contents**

- [Supported annotations](#supported-annotations)
- [Coordinate groups](#coordinate-groups)
- [Status at a glance](#status-at-a-glance)
- [Conditions are not independent](#conditions-are-not-independent)
- [Charges and bonds](#charges-and-bonds)

## Supported annotations

Each row below is a *condition* the model reads. The **CIF column** is the exact key written into
the file; its name encodes the condition and the level it applies to, for example
`condition_coordinate_1_atom` for per-atom coordinate conditioning.

Every condition has an **active mask** read off its value column (see
[Conditioning](../concepts/conditioning.md#every-condition-is-on-or-off-per-position) for how); where
the mask is true the model is given the value, where false it designs that value itself.

The **Training Definition** column matters most for getting good results — see
[Conditioning](../concepts/conditioning.md#condition-the-way-the-model-was-trained) for why.
The **Import** column gives the Python shorthand the
[Python API tutorial](../examples/python_api_tutorial.ipynb) uses. `C_CRD`, `C_SEQ` and
`C_IDX` come from `atomworks.ml.conditions`, and the rest from `rfproteina.conditions`.

| Condition | Import | CIF column | What it does | Values | Training Definition (where applicable) |
|---|---|---|---|---|---|
| **Fixed coordinates** (motif / "fixed atoms") | `C_CRD` | `condition_coordinate_1_atom` | Gives the model the ground truth xyz of the selected atoms. Everything else is generated. | per-atom xyz (Å), NaN where free |  |
| **Known sequence** (folding / fix identity) | `C_SEQ` | `condition_sequence_1_residue` | Reveals the residue identity. Where the value is `MASKED`, the sequence is designed. | per residue `res_name`, `MASKED` where free |  |
| **Sequence probabilities** | `SEQUENCE_PROBABILITY_CONDITIONS` | `condition_sequence-probability-{token}_1_residue` | Partially constrains probabilities over the 28 canonical polymer identities. Explicit zero excludes an identity; NaN leaves it flexible. `C_SEQ` fills the complete exact distribution and rejects conflicting values. | float in `[0, 1]`, NaN where unset; specified values sum to at most 1 | Eligible `C_SEQ` positions are diversified into exact, zero-only, soft-positive, or mixed zero-and-positive conditions. |
| **Index / unindexing** | `C_IDX` | `condition_index_1_residue` | `True` means the model knows this residue's position in the chain. **False means unindexed**, so the model places it. | bool (default **all on**) | Atomized tokens cannot be unindexed. |
| **Index group** | `C_IDXGRP` | `condition_index-group_1_token` | Groups residues whose *relative* offsets are known, so an unindexed motif can still be internally rigid. | int group id |  |
| **Hotspot** | `C_HOT` | `condition_hotspot_1_atom` | Marks target-interface atoms the designed binder should engage. | bool | A hotspot is a **target-chain atom within 4.5 Å of any binder-chain atom** (Euclidean). Count sampled `geometric(p=0.2)`, capped at 10 % of eligible atoms. 5 % of the time none are given. Trained alongside a **target conditioned on both coordinates and sequence**, with the binder generated. |
| **RASA bin** | `C_RAS` | `condition_rasa-binned_1_atom` | Tells the model how buried or exposed an atom should be. Useful for steering ligand pockets and surface design. | `RasaEnum`: `0`=UNSPECIFIED, `1`=BURIED, `2`=INTERMEDIATE, `3`=EXPOSED | Relative SASA, computed Shrake and Rupley with a 1.4 Å probe, element radii and 100 points. After rounding `rASA` to the sampling grid: `[0,0.02)`→BURIED, `[0.02,0.1]`→INTERMEDIATE, `>0.1`→EXPOSED. |
| **Hydrogen bond** | `C_HB` | `condition_hbond_1_atom` | Requests donor, acceptor, or both roles for an atom in a binding H-bond network. | `HBondEnum`: `0`=NONE, `1`=DONOR, `2`=ACCEPTOR, `3`=BOTH. | Uses interchain H-bond roles precomputed with AtomWorks' implementation of the PLIP definition. The model receives separate donor and acceptor channels; BOTH activates both. |
| **π-stacking** | `C_PISTACK` | `condition_pistack_1_atom` | Requests aromatic stacking contacts with the generated partner. | bool | Select every atom of the intended aromatic ring; training subsamples whole rings. |
| **π-cation** | `C_PICATION` | `condition_pication_1_atom` | Requests the aromatic or cation side of a π-cation interaction. | `PiCationEnum`: `0`=NONE, `1`=AROMATIC, `2`=CATION, `3`=BOTH | AROMATIC describes the selected atoms, not the desired partner. Select complete aromatic rings for AROMATIC, or single atoms for CATION; BOTH requests both roles in different interactions (e.g. a histidine whose positively-charged nitrogen is the cation in another pi-cation interaction). |
| **CAT fold** | `C_CATH` | `condition_cath-label_1_residue` | Requests a fold class, architecture, or topology for the selected residues using the CATH hierarchy. | Quoted C.A.T string: `1.x.x`, `2.60.40`; `x.x.x` clears it | Unspecified levels must be a trailing suffix. Select a whole chain or domain, with consistent values within each residue. |
| **Chain centroid** | `C_CENTROID` | `condition_ChainCentroid_1_atom` | Provides a target center of mass for a chain, so you can position a binder or ligand without fixing any atoms. | per-atom xyz (Å), broadcast per chain | Used in place of an explicit ORI token to guide COM placement. |
| **Secondary structure** | `C_SEC` | `condition_secondary-structure_1_residue` | Requests helix, sheet or loop at the selected residues. Leave `NONE` where you do not care. | `SSEnum`: `0`=NONE, `1`=HELIX, `2`=SHEET, `3`=LOOP | P-SEA assignment on the ground-truth structure. |
| **Internal coordinate** | `C_INTCRD` | `condition_InternalCoordinate_1_atom` | Gives the *internal* geometry of a group of atoms, centered and randomly rotated, without fixing where it sits globally. Mutually exclusive with `C_CRD` on the same atom. | per atom xyz (Å) in the group's own frame, NaN where unset | Trained by converting sampled motif regions from absolute to internal conditioning, so one design can carry several rigid motifs placed independently. |
| **Coordinate group** | `C_CRDGRP` | `condition_CoordinateGroup_1_atom` | Groups atoms with known *relative* coordinates, meaning internal structure without a fixed global dock. | int: `-1`=unknown, `0`=known-absolute, `≥1`=one internal-coordinate group per id | Each internal coord region gets its own id. Relative geometry is known within an id and unknown across ids. |

These conditions can be authored with the Python API or YAML/JSON condition specs, and stored in
annotated CIF/BCIF files. Do not infer conditioning support merely because an annotation can be
stored.

## Coordinate groups

See [Absolute vs. internal coordinates](../concepts/conditioning.md#absolute-vs-internal-coordinates)
for why `C_INTCRD` and `C_CRDGRP` exist. This section covers the mechanics of editing them.

Edit internal coordinate conditioning through `C_CRDGRP`. The reserved groups (`0` absolute, `-1`
unknown) are managed automatically. Newly authored groups get their targets during model-input
preparation, before unindexing. To move atoms out of group `0`, clear their absolute conditioning
through `C_CRD` first. Complete saved `C_INTCRD` targets remain unchanged until group membership
changes. A whole-group rename preserves targets. Combining independent inputs remaps group IDs so
unrelated saved coordinate frames cannot silently merge.

In Python, `C_CRDGRP.set_annotation(array, groups)` raises if an edit would invalidate saved
`C_INTCRD` targets. Pass `on_internal_coord_mismatch="clear"` to clear targets throughout affected
groups; unchanged whole groups retain their targets even when their IDs change. Pass a new group
array instead of modifying the existing annotation in place. Group `0` must match `C_CRD` under
both policies.

## Status at a glance

Which conditions the model actually acts on today.

| Conditioning | Status |
| --- | --- |
| **Coordinate (indexed motif)** | Supported |
| **Coordinate (unindexed motif)** | Supported |
| **Sequence constraint** | Supported |
| **Hotspot residues** | Supported |
| **Partial diffusion** | Supported |
| **Inverse folding** | Supported |
| **Small molecules and ligands** | Supported |
| **Internal coordinate** | Supported |
| **H-bond donor, acceptor, or BOTH** | Supported |
| **RASA (buried or exposed)** | Supported |
| **Secondary structure** | Supported |
| **Symmetry** | Planned |
| **CAT fold class, architecture, topology** | Supported |
| **Sequence probabilities** | Supported; probabilities in `[0, 1]`, NaN where unset |
| **π-stacking** | Supported |
| **π-cation** | Supported |

Supported conditions can be authored through Python, annotated CIFs, or
[condition-spec YAML/JSON](yaml-json.md).

Set `C_SEQ_PROB_CYS` to `0` on selected residues to request cysteine exclusion. Set it to NaN to
remove that identity's constraint; zero actively excludes it. Validation rejects invalid
probability sums, positive probabilities for the wrong polymer type, and conflicts with exact
sequence conditions. These are model conditions, not hard output constraints: inspect
sequence-probability adherence when exclusions matter.

## Conditions are not independent

Conditions are **not independent** — see [Conditions are not
independent](../concepts/conditioning.md#conditions-are-not-independent) for the couplings that get
enforced, including why a sidechain coordinate forces sequence conditioning and why atomized tokens
are always sequence conditioned. The same consistency checks used during training also run during
**input validation**, which reports violations before generation, with messages such as *"N token(s)
with non-backbone coordinate conditioning lack sequence conditioning"*.

## Charges and bonds

The model consumes both: formal charge per atom across 11 classes, and `bond_mask`, `bond_order`
(Kekulé, with coordination as its own class) and `is_aromatic` as pairwise features. See [Why charges
and bonds matter](../concepts/conditioning.md#why-charges-and-bonds-matter) for why this makes
annotated CIFs necessary for ligands, metals, covalent modifications, and non-canonicals.

**Kekulize your atomized regions.** Do not leave any bond type as `AROMATIC`, because those all get
cast to `SINGLE`. Edit them by hand to `AROMATIC_SINGLE` or `AROMATIC_DOUBLE` based on a valid
kekulization. Then both the aromatic information and the bond order reach the model intact. Atomized
regions cover any non-standard residue, ligand, or post-translational modification — including the
canonical residue carrying the PTM — and any other residue marked as atomized.
