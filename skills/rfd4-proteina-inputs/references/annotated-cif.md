# Annotated CIF reference

Source of truth: `docs/guides/annotated_cif_inputs.md` and
`docs/guides/tutorials/annotated_cif_tutorial.ipynb`.

An annotated CIF carries design intent as extra `_atom_site` columns and
categories. Naming scheme: `condition_{name}_{n_body}_{level}` for values,
`mask_{name}_{n_body}_{level}` for the on/off mask (boolean conditions use a
single `mask_...` column).

## Supported conditions (consumed by the model)

| Condition | Import | CIF column | Level | What it does |
| --- | --- | --- | --- | --- |
| Fixed coordinates | `C_CRD` | `mask_coordinate_1_atom` + `condition_coordinate_1_atom` | atom | Ground-truth xyz for selected atoms; rest generated |
| Known sequence | `C_SEQ` | `mask_sequence_1_residue` + `condition_sequence_1_residue` | residue | Reveals residue identity; free where mask off |
| Index / unindex | `C_IDX` | `mask_index_1_residue` | residue | On = position known; **off = unindexed**. Default all-on |
| Index group | `C_IDXGRP` | `condition_index-group_1_token` | token | Ties residues with known *relative* offsets (rigid but unindexed) |
| Hotspot | `C_HOT` | `mask_hotspot_1_atom` | atom | Target-interface atoms the binder should engage |
| RASA bin | `C_RAS` | `condition_rasa-binned_1_atom` | atom | `RasaEnum`: 0 UNSPECIFIED, 1 BURIED, 2 INTERMEDIATE, 3 EXPOSED |
| Hydrogen bond | `C_HB` | `condition_hbond_1_atom` | atom | `HBondEnum`: 0 NONE, 1 DONOR, 2 ACCEPTOR, 3 BOTH |
| Chain centroid | `C_CENTROID` | `condition_ChainCentroid_1_atom` | atom | Target center-of-mass for a chain (place a binder/ligand w/o fixing atoms) |
| Secondary structure | `C_SEC` | `condition_secondary-structure_1_residue` | residue | `SSEnum`: 0 NONE, 1 HELIX, 2 SHEET, 3 LOOP |
| Internal coordinate | `C_INTCRD` | `mask_InternalCoordinate_1_atom` + `condition_InternalCoordinate_1_atom` | atom | Internal geometry without a fixed global dock; mutually exclusive with `C_CRD` on the same atom |
| Coordinate group | `C_CRDGRP` | `condition_CoordinateGroup_1_atom` | atom | Groups atoms with known relative coords (−1 unknown, 0 known-absolute, ≥1 internal group id) |

`C_CRD`, `C_SEQ`, `C_IDX` import from `atomworks.ml.conditions`; the rest from
`rfproteina.conditions`.

Defined but **not consumed** by the current pipeline (round-trip only, no effect):
π-stacking, π-cation, salt bridge, hydrophobic, metal-coordination, antibody
CDR/heavy/light, pairwise distance, not-cysteine, CATH label, nucleic-acid
base-pairing.

## Interaction rules (enforced)

1. **Sidechain `C_CRD` => `C_SEQ`.** The atom37 layout fingerprints identity, so
   fixing any sidechain atom reveals that residue's sequence. **Exception:**
   backbone-only (`N/CA/C/O`) leaks nothing — this is how you fix a fold and
   redesign its sequence. Applies to `C_INTCRD` too.
2. **Atomized tokens are always sequence-conditioned.** Only the 20 standard amino
   acids are generated.

## Charges and bonds

The model embeds per-atom formal charge (11-class) and pairwise `bond_mask` /
`bond_order` (Kekulé; coordination is its own class) / `is_aromatic`. **Kekulize
atomized regions** — never leave bond type `AROMATIC` (cast to `SINGLE`); use
`AROMATIC_SINGLE`/`AROMATIC_DOUBLE`. Atomized regions = any non-standard residue,
ligand, PTM (including the canonical residue carrying it), or anything with
`S_ATM=True`.

## Expandable segments

Insert with the plugin (Main -> Insert Expandable Segment) or
`atomworks.ml.utils.expandable_segment.insert_expandable_segment` (`seg_min` /
`seg_max`). **Save collapsed, not realized** — a realized segment is frozen to one
length. Only polypeptide expandable segments are supported.

## Programmatic core (from the tutorial)

```python
import rfproteina.conditions  # noqa: F401  REQUIRED before any parse
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

## Output columns (provenance)

Generated designs echo where each atom came from (read with the aliases in
`rfproteina.utils.standard_annotations`):

| Alias | CIF column | Meaning |
| --- | --- | --- |
| `SA_SRC` | `annotation_src-component_1_atom` | Which input-spec part the atom originated from |
| `SA_CRDSRC` | `mapped_to_coordinate_conditioned_region` | True for coordinate-conditioned motif atoms |
| `SA_CRDGRPSRC` | `annotation_src-coord-group_1_atom` | Which input `CoordinateGroup` (−1 if none) |
| `SA_UII` | `is_unindexed_insertion` | True for atoms from an unindexed input motif |
