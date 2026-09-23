# Direct Python authoring

Use this reference after the intent plan is confirmed. The Python recipe is an
executable record of the build; it does not need a JSON or YAML companion.

## Load without losing annotations

Register RFD4-Proteína conditions before parsing, then choose the preset from the
artifact rather than from whether the file is used for training or inference:

- Use `ANNOTATED_CIF` when the source already carries StandardAnnotations or its
  exact authored atoms and topology must be preserved.
- Use `RCSB_WITH_METALS` for a raw or potentially incomplete experimental
  structure. It performs structural preparation, including missing-atom handling,
  component cleanup, MSE conversion, and alternate-location selection. Inspect
  and confirm the resulting assembly, atoms, components, and bonds before adding
  conditions.

```python
from pathlib import Path

import numpy as np

import rfproteina.conditions  # noqa: F401
from atomworks.io import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file
from rfproteina.utils.parse_presets import ANNOTATED_CIF, RCSB_WITH_METALS

source = Path("input.cif")
parse_config = ANNOTATED_CIF  # existing annotated/authored CIF
# parse_config = RCSB_WITH_METALS  # raw or potentially incomplete source
parsed = parse(source, config=parse_config)
assembly = parsed["assemblies"]["1"][0]  # choose the confirmed assembly key
aa = as_atom_array_plus(assembly.copy())
```

Do not blindly select assembly `"1"`; inspect the parse result and use the assembly
the user confirmed. Parsing still resolves a model and alternate location. Record
and confirm those choices when the source contains alternatives.

For an existing annotated CIF, take each current condition as the starting point.
Condition setters replace the complete annotation, so copy the current value,
change the selected entries, then write it back.

## Apply and edit conditions

```python
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from rfproteina.conditions import C_HB, C_HOT, C_IDXGRP, C_RAS, HBondEnum, RasaEnum

fixed_atoms = (aa.chain_id == "A") & (aa.res_id == 42) & np.isin(aa.atom_name, ["N", "CA", "C", "O"])
known_residue = (aa.chain_id == "A") & (aa.res_id == 42)

coord_mask = C_CRD.mask(aa, default="generate").copy()
coord_mask[fixed_atoms] = True
C_CRD.set_mask(aa, coord_mask)

seq_mask = C_SEQ.mask(aa, default="generate").copy()
seq_mask[known_residue] = True
C_SEQ.set_mask(aa, seq_mask)

index_mask = C_IDX.mask(aa, default="generate").copy()
index_mask[known_residue] = False
C_IDX.set_mask(aa, index_mask)

hotspot_mask = C_HOT.mask(aa, default="generate").copy()
hotspot_mask[target_atoms] = True
C_HOT.set_mask(aa, hotspot_mask)

rasa = C_RAS.annotation(aa, default="generate").copy()
rasa[pocket_atoms] = RasaEnum.BURIED.value
C_RAS.set_annotation(aa, rasa)

hbond = C_HB.annotation(aa, default="generate").copy()
hbond[donor_atoms] = HBondEnum.DONOR.value
hbond[acceptor_atoms] = HBondEnum.ACCEPTOR.value
hbond[donor_atoms & acceptor_atoms] = HBondEnum.BOTH.value
C_HB.set_annotation(aa, hbond)
```

`HBondEnum.BOTH` requests both donor and acceptor roles. Use it explicitly when an atom
must have both roles; sequential writes
of DONOR and ACCEPTOR replace each other. Run the model-input validator after authoring.

For fold conditioning, expose only CAT (class, architecture, topology) labels:

```python
from rfproteina.utils.condition_resolution import set_cath_condition

set_cath_condition(aa, np.flatnonzero(aa.chain_id == "A"), "2.60.40")
# Clear fold conditioning on that chain:
set_cath_condition(aa, np.flatnonzero(aa.chain_id == "A"), "x.x.x")
```

This helper preserves the internal C.A.T.x annotation format and prevents string truncation.

Masks for residue-level conditions must have one value across every atom in a
residue. Index-group annotations must be consistent across each token. When
unindexing, copy `C_IDXGRP.annotation(aa, default="generate")`, assign group IDs
that encode which pieces retain relative offsets, and set the full annotation
with `C_IDXGRP.set_annotation()`.

For internal-coordinate, centroid, secondary-structure, or coordinate-group
conditioning, follow the working examples in
`docs/examples/annotated_cif_tutorial.ipynb` and the implementations registered
in `src/rfproteina/conditions_reporting.py`.

## Add a variable segment

```python
from atomworks.ml.utils.expandable_segment import insert_expandable_segment

aa = insert_expandable_segment(
    aa,
    seg_min=100,
    seg_max=140,
    atom_idx=first_atom_of_anchor_residue,
    insert_before=True,
)
```

`atom_idx` must be the first atom of a residue when inserting before it and the
last atom when inserting after it. To append a new chain, pass `new_chain=True`, a
new chain ID, and a `ChainType`. Leave the returned sentinel collapsed in the file.

For a fixed de novo segment, construct coordinate-free residue stubs with
Biotite/AtomWorks primitives; do not create and realize an expandable range. Use
the AtomWorks backbone and tip-atom constants so the stub matches the model's
expected residue representation:

```python
from atomworks.constants import PROTEIN_BACKBONE_ATOM_NAMES, STANDARD_AA_TIP_ATOM_NAMES
from biotite.structure import Atom, array as struc_array

atom_names = tuple(name for name in PROTEIN_BACKBONE_ATOM_NAMES if name != "OXT")
atom_names += tuple(STANDARD_AA_TIP_ATOM_NAMES["ALA"])
atoms = [
    Atom(np.zeros(3, dtype=np.float32), res_name="ALA", res_id=res_id)
    for res_id in range(1, segment_length + 1)
    for _ in atom_names
]
segment = struc_array(atoms)
segment.atom_name = np.asarray(atom_names * segment_length)
segment.element = np.asarray(tuple(name[0] for name in atom_names) * segment_length)
```

Set the remaining required structure annotations by following the source array's
schema, then convert with `as_atom_array_plus()`. Zero coordinates are placeholders;
leave `C_CRD` false for the segment so the model generates them.

## Charges and bonds

Edit `aa.charge` only from confirmed values. Edit `aa.bonds` by exact atom indices
selected from the inspected array. Use `biotite.structure.BondType.COORDINATION`
only for an explicitly confirmed metal bond and use Kekule bond types
`AROMATIC_SINGLE` / `AROMATIC_DOUBLE` for aromatic atomized components. Never
choose bond partners from a distance cutoff.

After any topology edit, rerun the inspector and validator. A save/reload check is
mandatory because `struct_conn`, atom order, condition serialization, and bond
types must survive the CIF round trip.

## Save and reparse

```python
output = Path("annotated.cif")
to_cif_file(
    aa,
    output,
    save_standard_annotations=True,
    chain_disambiguation="chain_iid",
)

round_trip = parse(output, config=ANNOTATED_CIF)["assemblies"]["1"][0]
```

Assert the selections and values that implement the confirmed intent on
`round_trip`, not only on the pre-save array. For an edit, also compare source and
output condition summaries, residue/component inventories, charges, and bonds so
only requested changes remain.
