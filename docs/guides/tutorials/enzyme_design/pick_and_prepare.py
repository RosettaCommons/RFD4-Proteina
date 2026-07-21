#!/usr/bin/env python3
"""Select the best scaffold from a design run and carry the conditions onto it.

    PICK_OUT_DIR=<run dir> python pick_and_prepare.py

Designs are ranked by ``ame.pass``, then by ``design_unindexed_motif_allatom_rmsd`` — a direct
measure of how accurately the catalytic motif was placed, computed without the refolding oracle
and therefore deterministic. It is ``NaN`` exactly when the motif was not placed at all, a
failure that ``design_motif_allatom_rmsd`` can mask with a spuriously good number drawn from the
ligand alone.

A generated CIF records where the motif went but does not carry the conditions forward: on the
output, ``C_CRD`` and ``C_SEQ`` cover the ligand only. Passing it onward unchanged would let
inverse folding redesign the catalytic residues and partial diffusion let them drift. The
``mapped_to_coordinate_conditioned_region`` annotation marks exactly the atoms that came from the
input's conditioned motif, so the per-residue atom selection is inherited rather than re-derived.

Writes into ``<run dir>/best``:

    best_raw.cif          the chosen design, unchanged
    best_conditioned.cif  the same design with the conditions re-applied
    selection.csv         the full ranking, so callers need not re-derive the choice
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from atomworks.io import parse
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.conditions import C_CRD, C_SEQ

from rfproteina.utils.parse_presets import ANNOTATED_CIF

HERE = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("PICK_OUT_DIR", HERE / "output"))
METRICS = OUT_DIR / "inference_metrics" / "per_example_metrics.csv"
BEST_DIR = OUT_DIR / "best"
DST = BEST_DIR / "best_conditioned.cif"

PASS, KEY = "ame.pass", "design_unindexed_motif_allatom_rmsd.rmsd"
SHOW = [
    KEY,
    "design_motif_allatom_rmsd.rmsd",
    "refolded_backbone_rmsd.rmsd",
    "refolded_motif_allatom_rmsd.rmsd",
    "oracle.overall_plddt",
    PASS,
]

if not METRICS.exists():
    raise SystemExit(f"{METRICS} is missing; the design run has not finished.")

df = pd.read_csv(METRICS)
for c in (PASS, KEY):
    if c not in df.columns:
        raise SystemExit(f"{METRICS} has no column {c}; check metrics_tags.")

have = [c for c in SHOW if c in df.columns]
ranked = (
    df[["example_id", "designed_structure_path", *have]]
    .sort_values([PASS, KEY], ascending=[False, True])
    .reset_index(drop=True)
)

passed = ranked[ranked[PASS] == 1]
if passed.empty:
    raise SystemExit(
        f"No design passed ame.pass out of {len(ranked)} sampled; " f"review the run before selecting one."
    )

best = passed.iloc[0]
short = str(best["example_id"]).split("__")[-1].replace("_job0", "")
print(
    f"selected {short}: {len(passed)} of {len(ranked)} designs passed, and this one places "
    f"the motif most accurately ({KEY.split('.')[0]} = {best[KEY]:.3f} A)"
)

# ---- carry the conditions onto the chosen design ----------------------------
aa = parse(str(best["designed_structure_path"]), config=ANNOTATED_CIF)["assemblies"]["1"][0]
ensure_annotations(aa, "is_ligand")
ensure_annotations(aa, "is_standard_aa")

mapped_prot = aa.get_annotation("mapped_to_coordinate_conditioned_region") & ~aa.is_ligand
crd = mapped_prot | (C_CRD.mask(aa) & aa.is_ligand)
seq = mapped_prot | aa.is_ligand
C_CRD.set_mask(aa, crd)
C_SEQ.set_mask(aa, seq)
# C_IDX stays on throughout: the scaffold is now a concrete chain and the catalytic residues
# have a definite position in it, so they are no longer unindexed.

BEST_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(best["designed_structure_path"], BEST_DIR / "best_raw.cif")
to_cif_file(aa, str(DST), save_standard_annotations=True, chain_disambiguation="chain_iid")
ranked.to_csv(BEST_DIR / "selection.csv", index=False)

# ---- verify the round trip --------------------------------------------------
# The checks compare the file against what was just written rather than against any one
# benchmark entry, so they hold for every entry. The notebook additionally compares these
# counts with the input theozyme, which is what confirms the geometry carried through.
rt = parse(str(DST), config=ANNOTATED_CIF)["assemblies"]["1"][0]
ensure_annotations(rt, "is_ligand")
nan = np.isnan(rt.coord).any(axis=1)
problems = [
    f"{tag} {got} != {exp}"
    for tag, got, exp in (
        ("C_CRD", int(C_CRD.mask(rt).sum()), int(crd.sum())),
        ("C_SEQ", int(C_SEQ.mask(rt).sum()), int(seq.sum())),
    )
    if got != exp
]
problems += [
    f"{tag}: {val}"
    for tag, val in (
        ("NaN coordinates", int(nan.sum())),
        ("hydrogens", int((rt.element == "H").sum())),
        ("OXT atoms", int((rt.atom_name == "OXT").sum())),
        ("conditioned atoms with NaN", int((C_CRD.mask(rt) & nan).sum())),
    )
    if val != 0
]
if problems:
    raise SystemExit("round trip failed: " + "; ".join(problems))
