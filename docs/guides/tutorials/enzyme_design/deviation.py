#!/usr/bin/env python3
"""Measure how far a generated structure has moved from its parent scaffold.

Inverse folding and partial diffusion both start from the scaffold chosen in the design step, so
the distance from that starting point is what separates them: inverse folding holds the backbone
and changes only the sequence, while partial diffusion resamples both modalities and moves the
backbone as well.

Two quantities are reported:

    ca_rmsd_to_parent    C-alpha RMSD after optimal rigid superposition, in Angstrom
    seq_ident_to_parent  fraction of positions whose residue is unchanged; one minus this is the
                         extent of the sequence change

Both compare residue by residue in order and therefore require the two structures to have the
same residue count, which holds for redesigns of one scaffold.
"""

from __future__ import annotations

import biotite.structure as struc
import numpy as np
import pandas as pd
from atomworks.io import parse
from atomworks.io.utils.annotator import ensure_annotations

from rfproteina.utils.parse_presets import ANNOTATED_CIF


def _ca_and_seq(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the protein C-alpha coordinates and per-residue names, skipping ligands and gaps."""
    aa = parse(str(path), config=ANNOTATED_CIF)["assemblies"]["1"][0]
    ensure_annotations(aa, "is_ligand")
    sel = (~aa.is_ligand) & (aa.atom_name == "CA") & ~np.isnan(aa.coord).any(axis=1)
    return aa.coord[sel], aa.res_name[sel]


def deviation(generated: str, reference: str) -> dict[str, float]:
    """Return ``ca_rmsd_to_parent`` and ``seq_ident_to_parent`` for one structure.

    Returns NaN when the residue counts differ, rather than truncating to a common length and
    comparing positions that do not correspond.
    """
    gc, gs = _ca_and_seq(generated)
    rc, rs = _ca_and_seq(reference)
    if len(gc) != len(rc) or len(gc) == 0:
        return {"ca_rmsd_to_parent": float("nan"), "seq_ident_to_parent": float("nan"), "n_res": len(gc)}
    fitted, _ = struc.superimpose(rc, gc)
    rmsd = float(np.sqrt(((fitted - rc) ** 2).sum(-1).mean()))
    ident = float((gs == rs).mean())
    return {"ca_rmsd_to_parent": rmsd, "seq_ident_to_parent": ident, "n_res": len(gc)}


def deviation_table(metrics_csv: str, reference: str) -> pd.DataFrame:
    """Add ``ca_rmsd_to_parent`` and ``seq_ident_to_parent`` columns to a metrics table."""
    d = pd.read_csv(metrics_csv)
    rows = [deviation(p, reference) for p in d["designed_structure_path"]]
    for k in ("ca_rmsd_to_parent", "seq_ident_to_parent"):
        d[k] = [r[k] for r in rows]
    return d
