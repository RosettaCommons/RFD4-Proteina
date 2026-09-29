"""Load a glycan query component with its directly bonded attachment residues."""

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from atomworks.ml.datasets.loaders import create_structure_loader
from biotite.structure import get_residue_positions


def _load_glycan(row: pd.Series, loader: Callable) -> dict[str, Any]:
    data = loader(row)
    atoms = data["atom_array"]
    seed = atoms.pn_unit_iid == str(row["q_pn_unit_iid"])
    if not seed.any() or atoms.bonds is None:
        raise ValueError("Glycan selection requires a present query PN unit and explicit bonds")
    edges = atoms.bonds.as_array()[:, :2]
    boundary = edges[seed[edges[:, 0]] != seed[edges[:, 1]]].ravel()
    attachments = boundary[~seed[boundary]]
    residues = get_residue_positions(atoms, np.arange(len(atoms)))
    data["atom_array"] = atoms[seed | np.isin(residues, residues[attachments])]
    # The selected atoms, not parent-assembly metadata, define the loaded object.
    data.get("extra_info", {}).pop("all_pn_unit_iids_after_processing", None)
    data.pop("atom_array_stack", None)
    return data


def create_glycan_loader(**kwargs: Any) -> Callable:
    """Keep the query PN unit and whole residues directly bonded to it, preserving bonds."""
    return partial(_load_glycan, loader=create_structure_loader(**kwargs))
