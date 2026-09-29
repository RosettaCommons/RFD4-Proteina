"""Bond-preserving selection for published glycans and their attachment residues."""

import numpy as np
import pandas as pd
import pytest
from biotite.structure import AtomArray, BondList

from rfproteina.datapipes.glycan import create_glycan_loader


def test_glycan_selection(monkeypatch):
    atoms = AtomArray(6)
    atoms.chain_id[:] = "A"
    atoms.res_id = np.array([1, 1, 2, 2, 3, 4])
    atoms.set_annotation("pn_unit_iid", np.array(["glycan", "glycan", "protein", "protein", "protein", "ion"]))
    atoms.bonds = BondList(6, np.array([[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1]]))

    def load(row):
        return {
            "atom_array": atoms,
            "atom_array_stack": [atoms],
            "extra_info": {"all_pn_unit_iids_after_processing": ["glycan", "protein", "ion"]},
        }

    monkeypatch.setattr("rfproteina.datapipes.glycan.create_structure_loader", lambda **kwargs: load)
    row = pd.Series({"example_id": "test", "q_pn_unit_iid": "glycan"})
    loader = create_glycan_loader()
    data = loader(row)
    assert np.array_equal(data["atom_array"].res_id, [1, 1, 2, 2])
    assert np.array_equal(data["atom_array"].bonds.as_array(), [[0, 1, 1], [1, 2, 1], [2, 3, 1]])
    assert "atom_array_stack" not in data
    assert not data["extra_info"]
    with pytest.raises(ValueError, match="present query"):
        loader(pd.Series({**row, "q_pn_unit_iid": "missing"}))
    atoms.bonds = BondList(6, np.array([[0, 1, 1]]))
    assert np.array_equal(loader(row)["atom_array"].res_id, [1, 1])
    atoms.bonds = None
    with pytest.raises(ValueError, match="explicit bonds"):
        loader(row)
