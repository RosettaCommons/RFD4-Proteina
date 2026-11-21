"""Small deterministic structures for surface-conditioning boundary tests."""

import numpy as np
import pytest
from biotite.structure import AtomArray, BondList


@pytest.fixture
def surface_atoms() -> AtomArray:
    """Include repeated residue numbers across chains and an insertion code."""
    atoms = AtomArray(12)
    atoms.chain_id = np.repeat(["A", "A", "B"], 4)
    atoms.res_id = np.repeat([7, 7, 7], 4)
    atoms.ins_code = np.repeat(["", "A", ""], 4)
    atoms.res_name = np.repeat(["LYS", "SER", "ASP"], 4)
    atoms.atom_name = np.tile(["N", "CA", "C", "O"], 3)
    atoms.element = np.tile(["N", "C", "C", "O"], 3)
    atoms.coord = np.arange(36, dtype=np.float32).reshape(12, 3)
    atoms.bonds = BondList(
        12,
        np.array(
            [(i + a, i + b, order) for i in (0, 4, 8) for a, b, order in ((0, 1, 1), (1, 2, 1), (2, 3, 2))],
            dtype=np.uint32,
        ),
    )
    return atoms


@pytest.fixture
def campaign_manifest() -> dict[str, object]:
    """One mutable insertion-coded site, one functional and one other-chain site."""
    return {
        "target": "fixture",
        "residues": [
            {"key": ["A", 7, ""], "name": "LYS", "eligible": False, "excluded_by": ["functional"]},
            {"key": ["A", 8, ""], "name": "SER", "eligible": True, "excluded_by": []},
            {"key": ["B", 7, ""], "name": "ASP", "eligible": False, "excluded_by": ["not_surface"]},
        ],
    }


@pytest.fixture
def indexed_surface_atoms(surface_atoms: AtomArray) -> AtomArray:
    """CIF-safe IDs; AtomWorks 3.0's writer explicitly clears insertion codes."""
    result = surface_atoms.copy()
    result.res_id[4:8] = 8
    result.ins_code[:] = ""
    return result


@pytest.fixture
def protected_histidine_atoms(indexed_surface_atoms: AtomArray) -> AtomArray:
    """Canonical protected His heavy atoms, retaining the campaign's three CAs."""
    atoms = indexed_surface_atoms.copy()
    atoms.res_name[:4] = "HIS"
    sidechain = AtomArray(6)
    sidechain.chain_id[:] = "A"
    sidechain.res_id[:] = 7
    sidechain.res_name[:] = "HIS"
    sidechain.atom_name[:] = ["CB", "CG", "ND1", "CD2", "CE1", "NE2"]
    sidechain.element[:] = ["C", "C", "N", "C", "C", "N"]
    sidechain.coord[:] = np.arange(18, dtype=np.float32).reshape(6, 3)
    return atoms + sidechain


@pytest.fixture
def confirmation_rows() -> list[dict[str, str | float | int]]:
    """Measured four-stratum candidates expose incomplete/unsafe promotions."""
    rows = []
    for target in ["tll", "bacillus"]:
        for mode in ["inverse", "partial035"]:
            for arm, shift in [
                ("baseline", 0.0),
                ("xtr_weak", 0.1),
                ("xtr_moderate", 0.2),
                ("notion_moderate", 0.3),
            ]:
                rows.append(
                    {
                        "target": target,
                        "mode": mode,
                        "arm": arm,
                        "physics_status": "valid",
                        "passes_geometry_gate": 1,
                        "surface_fraction_DETA": 0.3 + shift,
                        "n_surface_mutations": 10,
                        "n_surface": 100,
                        "hydrophobic_sasa_A2": 100 - shift,
                    }
                )
    return rows


@pytest.fixture
def bond_valid_surface_atoms(indexed_surface_atoms: AtomArray) -> AtomArray:
    """A real peptide bond and ordinary backbone lengths exercise validation."""
    atoms = indexed_surface_atoms.copy()
    atoms.coord = np.array(
        [
            [0, 0, 0],
            [1.45, 0, 0],
            [2.05, 1.4, 0],
            [1.4, 2.4, 0],
            [3.37, 1.48, 0],
            [4.05, 2.76, 0],
            [5.55, 2.6, 0],
            [6.2, 3.65, 0],
            [0, 10, 0],
            [1.45, 10, 0],
            [2.05, 11.4, 0],
            [1.4, 12.4, 0],
        ],
        dtype=np.float32,
    )
    return atoms
