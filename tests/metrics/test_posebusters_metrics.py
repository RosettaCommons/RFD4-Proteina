"""Tests for PoseBusters ligand validation metric."""

import numpy as np
import pytest
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.ml.utils.testing import cached_parse

from proteinfoundation.metrics.posebusters_metrics import validate_single_ligand


def _has_posebusters() -> bool:
    """Check if PoseBusters is installed."""
    try:
        from posebusters import PoseBusters  # noqa: F401

        return True
    except ImportError:
        return False


skip_if_no_posebusters = pytest.mark.skipif(not _has_posebusters(), reason="PoseBusters not installed")


def _extract_sa8_ligand():
    """Extract SA8 ligand from 6wtf structure."""
    atom_array = cached_parse("6wtf")["atom_array"]
    ensure_annotations(atom_array, "is_ligand")
    ligand_atoms = atom_array[atom_array.is_ligand]
    sa8_mask = (ligand_atoms.chain_id == "E") & (ligand_atoms.res_id == 803) & (ligand_atoms.res_name == "SA8")
    return ligand_atoms[sa8_mask]


@skip_if_no_posebusters
@pytest.mark.parametrize(
    "transform,expected_pass",
    [
        ("identity", True),
        ("noise", False),
        ("mirror", True),
    ],
)
def test_validate_sa8(transform: str, expected_pass: bool):
    """SA8 ligand: identity and mirror pass PoseBusters; large noise fails."""
    sa8 = _extract_sa8_ligand().copy()

    if transform == "noise":
        rng = np.random.default_rng(42)
        sa8.coord = sa8.coord + rng.normal(scale=2.0, size=sa8.coord.shape).astype(sa8.coord.dtype)
    elif transform == "mirror":
        sa8.coord = sa8.coord.copy()
        sa8.coord[:, 0] = -sa8.coord[:, 0]

    result = validate_single_ligand(sa8)
    assert result["passes_validation"] is expected_pass
