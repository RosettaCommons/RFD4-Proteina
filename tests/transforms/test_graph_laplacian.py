"""Tests for graph Laplacian positional encoding transform."""

import numpy as np
import pytest
import torch
from atomworks.constants import STANDARD_AA
from atomworks.ml.transforms.atom_array import AddGlobalTokenIdAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.utils.testing import cached_parse
from atomworks.ml.utils.token import get_token_starts

from proteinfoundation.transforms.graph_laplacian import (
    FeaturizeBondsAsLaplacianPositionalEncoding,
    featurize_bonds_as_laplacian_positional_encoding,
)


def _setup_atomized_data(pdb_id: str) -> dict:
    """Helper to load and atomize a structure."""
    data = cached_parse(pdb_id)
    atomize = AtomizeByCCDName(
        atomize_by_default=True,
        res_names_to_ignore=STANDARD_AA,
    )
    add_token_id = AddGlobalTokenIdAnnotation()
    data = atomize(data)
    return add_token_id(data)


@pytest.mark.parametrize("pdb_id", ["5ocm", "6wtf"])
def test_non_atomized_tokens_are_zeros(pdb_id: str):
    """Test that non-atomized tokens (proteins) have zero PE vectors."""
    data = _setup_atomized_data(pdb_id)
    atom_array = data["atom_array"]
    token_starts = get_token_starts(atom_array, add_exclusive_stop=False)

    # Apply transform
    transform = FeaturizeBondsAsLaplacianPositionalEncoding(encoding_dim=32)
    data = transform(data)
    pe = data["feats"]["graph_laplacian_pe_token_level"]

    # Get atomize status at token level
    atomize_token_level = atom_array.atomize[token_starts]

    # Non-atomized tokens should have zero PE
    non_atomized_pe = pe[~atomize_token_level]
    assert torch.allclose(
        non_atomized_pe, torch.zeros_like(non_atomized_pe)
    ), "Non-atomized tokens should have zero PE vectors"


@pytest.mark.parametrize("pdb_id", ["5ocm", "6wtf"])
def test_atomized_tokens_have_nonzero_encodings(pdb_id: str):
    """Test that atomized molecules with >3 atoms have non-zero encodings.

    Note: Some atoms in symmetric molecules may have identical encodings due to
    graph symmetry (equivalent positions), so we only check that encodings are non-zero.
    """
    data = _setup_atomized_data(pdb_id)
    atom_array = data["atom_array"]
    token_starts = get_token_starts(atom_array, add_exclusive_stop=False)

    # Apply transform
    pe = featurize_bonds_as_laplacian_positional_encoding(atom_array, encoding_dim=32)

    # Get molecule_iid at token level
    molecule_iid_token_level = atom_array.molecule_iid[token_starts]
    atomize_token_level = atom_array.atomize[token_starts]

    # Get atomized molecule IDs
    atomized_molecule_iids = np.unique(molecule_iid_token_level[atomize_token_level])

    for mol_iid in atomized_molecule_iids:
        mol_mask = molecule_iid_token_level == mol_iid
        mol_pe = pe[mol_mask]
        n_atoms = mol_pe.shape[0]

        # Skip small molecules (<=3 atoms get zeros) and single atoms (ions)
        if n_atoms <= 3:
            continue

        # Check that molecule has non-zero encodings (not all zeros)
        assert not torch.allclose(
            mol_pe, torch.zeros_like(mol_pe)
        ), f"Molecule {mol_iid} with {n_atoms} atoms should have non-zero encodings"

        # Check that there's variation in encodings (not all identical rows)
        # At least some atoms should have different encodings
        unique_rows = torch.unique(mol_pe, dim=0)
        assert unique_rows.shape[0] > 1, f"Molecule {mol_iid} should have varied encodings across atoms"


@pytest.mark.parametrize("pdb_id", ["5ocm", "6wtf"])
@pytest.mark.parametrize("encoding_dim", [8, 16, 32, 64])
def test_output_shape(pdb_id: str, encoding_dim: int):
    """Test that output has correct shape (n_tokens, encoding_dim)."""
    data = _setup_atomized_data(pdb_id)
    atom_array = data["atom_array"]
    token_starts = get_token_starts(atom_array, add_exclusive_stop=False)
    n_tokens = len(token_starts)

    # Apply transform
    transform = FeaturizeBondsAsLaplacianPositionalEncoding(encoding_dim=encoding_dim)
    data = transform(data)
    pe = data["feats"]["graph_laplacian_pe_token_level"]

    assert pe.shape == (n_tokens, encoding_dim), f"Expected shape ({n_tokens}, {encoding_dim}), got {pe.shape}"
    assert pe.dtype == torch.float32, f"Expected dtype float32, got {pe.dtype}"


@pytest.mark.parametrize("pdb_id", ["5ocm", "6wtf"])
def test_small_molecule_padding(pdb_id: str):
    """Test that small molecules have proper zero-padding in higher dimensions.

    For a molecule with n_atoms where 3 < n_atoms < encoding_dim:
    - Can have at most (n_atoms - 1) non-trivial eigenvectors
    - Columns [n_atoms-1:encoding_dim] should be zeros (padding)
    """
    data = _setup_atomized_data(pdb_id)
    atom_array = data["atom_array"]
    token_starts = get_token_starts(atom_array, add_exclusive_stop=False)
    encoding_dim = 32

    # Apply transform
    pe = featurize_bonds_as_laplacian_positional_encoding(atom_array, encoding_dim=encoding_dim)

    # Get molecule_iid at token level
    molecule_iid_token_level = atom_array.molecule_iid[token_starts]
    atomize_token_level = atom_array.atomize[token_starts]

    # Get atomized molecule IDs
    atomized_molecule_iids = np.unique(molecule_iid_token_level[atomize_token_level])

    for mol_iid in atomized_molecule_iids:
        mol_mask = molecule_iid_token_level == mol_iid
        mol_pe = pe[mol_mask]
        n_atoms = mol_pe.shape[0]

        # Only check molecules where 3 < n_atoms < encoding_dim
        if n_atoms <= 3 or n_atoms >= encoding_dim:
            continue

        # Max eigenvectors we can compute is (n_atoms - 1)
        # Columns from index (n_atoms - 1) onward should be zeros
        padding_start = n_atoms - 1
        padding_columns = mol_pe[:, padding_start:]

        assert torch.allclose(
            padding_columns, torch.zeros_like(padding_columns)
        ), f"Molecule {mol_iid} with {n_atoms} atoms should have zeros in columns [{padding_start}:{encoding_dim}]"


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
