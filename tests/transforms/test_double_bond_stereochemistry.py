"""Topology detection, idealized E/Z conditioning, and stereochemistry accuracy."""

import numpy as np
import pytest
from atomworks.io.tools.rdkit import atom_array_from_rdkit
from atomworks.io.utils.ccd import custom_ccd_residues
from atomworks.ml.utils.annotator import get_stereogenic_bond_atoms
from biotite.structure import stack
from rdkit import Chem
from rdkit.Chem import AllChem

from rfproteina.metrics.chiral_metric import compute_chiral_metrics
from rfproteina.transforms.chirality_feature import AddChiralityFeature, build_chirality_block_diagonal
from rfproteina.utils.chirals import classify_double_bond_stereoisomer


def molecule(smiles: str, explicit_hydrogens: bool = False):
    """Embed and annotate a custom CCD reference, preserving SMILES atom order."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=9) == 0
    atoms = atom_array_from_rdkit(mol, remove_hydrogens=not explicit_hydrogens)
    atoms.set_annotation("chiral_type", np.zeros(len(atoms), dtype=int))
    atoms.set_annotation("atomize", np.ones(len(atoms), dtype=bool))
    atoms.res_name[:] = "EZ1"
    with custom_ccd_residues({"EZ1": atoms}):
        get_stereogenic_bond_atoms(atoms)
    return atoms


@pytest.mark.parametrize("smiles,side", [("C/C=C/C", -1), (r"C/C=C\C", 1)])
@pytest.mark.parametrize("is_inference", [False, True])
def test_feature_represents_isomer_in_shared_frame(smiles, side, is_inference):
    """Check that training and inference features preserve the expected E/Z double-bond configuration."""
    atoms = molecule(smiles)
    feature = AddChiralityFeature().forward({"atom_array": atoms, "is_inference": is_inference})["feats"]["chirality"]
    assert feature.shape == (4, 18)
    assert feature.dtype == np.float32
    for block in feature.reshape(4, 6, 3).transpose(1, 0, 2):
        assert classify_double_bond_stereoisomer(block[[1, 2, 0, 3]]) == side
    if is_inference:
        np.testing.assert_array_equal(feature, build_chirality_block_diagonal(atoms, is_inference=True))


@pytest.mark.parametrize("degeneracy", ["nan", "collapsed", "collinear", "orthogonal"])
def test_ambiguous_geometry_is_not_encoded_or_scored_as_correct(degeneracy):
    """Check that ambiguous geometry produces no E/Z feature and cannot count as correct stereochemistry."""
    reference = molecule("C/C=C/C")
    atoms = reference.copy()
    atoms.coord = np.array([[-1, 1, 0], [0, 0, 0], [1, 0, 0], [2, -1, 0]], dtype=np.float32)
    if degeneracy == "nan":
        atoms.coord[0] = np.nan
    elif degeneracy == "collapsed":
        atoms.coord[2] = atoms.coord[1]
    elif degeneracy == "collinear":
        atoms.coord[0] = [-1, 0, 0]
    else:
        atoms.coord[3] = [2, 0, 1]
    assert classify_double_bond_stereoisomer(atoms.coord[[1, 2, 0, 3]]) is None
    assert not build_chirality_block_diagonal(atoms, is_inference=True).any()
    assert compute_chiral_metrics(reference, atoms) == {}
    assert compute_chiral_metrics(atoms, reference)["double_bond_percent_correct_stereochemistry"] == 0


def test_accuracy_and_overlapping_bond_frames():
    """Check E/Z accuracy averaging and separate feature frames for double bonds that share atoms."""
    trans = molecule("C/C=C/C=C/C")
    flipped = molecule(r"C/C=C\C=C/C")
    assert compute_chiral_metrics(trans, trans)["double_bond_percent_correct_stereochemistry"] == 1
    # This SMILES changes both bond configurations; a mixed prediction stack averages to 1/2.
    assert compute_chiral_metrics(flipped, trans)["double_bond_percent_correct_stereochemistry"] == 0
    assert compute_chiral_metrics(stack([trans, flipped]), trans)["double_bond_percent_correct_stereochemistry"] == 0.5
    feature = build_chirality_block_diagonal(trans, is_inference=True)
    for indices, block in [([1, 2, 0, 3], 0), ([3, 4, 2, 5], 1)]:
        assert classify_double_bond_stereoisomer(feature[indices, 3 * block : 3 * block + 3]) == -1
