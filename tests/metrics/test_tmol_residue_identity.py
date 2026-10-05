"""Strict instance-aware residue selections agree with TMol pose identities."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

pytest.importorskip("tmol")

from rfproteina.metrics.physics.tmol_ddg_core import atom_sel_to_block_mask  # noqa: E402


def identity_fixture(use_instances=True):
    atoms = AtomArray(8)
    atoms.chain_id = np.repeat(["H", "H", "H", "H"], 2)
    atoms.res_id = np.repeat([1, 2, 2, 1], 2)
    atoms.ins_code = np.repeat(["", "A", "B", ""], 2)
    atoms.res_name[:] = "ALA"
    labels = np.repeat(["H_1", "H_1", "H_1", "H_2"], 2)
    if use_instances:
        atoms.set_annotation("chain_iid", labels)
    else:
        atoms.chain_id = labels
    # Reordered blocks force identity mapping, rather than positional assignment.
    pose = SimpleNamespace(
        block_type_ind=torch.tensor([[0, 0, 0, 0]]),
        max_n_blocks=4,
        pdb_info=SimpleNamespace(
            chain_labels=np.array([["H_2", "H_1", "H_1", "H_1"]]),
            residue_labels=np.array([[1, 2, 1, 2]]),
            residue_insertion_codes=np.array([["", "B", "", "A"]]),
        ),
    )
    return atoms, pose


@pytest.mark.parametrize("use_instances", [False, True])
def test_instance_and_insertion_identity_maps_exact_reordered_blocks(use_instances):
    atoms, pose = identity_fixture(use_instances)
    original = atoms.copy()
    selection = np.array([False, False, True, True, False, False, True, True])
    actual = atom_sel_to_block_mask(pose, atoms, selection, torch.device("cpu"))
    assert actual.tolist() == [[True, False, False, True]]
    for name in atoms.get_annotation_categories():
        np.testing.assert_array_equal(atoms.get_annotation(name), original.get_annotation(name))


def test_adjacent_instances_with_identical_author_residue_ids_stay_separate():
    atoms, pose = identity_fixture()
    atoms = atoms[np.array([0, 1, 6, 7])]
    actual = atom_sel_to_block_mask(pose, atoms, np.array([False, False, True, True]), torch.device("cpu"))
    assert actual.tolist() == [[True, False, False, False]]


def test_duplicate_effective_residue_identity_is_rejected():
    atoms, pose = identity_fixture()
    atoms.chain_iid[:] = "H_1"
    with pytest.raises(ValueError, match="residues sharing"):
        atom_sel_to_block_mask(pose, atoms, np.ones(len(atoms), dtype=bool), torch.device("cpu"))


def test_missing_selected_insertion_code_is_not_silently_dropped():
    atoms, pose = identity_fixture()
    pose.pdb_info.residue_insertion_codes[0, 3] = "Z"
    selection = np.array([False, False, True, True, False, False, False, False])
    with pytest.raises(ValueError, match=r"1 selected residue\(s\) dropped"):
        atom_sel_to_block_mask(pose, atoms, selection, torch.device("cpu"))
