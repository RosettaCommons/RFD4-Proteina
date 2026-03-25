import numpy as np
import pytest
from biotite.structure import AtomArray

from proteinfoundation.transforms.design_task import TipAtomActiveSiteTask


@pytest.mark.fast
def test_protein_pocket_sampling():
    """
    Tests TipAtomActiveSiteTask.sample_protein_pocket() by making a dummy atom array
    with 4 atoms (one in the center and three far away). Ensures that we sample the center atom
    if it is eligible and that we don't when it isn't eligible. Note it isn't guarenteed we sample
    the center atom, but it is based on distance away from the convex hull boundary, so it
    should be super high likelihood.
    """

    # make a dummy atom array with 4 atoms (one in the center and three far away)
    atom_array = AtomArray(4)
    atom_array.coord = np.array([[0, 0, 0], [999, 0, 0], [0, 999, 0], [0, 0, 999]])
    atom_array.chain_id = np.array(["A"] * 4)
    atom_array.res_name = np.array(["ALA"] * 4)
    atom_array.atom_name = np.array(["CA"] * 4)

    eligible_mask = np.array([True, True, True, True])

    task = TipAtomActiveSiteTask()
    task.sampled_chain = "A"  # set sampled chain manually for testing

    # sample the pocket three times and ensure we get the center atom at least twice
    sampled_tip_atom_inds = []
    for _ in range(3):
        sampled_tip_atom_ind = task.sample_protein_pocket(atom_array, eligible_mask)
        sampled_tip_atom_inds.append(sampled_tip_atom_ind)

    assert np.sum(np.array(sampled_tip_atom_inds) == 0) >= 2

    eligible_mask[0] = False

    # sample the pocket 3 times and ensure we don't get the center atom
    sampled_tip_atom_inds = []
    for _ in range(3):
        sampled_tip_atom_ind = task.sample_protein_pocket(atom_array, eligible_mask)
        sampled_tip_atom_inds.append(sampled_tip_atom_ind)

    assert np.sum(np.array(sampled_tip_atom_inds) == 0) == 0
