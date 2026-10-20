"""Dropping a NaN token must not lengthen its neighbours' bond-graph separation."""

import numpy as np
import pytest
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.token import get_token_starts

from rfproteina.datapipes.pipeline import build_design_train_pipeline
from tests.pipeline.test_pipeline_invariants import DESIGN_TASKS, _parse_test_structure


@pytest.mark.parametrize("crop_size", [None, 2048], ids=["no_crop", "crop"])
@pytest.mark.parametrize("damage", ["nan", "missing_reference"])
@pytest.mark.parametrize(
    # Residues in the middle of a protein chain whose heavy atoms, and both neighbours', are all
    # resolved in the deposit -- so the only NaN in the neighbourhood is the one injected here.
    "pdb_id, chain_iid, res_id",
    [("5ocm", "A_1", 147), ("6wtf", "A_1", 285)],
)
def test_flanking_residues_of_a_dropped_nan_token_stay_two_hops_apart(
    pdb_id: str, chain_iid: str, res_id: int, crop_size: int | None, damage: str
) -> None:
    """A NaN'd residue's neighbours are still two peptide bonds apart once it is dropped.

    ``crop_size=2048`` exceeds the token count of either structure, so the crop keeps everything
    while still populating ``crop_info`` -- which exercises composing the crop's token indices
    through the snapshot's. ``crop_size=None`` leaves the snapshot's indices to stand alone.
    """
    path_length_cutoff = 64

    input_data = _parse_test_structure(pdb_id)
    input_data["example_id"] = pdb_id
    atom_array = input_data["atom_array"]
    ensure_annotations(atom_array, "is_protein")

    # Knock out a single side-chain coordinate of the intermediate residue, the way an unresolved
    # atom arrives from the PDB. That is enough for the whole token to be dropped.
    atom_name = "CB" if damage == "nan" else "CA"
    victim = (atom_array.chain_iid == chain_iid) & (atom_array.res_id == res_id) & (atom_array.atom_name == atom_name)
    assert victim.sum() == 1
    if damage == "nan":
        atom_array.coord[victim] = np.nan
    else:
        input_data["atom_array"] = atom_array[~victim]

    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)):
        pipeline = build_design_train_pipeline(
            crop_size=crop_size,
            design_tasks={"unconditional": DESIGN_TASKS["unconditional"]},
            is_inference=False,
            p_random_rotation=0.0,
            atomization_probabilities={},
            path_length_search_cutoff=path_length_cutoff,
        )
        output = pipeline(input_data)

    out_atom_array = output["atom_array"]
    assert not (
        (out_atom_array.chain_iid == chain_iid) & (out_atom_array.res_id == res_id)
    ).any(), "the NaN token was expected to be dropped, so this test would not be testing anything"

    token_starts = get_token_starts(out_atom_array)

    def token_of(flank_res_id: int) -> int:
        (found,) = np.where(
            (out_atom_array.chain_iid[token_starts] == chain_iid)
            & (out_atom_array.res_id[token_starts] == flank_res_id)
        )
        assert len(found) == 1, f"expected residue {flank_res_id} to be exactly one token, got {len(found)}"
        return int(found[0])

    before, after = token_of(res_id - 1), token_of(res_id + 1)
    path_lengths = output["feats"]["token_bond_path_length"]
    assert path_lengths[before, after] == 2, "flanking residues must still read as two peptide bonds apart"
