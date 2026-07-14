"""End-to-end tests for :py:class:`EnforceCropBudget`."""

import copy

import pytest
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_IDX
from atomworks.ml.transforms.base import Compose
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.token import get_token_count

from rfproteina.datapipes.pipeline import _build_design_clean_and_sample_pipeline
from rfproteina.transforms.budget import plan_unindex_expansion
from rfproteina.transforms.design_task import MotifScaffoldTask


@pytest.mark.parametrize("crop_size", [60, 1000], ids=["tight_forces_recrop", "loose_fast_path"])
def test_enforce_crop_budget(protein_protein_interface, crop_size):
    """Full train pipeline (crop → design task → EnforceCropBudget) preserves unindexed
    atoms, fits the budget, and keeps crop_info consistent.
    """
    seeds = create_rng_state_from_seeds(np_seed=0, torch_seed=0, py_seed=0)
    with rng_state(seeds, include_cuda=False):
        data = copy.deepcopy(protein_protein_interface)
        data["atom_array"] = as_atom_array_plus(data["atom_array"])
        data.setdefault("example_id", "1mso")

        design_tasks = {
            "motif_scaffold": {
                "transform": MotifScaffoldTask(p_unindex_any=1.0, max_unindexed_tokens=20),
                "frequency": 1.0,
            },
        }
        pipeline = Compose(
            _build_design_clean_and_sample_pipeline(
                crop_size=crop_size,
                max_atoms_in_crop=10000,
                design_tasks=design_tasks,
                p_random_rotation=0.0,
                is_inference=False,
            )
        )
        data = pipeline(data)

    atom_array = data["atom_array"]
    assert (~C_IDX.mask(atom_array, default="generate")).any(), "precondition: task must unindex"

    extra_tokens, _ = plan_unindex_expansion(atom_array)
    assert get_token_count(atom_array) + extra_tokens <= crop_size

    assert len(data["crop_info"]["crop_atom_idxs"]) == atom_array.array_length()
    assert len(data["crop_info"]["crop_token_idxs"]) == get_token_count(atom_array)
