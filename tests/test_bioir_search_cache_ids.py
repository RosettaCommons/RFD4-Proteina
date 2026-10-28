"""BioIR search-evaluation cache ids must be unique across the batches that share a cache.

The oracle cache is keyed on the example id and skips the fold when a confidence JSON for that
id already exists. BioIR numbers its requests from ``req0`` again in every batch, so an id taken
straight from BioIR makes batch 1's candidate read batch 0's refold -- a real measurement of a
different molecule -- instead of folding its own.
"""

from pathlib import Path
from types import SimpleNamespace

import torch

from rfproteina.nn.bioir_full import NativeMetricEvaluator
from rfproteina.utils.common import design_output_dir


def _evaluator(example_ids: list[str]) -> NativeMetricEvaluator:
    return NativeMetricEvaluator(SimpleNamespace(), {}, Path("bioir_search"), example_ids)


def _features(rows: list[int]) -> dict[str, torch.Tensor]:
    return {"_native_row_index": torch.tensor(rows)}


def test_cache_ids_do_not_collide_across_batches():
    # BioIR hands out the same request ids in both batches; the inputs behind them differ.
    bioir_ids = ["req0_step60_b0_r0", "req1_step60_b0_r0"]
    context = SimpleNamespace(request_ids=bioir_ids, checkpoint=60)

    first = _evaluator(["pdl1_rep0", "pdl1_rep1"]).candidate_ids(_features([0, 1]), context)
    second = _evaluator(["pdl1_rep2", "pdl1_rep3"]).candidate_ids(_features([0, 1]), context)

    assert not set(first) & set(second), f"cache ids reused across batches: {sorted(set(first) & set(second))}"
    assert len(set(first + second)) == 4

    # The ids are what the oracle cache keys on, so distinct ids must mean distinct directories:
    # a shared one is the collision itself, where the second candidate finds the first's refold.
    cache = Path("runs/out/bioir_search")
    dirs = [design_output_dir(cache, candidate_id) for candidate_id in first + second]
    assert len(set(dirs)) == 4


def test_cache_id_matches_the_final_output_prefix():
    """Search and final output name the same candidate the same way, so the fold is reused once."""
    context = SimpleNamespace(request_ids=["req0_step60_b0_r0"], checkpoint=60)

    (candidate_id,) = _evaluator(["pdl1_rep0"]).candidate_ids(_features([0]), context)

    assert candidate_id == "pdl1_rep0_req0_step60_b0_r0"


def test_cache_ids_follow_the_row_a_branch_came_from():
    """Branches of one request share its input id; branches of different requests never do."""
    context = SimpleNamespace(
        request_ids=["req0_step60_b0_r0", "req0_step60_b1_r0", "req1_step60_b0_r0"], checkpoint=60
    )

    ids = _evaluator(["pdl1_rep0", "pdl1_rep1"]).candidate_ids(_features([0, 0, 1]), context)

    assert ids == [
        "pdl1_rep0_req0_step60_b0_r0",
        "pdl1_rep0_req0_step60_b1_r0",
        "pdl1_rep1_req1_step60_b0_r0",
    ]
    assert len(set(ids)) == 3
