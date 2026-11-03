"""A BioIR search must score its candidates against its own refolds, never an earlier search's.

The oracle skips folding when a confidence JSON for the candidate's ``example_id`` already exists.
BioIR numbers its requests from ``req0`` again in every search, so a shared cache directory lets a
later search find an earlier one's result under the same name and score a different molecule with
it. Nothing downstream re-checks that a cached artifact belongs to the candidate being scored, so
the failure is silent unless a structure-comparison reward happens to notice the shape change.

The fake oracle below keeps ``OracleMetric``'s real disk-cache and load path and replaces only the
fold, so these exercise the cache that actually ships. The harness follows the one @RazzyChen wrote
for the same bug in #71.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from atomworks.enums import ChainType
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, set_structure

from rfproteina.metrics import MetricManager
from rfproteina.metrics.oracles.oracle_metric import OracleMetric
from rfproteina.nn import bioir_full
from rfproteina.rewards.reward_model import RewardModel


def _structure(length=3, residue="ALA", offset=0.0):
    array = as_atom_array_plus(AtomArray(length))
    array.chain_id[:] = "A"
    array.res_id[:] = np.arange(1, length + 1)
    array.res_name[:] = residue
    array.atom_name[:] = "CA"
    array.element[:] = "C"
    array.coord[:] = 0.0
    array.coord[:, 0] = np.arange(length) * 3.8 + offset
    array.set_annotation("chain_type", np.full(length, ChainType.POLYPEPTIDE_L))
    return array


def _score(array):
    """A value that differs whenever the structure does, so a stale reuse shows up as a reward."""
    return float(len(array) + array.coord.sum() + 0.25 * (array.res_name == "SER").sum())


class CountingOracle(OracleMetric):
    """Replace folding only; retain OracleMetric's real disk cache and load path."""

    def __init__(self):
        super().__init__(oracle_name="probe")
        self.jobs = []

    def fold_one(self, job):
        self.jobs.append(job)
        result = {"confidence": _score(job.atom_array)}
        self._save_confidence_json(result, self.confidence_path(job.out_dir, job.example_id, job.save_id))
        cif = CIFFile()
        set_structure(cif, job.atom_array)
        cif.write(self.refolded_cif_path(job.out_dir, job.example_id, job.save_id))
        return result

    def fold_batch(self, jobs):
        for job in jobs:
            self.fold_one(job)
        return len(jobs)


@pytest.fixture
def formatting_seam(monkeypatch):
    """Model-free decoding, preserving the real evaluator, annotation and oracle pipeline."""

    def features(batch):
        return {
            "padding_mask": batch["padding_mask"],
            "_native_row_index": torch.arange(len(batch["padding_mask"])),
        }

    def format_prediction(native, prediction, features, batch, example_ids=None, **kwargs):
        assert len(prediction) == len(example_ids)
        return [
            {
                "generated_atom_array": array.copy(),
                "input_atom_array": batch["structures"][row].copy(),
                "example_id": name,
                "orig_batch_idx": row,
            }
            for array, name, row in zip(prediction, example_ids, features["_native_row_index"].tolist(), strict=True)
        ]

    monkeypatch.setattr(bioir_full, "_features", features)
    monkeypatch.setattr(bioir_full, "format_prediction", format_prediction)


class SearchProbe:
    """Stand in for BioIR: hand the evaluator the ids BioIR would, restarting at ``req0``."""

    def __init__(self, *, repeat=False):
        self.repeat = repeat

    def search(self, features, config, *, evaluators, **kwargs):
        evaluator = evaluators[0]
        rows = list(range(len(features["padding_mask"])))
        local_ids = [f"req{row}_step60_b0_r0" for row in rows]
        arrays = [evaluator.batch["structures"][row] for row in rows]
        context = SimpleNamespace(checkpoint=60, request_ids=tuple(local_ids))
        metrics = evaluator.evaluate(arrays, features, context)
        if self.repeat:
            assert evaluator.evaluate(arrays, features, context) == metrics
        reward_model = evaluator.native.search_strategy.reward_models[60]
        return SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    request_index=row,
                    candidate_id=local_id,
                    prediction=[array],
                    metrics=row_metrics,
                    reward=reward_model.aggregate_metrics_output(row_metrics),
                    checkpoint=60,
                    seed=123,
                    is_intermediate=False,
                )
                for row, local_id, array, row_metrics in zip(rows, local_ids, arrays, metrics, strict=True)
            ]
        )


def _adapter(root, oracle, *, probe=None):
    native = SimpleNamespace(
        store_dir=str(root),
        search_strategy=SimpleNamespace(
            checkpoints={60: {"metrics": ["oracle"]}},
            reward_models={60: RewardModel({"oracle.confidence": {}})},
        ),
        design_metric_manager=None,
        oracle_metric_manager=MetricManager({"oracle": oracle}),
        refolding_metric_manager=None,
    )
    adapter = object.__new__(bioir_full.FullBioIRAdapter)
    adapter.native = native
    adapter.model = probe or SearchProbe()
    adapter.search = object()
    adapter.partial = None
    adapter.completed_calls = 0
    return adapter


def _batch(arrays, names=None):
    batch = {"padding_mask": torch.ones(len(arrays), 1, dtype=torch.bool), "structures": arrays}
    if names is not None:
        batch["example_id_batch"] = names
    return batch


# A longer binder is what produced the reported length-mismatch warning; the equal-length cases are
# the silent ones, where a stale reward is a real measurement of the wrong molecule.
@pytest.mark.parametrize("second", [_structure(5), _structure(residue="SER"), _structure(offset=2.0)])
def test_successive_searches_fold_their_own_structure(tmp_path, formatting_seam, second):
    oracle = CountingOracle()
    adapter = _adapter(tmp_path, oracle)
    first = _structure()

    outputs = [adapter.predict_step(_batch([array]))[0] for array in (first, second)]

    assert len(oracle.jobs) == 2, "the second search reused the first's refold instead of folding"
    assert [output["reward"] for output in outputs] == pytest.approx([_score(first), _score(second)])


def test_rerun_after_a_failed_search_does_not_reuse_its_cache(tmp_path, formatting_seam):
    """A retry draws a fresh seed, so the same ids name a different molecule."""
    oracle = CountingOracle()
    adapter = _adapter(tmp_path, oracle)
    succeed = adapter.model.search

    def fail_after_scoring(*args, **kwargs):
        succeed(*args, **kwargs)
        raise RuntimeError("search failed after the oracle wrote its result")

    adapter.model.search = fail_after_scoring
    with pytest.raises(RuntimeError, match="search failed"):
        adapter.predict_step(_batch([_structure()]))

    adapter.model.search = succeed
    second = _structure(offset=1.0)
    output = adapter.predict_step(_batch([second]))[0]

    assert len(oracle.jobs) == 2
    assert output["reward"] == pytest.approx(_score(second))


def test_one_search_still_reuses_its_own_cache(tmp_path, formatting_seam):
    """Isolating invocations must not disable the caching that makes re-scoring cheap."""
    oracle = CountingOracle()
    adapter = _adapter(tmp_path, oracle, probe=SearchProbe(repeat=True))

    adapter.predict_step(_batch([_structure(), _structure(offset=1.0)]))

    assert len(oracle.jobs) == 2, "the repeated evaluation refolded instead of hitting its own cache"


def test_allocating_the_cache_does_not_disturb_the_sampler_stream(tmp_path, formatting_seam):
    """The search seed must stay a pure function of the run seed and the batch index."""
    oracle = CountingOracle()
    adapter = _adapter(tmp_path, oracle)
    state = torch.get_rng_state()
    try:
        torch.manual_seed(13)
        before = torch.get_rng_state()
        expected_seed = bioir_full._fresh_seed()
        expected_state = torch.get_rng_state()
        torch.set_rng_state(before)

        observed = []
        succeed = adapter.model.search

        def search(*args, seed, **kwargs):
            observed.append(seed)
            return succeed(*args, seed=seed, **kwargs)

        adapter.model.search = search
        adapter.predict_step(_batch([_structure()]))

        assert observed == [expected_seed]
        assert torch.equal(torch.get_rng_state(), expected_state)
        assert adapter.completed_calls == 1
    finally:
        torch.set_rng_state(state)
