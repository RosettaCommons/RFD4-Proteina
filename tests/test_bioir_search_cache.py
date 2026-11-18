"""Search-local BioIR IDs must never select another invocation's oracle result."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from atomworks.enums import ChainType
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, set_structure

from rfproteina.metrics import MetricManager
from rfproteina.metrics.metric import Metric
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
    def __init__(self, *, indices=None, repeat=False, chunk_size=None):
        self.indices = indices
        self.repeat = repeat
        self.chunk_size = chunk_size
        self.evaluations = []

    def search(self, features, config, *, evaluators, **kwargs):
        evaluator = evaluators[0]
        indices = list(range(len(features["padding_mask"]))) if self.indices is None else self.indices
        local_ids = [
            f"req{row}_step60_b{index // len(features['padding_mask'])}_r0" for index, row in enumerate(indices)
        ]
        scored_features = {key: value[indices] for key, value in features.items()}
        arrays = [evaluator.batch["structures"][row] for row in indices]
        rows = []
        chunk_size = self.chunk_size or len(arrays)
        for start in range(0, len(arrays), chunk_size):
            end = start + chunk_size
            context = SimpleNamespace(checkpoint=60, request_ids=tuple(local_ids[start:end]))
            chunk_features = {key: value[start:end] for key, value in scored_features.items()}
            chunk_rows = evaluator.evaluate(arrays[start:end], chunk_features, context)
            if self.repeat:
                assert evaluator.evaluate(arrays[start:end], chunk_features, context) == chunk_rows
            rows.extend(chunk_rows)
        self.evaluations.append((evaluator.output_dir, rows))
        candidates = [
            SimpleNamespace(
                request_index=row,
                candidate_id=local_id,
                prediction=[array],
                metrics=metrics,
                reward=evaluator.native.search_strategy.reward_models[60].aggregate_metrics_output(metrics),
                checkpoint=60,
                seed=123,
                is_intermediate=False,
            )
            for row, local_id, array, metrics in zip(indices, local_ids, arrays, rows, strict=True)
        ]
        return SimpleNamespace(candidates=candidates)


def _adapter(root, oracle, *, rank=0, probe=None):
    reward = RewardModel({"oracle.confidence": {}} if oracle is not None else {})
    native = SimpleNamespace(
        store_dir=str(root),
        global_rank=rank,
        search_strategy=SimpleNamespace(
            checkpoints={60: {"metrics": ["oracle"] if oracle is not None else []}}, reward_models={60: reward}
        ),
        design_metric_manager=None,
        oracle_metric_manager=MetricManager({"oracle": oracle}) if oracle is not None else None,
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


@pytest.mark.parametrize("second", [_structure(5), _structure(residue="SER"), _structure(offset=2.0)])
def test_successive_batches_fold_their_own_structure(tmp_path, formatting_seam, second):
    oracle = CountingOracle()
    adapter = _adapter(tmp_path, oracle)
    first = _structure()
    outputs = [
        adapter.predict_step(_batch([array], [name]))[0]
        for array, name in ((first, "sample_alpha"), (second, "sample_beta"))
    ]

    assert len(oracle.jobs) == 2
    assert [output["reward"] for output in outputs] == pytest.approx([_score(first), _score(second)])
    assert len({job.out_dir for job in oracle.jobs}) == 2
    assert outputs[0]["example_id"] == "sample_alpha_req0_step60_b0_r0"
    assert outputs[1]["example_id"] == "sample_beta_req0_step60_b0_r0"


@pytest.mark.parametrize("second_rank", [0, 1])
def test_new_adapters_with_repeated_sample_ids_do_not_share_cache(tmp_path, formatting_seam, second_rank):
    oracle = CountingOracle()
    arrays = [_structure(), _structure(offset=2.0)]
    outputs = []
    for rank, array in zip((0, second_rank), arrays, strict=True):
        adapter = _adapter(tmp_path, oracle, rank=rank)
        outputs.extend(adapter.predict_step(_batch([array], ["repeated_sample"])))

    assert len(oracle.jobs) == 2
    assert len({job.out_dir for job in oracle.jobs}) == 2
    assert [output["reward"] for output in outputs] == pytest.approx([_score(array) for array in arrays])


def test_retry_after_failed_search_does_not_reuse_partial_cache(tmp_path, formatting_seam):
    oracle = CountingOracle()
    adapter = _adapter(tmp_path, oracle)
    search = adapter.model.search

    def fail_after_scoring(*args, **kwargs):
        search(*args, **kwargs)
        raise RuntimeError("search failed after the oracle wrote its result")

    adapter.model.search = fail_after_scoring
    with pytest.raises(RuntimeError, match="search failed"):
        adapter.predict_step(_batch([_structure()], ["sample_alpha"]))
    assert adapter.completed_calls == 0
    adapter.model.search = search
    second = _structure(offset=1.0)
    output = adapter.predict_step(_batch([second], ["sample_alpha"]))[0]
    assert len(oracle.jobs) == 2
    assert len({job.out_dir for job in oracle.jobs}) == 2
    assert output["reward"] == pytest.approx(_score(second))


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_reordered_chunks_and_branches_keep_original_sample_ids(tmp_path, formatting_seam, chunk_size):
    oracle = CountingOracle()
    adapter = _adapter(tmp_path, oracle, probe=SearchProbe(indices=[1, 0, 1, 0], repeat=True, chunk_size=chunk_size))
    arrays = [_structure(), _structure(offset=1.0)]
    outputs = adapter.predict_step(_batch(arrays, ["sample_alpha", "sample_beta"]))

    assert len(oracle.jobs) == 4  # The repeated evaluation in this search hits its own cache.
    assert [job.example_id for job in oracle.jobs] == [output["example_id"] for output in outputs]
    assert [job.example_id for job in oracle.jobs] == [
        "sample_beta_req1_step60_b0_r0",
        "sample_alpha_req0_step60_b0_r0",
        "sample_beta_req1_step60_b1_r0",
        "sample_alpha_req0_step60_b1_r0",
    ]
    assert [output["reward"] for output in outputs] == pytest.approx([_score(arrays[i]) for i in [1, 0, 1, 0]])


def test_search_without_oracle_keeps_fallback_ids_and_seed_stream(tmp_path, formatting_seam):
    adapter = _adapter(tmp_path, None)
    state = torch.get_rng_state()
    try:
        torch.manual_seed(13)
        before = torch.get_rng_state()
        expected_seed = bioir_full._fresh_seed()
        expected_state = torch.get_rng_state()
        torch.set_rng_state(before)
        observed = []
        original_search = adapter.model.search

        def search(*args, seed, **kwargs):
            observed.append(seed)
            return original_search(*args, seed=seed, **kwargs)

        adapter.model.search = search
        outputs = adapter.predict_step(_batch([_structure()]))
        assert observed == [expected_seed]
        assert torch.equal(torch.get_rng_state(), expected_state)
        assert outputs[0]["example_id"] == "sample0_req0_step60_b0_r0"
        assert adapter.completed_calls == 1
    finally:
        torch.set_rng_state(state)


def test_design_only_rewards_receive_the_same_sample_identity(tmp_path, formatting_seam):
    seen = []

    class SizeMetric(Metric):
        @property
        def kwargs_to_compute_args(self):
            return {"array": ("predicted_atom_array",), "example_id": ("extra_info", "example_id")}

        def compute(self, array, example_id):
            seen.append(example_id)
            return {"n_atoms": float(len(array))}

    adapter = _adapter(tmp_path, None, probe=SearchProbe(indices=[1, 0]))
    adapter.native.design_metric_manager = MetricManager({"size": SizeMetric()})
    adapter.native.search_strategy.checkpoints[60]["metrics"] = ["size"]
    adapter.native.search_strategy.reward_models[60] = RewardModel({"size.n_atoms": {}})
    outputs = adapter.predict_step(_batch([_structure(3), _structure(5)], ["sample_alpha", "sample_beta"]))

    assert seen == [output["example_id"] for output in outputs]
    assert seen == ["sample_beta_req1_step60_b0_r0", "sample_alpha_req0_step60_b0_r0"]
    assert [output["reward"] for output in outputs] == [5.0, 3.0]
