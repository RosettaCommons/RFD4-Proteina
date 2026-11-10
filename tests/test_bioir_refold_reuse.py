"""Reuse the selected search candidate's refold through the real final-fold pipeline."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from atomworks.io.utils.atom_array_plus import AnnotationList2D
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import BondList

from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput
from rfproteina.metrics.oracles.search_refold import SearchRefold
from rfproteina.metrics.utils.metrics_utils import (
    _stage_generated_structure,
    fold_designs_for_oracles,
    run_oracle_and_refolding_metrics,
)
from tests.test_bioir_search_cache import SearchProbe, _adapter, _batch, _score, formatting_seam  # noqa: F401
from tests.test_bioir_search_cache import _structure as _base_structure


def _structure(*args, **kwargs):
    array = _base_structure(*args, **kwargs)
    array.bonds = BondList(len(array))
    return array


@pytest.fixture
def oracle(monkeypatch):
    oracle = AF3Oracle(oracle_name="of3", seed=17)
    oracle.jobs = []

    def fold_batch(jobs):
        for job in jobs:
            if oracle.already_refolded(job.out_dir, job.example_id, job.save_id):
                continue
            oracle.jobs.append(job)
            cif = oracle.refolded_cif_path(job.out_dir, job.example_id, job.save_id)
            cif.parent.mkdir(parents=True, exist_ok=True)
            to_cif_file(job.atom_array, cif)
            oracle._save_confidence_json(
                {"confidence": _score(job.atom_array)},
                oracle.confidence_path(job.out_dir, job.example_id, job.save_id),
            )
        return len(jobs)

    monkeypatch.setattr(oracle, "fold_batch", fold_batch)
    return oracle


def _final(adapter, batch, *, suffix="_job3"):
    entries = adapter.predict_step(batch)
    results = []
    for entry in entries:
        name = entry["example_id"] + suffix
        payload = dict(entry, example_id=name, save_input=False)
        staged = _stage_generated_structure(payload)
        results.append(
            dict(
                staged,
                example_id=name,
                output_base_dir=str(adapter.native.store_dir / "final"),
                metrics_tags="",
                search_refolds=entry.get("search_refolds", {}),
            )
        )
    return results


@pytest.mark.usefixtures("formatting_seam")
def test_two_batches_reuse_their_selected_candidates(tmp_path, oracle):
    adapter = _adapter(tmp_path, oracle)
    adapter.native.store_dir = tmp_path
    for arrays in ([_structure(), _structure(offset=2)], [_structure(5), _structure(residue="SER")]):
        final = _final(adapter, _batch(arrays))
        before = len(oracle.jobs)
        fold_designs_for_oracles(final, adapter.native.oracle_metric_manager)
        assert len(oracle.jobs) == before
        for entry, array in zip(final, arrays, strict=True):
            metrics = run_oracle_and_refolding_metrics(
                entry["designed_atom_array"],
                entry["example_id"],
                tmp_path / "final",
                adapter.native.oracle_metric_manager,
                None,
            )
            assert metrics["oracle.confidence"] == pytest.approx(_score(array))
            assert oracle.already_refolded(tmp_path / "final", entry["example_id"])
    assert len(oracle.jobs) == 4


@pytest.fixture
def handoff(tmp_path, oracle):
    array = _structure()
    source = OracleJobInput(array, "req0_step45_b2_r1", tmp_path / "search")
    oracle.fold_batch([source])
    record = SearchRefold.capture(oracle, array, source.out_dir, source.example_id)
    target = replace(source, example_id="input_req0_step45_b2_r1_job3", out_dir=tmp_path / "final")
    return record, source, target


@pytest.mark.parametrize(
    "change",
    [
        "coords",
        "sequence",
        "conditioning",
        "system_condition",
        "pair_condition",
        "bonds",
        "seed",
        "template",
        "flags",
        "exclude",
        "numbering",
    ],
)
def test_changed_input_or_settings_refolds(oracle, handoff, change):
    record, source, target = handoff
    # Pre-existing final artifacts must not hide a rejected handoff.
    oracle.output_cif_suffix = ".cif.gz"
    oracle.fold_batch([target])
    oracle.output_cif_suffix = ".cif"
    if change == "coords":
        target.atom_array.coord[0, 0] += 1
    elif change == "sequence":
        target.atom_array.res_name[0] = "SER"
    elif change == "conditioning":
        target.atom_array.set_annotation("condition", np.ones(len(target.atom_array), dtype=int))
    elif change == "system_condition":
        target.atom_array.set_annotation("condition", 1, n_body=0)
    elif change == "pair_condition":
        target.atom_array.set_annotation(
            "condition", AnnotationList2D(len(target.atom_array), np.array([[0, 1]]), np.array([1.0])), n_body=2
        )
    elif change == "bonds":
        target.atom_array.bonds.add_bond(0, 1, 1)
    elif change == "seed":
        oracle.seed += 1
    elif change == "template":
        oracle.template_query = "all"
    elif change == "flags":
        oracle.extra_flags.append("--different")
    elif change == "exclude":
        oracle.exclude_ccds.append("HOH")
    elif change == "numbering":
        oracle.preserve_res_id = False
    assert not record.reuse(oracle, target)
    assert not oracle.already_refolded(target.out_dir, target.example_id)
    oracle.fold_batch([target])
    assert len(oracle.jobs) == 3


@pytest.mark.parametrize("damage", ["missing_cif", "bad_cif", "missing_json", "bad_json", "list_json", "empty_json"])
def test_incomplete_or_unreadable_source_refolds(oracle, handoff, damage):
    record, source, target = handoff
    cif = oracle.refolded_cif_path(source.out_dir, source.example_id)
    confidence = oracle.confidence_path(source.out_dir, source.example_id)
    if damage == "missing_cif":
        cif.unlink()
    elif damage == "bad_cif":
        cif.write_text("not a structure")
    elif damage == "missing_json":
        confidence.unlink()
    else:
        confidence.write_text({"list_json": "[1]", "empty_json": "{}", "bad_json": "{"}[damage])
    assert not record.reuse(oracle, target)
    assert not oracle.already_refolded(target.out_dir, target.example_id)
    oracle.fold_batch([target])
    assert len(oracle.jobs) == 2


@pytest.mark.parametrize("suffix", [".cif", ".cif.gz", ".cif.zst"])
def test_preserves_encoding_and_replaces_old_final_artifacts(oracle, handoff, suffix):
    record, source, target = handoff
    oracle.fold_batch([target])
    original = oracle.refolded_cif_path(source.out_dir, source.example_id)
    original.unlink()
    encoded = original.with_name(original.stem + suffix)
    to_cif_file(source.atom_array, encoded)
    assert record.reuse(oracle, target)
    assert oracle.existing_refolded_cif_path(target.out_dir, target.example_id).name.endswith(suffix)
    assert oracle.already_refolded(target.out_dir, target.example_id)
    assert len(list(oracle.confidence_path(target.out_dir, target.example_id).parent.glob("*refolded*"))) == 1


def test_failed_publication_leaves_no_completion_marker(oracle, handoff, monkeypatch):
    record, _, target = handoff
    oracle.fold_batch([target])
    import rfproteina.metrics.oracles.search_refold as module

    original = module.os.replace

    def fail_json(source, destination):
        if str(destination).endswith(".json"):
            raise OSError("injected publication failure")
        original(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_json)
    assert not record.reuse(oracle, target)
    assert not oracle.already_refolded(target.out_dir, target.example_id)
    oracle.fold_batch([target])
    assert len(oracle.jobs) == 3


def test_transforms_are_not_reused(oracle, handoff):
    record, source, target = handoff
    oracle.pre_oracle_transform = SimpleNamespace()
    assert SearchRefold.capture(oracle, source.atom_array, source.out_dir, source.example_id) is None
    assert not record.reuse(oracle, target)


@pytest.mark.usefixtures("formatting_seam")
def test_intermediate_candidates_are_not_reused(tmp_path, oracle):
    class IntermediateProbe(SearchProbe):
        def search(self, *args, **kwargs):
            output = super().search(*args, **kwargs)
            output.candidates[0].is_intermediate = True
            return output

    adapter = _adapter(tmp_path, oracle, probe=IntermediateProbe())
    adapter.native.store_dir = tmp_path
    final = _final(adapter, _batch([_structure()]))
    assert not final[0]["search_refolds"]
    fold_designs_for_oracles(final, adapter.native.oracle_metric_manager)
    assert len(oracle.jobs) == 2


@pytest.mark.usefixtures("formatting_seam")
def test_uses_actual_selected_id_and_checkpoint(tmp_path, oracle):
    class SelectionProbe:
        def search(self, features, config, *, evaluators, **kwargs):
            evaluator = evaluators[0]
            arrays = evaluator.batch["structures"]
            ids = ("req0_step37_b4_r1", "req1_step37_b3_r1")
            rows = evaluator.evaluate(arrays, features, SimpleNamespace(checkpoint=37, request_ids=ids))
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        request_index=1,
                        candidate_id=ids[1],
                        prediction=[arrays[1]],
                        metrics=rows[1],
                        reward=1.0,
                        checkpoint=37,
                        seed=123,
                        is_intermediate=False,
                    )
                ]
            )

    adapter = _adapter(tmp_path, oracle, probe=SelectionProbe())
    adapter.native.store_dir = tmp_path
    strategy = adapter.native.search_strategy
    strategy.checkpoints[37] = strategy.checkpoints.pop(60)
    strategy.reward_models[37] = strategy.reward_models.pop(60)
    final = _final(adapter, _batch([_structure(), _structure(offset=7)], ["first", "second"]))
    assert len(final) == 1
    assert final[0]["example_id"] == "second_req1_step37_b3_r1_job3"
    fold_designs_for_oracles(final, adapter.native.oracle_metric_manager)
    assert len(oracle.jobs) == 2
    confidence = oracle._load_confidence_json(oracle.confidence_path(tmp_path / "final", final[0]["example_id"]))
    assert confidence["confidence"] == pytest.approx(_score(_structure(offset=7)))


def test_saving_aligned_cif_preserves_oracle_input(tmp_path, monkeypatch):
    """The file writer may align its array in place; final folding uses the original candidate."""
    import rfproteina.metrics.utils.metrics_utils as module

    array = _structure()

    def align_in_place(value):
        value.coord += 10
        return value

    monkeypatch.setattr(module, "align_atom_array_to_input_frame", align_in_place)
    staged = _stage_generated_structure(
        {
            "example_id": "aligned",
            "generated_atom_array": array,
            "generated_output_path": tmp_path / "aligned_generated.cif",
            "save_input": False,
            "save_chain_of_interest": False,
        }
    )
    np.testing.assert_array_equal(staged["designed_atom_array"].coord, array.coord)
    saved = module._parse_structure(tmp_path / "aligned_generated.cif", config=module.ANNOTATED_CIF)["assemblies"]["1"][
        0
    ]
    np.testing.assert_allclose(saved.coord, array.coord + 10, atol=0.001)
