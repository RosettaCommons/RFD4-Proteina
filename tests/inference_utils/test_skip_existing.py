"""Regression tests for resuming partially evaluated generation runs."""

import os
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import rfproteina.generate as generate
from rfproteina.inference_utils.aggregate_metrics import concatenate_results


def _write_metrics(path, rows) -> None:
    frame = pd.DataFrame(rows).assign(task="test", dataset_name="test")
    frame.to_csv(path, index=False)


def test_partition_existing_outputs_routes_existing_designs_to_evaluation(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "inference_samples" / "batch"
    output_dir.mkdir(parents=True)
    (output_dir / "second_job4_generated.cif").write_text("data_second\n")
    (output_dir / "third_job4_generated.cif").touch()
    (output_dir / "first_job5_generated.cif").write_text("data_sibling_job\n")
    staging_dir = output_dir / ".generated-cif-interrupted"
    staging_dir.mkdir()
    (staging_dir / "third_job4_generated.cif").write_text("partial staged CIF")

    dataset = torch.utils.data.TensorDataset(torch.arange(3))
    monkeypatch.setattr(generate, "_get_example_ids_from_dataset", lambda _: ["first", "second", "third"])

    to_generate, to_evaluate = generate._partition_existing_outputs(
        dataset,
        SimpleNamespace(store_dir=tmp_path, job_id=4),
    )

    assert to_generate.indices == [0, 2]
    assert to_evaluate is not None
    assert to_evaluate.indices == [1]


@pytest.mark.parametrize("job_id", [None, 4])
def test_partition_existing_outputs_regenerates_empty_cifs(tmp_path, monkeypatch, job_id) -> None:
    output_dir = tmp_path / "inference_samples" / "batch"
    output_dir.mkdir(parents=True)
    suffix = f"_job{job_id}" if job_id is not None else ""
    (output_dir / f"first{suffix}_generated.cif").touch()
    dataset = torch.utils.data.TensorDataset(torch.arange(1))
    monkeypatch.setattr(generate, "_get_example_ids_from_dataset", lambda _: ["first"])

    to_generate, to_evaluate = generate._partition_existing_outputs(
        dataset, SimpleNamespace(store_dir=tmp_path, job_id=job_id)
    )

    assert to_generate is dataset
    assert to_evaluate is None


def test_concatenate_results_merges_archived_and_newest_metrics(tmp_path) -> None:
    archive = tmp_path / "individual_job_results"
    archive.mkdir()
    old_path = archive / "per_example_metrics_job0.csv"
    new_path = tmp_path / "per_example_metrics_job0_reeval.csv"
    _write_metrics(old_path, [{"example_id": "old-only", "score": 1.0}, {"example_id": "rerun", "score": 2.0}])
    _write_metrics(new_path, [{"example_id": "rerun", "score": 3.0}])
    old_mtime = old_path.stat().st_mtime
    os.utime(new_path, (old_mtime + 1, old_mtime + 1))

    concatenate_results(str(tmp_path), cleanup=False)

    result = pd.read_csv(tmp_path / "per_example_metrics.csv").set_index("example_id")
    assert result["score"].to_dict() == {"old-only": 1.0, "rerun": 3.0}
