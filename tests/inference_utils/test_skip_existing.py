"""Regression tests for resuming partially evaluated generation runs."""

import os
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from atomworks.io.utils.compression import open_compressed

import rfproteina.generate as generate
from rfproteina.inference_utils.aggregate_metrics import concatenate_results
from rfproteina.metrics.utils.metrics_utils import _write_chain_only_cif_if_unambiguous
from rfproteina.utils.structure_files import iter_structure_files


def _write_metrics(path, rows) -> None:
    frame = pd.DataFrame(rows).assign(task="test", dataset_name="test")
    frame.to_csv(path, index=False)


@pytest.mark.parametrize("suffix", [".cif", ".cif.gz", ".cif.zst"])
@pytest.mark.parametrize("subdir", ["batch", ".", "archive/batch"])
def test_partition_existing_outputs_routes_existing_designs_to_evaluation(
    tmp_path, monkeypatch, suffix, subdir
) -> None:
    output_dir = tmp_path / "inference_samples" / subdir
    output_dir.mkdir(parents=True)
    for stem in ["second_job4", "first_job5"]:
        with open_compressed(output_dir / f"{stem}_generated{suffix}", "wt") as stream:
            stream.write(f"data_{stem}\n")
    (output_dir / f"third_job4_generated{suffix}").touch()
    staging_dir = tmp_path / "inference_samples" / "batch" / ".generated-cif-interrupted"
    staging_dir.mkdir(parents=True)
    (staging_dir / f"third_job4_generated{suffix}").write_text("partial staged CIF")

    dataset = torch.utils.data.TensorDataset(torch.arange(3))
    monkeypatch.setattr(generate, "_get_example_ids_from_dataset", lambda _: ["first", "second", "third"])

    to_generate, to_evaluate = generate._partition_existing_outputs(
        dataset,
        SimpleNamespace(store_dir=tmp_path, job_id=4),
    )

    if subdir != "batch":
        assert to_generate is dataset
        assert to_evaluate is None
        return
    assert to_generate.indices == [0, 2]
    assert to_evaluate is not None
    assert to_evaluate.indices == [1]


@pytest.mark.parametrize("suffix", [".cif", ".cif.gz", ".cif.zst"])
def test_resumed_chain_export_preserves_encoding(tmp_path, lysozyme, suffix) -> None:
    original = _write_chain_only_cif_if_unambiguous(lysozyme, "design", tmp_path, suffix)
    assert original is not None
    new_suffix = ".cif.zst" if suffix == ".cif" else ".cif"
    assert _write_chain_only_cif_if_unambiguous(lysozyme, "design", tmp_path, new_suffix) == original
    assert iter_structure_files(tmp_path, stem_suffix="_chain") == [original]


@pytest.mark.parametrize("suffix", [".cif", ".cif.gz", ".cif.zst"])
@pytest.mark.parametrize("job_id", [None, 4])
def test_partition_existing_outputs_regenerates_empty_cifs(tmp_path, monkeypatch, job_id, suffix) -> None:
    output_dir = tmp_path / "inference_samples" / "batch"
    output_dir.mkdir(parents=True)
    job_suffix = f"_job{job_id}" if job_id is not None else ""
    (output_dir / f"first{job_suffix}_generated{suffix}").touch()
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
