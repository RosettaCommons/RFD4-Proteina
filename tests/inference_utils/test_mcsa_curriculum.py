import csv
import json
from pathlib import Path

import pytest

from script_utils.mcsa_curriculum import build_curriculum, load_tasks, summarize

MANIFEST = Path(__file__).resolve().parents[2] / "data/benchmarks/mcsa/curriculum/tasks.json"


def test_real_curriculum_sizes_segments_and_fixed_length(tmp_path):
    spec = load_tasks(MANIFEST)
    tasks = spec["tasks"][:7]
    assert [t["motif_size"] for t in tasks] == list(range(8, 33, 4))
    assert [len(t["segments"]) for t in tasks] == [1, 2, 3, 4, 6, 7, 8]
    inputs = build_curriculum(MANIFEST, tmp_path / "run", 200, 2, 42)
    designs = json.loads(inputs.read_text())
    assert len(designs) == 14
    for task in spec["tasks"]:
        tokens = designs[task["id"]]["contig"].split(", ")
        inserted = sum(int(t.split(":")[1]) for t in tokens if ":" in t)
        assert inserted + task["motif_size"] == 200
        assert all(int(t.split(":")[1]) >= 1 for t in tokens if ":" in t)
        assert "is_ligand" not in str(designs[task["id"]]["conditions"])
    ledger = json.loads((inputs.parent / "run_manifest.json").read_text())
    assert len(ledger["attempts"]) == 28
    assert ledger["attempts"][0]["example_id"] == "mcsa_l1_08_0"
    with pytest.raises(FileExistsError):
        build_curriculum(MANIFEST, inputs.parent, 200, 2, 42)


def test_rejects_too_short_scaffold_and_modified_reference(tmp_path):
    with pytest.raises(ValueError, match="length"):
        build_curriculum(MANIFEST, tmp_path / "run", 32, 1, 0)
    assert not (tmp_path / "run").exists()
    spec = json.loads(MANIFEST.read_text())
    spec["reference"]["file"] = str((MANIFEST.parent / spec["reference"]["file"]).resolve())
    spec["reference"]["sha256"] = "wrong"
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="checksum"):
        load_tasks(altered)


def _write_metrics(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _passing_row():
    return {
        "example_id": "mcsa_l1_08_0",
        "curriculum_geometry.scaffold_ca_rmsd": "1.0",
        "curriculum_geometry.motif_bb_rmsd": "0.5",
        "curriculum_geometry.catalytic_rmsd": "0.7",
        "curriculum_geometry.max_catalytic_displacement": "0.9",
        "curriculum_geometry.max_catalytic_distance_error": "0.3",
        "curriculum_geometry.worst_segment_bb_rmsd": "0.6",
        "sequence_adherence.n_violations": "0",
    }


def test_summary_keeps_missing_attempts_in_denominator(tmp_path):
    build_curriculum(MANIFEST, tmp_path / "run", 200, 2, 0)
    metrics = tmp_path / "metrics.csv"
    _write_metrics(metrics, [_passing_row()])
    output = tmp_path / "summary.csv"
    summarize(tmp_path / "run/run_manifest.json", [metrics], output)
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 14
    assert rows[0]["attempted"] == "2"
    assert rows[0]["evaluated"] == "1"
    assert rows[0]["site_geometry_success_rate"] == "0.5"
    assert rows[1]["site_geometry_success_rate"] == "0.0"
    assert rows[0]["unique_site_solutions"] == ""
    with pytest.raises(ValueError, match="Duplicate metrics"):
        summarize(tmp_path / "run/run_manifest.json", [metrics, metrics], tmp_path / "duplicate.csv")


@pytest.mark.parametrize("value", ["", "nan", "inf", "-inf", "False"])
def test_missing_or_invalid_geometry_cannot_pass(tmp_path, value):
    build_curriculum(MANIFEST, tmp_path / "run", 200, 1, 0)
    row = _passing_row()
    row["curriculum_geometry.catalytic_rmsd"] = value
    metrics = tmp_path / "metrics.csv"
    _write_metrics(metrics, [row])
    output = tmp_path / "summary.csv"
    summarize(tmp_path / "run/run_manifest.json", [metrics], output)
    with output.open() as handle:
        first = next(csv.DictReader(handle))
    assert first["site_geometry_successes"] == "0"
    assert first["evaluated"] == "0"


def test_summary_rejects_foreign_runs(tmp_path):
    build_curriculum(MANIFEST, tmp_path / "run", 200, 1, 0)
    row = _passing_row()
    row["example_id"] = "another_run"
    metrics = tmp_path / "metrics.csv"
    _write_metrics(metrics, [row])
    with pytest.raises(ValueError, match="Unplanned"):
        summarize(tmp_path / "run/run_manifest.json", [metrics], tmp_path / "summary.csv")


def test_source_has_every_requested_backbone_and_catalytic_atom(tmp_path):
    """Validate the actual observed experimental atoms, not invented residue ranges."""
    import io

    import numpy as np
    import zstandard
    from biotite.structure.io.pdbx import CIFFile, get_structure

    spec = load_tasks(MANIFEST)
    source = MANIFEST.parent / spec["reference"]["file"]
    with source.open("rb") as handle:
        raw = zstandard.ZstdDecompressor().stream_reader(handle).read().decode()
    aa = get_structure(CIFFile.read(io.StringIO(raw)), model=1, use_author_fields=False)
    for task in spec["tasks"]:
        for start, end in task["segments"]:
            for residue in range(start, end + 1):
                selected = aa[(aa.chain_id == "A") & (aa.res_id == residue)]
                assert {"N", "CA", "C", "O"} <= set(selected.atom_name)
                assert np.isfinite(selected.coord).all()
    assert set(aa.res_name[(aa.chain_id == "A") & (aa.res_id == 22)]) == {"GLU"}
    assert set(aa.res_name[(aa.chain_id == "A") & (aa.res_id == 100)]) == {"LYS"}


def test_curriculum_config_composes_and_keeps_default_ame_criteria(monkeypatch):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from rfproteina.metrics.motif_geometry import MotifGeometryMetric

    monkeypatch.setenv("DATA_PATH", "/unused")
    configs = MANIFEST.parents[4] / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base="1.3"):
        cfg = compose(
            config_name="inference",
            overrides=[
                "inputs=unused",
                "metrics/refolding_metrics=mcsa_curriculum",
                "metrics/pass_criteria=mcsa_curriculum",
            ],
        )
        baseline = compose(config_name="inference", overrides=["inputs=unused"])
    assert cfg.metrics.pass_criteria.ame == baseline.metrics.pass_criteria.ame
    assert "curriculum_geometry" not in baseline.metrics.refolding_metrics
    metric = instantiate(cfg.metrics.refolding_metrics.curriculum_geometry)
    assert isinstance(metric, MotifGeometryMetric)
    assert metric.applies_to_tags({"mcsa_curriculum", "is_refold:untransformed"})
    assert not metric.applies_to_tags({"mcsa_curriculum"})
    row = {k: float(v) for k, v in _passing_row().items() if k != "example_id"}
    for name in ("curriculum_backbone", "curriculum_site_geometry"):
        criterion = instantiate(cfg.metrics.pass_criteria[name])
        assert criterion.compute(row)["pass"] == 1


@pytest.mark.parametrize("task_id,size,islands", [("mcsa_l1_08", 8, 1), ("mcsa_l7_32", 32, 8)])
def test_real_input_build_and_refolding_metric_adapter(tmp_path, task_id, size, islands):
    import numpy as np
    from atomworks.io.utils.annotator import ensure_annotations
    from atomworks.ml.conditions import C_CRD

    from rfproteina.inference_utils.condition_spec import build_annotated_atom_array_from_condition_spec
    from rfproteina.metrics.motif_geometry import MotifGeometryMetric
    from rfproteina.utils.standard_annotations import SA_CRDSRC, SA_CWI

    inputs = build_curriculum(MANIFEST, tmp_path / "run", 200, 1, 0)
    spec = json.loads(inputs.read_text())[task_id]
    data = build_annotated_atom_array_from_condition_spec(spec, spec_dir=inputs.parent, validate=True)
    aa = data["atom_array"]
    ensure_annotations(aa, "is_ligand", "is_protein_backbone")
    assert len(set(aa.res_name[aa.element == "X"])) == islands + 1
    aa = aa[aa.element != "X"]
    mask = C_CRD.mask(aa, default="raise")
    assert int((mask & (aa.atom_name == "CA")).sum()) == size
    assert not aa.is_ligand.any()
    provenance = SA_CRDSRC.default_annotation(aa)
    provenance[mask] = "motif"
    SA_CRDSRC.set_annotation(aa, provenance)
    reference = SA_CWI.default_annotation(aa)
    reference[mask] = aa.coord[mask]
    SA_CWI.set_annotation(aa, reference)
    prediction = aa.copy()
    prediction.coord += [20, 10, 5]
    scores = MotifGeometryMetric().compute(aa, prediction)
    assert max(scores.values()) < 1e-4
    prediction.coord[np.flatnonzero(mask & ~aa.is_protein_backbone)[0]] += [5, 0, 0]
    scores = MotifGeometryMetric().compute(aa, prediction)
    assert scores["max_catalytic_displacement"] > 4.9
