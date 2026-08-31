"""Regression tests for LigandCentroidDistanceMetric.

Guards the metric's end-to-end path (including the shared ``resolve_mask``
wiring) — comparing a ligand-bearing structure against itself must succeed and yield
a ~zero centroid distance.
"""

from pathlib import Path

from atomworks.io import parse

from rfproteina.metrics.gt_structure_comparison import LigandCentroidDistanceMetric


def test_ligand_centroid_distance_on_self_is_zero():
    cif_path = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "enz_design" / "lenacap_partial_sc.cif"
    atom_array = parse(str(cif_path))["assemblies"]["1"][0]

    metric = LigandCentroidDistanceMetric()
    out = metric.compute(ground_truth_atom_array=atom_array, predicted_atom_array=atom_array)

    assert "ligand_centroid_distance" in out
    assert out["ligand_centroid_distance"] < 1e-3
