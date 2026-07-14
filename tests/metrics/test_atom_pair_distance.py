"""Tests for atom-pair distance constraints used in enzyme design."""

from pathlib import Path

from atomworks.io import parse

from rfproteina.metrics.atom_pair_distance_metric import AtomPairDistanceMetric


def test_lenacap_reference_satisfies_catalytic_distance_constraints():
    cif_path = (
        Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "enz_design_cifs" / "lenacap_partial_sc.cif"
    )
    atom_array = parse(str(cif_path))["assemblies"]["1"][0]

    metric = AtomPairDistanceMetric(
        constraints=[
            {
                "name": "oxanion_cys",
                "max_distance_angstrom": 3.0,
                "atom_a": {"chain_id": "B", "res_name": "lpr", "res_id": 1, "atom_name": "O29"},
                "atom_b": {"chain_id": "A", "res_name": "CYS", "res_id": 580, "atom_name": "N"},
            },
            {
                "name": "oxanion_gln",
                "max_distance_angstrom": 3.5,
                "atom_a": {"chain_id": "B", "res_name": "lpr", "res_id": 1, "atom_name": "O29"},
                "atom_b": {"chain_id": "A", "res_name": "GLN", "res_id": 574, "atom_name": "NE2"},
            },
            {
                "name": "triad_his_cys",
                "max_distance_angstrom": 4.0,
                "atom_a": {"chain_id": "A", "res_name": "HIS", "res_id": 514, "atom_name": "ND1"},
                "atom_b": {"chain_id": "A", "res_name": "CYS", "res_id": 580, "atom_name": "SG"},
            },
            {
                "name": "triad_his_asp",
                "max_distance_angstrom": 3.2,
                "atom_a": {"chain_id": "A", "res_name": "HIS", "res_id": 514, "atom_name": "NE2"},
                "atom_b": {"chain_id": "A", "res_name": "ASP", "res_id": 531, "atom_name": "OD1"},
            },
        ]
    )
    out = metric.compute(predicted_atom_array=atom_array)

    assert out["n_violated"] == 0
    assert out["all_satisfied"] == 1.0
    assert out["worst_violation"] == 0.0
    assert out["soft_min_score"] == 1.0


def test_atom_a_and_b_options_use_minimum_pair_distance():
    cif_path = (
        Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "enz_design_cifs" / "lenacap_partial_sc.cif"
    )
    atom_array = parse(str(cif_path))["assemblies"]["1"][0]

    metric = AtomPairDistanceMetric(
        constraints=[
            {
                "name": "triad_his_asp",
                "max_distance_angstrom": 3.0,
                "atom_a": [
                    {"chain_id": "A", "res_name": "HIS", "res_id": 514, "atom_name": "ND1"},
                    {"chain_id": "A", "res_name": "HIS", "res_id": 514, "atom_name": "NE2"},
                ],
                "atom_b": [
                    {"chain_id": "A", "res_name": "ASP", "res_id": 531, "atom_name": "OD1"},
                    {"chain_id": "A", "res_name": "ASP", "res_id": 531, "atom_name": "OD2"},
                ],
            }
        ]
    )
    out = metric.compute(predicted_atom_array=atom_array)
    assert out["triad_his_asp_distance"] > 0.0
    assert out["triad_his_asp_satisfied"] == 1.0


def test_missing_atoms_yield_inf_distance_not_crash():
    """When a constrained atom is missing, the constraint is violated (inf distance, zero soft score)."""
    cif_path = (
        Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "enz_design_cifs" / "lenacap_partial_sc.cif"
    )
    atom_array = parse(str(cif_path))["assemblies"]["1"][0]
    atom_array.res_name[atom_array.res_name == "ASP"] = "ALA"

    metric = AtomPairDistanceMetric(
        constraints=[
            {
                "name": "triad_his_asp",
                "max_distance_angstrom": 3.2,
                "atom_a": {"chain_id": "A", "res_name": "HIS", "res_id": 514, "atom_name": "NE2"},
                "atom_b": {"chain_id": "A", "res_name": "ASP", "res_id": 531, "atom_name": "OD1"},
            },
        ],
        soft_margin=1.0,
    )
    out = metric.compute(predicted_atom_array=atom_array)

    assert out["triad_his_asp_distance"] == float("inf")
    assert out["triad_his_asp_satisfied"] == 0.0
    assert out["triad_his_asp_soft_score"] == 0.0
    assert out["soft_min_score"] == 0.0
    assert out["n_violated"] == 1.0


def test_soft_min_score_decays_linearly_past_threshold():
    cif_path = (
        Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "enz_design_cifs" / "lenacap_partial_sc.cif"
    )
    atom_array = parse(str(cif_path))["assemblies"]["1"][0]

    metric = AtomPairDistanceMetric(
        constraints=[
            {
                "name": "oxanion_cys",
                "max_distance_angstrom": 2.0,
                "atom_a": {"chain_id": "B", "res_name": "lpr", "res_id": 1, "atom_name": "O29"},
                "atom_b": {"chain_id": "A", "res_name": "CYS", "res_id": 580, "atom_name": "N"},
            },
        ],
        soft_margin=1.0,
    )
    out = metric.compute(predicted_atom_array=atom_array)
    assert out["oxanion_cys_distance"] > 2.0
    assert 0.0 < out["oxanion_cys_soft_score"] < 1.0
    assert out["soft_min_score"] == out["oxanion_cys_soft_score"]
