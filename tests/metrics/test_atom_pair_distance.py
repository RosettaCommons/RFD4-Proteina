"""Tests for atom-pair distance constraints used in enzyme design."""

from pathlib import Path

from atomworks.io import parse

from rfproteina.metrics.atom_pair_distance_metric import AtomPairDistanceMetric


def _atom(chain_id: str, res_name: str, res_id: int, atom_name: str) -> str:
    """Build an AtomWorks query string selecting a single atom by its identity."""
    return (
        f"chain_id == '{chain_id}' and res_name == '{res_name}' and res_id == {res_id} and atom_name == '{atom_name}'"
    )


def test_lenacap_reference_satisfies_catalytic_distance_constraints():
    cif_path = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "enz_design" / "lenacap_partial_sc.cif"
    atom_array = parse(str(cif_path))["assemblies"]["1"][0]

    metric = AtomPairDistanceMetric(
        constraints=[
            {
                "name": "oxanion_cys",
                "max_distance_angstrom": 3.0,
                "atom_a": _atom("B", "lpr", 1, "O29"),
                "atom_b": _atom("A", "CYS", 5, "N"),
            },
            {
                "name": "oxanion_gln",
                "max_distance_angstrom": 3.5,
                "atom_a": _atom("B", "lpr", 1, "O29"),
                "atom_b": _atom("A", "GLN", 3, "NE2"),
            },
            {
                "name": "triad_his_cys",
                "max_distance_angstrom": 4.0,
                "atom_a": _atom("A", "HIS", 1, "ND1"),
                "atom_b": _atom("A", "CYS", 5, "SG"),
            },
            {
                "name": "triad_his_asp",
                "max_distance_angstrom": 3.2,
                "atom_a": _atom("A", "HIS", 1, "NE2"),
                "atom_b": _atom("A", "ASP", 2, "OD1"),
            },
        ]
    )
    out = metric.compute(predicted_atom_array=atom_array)

    assert out["n_violated"] == 0
    assert out["all_satisfied"] == 1.0
    assert out["worst_violation"] == 0.0
    assert out["soft_min_score"] == 1.0


def test_atom_a_and_b_options_use_minimum_pair_distance():
    cif_path = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "enz_design" / "lenacap_partial_sc.cif"
    atom_array = parse(str(cif_path))["assemblies"]["1"][0]

    metric = AtomPairDistanceMetric(
        constraints=[
            {
                "name": "triad_his_asp",
                "max_distance_angstrom": 3.0,
                "atom_a": [
                    _atom("A", "HIS", 1, "ND1"),
                    _atom("A", "HIS", 1, "NE2"),
                ],
                "atom_b": [
                    _atom("A", "ASP", 2, "OD1"),
                    _atom("A", "ASP", 2, "OD2"),
                ],
            }
        ]
    )
    out = metric.compute(predicted_atom_array=atom_array)
    assert out["triad_his_asp_distance"] > 0.0
    assert out["triad_his_asp_satisfied"] == 1.0


def test_atom_options_skip_unresolved_queries():
    cif_path = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "enz_design" / "lenacap_partial_sc.cif"
    atom_array = parse(str(cif_path))["assemblies"]["1"][0]

    metric = AtomPairDistanceMetric(
        constraints=[
            {
                "name": "triad_his_asp",
                "max_distance_angstrom": 3.2,
                "atom_a": [
                    _atom("A", "HIS", 1, "MISSING"),
                    _atom("A", "HIS", 1, "NE2"),
                ],
                "atom_b": _atom("A", "ASP", 2, "OD1"),
            }
        ]
    )

    out = metric.compute(predicted_atom_array=atom_array)
    assert out["triad_his_asp_distance"] < float("inf")
    assert out["triad_his_asp_satisfied"] == 1.0


def test_missing_atoms_yield_inf_distance_not_crash():
    """When a constrained atom is missing, the constraint is violated (inf distance, zero soft score)."""
    cif_path = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "enz_design" / "lenacap_partial_sc.cif"
    atom_array = parse(str(cif_path))["assemblies"]["1"][0]
    atom_array.res_name[atom_array.res_name == "ASP"] = "ALA"

    metric = AtomPairDistanceMetric(
        constraints=[
            {
                "name": "triad_his_asp",
                "max_distance_angstrom": 3.2,
                "atom_a": _atom("A", "HIS", 1, "NE2"),
                "atom_b": _atom("A", "ASP", 2, "OD1"),
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
    cif_path = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "enz_design" / "lenacap_partial_sc.cif"
    atom_array = parse(str(cif_path))["assemblies"]["1"][0]

    metric = AtomPairDistanceMetric(
        constraints=[
            {
                "name": "oxanion_cys",
                "max_distance_angstrom": 2.0,
                "atom_a": _atom("B", "lpr", 1, "O29"),
                "atom_b": _atom("A", "CYS", 5, "N"),
            },
        ],
        soft_margin=1.0,
    )
    out = metric.compute(predicted_atom_array=atom_array)
    assert out["oxanion_cys_distance"] > 2.0
    assert 0.0 < out["oxanion_cys_soft_score"] < 1.0
    assert out["soft_min_score"] == out["oxanion_cys_soft_score"]
