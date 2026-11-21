"""Tests catch off-by-one masks, fixed-site edits and conditioning drift."""

import numpy as np
import pytest
import torch
from atomworks.ml.conditions import C_SEQ
from biotite.structure import AtomArray
from hydra.core.override_parser.overrides_parser import OverridesParser

from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS
from rfproteina.encoding import UNIFIED_ATOM37_ENCODING
from rfproteina.metrics.surface_preference import ExpectedSurfacePreferenceMetric, SurfacePreferenceMetric
from rfproteina.surface_engineering import (
    AA_NAMES,
    annotate_probabilities,
    audit_guidance_delta,
    bias_sequence_logits,
    complete_probability_profile,
    logit_bias_override,
    residue_keys,
    reweight_probabilities,
    select_surface_pool,
    surface_metrics,
)
from rfproteina.transforms.sequence_conditioning import CanonicalizeSequenceProbabilityConditions


class TestProbabilityProfiles:
    def test_partial_completion_preserves_explicit_targets_and_zero(self) -> None:
        profile = complete_probability_profile({"ASP": 0.2, "SER": 0.1, "GLU": 0.0}, dict.fromkeys(AA_NAMES, 0.05))
        assert sum(profile.values()) == pytest.approx(1)
        assert profile["ASP"] == 0.2 and profile["SER"] == 0.1 and profile["GLU"] == 0
        assert profile["ALA"] == pytest.approx(0.7 / 17)

    def test_zero_strength_is_identity_and_known_odds_ratio(self) -> None:
        base = {aa: 0.05 for aa in AA_NAMES}
        assert reweight_probabilities(base, {"ASP": 1.0}, 0.0) == base
        shifted = reweight_probabilities(base, {"ASP": 1.0, "ASN": -1.0}, 0.5)
        assert sum(shifted.values()) == pytest.approx(1.0)
        assert shifted["ASP"] / shifted["ASN"] == pytest.approx(np.e)

    @pytest.mark.parametrize("strength", [float("nan"), float("inf"), -1.0])
    def test_invalid_strength_is_rejected(self, strength: float) -> None:
        with pytest.raises(ValueError):
            reweight_probabilities({aa: 0.05 for aa in AA_NAMES}, {}, strength)

    def test_zero_base_probability_stays_excluded(self) -> None:
        base = {aa: 1 / 19 for aa in AA_NAMES if aa != "CYS"} | {"CYS": 0.0}
        assert reweight_probabilities(base, {"CYS": 100.0}, 1.0)["CYS"] == 0.0


class TestResidueConditioning:
    def test_chain_and_insertion_code_are_part_of_identity(self, surface_atoms: AtomArray) -> None:
        assert residue_keys(surface_atoms) == [("A", 7, ""), ("A", 7, "A"), ("B", 7, "")]

    def test_reannotation_clears_omitted_selected_entries_but_preserves_other_sites(
        self, surface_atoms: AtomArray
    ) -> None:
        annotate_probabilities(surface_atoms, [("A", 7, "A"), ("B", 7, "")], {"GLU": 0.4, "ASP": 0.6})
        annotate_probabilities(surface_atoms, [("A", 7, "A")], {"ASP": 0.7, "ASN": 0.1})
        glutamate = next(c for c in SEQUENCE_PROBABILITY_CONDITIONS if c.token_name == "GLU")
        values = np.asarray(glutamate.annotation(surface_atoms, default="generate"))
        assert np.isnan(values[4:8]).all()
        assert np.all(values[8:] == 0.4)

    def test_real_canonicalizer_preserves_fixed_and_partial_probabilities(self, surface_atoms: AtomArray) -> None:
        C_SEQ.set_annotation_from_ground_truth(surface_atoms, surface_atoms.chain_id == "B")
        annotate_probabilities(surface_atoms, [("A", 7, "A")], {"ASP": 0.3, "ASN": 0.0})
        result = CanonicalizeSequenceProbabilityConditions()({"atom_array": surface_atoms})["atom_array"]
        asp = next(c for c in SEQUENCE_PROBABILITY_CONDITIONS if c.token_name == "ASP")
        asn = next(c for c in SEQUENCE_PROBABILITY_CONDITIONS if c.token_name == "ASN")
        lys = next(c for c in SEQUENCE_PROBABILITY_CONDITIONS if c.token_name == "LYS")
        np.testing.assert_allclose(asp.annotation(result)[4:8], 0.3)
        np.testing.assert_allclose(asn.annotation(result)[4:8], 0.0)
        assert np.isnan(lys.annotation(result)[4:8]).all(), "Omission must not become exclusion"
        assert np.isnan(asp.annotation(result)[:4]).all(), "Same number without insertion must be untouched"
        np.testing.assert_allclose(asp.annotation(result)[8:], 1.0)

    def test_fixed_or_missing_sites_fail_closed(self, surface_atoms: AtomArray) -> None:
        C_SEQ.set_annotation_from_ground_truth(surface_atoms, surface_atoms.chain_id == "B")
        for keys in [[("B", 7, "")], [("A", 999, "")]]:
            with pytest.raises(ValueError):
                annotate_probabilities(surface_atoms, keys, {"ASP": 0.5})

    def test_conflicting_probability_mass_is_rejected(self, surface_atoms: AtomArray) -> None:
        with pytest.raises(ValueError):
            annotate_probabilities(surface_atoms, [("A", 7, "")], {"ASP": 0.8, "GLU": 0.8})


class TestSequenceBias:
    def test_real_hydra_parser_accepts_bias_dictionary(self) -> None:
        override = logit_bias_override({"ASP": 0.5, "ASN": -0.5})
        parsed = OverridesParser.create().parse_override(override)
        assert parsed.value() == {"ASP": 0.5, "ASN": -0.5}

    def test_masks_preserve_fixed_padding_and_illegal_identities(self) -> None:
        logits = torch.zeros(1, 3, 4)
        fixed = torch.tensor([[False, True, False]])
        selected = torch.tensor([[True, True, False]])
        bias = torch.tensor([0.0, 2.0, 0.0, 100.0])
        legal = torch.tensor([[[True, True, True, False]]]).expand_as(logits)
        result = bias_sequence_logits(logits, bias, selected, fixed, legal)
        assert result[0, 0].argmax() == 1
        torch.testing.assert_close(result[0, 1, :3], logits[0, 1, :3])
        torch.testing.assert_close(result[0, 2, :3], logits[0, 2, :3])
        assert not torch.isfinite(result[..., 3]).any()
        torch.testing.assert_close(logits, torch.zeros_like(logits))


class TestSurfaceMetrics:
    def test_only_reference_surface_is_counted_and_protected_edit_fails(self) -> None:
        ref = {("A", 7, ""): "LYS", ("A", 7, "A"): "SER", ("B", 7, ""): "ASP"}
        changed = ref | {("A", 7, "A"): "THR"}
        metrics = surface_metrics(ref, changed, [("A", 7, "A")], [("B", 7, "")])
        assert metrics["surface_fraction_THR"] == 1.0
        assert metrics["n_surface"] == 1
        assert metrics["n_mutations"] == 1
        assert metrics["surface_sidechain_TDS_delta_kcal_mol"] == pytest.approx(-0.08)
        with pytest.raises(ValueError, match="protected"):
            surface_metrics(ref, changed | {("B", 7, ""): "ALA"}, [("A", 7, "A")], [("B", 7, "")])

    def test_missing_mapping_and_empty_selection_are_errors(self) -> None:
        ref = {("A", 7, ""): "LYS"}
        with pytest.raises(ValueError):
            surface_metrics(ref, {}, [("A", 7, "")], [])
        with pytest.raises(ValueError):
            surface_metrics(ref, ref, [], [])


class TestSurfacePreferenceMetrics:
    def test_nonfinite_candidate_is_rejected_before_search_scoring(self, surface_atoms: AtomArray) -> None:
        surface_atoms.coord[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            SurfacePreferenceMetric(selected_keys=[["A", 7, "A"]]).compute(surface_atoms)

    def test_physical_entropy_objective_uses_published_energy_units(self) -> None:
        logits = torch.zeros(1, 1, UNIFIED_ATOM37_ENCODING.n_tokens)
        logits[0, 0, UNIFIED_ATOM37_ENCODING.token_to_idx["ASP"]] = 100.0
        logits.requires_grad_()
        value = ExpectedSurfacePreferenceMetric([0], objective="sidechain_entropy").compute(logits)["cost"]
        assert value.item() == pytest.approx(1.25)  # Pickett/Sternberg magnitude, kcal/mol

    def test_search_rejects_protected_identity_before_ranking(self, surface_atoms: AtomArray) -> None:
        metric = SurfacePreferenceMetric(selected_keys=[["A", 7, "A"]])
        metric.reference = surface_atoms.copy()
        candidate = surface_atoms.copy()
        candidate.res_name[candidate.chain_id == "B"] = "GLU"
        with pytest.raises(ValueError, match="Protected"):
            metric.compute(candidate)

    def test_discrete_metric_excludes_other_chains_and_uses_minimization_sign(self, surface_atoms: AtomArray) -> None:
        metric = SurfacePreferenceMetric(selected_keys=[["A", 7, "A"]])
        assert metric.compute(surface_atoms)["cost"] == 1.0  # SER has XTR bias -1
        changed = surface_atoms.copy()
        changed.res_name[4:8] = "ASP"
        assert metric.compute(changed)["cost"] == -1.0  # ASP has XTR bias +1

    def test_expected_cost_keeps_gradients_and_ignores_unselected_tokens(self) -> None:
        logits = torch.zeros(1, 3, UNIFIED_ATOM37_ENCODING.n_tokens, requires_grad=True)
        metric = ExpectedSurfacePreferenceMetric(token_indices=[1])
        value = metric.compute(logits)["cost"]
        (gradient,) = torch.autograd.grad(value, logits)
        assert torch.isfinite(gradient).all()
        assert gradient[0, 1].abs().sum() > 0
        assert not gradient[0, 0].any() and not gradient[0, 2].any()
        with pytest.raises(ValueError, match="gradient"):
            metric.compute(logits.detach())


class TestPosthocSurfacePool:
    def test_rank_and_seeded_random_use_the_identical_measured_pool(self, surface_atoms: AtomArray) -> None:
        metric = SurfacePreferenceMetric(selected_keys=[["A", 7, "A"]])
        scores = {}
        for name in ["SER", "GLU", "ASP"]:
            candidate = surface_atoms.copy()
            candidate.res_name[4:8] = name
            scores[name] = -metric.compute(candidate)["cost"]
        assert select_surface_pool(scores, 2) == ["ASP", "GLU"]
        random_ids = select_surface_pool(scores, 2, random_seed=20261120)
        assert len(set(random_ids)) == 2 and set(random_ids) <= scores.keys()
        assert select_surface_pool(dict(reversed(list(scores.items()))), 2, random_seed=20261120) == random_ids
        assert (
            select_surface_pool({key: -value for key, value in scores.items()}, 2, random_seed=20261120) == random_ids
        )

    @pytest.mark.parametrize("scores,count", [({}, 1), ({"a": 1.0}, 2), ({"a": float("nan")}, 1)])
    def test_invalid_pool_is_not_silently_ranked(self, scores: dict[str, float], count: int) -> None:
        with pytest.raises(ValueError):
            select_surface_pool(scores, count)


class TestNativeGuidanceAudit:
    def test_actual_delta_requires_finite_values_and_zero_force_outside_scope(self) -> None:
        mutable = torch.tensor([[False, True, False]])
        delta = torch.zeros(1, 3, 2)
        delta[:, 1] = 1
        assert audit_guidance_delta({"local_latents": delta}, mutable)["gradient_l1"] == 2
        delta[:, 0] = 0.001
        with pytest.raises(ValueError, match="outside"):
            audit_guidance_delta({"local_latents": delta}, mutable)
        delta[:, 0] = 0
        delta[:, 1] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            audit_guidance_delta({"local_latents": delta}, mutable)
