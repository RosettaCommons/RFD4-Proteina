"""Model-free contracts for the experimental AF2 gradient bridge."""

import runpy
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from atomworks.ml.conditions import C_SEQ
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
from rfproteina.guidance.colabdesign import (
    AF_RESTYPES,
    ColabDesignBinderLoss,
    ColabDesignSequenceMetric,
    _ExternalSequenceLoss,
)
from rfproteina.guidance.guidance_model import GuidanceModel, build_guidance_terms
from rfproteina.metrics.metric import MetricManager
from rfproteina.partial_autoencoder.autoencoder import AutoEncoder
from rfproteina.partial_autoencoder.decoder import get_atom_mask
from rfproteina.proteina import Proteina
from tests.helpers import CONFIGS_DIR


def quadratic(logits, target):
    return np.asarray(0.5 * np.square(logits).sum()), logits.copy()


def test_prepared_example_through_real_input_pipeline(tmp_path, monkeypatch):
    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.generate import get_contig_or_design_problem_dataset

    script = Path(__file__).resolve().parents[1] / "examples/colabdesign_guidance/prepare_example.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--out-dir", str(tmp_path)])
    runpy.run_path(str(script), run_name="__main__")
    dataset = get_contig_or_design_problem_dataset(
        str(tmp_path / "binder.yaml"),
        num_replicates=1,
        transform=build_design_validation_pipeline(metrics_tags=None),
    )
    example = collate_batch([dataset[0]], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
    logits = torch.ones(*example["padding_mask"].shape, ENC.n_tokens, requires_grad=True)

    def objective(values, target):
        assert values.shape == (1, 60, 20)
        assert target.shape == (115,)
        return quadratic(values, target)

    loss = ColabDesignSequenceMetric(objective=objective).compute(logits, example)["loss"]
    gradient = torch.autograd.grad(loss, logits)[0]
    assert torch.count_nonzero(gradient) == 60 * 20
    with pytest.raises(FileExistsError):
        runpy.run_path(str(script), run_name="__main__")


def batch():
    # Target first, binder second, then padding: do not rely on binder-first order.
    return {
        "padding_mask": torch.tensor([[True, True, True, True, False]]),
        "is_in_context_conditioning": torch.zeros(1, 5, dtype=torch.bool),
        "supports_local_latent": torch.ones(1, 5, dtype=torch.bool),
        "token_type": torch.zeros(1, 5, dtype=torch.long),
        "chain_iid": torch.tensor([[7, 7, 9, 9, -1]]),
        C_SEQ.full_name: torch.tensor(
            [
                [
                    ENC.token_to_idx["GLY"],
                    ENC.token_to_idx["TRP"],
                    ENC.token_to_idx["<M>"],
                    ENC.token_to_idx["<M>"],
                    ENC.token_to_idx["<M>"],
                ]
            ]
        ),
    }


def test_external_vjp_and_finite_difference():
    x = torch.tensor([[[0.2, -0.4], [0.7, 0.1]]], requires_grad=True)
    target = torch.tensor([0])
    loss = _ExternalSequenceLoss.apply(x, target, quadratic)
    (gradient,) = torch.autograd.grad(3.0 * loss, x)
    torch.testing.assert_close(gradient, 3 * x)
    direction = torch.tensor([[[0.3, 0.4], [-0.1, 0.2]]])
    eps = 1e-3
    finite_difference = (
        quadratic((x + eps * direction).detach().numpy(), None)[0]
        - quadratic((x - eps * direction).detach().numpy(), None)[0]
    ) / (2 * eps)
    assert finite_difference == pytest.approx(float((x * direction).sum().detach()), abs=1e-4)


@pytest.mark.parametrize("bad", ["nan_loss", "nan_gradient", "shape", "vector_loss"])
def test_invalid_external_results_fail(bad):
    def objective(x, target):
        loss, gradient = quadratic(x, target)
        if bad == "nan_loss":
            loss = np.asarray(np.nan)
        if bad == "nan_gradient":
            gradient[:] = np.inf
        if bad == "shape":
            gradient = gradient[..., :1]
        if bad == "vector_loss":
            loss = np.ones(2)
        return loss, gradient

    with pytest.raises(ValueError):
        _ExternalSequenceLoss.apply(torch.ones(1, 2, 20, requires_grad=True), torch.tensor([0]), objective)


def test_vocabulary_order_and_binder_selection():
    seen = []

    def objective(x, target):
        seen.append((x.copy(), target.copy()))
        return quadratic(x, target)

    logits = torch.arange(5 * ENC.n_tokens, dtype=torch.float32).reshape(1, 5, -1).requires_grad_()
    loss = ColabDesignSequenceMetric(objective).compute(logits, batch())["loss"]
    indices = [ENC.token_to_idx[a] for a in AF_RESTYPES]
    np.testing.assert_array_equal(seen[0][0], logits.detach().numpy()[:, 2:4, indices])
    np.testing.assert_array_equal(seen[0][1], [7, 17])  # AF2 GLY, TRP
    (gradient,) = torch.autograd.grad(loss, logits)
    assert not gradient[:, :2].any() and not gradient[:, 4:].any()
    torch.testing.assert_close(gradient[:, 2:4, indices], logits[:, 2:4, indices])
    other = sorted(set(range(ENC.n_tokens)) - set(indices))
    assert not gradient[..., other].any()


@pytest.mark.parametrize("case", ["context", "ligand", "partial_binder", "extra_chain", "batch", "nonstandard"])
def test_unsupported_inputs_fail_before_loading_af(case):
    b = batch()
    logits = torch.zeros(1, 5, ENC.n_tokens, requires_grad=True)
    if case == "context":
        b["is_in_context_conditioning"][0, 2] = True
    if case == "ligand":
        b["token_type"][0, 2] = 3
    if case == "partial_binder":
        b[C_SEQ.full_name][0, 2] = ENC.token_to_idx["ALA"]
    if case == "extra_chain":
        b["chain_iid"][0, 3] = 10
    if case == "batch":
        logits = logits.expand(2, -1, -1)
    if case == "nonstandard":
        b[C_SEQ.full_name][0, 0] = ENC.token_to_idx["UNK"]
    objective = Mock(side_effect=AssertionError("must not load AF2"))
    with pytest.raises(ValueError):
        ColabDesignSequenceMetric(objective).compute(logits, b)
    objective.assert_not_called()


def test_decoder_to_guidance_under_inference_mode():
    """Exercise the real decode/format/provider/GuidanceModel path, with a small decoder stub."""
    weights = torch.arange(2 * ENC.n_tokens, dtype=torch.float32).reshape(2, -1) / 100

    def decoder(b):
        emit = b["padding_mask"] & ~b["is_in_context_conditioning"] & b["supports_local_latent"]
        seq = b["z_latent"] @ weights
        return {
            "seq_logits": seq * b["padding_mask"][..., None],
            "emit_mask": emit,
            "aatype_max": torch.zeros(1, 5, dtype=torch.long),
            "atom_presence_logits": torch.ones(1, 5, 37),
            "coors_nm": b["ca_coors_nm"][..., None, :].expand(-1, -1, 37, -1),
        }

    decoder.residue_atom_mask = get_atom_mask()
    ae = SimpleNamespace(decoder=decoder)
    ae.decode = MethodType(AutoEncoder.decode, ae)
    model = SimpleNamespace(autoencoder=ae)
    model._format_sample_local_latents = MethodType(Proteina._format_sample_local_latents, model)
    decode = MethodType(Proteina.decode_for_guidance, model)
    objective = Mock(side_effect=quadratic)
    guidance = GuidanceModel(
        MetricManager({"af": ColabDesignSequenceMetric(objective)}),
        {"af.loss": {"local_latents": lambda t: -0.001 if t >= 0.8 else 0}},
        decode_fn=decode,
    )
    with torch.inference_mode():
        b = batch()
        state = {"local_latents": torch.ones(1, 5, 2), "bb_ca": torch.ones(1, 5, 3)}
        assert guidance(state, {"batch": b, "t": {"local_latents": 0.5}}) == {}
        objective.assert_not_called()
        delta = guidance(state, {"batch": b, "t": {"local_latents": 0.82}})
    assert set(delta) == {"local_latents"}
    assert torch.isfinite(delta["local_latents"]).all()
    assert not delta["local_latents"][:, :2].any()
    assert not delta["local_latents"][:, 4:].any()
    x = state["local_latents"].clone()
    indices = [ENC.token_to_idx[a] for a in AF_RESTYPES]
    before = ((x @ weights)[:, 2:4, indices] ** 2).sum()
    after = (((x + delta["local_latents"]) @ weights)[:, 2:4, indices] ** 2).sum()
    assert after < before


def test_colabdesign_adapter_runs_backprop_without_optimizer():
    model = SimpleNamespace(
        _inputs={"batch": {"aatype": np.array([7, 17])}},
        _target_len=2,
        set_seed=Mock(),
        set_seq=Mock(),
        run=Mock(),
        step=Mock(),
        aux={"loss": np.asarray(0.5), "grad": {"seq": np.ones((1, 2, 20))}},
    )
    objective = ColabDesignBinderLoss("unused", "unused", binder_length=2)
    objective._model = model
    logits = np.zeros((1, 2, 20), dtype=np.float32)
    loss, gradient = objective(logits, np.array([7, 17]))
    assert loss == 0.5 and gradient.shape == logits.shape
    model.run.assert_called_once_with(num_recycles=0, models=[0], num_models=1, sample_models=False, backprop=True)
    model.step.assert_not_called()
    with pytest.raises(ValueError, match="sequence/order"):
        objective(logits, np.array([17, 7]))


def test_config_composes_and_constructs_without_colabdesign():
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="inference",
            overrides=[
                "+guidance=colabdesign_binder",
                "guidance.metrics.colabdesign.objective.target_pdb=/not/loaded.pdb",
                "guidance.metrics.colabdesign.objective.data_dir=/not/loaded",
            ],
        )
    metric = instantiate(cfg.guidance.metrics.colabdesign)
    assert metric.objective._model is None
    terms = build_guidance_terms(cfg.guidance.terms)
    schedule = terms["colabdesign.loss"]["local_latents"]
    assert schedule(0.5) == 0 and schedule(0.82) < 0 and schedule(0.9) == 0
