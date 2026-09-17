from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rfproteina.flow_matching import product_space_flow_matcher
from rfproteina.flow_matching.product_space_flow_matcher import (
    ProductSpaceFlowMatcher,
    sample_noised_target_backbone,
)
from rfproteina.flow_matching.rdn_flow_matcher import RDNFlowMatcher
from rfproteina.search.vanilla import (
    RESAMPLED_FORWARD_MARGINAL,
    resolve_inverse_folding_method,
    resolve_partial_t,
)


def test_forward_marginal_uses_centering_once_with_groups():
    matcher = RDNFlowMatcher(
        dim=3,
        stochastic_centering_mode="x_0",
        stochastic_centering_scale=0.2,
        stochastic_centering_scope="per_group",
        stochastic_centering_group_keys=["molecule_iid"],
    )
    fm = SimpleNamespace(
        base_flow_matchers={"bb_ca": matcher},
    )
    target = torch.zeros((1, 4, 3))
    batch = {
        "padding_mask": torch.ones((1, 4), dtype=torch.bool),
        "is_in_context_conditioning": torch.zeros((1, 4), dtype=torch.bool),
        "supports_local_latent": torch.ones((1, 4), dtype=torch.bool),
        "coord_cond_mask": torch.zeros((1, 4), dtype=torch.bool),
        "molecule_iid": torch.tensor([[0, 0, 1, 1]]),
    }

    with patch.object(matcher, "add_stochastic_noise", wraps=matcher.add_stochastic_noise) as add_centering:
        sampled = sample_noised_target_backbone(fm, batch, target, torch.tensor([0.5]))

    assert sampled.shape == target.shape
    assert add_centering.call_count == 1
    assert add_centering.call_args.kwargs["groups"]["molecule_iid"].equal(batch["molecule_iid"])


def test_inverse_folding_resolution_is_immutable():
    args = {
        "inverse_folding": True,
        "inverse_folding_method": RESAMPLED_FORWARD_MARGINAL,
        "partial_t_backbone": None,
        "partial_t_latent": None,
        "couple_partial_t": False,
        "partial_trajectory_fraction_steps": None,
    }
    original = deepcopy(args)
    fm = SimpleNamespace(data_modes=["bb_ca", "local_latents"])

    assert resolve_inverse_folding_method(args) == RESAMPLED_FORWARD_MARGINAL
    assert resolve_partial_t(args, fm, sampling_model_args={}) == {
        "bb_ca": 0.0,
        "local_latents": 0.0,
    }
    assert args == original


def test_inverse_folding_rejects_partial_diffusion_overrides():
    args = {
        "inverse_folding": True,
        "inverse_folding_method": RESAMPLED_FORWARD_MARGINAL,
        "partial_t_backbone": 0.2,
        "partial_t_latent": None,
        "couple_partial_t": False,
        "partial_trajectory_fraction_steps": None,
    }
    fm = SimpleNamespace(data_modes=["bb_ca", "local_latents"])

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_partial_t(args, fm, sampling_model_args={})


def test_inverse_folding_rejects_unknown_method():
    with pytest.raises(ValueError, match="inverse_folding_method"):
        resolve_inverse_folding_method(
            {
                "inverse_folding": True,
                "inverse_folding_method": "m3",
            }
        )


def test_partial_simulation_uses_resampled_backbone():
    target = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    seen_batches = []

    class FakeFlowMatcher:
        data_modes = ["bb_ca", "local_latents"]

        def get_clean_pred_n_guided_vector(self, batch, **_kwargs):
            seen_batches.append(
                {
                    "x_t": {key: value.clone() for key, value in batch["x_t"].items()},
                }
            )
            return {
                "bb_ca": torch.full_like(batch["x_t"]["bb_ca"], 99.0),
                "local_latents": torch.full_like(batch["x_t"]["local_latents"], 7.0),
            }

        def nn_out_to_clean_sample_prediction(self, batch, nn_out):
            return {key: value.clone() for key, value in nn_out.items()}

        def simulation_step(self, x_t, **_kwargs):
            return {key: value + 1.0 for key, value in x_t.items()}

        def apply_flow_emit_masks(self, x, batch):
            bb_ca_emit_mask = batch["padding_mask"] & ~batch["is_in_context_conditioning"]
            return {key: value * bb_ca_emit_mask[..., None] for key, value in x.items()}

    fake_fm = FakeFlowMatcher()
    x = {
        "bb_ca": torch.zeros_like(target),
        "local_latents": torch.zeros((1, 2, 2)),
    }
    batch = {
        "x_1": {"bb_ca": target, "local_latents": torch.zeros((1, 2, 2))},
        "padding_mask": torch.ones((1, 2), dtype=torch.bool),
        "supports_local_latent": torch.ones((1, 2), dtype=torch.bool),
        "is_in_context_conditioning": torch.tensor([[False, True]]),
        "atom37_mask": torch.ones((1, 2, 37), dtype=torch.bool),
    }
    ts = {
        "bb_ca": torch.tensor([0.0, 0.5, 1.0]),
        "local_latents": torch.tensor([0.0, 0.5, 1.0]),
    }
    gt = {
        "bb_ca": torch.zeros(2),
        "local_latents": torch.zeros(2),
    }

    def deterministic_marginal(_fm, _batch, target_backbone, t_bb):
        return target_backbone * t_bb[:, None, None]

    with patch.object(
        product_space_flow_matcher,
        "sample_noised_target_backbone",
        side_effect=deterministic_marginal,
    ) as sample_backbone:
        final_x, _, _ = ProductSpaceFlowMatcher.partial_simulation(
            fake_fm,
            batch=batch,
            x=x,
            predict_for_sampling=lambda _: None,
            recycle_depths=(0, 0),
            nsteps=2,
            ts=ts,
            gt=gt,
            simulation_step_params={},
            device=torch.device("cpu"),
            resample_target_backbone=True,
            target_backbone=target,
        )

    assert sample_backbone.call_count == 2
    assert torch.equal(seen_batches[0]["x_t"]["bb_ca"], torch.zeros_like(target))
    masked_target = target * ~batch["is_in_context_conditioning"][..., None]
    assert torch.equal(seen_batches[1]["x_t"]["bb_ca"], masked_target * 0.5)
    assert torch.equal(final_x["bb_ca"], masked_target)
