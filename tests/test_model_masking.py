"""Executable contract for autoencoder and flow-model token masking.

``P`` is the padding mask, ``C`` means the token is used for in-context conditioning,
``L`` means the token is a generated residue type represented by the AE latent alphabet, and
``R`` means every present atom is resolved.

Autoencoder contract

| Token      | Attended | Emit BBCA | Emit latent | Supervise BBCA | Supervise latent |
|------------|----------|-----------|-------------|----------------|------------------|
| Generated  | yes      | yes       | yes         | yes            | yes              |
| Atomized   | yes      | no        | no          | no             | no               |
| Context    | no       | no        | no          | no             | no               |
| Padding    | no       | no        | no          | no             | no               |
| Unresolved | yes      | yes       | yes         | no             | no               |

Flow-model contract

| Token      | Attended | Emit BBCA | Emit latent | Supervise BBCA | Supervise latent |
|------------|----------|-----------|-------------|----------------|------------------|
| Generated  | yes      | yes       | yes         | yes            | yes              |
| Atomized   | yes      | yes       | no          | yes            | no               |
| Context    | yes      | no        | no          | no             | no               |
| Padding    | no       | no        | no          | no             | no               |
| Unresolved | yes      | yes       | yes         | no             | no               |

Expected values below are literal and independent of the production mask formulas.
Every behavioral test reuses this oracle so a failing assertion names the exact grid column.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
from rfproteina.guidance.guidance_model import _flow_emit_mask
from rfproteina.model_masks import (
    compute_ae_masks,
    compute_ae_supervision_mask,
    compute_flow_emit_masks,
    compute_flow_supervision_masks,
)
from rfproteina.nn.local_latents_transformer import LocalLatentsTransformer
from rfproteina.partial_autoencoder.autoencoder import AutoEncoder
from rfproteina.partial_autoencoder.decoder import DecoderTransformer
from rfproteina.partial_autoencoder.encoder import EncoderTransformer
from rfproteina.proteina import Proteina
from rfproteina.search.beam_search import BeamSearchStrategy

TOKEN_NAMES = ("generated", "atomized", "context", "padding", "unresolved")
# Columns are literal oracles in the token order above. The padding token's L/R
# values are deliberately true so padding must dominate every derived mask.
GRID = {
    "padding": (True, True, True, False, True),
    "is_in_context_conditioning": (False, False, True, False, False),
    "supports_local_latent": (True, False, False, True, True),
    "resolved": (True, True, True, True, False),
    "ae_attended": (True, True, False, False, True),
    "ae_latent_emitted": (True, False, False, False, True),
    "flow_attended": (True, True, True, False, True),
    "flow_bb_emit": (True, True, False, False, True),
    "flow_latent_emit": (True, False, False, False, True),
    "bb_supervised": (True, True, False, False, False),
    "latent_supervised": (True, False, False, False, False),
}
PERMUTATION = torch.tensor([4, 2, 0, 3, 1])


def _matrix(field: str) -> torch.Tensor:
    """Return a two-example literal oracle with a permuted second row."""
    row = torch.tensor(GRID[field], dtype=torch.bool)
    return torch.stack((row, row[PERMUTATION]))


def _assert_grid(actual: torch.Tensor, field: str) -> None:
    """Compare a mask with one named contract column."""
    expected = _matrix(field).to(actual.device)
    if not torch.equal(actual, expected):
        pytest.fail(f"{field}: expected {expected.tolist()}, got {actual.tolist()}")


def _assert_flow_emit_masks(emit_masks: dict[str, torch.Tensor]) -> None:
    _assert_grid(emit_masks["bb_ca"], "flow_bb_emit")
    _assert_grid(emit_masks["local_latents"], "flow_latent_emit")


@pytest.fixture
def token_batch() -> dict[str, torch.Tensor]:
    """Two copies of the five token kinds, with the second copy permuted."""
    batch = {
        "padding_mask": _matrix("padding"),
        "is_in_context_conditioning": _matrix("is_in_context_conditioning"),
        "supports_local_latent": _matrix("supports_local_latent"),
        "is_resolved_mask": _matrix("resolved"),
        "atom37_mask": torch.zeros((2, len(TOKEN_NAMES), 37), dtype=torch.bool),
    }
    batch["atom37_mask"][..., :2] = True
    return batch


@pytest.fixture
def flow_matcher() -> ProductSpaceFlowMatcher:
    return ProductSpaceFlowMatcher(
        OmegaConf.create({"product_flowmatcher": {"bb_ca": {"dim": 3}, "local_latents": {"dim": 2}}})
    )


@pytest.fixture
def simulation_probe(flow_matcher, monkeypatch):
    """Run real trajectory methods with deterministic, observable boundaries."""
    calls = SimpleNamespace(sample_emit_masks=[], step_emit_masks=[], model_inputs=[])
    dims = {"bb_ca": 3, "local_latents": 2}

    def sample_schedule(nsteps, sampling_model_args):
        del sampling_model_args
        times = torch.linspace(0, 1, nsteps + 1)
        return ({mode: times for mode in flow_matcher.data_modes}, {mode: torch.zeros(nsteps) for mode in dims})

    def sample_noise(_n, shape, emit_masks, **_kwargs):
        calls.sample_emit_masks.append({mode: value.clone() for mode, value in emit_masks.items()})
        return {mode: torch.ones(*shape, len(TOKEN_NAMES), dim) for mode, dim in dims.items()}

    def predict(batch, **_kwargs):
        calls.model_inputs.append({mode: value.clone() for mode, value in batch["x_t"].items()})
        return {"bb_ca": {}, "local_latents": {}, "insertion": {}}

    def clean_prediction(batch, nn_out):
        del nn_out
        emit_masks = compute_flow_emit_masks(batch)
        return {
            mode: torch.ones(*emit_masks[mode].shape, dim) * emit_masks[mode][..., None] for mode, dim in dims.items()
        }

    def simulation_step(x_t, emit_masks, **_kwargs):
        calls.step_emit_masks.append({mode: value.clone() for mode, value in emit_masks.items()})
        return {mode: torch.ones_like(value) for mode, value in x_t.items()}

    monkeypatch.setattr(flow_matcher, "sample_schedule", sample_schedule)
    monkeypatch.setattr(flow_matcher, "sample_noise", sample_noise)
    monkeypatch.setattr(flow_matcher, "get_clean_pred_n_guided_vector", predict)
    monkeypatch.setattr(flow_matcher, "nn_out_to_clean_sample_prediction", clean_prediction)
    monkeypatch.setattr(flow_matcher, "simulation_step", simulation_step)
    return flow_matcher, calls


def test_current_helpers_match_every_grid_cell(token_batch) -> None:
    """Every current-mode mask equals its independent literal contract column."""
    before = {key: value.clone() for key, value in token_batch.items()}

    ae_attended, ae_latent_emitted = compute_ae_masks(token_batch)
    flow_emit_masks = compute_flow_emit_masks(token_batch)
    ae_supervision = compute_ae_supervision_mask(token_batch)
    flow_supervision = compute_flow_supervision_masks(token_batch)

    _assert_grid(ae_attended, "ae_attended")
    _assert_grid(ae_latent_emitted, "ae_latent_emitted")
    _assert_grid(flow_emit_masks["bb_ca"], "flow_bb_emit")
    _assert_grid(flow_emit_masks["local_latents"], "flow_latent_emit")
    _assert_grid(ae_supervision, "latent_supervised")
    _assert_grid(flow_supervision["bb_ca"], "bb_supervised")
    _assert_grid(flow_supervision["local_latents"], "latent_supervised")
    assert all(torch.equal(token_batch[key], value) for key, value in before.items())


def test_decoder_rejects_nonunified_residue_encoding() -> None:
    with pytest.raises(ValueError, match="requires the 33-class unified residue encoding"):
        DecoderTransformer(n_residue_types=20, decoder={})


def test_autoencoder_rejects_nonunified_residue_encoding(tmp_path) -> None:
    cfg = SimpleNamespace(nn_ae={"n_residue_types": 20})
    with pytest.raises(ValueError, match="requires the 33-class unified residue encoding"):
        AutoEncoder(cfg, store_dir=str(tmp_path))


def test_flow_passes_full_axis_through_autoencoder(token_batch) -> None:
    """Flow target encoding keeps the full padded token axis."""
    encoded_batches = []

    def encode(batch):
        encoded_batches.append(batch)
        _, emit = compute_ae_masks(batch)
        return {"z_latent": torch.ones(*emit.shape, 2) * emit[..., None]}

    model = SimpleNamespace(
        autoencoder=SimpleNamespace(encode=encode),
        cfg_exp=SimpleNamespace(product_flowmatcher=SimpleNamespace(local_latents={"dim": 2})),
    )
    batch = {
        **token_batch,
        "coords_nm": torch.zeros((2, len(TOKEN_NAMES), 37, 3)),
        "residue_type": torch.ones((2, len(TOKEN_NAMES)), dtype=torch.long),
    }

    latents = Proteina._get_clean_sample(model, batch, "local_latents")

    assert len(encoded_batches) == 1
    assert encoded_batches[0]["padding_mask"].shape == (2, len(TOKEN_NAMES))
    assert torch.equal(encoded_batches[0]["is_in_context_conditioning"], token_batch["is_in_context_conditioning"])
    assert torch.equal(encoded_batches[0]["supports_local_latent"], token_batch["supports_local_latent"])
    _assert_grid(latents.ne(0).any(dim=-1), "flow_latent_emit")


@pytest.mark.parametrize(
    ("token", "matters"),
    (("atomized", True), ("context", False), ("padding", False), ("unresolved", True)),
)
@pytest.mark.parametrize(
    ("forward", "select"),
    (
        (EncoderTransformer.forward, lambda output: output["mean"][0, 0]),
        (DecoderTransformer.forward, lambda output: output["coors_nm"][0, 0, 0]),
    ),
)
def test_autoencoder_context_influence_matches_grid(token_batch, token, matters, forward, select) -> None:
    """Atomized context matters; in-context-conditioning and padding values cannot affect output."""
    batch = {key: value.clone() for key, value in token_batch.items()}
    batch.update(
        {
            "probe": torch.ones(2, len(TOKEN_NAMES), 1),
            "ca_coors_nm": torch.zeros(2, len(TOKEN_NAMES), 3),
            "z_latent": torch.zeros(2, len(TOKEN_NAMES), 1),
        }
    )
    changed = {key: value.clone() for key, value in batch.items()}
    changed["probe"][0, TOKEN_NAMES.index(token)] += 1

    if forward is EncoderTransformer.forward:
        extras = {
            "latent_decoder_mean_n_log_scale": lambda values: torch.cat((values, torch.zeros_like(values)), dim=-1),
            "ln_z": torch.nn.Identity(),
        }
    else:

        def residue_logits(values):
            logits = torch.zeros(*values.shape[:2], 33)
            logits[..., 1] = 1.0
            return logits

        extras = {
            "logit_linear": residue_logits,
            "struct_linear": lambda values: values.repeat(1, 1, 37 * 3),
            "n_residue_types": 33,
            "abs_coors": True,
        }

    model = SimpleNamespace(
        cond_factory=lambda input_batch: torch.ones(*input_batch["padding_mask"].shape, 1),
        transition_c_1=lambda value, _mask: value,
        transition_c_2=lambda value, _mask: value,
        init_repr_factory=lambda input_batch: input_batch["probe"],
        pair_rep_factory=None,
        transformer_layers=[lambda seqs, _pair, _cond, mask: seqs + (seqs * mask[..., None]).sum(dim=1, keepdim=True)],
        nlayers=1,
        update_pair_repr=False,
        **extras,
    )
    baseline = forward(model, batch)
    observed = forward(model, changed)

    _assert_grid(baseline["emit_mask"], "ae_latent_emitted")
    if forward is EncoderTransformer.forward:
        _assert_grid(baseline["mean"].ne(0).any(dim=-1), "ae_latent_emitted")
        _assert_grid(baseline["z_latent_pre_ln"].ne(0).any(dim=-1), "ae_latent_emitted")
    else:
        for field in ("coors_nm", "seq_logits", "atom_mask"):
            values = baseline[field]
            _assert_grid(values.ne(0).any(dim=tuple(range(2, values.ndim))), "ae_latent_emitted")
        _assert_grid(baseline["aatype_max"].ne(0), "ae_latent_emitted")
    assert torch.equal(select(baseline), select(observed)) == (not matters)


def test_decoder_coordinates_are_zero_outside_emit_mask(token_batch) -> None:
    """CA composition cannot populate coordinates for tokens the decoder does not emit."""
    batch = {
        **token_batch,
        "probe": torch.ones(2, len(TOKEN_NAMES), 1),
        "ca_coors_nm": torch.ones(2, len(TOKEN_NAMES), 3),
    }
    model = SimpleNamespace(
        init_repr_factory=lambda input_batch: input_batch["probe"],
        logit_linear=lambda values: values.repeat(1, 1, 33),
        struct_linear=lambda values: values.repeat(1, 1, 37 * 3),
        cond_factory=lambda input_batch: torch.ones(*input_batch["padding_mask"].shape, 1),
        transition_c_1=lambda value, _mask: value,
        transition_c_2=lambda value, _mask: value,
        pair_rep_factory=None,
        transformer_layers=[],
        nlayers=0,
        update_pair_repr=False,
        n_residue_types=33,
        abs_coors=False,
    )

    output = DecoderTransformer.forward(model, batch)

    _assert_grid(output["coors_nm"].ne(0).any(dim=(-1, -2)), "ae_latent_emitted")


@pytest.mark.parametrize(
    ("token", "matters"),
    (("atomized", True), ("context", True), ("padding", False), ("unresolved", True)),
)
def test_flow_attention_includes_every_nonpadding_context(token_batch, token, matters) -> None:
    """Every non-padding token can influence a generated flow prediction."""
    trunk_masks = []

    def run_trunk(_batch, seqs, pair, _cond, mask):
        trunk_masks.append(mask.clone())
        context = (seqs * mask[..., None]).sum(dim=1, keepdim=True)
        return seqs + context, pair

    model = SimpleNamespace(
        cond_factory=lambda batch: torch.ones(*batch["padding_mask"].shape, 2),
        transition_c_1=lambda value, _mask: value,
        transition_c_2=lambda value, _mask: value,
        init_repr_factory=lambda batch: batch["probe"],
        pair_repr_builder=lambda batch: torch.zeros(*batch["padding_mask"].shape, batch["padding_mask"].shape[-1], 1),
        local_latents_linear=lambda values: values,
        ca_linear=lambda values: torch.cat((values, values[..., :1]), dim=-1),
        insertion_q_proj=lambda values: values[..., :1],
        insertion_k_proj=lambda values: values[..., :1],
        _insertion_head_scale=1.0,
        output_param={"bb_ca": "v", "local_latents": "v"},
        _run_trunk=run_trunk,
    )
    batch = {**token_batch, "probe": torch.ones(2, len(TOKEN_NAMES), 2)}
    changed = {**batch, "probe": batch["probe"].clone()}
    changed["probe"][0, TOKEN_NAMES.index(token)] += 1

    baseline = LocalLatentsTransformer._forward_impl(model, batch)
    output = LocalLatentsTransformer._forward_impl(model, changed)

    _assert_grid(trunk_masks[-1], "flow_attended")
    _assert_grid(baseline["bb_ca"]["v"].ne(0).any(dim=-1), "flow_bb_emit")
    _assert_grid(baseline["local_latents"]["v"].ne(0).any(dim=-1), "flow_latent_emit")
    changed_indexed = not torch.equal(output["bb_ca"]["v"][0, 0], baseline["bb_ca"]["v"][0, 0])
    assert changed_indexed is matters


def test_flow_corruption_uses_flow_emit_masks(token_batch, flow_matcher) -> None:
    """Clean targets, noise, and interpolated state stay zero outside each modality."""
    values = {"bb_ca": torch.ones(2, len(TOKEN_NAMES), 3), "local_latents": torch.ones(2, len(TOKEN_NAMES), 2)}
    batch = {**token_batch, "x_1": values}
    for matcher in flow_matcher.base_flow_matchers.values():
        matcher.sample_noise = Mock(wraps=matcher.sample_noise)
        matcher.interpolate = Mock(wraps=matcher.interpolate)
        matcher.simulation_step = Mock(wraps=matcher.simulation_step)

    processed_x_1, *_ = flow_matcher.process_batch(batch)
    _assert_grid(processed_x_1["bb_ca"].ne(0).any(dim=-1), "flow_bb_emit")
    _assert_grid(processed_x_1["local_latents"].ne(0).any(dim=-1), "flow_latent_emit")

    corrupted = flow_matcher.corrupt_batch(
        batch,
        t={"bb_ca": torch.full((2,), 0.5), "local_latents": torch.full((2,), 0.5)},
    )
    for state_name in ("x_0", "x_1", "x_t"):
        _assert_grid(corrupted[state_name]["bb_ca"].ne(0).any(dim=-1), "flow_bb_emit")
        _assert_grid(corrupted[state_name]["local_latents"].ne(0).any(dim=-1), "flow_latent_emit")

    updated = flow_matcher.simulation_step(
        x_t=values,
        nn_out={mode: {"v": torch.ones_like(value)} for mode, value in values.items()},
        t={"bb_ca": torch.zeros(2), "local_latents": torch.zeros(2)},
        dt={"bb_ca": 1.0, "local_latents": 1.0},
        gt={"bb_ca": 0.0, "local_latents": 0.0},
        simulation_step_params={
            mode: {
                "sampling_mode": "vf",
                "sc_scale_noise": 0.0,
                "sc_scale_score": 1.0,
                "t_lim_ode": 1.0,
                "t_lim_ode_below": 0.0,
                "center_every_step": False,
            }
            for mode in values
        },
        emit_masks=compute_flow_emit_masks(token_batch),
    )
    for mode, field in (("bb_ca", "flow_bb_emit"), ("local_latents", "flow_latent_emit")):
        matcher = flow_matcher.base_flow_matchers[mode]
        _assert_grid(matcher.sample_noise.call_args.kwargs["emit_mask"], field)
        _assert_grid(matcher.interpolate.call_args.kwargs["emit_mask"], field)
        _assert_grid(matcher.simulation_step.call_args.kwargs["emit_mask"], field)
        _assert_grid(updated[mode].ne(0).any(dim=-1), field)


def test_full_simulation_preserves_flow_emit_masks_through_trajectory(token_batch, simulation_probe) -> None:
    """Noise, integration, guidance, and snapshots obey the grid."""
    model, calls = simulation_probe
    guidance_emit_masks = []

    def guidance(x_1_pred, extra_info):
        assert "flow_emit_masks" not in extra_info
        guidance_emit_masks.append(
            {mode: value.clone() for mode, value in compute_flow_emit_masks(extra_info["batch"]).items()}
        )
        return {mode: torch.ones_like(value) for mode, value in x_1_pred.items()}

    sampling_args = {mode: {"simulation_step_params": {}} for mode in model.data_modes}

    state, info = model.full_simulation(
        batch=dict(token_batch),
        predict_for_sampling=lambda *_args, **_kwargs: None,
        recycle_depths=(1, 2),
        nsteps=2,
        nsamples=2,
        n=len(TOKEN_NAMES),
        sampling_model_args=sampling_args,
        device=torch.device("cpu"),
        save_trajectory_every=1,
        dump_noisy_trajectories=True,
        guidance_model=guidance,
    )

    _assert_flow_emit_masks(calls.sample_emit_masks[0])
    for emit_masks in calls.step_emit_masks + guidance_emit_masks:
        _assert_flow_emit_masks(emit_masks)
    for mode, field in (("bb_ca", "flow_bb_emit"), ("local_latents", "flow_latent_emit")):
        _assert_grid(calls.model_inputs[0][mode].ne(0).any(dim=-1), field)
        _assert_grid(state[mode].ne(0).any(dim=-1), field)
        for snapshot in info["trajectory_steps"]:
            _assert_grid(snapshot["x_1"][mode].ne(0).any(dim=-1), field)
            _assert_grid(snapshot["x_t"][mode].ne(0).any(dim=-1), field)


def test_partial_simulation_passes_flow_emit_masks_at_each_step(token_batch, simulation_probe) -> None:
    """Segmented trajectories use the same modality masks as full simulation."""
    model, calls = simulation_probe
    times = {mode: torch.linspace(0, 1, 3) for mode in model.data_modes}
    zeros = {mode: torch.zeros(2) for mode in model.data_modes}
    initial = {
        "bb_ca": torch.ones(2, len(TOKEN_NAMES), 3),
        "local_latents": torch.ones(2, len(TOKEN_NAMES), 2),
    }

    state, _, _ = model.partial_simulation(
        batch=dict(token_batch),
        x=initial,
        predict_for_sampling=lambda *_args, **_kwargs: None,
        recycle_depths=(1, 2),
        nsteps=2,
        ts=times,
        gt=zeros,
        simulation_step_params={mode: {} for mode in model.data_modes},
        device=torch.device("cpu"),
    )

    for emit_masks in calls.step_emit_masks:
        _assert_flow_emit_masks(emit_masks)
    for mode, field in (("bb_ca", "flow_bb_emit"), ("local_latents", "flow_latent_emit")):
        _assert_grid(calls.model_inputs[0][mode].ne(0).any(dim=-1), field)
        _assert_grid(state[mode].ne(0).any(dim=-1), field)


def test_beam_initialization_uses_modality_flow_emit_masks(token_batch) -> None:
    """Beam noise receives the tiled BBCA and latent masks before any rollout."""
    fm = SimpleNamespace(
        data_modes=("bb_ca", "local_latents"),
        sample_schedule=lambda **_kwargs: (
            {mode: torch.linspace(0, 1, 3) for mode in fm.data_modes},
            {mode: torch.zeros(2) for mode in fm.data_modes},
        ),
    )

    def sample_noise(_n, *, emit_masks, **_kwargs):
        _assert_flow_emit_masks({mode: value.reshape(2, 2, -1)[0] for mode, value in emit_masks.items()})
        # Purely to early-exit the beam search, proves that we passed the above assertion
        raise RuntimeError("beam initialized")

    fm.sample_noise = sample_noise
    model = SimpleNamespace(
        fm=fm,
        inf_cfg=OmegaConf.create(
            {
                "args": {"nsteps": 2},
                "model": {mode: {"simulation_step_params": {}} for mode in fm.data_modes},
            }
        ),
        device=torch.device("cpu"),
        preprocess_batch_for_generation=lambda batch: batch,
        recycle_depths=lambda nsteps, full_nsteps=None: (1,) * nsteps,
    )
    reward = SimpleNamespace(validate_checkpoint_steps=lambda _steps: None)
    batch = {
        **token_batch,
        "coords_nm": torch.zeros(2, len(TOKEN_NAMES), 37, 3),
    }

    with pytest.raises(RuntimeError, match="beam initialized"):
        BeamSearchStrategy(step_checkpoints=[0, 2], beam_width=2).generate(model, batch, reward)


def test_flow_predictions_and_simulation_vectors_use_flow_emit_masks(token_batch, flow_matcher) -> None:
    """Both prediction conversions receive modality-specific flow emit masks."""
    batch = {
        **token_batch,
        "x_t": {
            "bb_ca": torch.ones(2, len(TOKEN_NAMES), 3),
            "local_latents": torch.ones(2, len(TOKEN_NAMES), 2),
        },
        "t": {"bb_ca": torch.zeros(2), "local_latents": torch.zeros(2)},
    }
    predictions = {mode: {"v": torch.ones_like(value)} for mode, value in batch["x_t"].items()}

    clean = flow_matcher.nn_out_add_clean_sample_prediction(batch, predictions)
    simulation = flow_matcher.nn_out_add_simulation_tensor(batch, predictions)

    for mode, field in (("bb_ca", "flow_bb_emit"), ("local_latents", "flow_latent_emit")):
        _assert_grid(clean[mode]["x_1"].ne(0).any(dim=-1), field)
        _assert_grid(simulation[mode]["v"].ne(0).any(dim=-1), field)


@pytest.mark.parametrize(
    ("mode", "field"),
    (("bb_ca", "flow_bb_emit"), ("local_latents", "flow_latent_emit")),
)
def test_guidance_delta_uses_flow_emit_mask(mode, field) -> None:
    """Guidance cannot update a token that carries no state for its modality."""
    batch = {
        "padding_mask": _matrix("padding"),
        "is_in_context_conditioning": _matrix("is_in_context_conditioning"),
        "supports_local_latent": _matrix("supports_local_latent"),
    }
    ref = torch.ones(2, len(TOKEN_NAMES), 2)
    emit_mask = _flow_emit_mask(mode, {"batch": batch}, ref)
    _assert_grid(emit_mask.bool().squeeze(-1), field)


def test_flow_loss_changes_only_for_supervised_grid_cells(token_batch, flow_matcher) -> None:
    """A large one-token prediction error affects exactly the supervised modalities."""
    n = len(TOKEN_NAMES)
    rows = {key: value[:1].expand(n, *value.shape[1:]) for key, value in token_batch.items()}
    batch = {
        **rows,
        "x_0": {
            "bb_ca": torch.zeros(n, n, 3),
            "local_latents": torch.zeros(n, n, 2),
        },
        "x_1": {
            "bb_ca": torch.zeros(n, n, 3),
            "local_latents": torch.zeros(n, n, 2),
        },
        "x_t": {
            "bb_ca": torch.zeros(n, n, 3),
            "local_latents": torch.zeros(n, n, 2),
        },
        "t": {"bb_ca": torch.full((n,), 0.5), "local_latents": torch.full((n,), 0.5)},
    }
    predictions = {
        "bb_ca": {"x_1": torch.zeros_like(batch["x_1"]["bb_ca"])},
        "local_latents": {"x_1": torch.zeros_like(batch["x_1"]["local_latents"])},
    }
    diagonal = torch.arange(n)
    predictions["bb_ca"]["x_1"][diagonal, diagonal] = 10.0
    predictions["local_latents"]["x_1"][diagonal, diagonal] = 10.0

    losses = flow_matcher.compute_fm_loss(batch, predictions)

    assert torch.equal(losses["bb_ca"].ne(0), torch.tensor(GRID["bb_supervised"]))
    assert torch.equal(losses["local_latents"].ne(0), torch.tensor(GRID["latent_supervised"]))
    assert torch.equal(
        predictions["bb_ca"]["x_1"][diagonal, diagonal].ne(0).any(dim=-1),
        torch.tensor(GRID["flow_bb_emit"]),
    )
    assert torch.equal(
        predictions["local_latents"]["x_1"][diagonal, diagonal].ne(0).any(dim=-1),
        torch.tensor(GRID["flow_latent_emit"]),
    )


def test_autoencoder_losses_change_only_for_supervised_grid_cells(token_batch) -> None:
    """Structure, sequence, and KL losses ignore every unsupervised AE token."""
    n = len(TOKEN_NAMES)
    rows = {key: value[:1].expand(n, *value.shape[1:]) for key, value in token_batch.items()}
    loss_mask = compute_ae_supervision_mask(rows)
    atom_mask = rows["atom37_mask"]
    coords = torch.zeros(n, n, 37, 3)
    pred = coords.clone()
    diagonal = torch.arange(n)
    pred[diagonal, diagonal] = 1.0
    output = {
        "coors_nm": pred,
        "atom_mask": atom_mask,
        "seq_logits": torch.zeros(n, n, 33),
        "aatype_max": torch.zeros(n, n, dtype=torch.long),
    }
    batch = {**rows, "coords_nm": coords, "residue_type": torch.zeros(n, n, dtype=torch.long)}

    struct = AutoEncoder.compute_struct_rec_loss(output, batch, loss_mask=loss_mask)["struct_l2"]
    baseline_seq = AutoEncoder.compute_seq_rec_loss(None, output, batch, loss_mask=loss_mask)["ce_seq"]
    output["seq_logits"][diagonal, diagonal, 0] = 10.0
    changed_seq = AutoEncoder.compute_seq_rec_loss(None, output, batch, loss_mask=loss_mask)["ce_seq"]
    mean = torch.zeros(n, n, 2)
    mean[diagonal, diagonal] = 3.0
    kl_components = AutoEncoder._per_component_kl(None, mean, torch.zeros_like(mean), loss_mask)

    expected = torch.tensor(GRID["latent_supervised"])
    assert torch.equal(struct.ne(0), expected)
    assert torch.equal(changed_seq.ne(baseline_seq), expected)
    assert torch.equal(kl_components.any(dim=(-1, -2)), expected)


def test_final_formatting_matches_flow_emit_contract(token_batch, flow_matcher) -> None:
    """Formatting retains atomized BBCA while suppressing context and padding output."""
    decoded_batches = []

    def decode(batch):
        decoded_batches.append(batch)
        _, emit = compute_ae_masks(batch)
        return {
            "coors_nm": torch.ones(2, len(TOKEN_NAMES), 37, 3) * emit[..., None, None],
            "residue_type": torch.ones(2, len(TOKEN_NAMES), dtype=torch.long) * emit,
            "emit_mask": emit,
        }

    flow_state = flow_matcher.apply_flow_emit_masks(
        {
            "bb_ca": torch.ones(2, len(TOKEN_NAMES), 3),
            "local_latents": torch.ones(2, len(TOKEN_NAMES), 2),
        },
        token_batch,
    )
    model = SimpleNamespace(autoencoder=SimpleNamespace(decode=decode))
    result = Proteina._format_sample_local_latents(
        model,
        x=flow_state,
        ret_mode="coors37_n_aatype",
        batch=token_batch,
    )

    assert len(decoded_batches) == 1
    assert decoded_batches[0]["padding_mask"].shape == (2, len(TOKEN_NAMES))
    assert torch.equal(decoded_batches[0]["is_in_context_conditioning"], token_batch["is_in_context_conditioning"])
    assert torch.equal(decoded_batches[0]["supports_local_latent"], token_batch["supports_local_latent"])
    _assert_grid(result["coors"].ne(0).any(dim=(-1, -2)), "flow_bb_emit")
    _assert_grid(result["residue_type"] >= 0, "flow_latent_emit")

    atomized = _matrix("flow_bb_emit") & ~_matrix("flow_latent_emit")
    assert torch.equal(result["coors"][..., 1, :][atomized], flow_state["bb_ca"][atomized] * 10)
    atomized_coords = result["coors"][atomized].clone()
    atomized_coords[:, 1] = 0
    assert not atomized_coords.any()
