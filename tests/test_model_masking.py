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
from atomworks.ml.conditions import C_SEQ
from omegaconf import OmegaConf

from rfproteina.encoding import UNIFIED_ATOM37_ENCODING
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
from rfproteina.partial_autoencoder.decoder import DecoderTransformer, get_allowed_residue_type_mask, get_atom_mask
from rfproteina.partial_autoencoder.encoder import EncoderTransformer
from rfproteina.proteina import Proteina
from rfproteina.search.beam_search import BeamSearchStrategy

TOKEN_NAMES = ("generated", "atomized", "context", "padding", "unresolved")
N_RESIDUE_TYPES = UNIFIED_ATOM37_ENCODING.n_tokens
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
        logits = torch.full((*batch["padding_mask"].shape, N_RESIDUE_TYPES), float(len(calls.model_inputs)))
        return {"bb_ca": {}, "local_latents": {}, "insertion": {}, "sequence": {"logits": logits}}

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


def test_decoder_rejects_mismatched_residue_encoding() -> None:
    with pytest.raises(ValueError, match=f"requires {N_RESIDUE_TYPES} classes"):
        DecoderTransformer(n_residue_types=20, decoder={})


def test_autoencoder_rejects_mismatched_residue_encoding(tmp_path) -> None:
    cfg = SimpleNamespace(nn_ae={"n_residue_types": 20})
    with pytest.raises(ValueError, match=f"requires {N_RESIDUE_TYPES} classes"):
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
            logits = torch.zeros(*values.shape[:2], N_RESIDUE_TYPES)
            logits[..., 1] = 1.0
            return logits

        extras = {
            "logit_linear": residue_logits,
            "struct_linear": lambda values: values.repeat(1, 1, 37 * 3),
            "atom_presence_linear": lambda values: values.repeat(1, 1, 37),
            "n_residue_types": N_RESIDUE_TYPES,
            "abs_coors": True,
            "allowed_residue_types": get_allowed_residue_type_mask(),
            "residue_atom_mask": get_atom_mask(),
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
        logit_linear=lambda values: values.repeat(1, 1, N_RESIDUE_TYPES),
        struct_linear=lambda values: values.repeat(1, 1, 37 * 3),
        atom_presence_linear=lambda values: values.repeat(1, 1, 37),
        cond_factory=lambda input_batch: torch.ones(*input_batch["padding_mask"].shape, 1),
        transition_c_1=lambda value, _mask: value,
        transition_c_2=lambda value, _mask: value,
        pair_rep_factory=None,
        transformer_layers=[],
        nlayers=0,
        update_pair_repr=False,
        n_residue_types=N_RESIDUE_TYPES,
        abs_coors=False,
        allowed_residue_types=get_allowed_residue_type_mask(),
        residue_atom_mask=get_atom_mask(),
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
        sequence_readout_enabled=False,
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
        assert torch.all(extra_info["sequence_logits"] == len(calls.model_inputs))
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

    assert torch.all(info["sequence_logits"] == 2)
    for snapshot in info["trajectory_steps"]:
        assert torch.all(snapshot["sequence_logits"] == snapshot["step"] + 1)
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

    state, _, info = model.partial_simulation(
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

    assert torch.all(info["sequence_logits"] == 2)
    for emit_masks in calls.step_emit_masks:
        _assert_flow_emit_masks(emit_masks)
    for mode, field in (("bb_ca", "flow_bb_emit"), ("local_latents", "flow_latent_emit")):
        _assert_grid(calls.model_inputs[0][mode].ne(0).any(dim=-1), field)
        _assert_grid(state[mode].ne(0).any(dim=-1), field)


def test_beam_initialization_uses_masks_and_centering_groups(token_batch) -> None:
    """Beam noise receives tiled modality masks and stochastic-centering metadata."""
    fm = SimpleNamespace(
        data_modes=("bb_ca", "local_latents"),
        sample_schedule=lambda **_kwargs: (
            {mode: torch.linspace(0, 1, 3) for mode in fm.data_modes},
            {mode: torch.zeros(2) for mode in fm.data_modes},
        ),
    )

    molecule_iid = torch.tensor([[0, 0, 1, 1, 1], [2, 2, 2, 3, 3]])
    coord_cond_mask = torch.tensor(
        [[False, False, True, True, False], [True, False, False, False, True]], dtype=torch.bool
    )

    def sample_noise(_n, *, emit_masks, groups, coord_cond_mask, **_kwargs):
        _assert_flow_emit_masks({mode: value.reshape(2, 2, -1)[0] for mode, value in emit_masks.items()})
        assert torch.equal(groups["molecule_iid"], molecule_iid.repeat(2, 1))
        assert torch.equal(coord_cond_mask, batch["coord_cond_mask"].repeat(2, 1))
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
    batch = {
        **token_batch,
        "coords_nm": torch.zeros(2, len(TOKEN_NAMES), 37, 3),
        "molecule_iid": molecule_iid,
        "coord_cond_mask": coord_cond_mask,
    }

    with pytest.raises(RuntimeError, match="beam initialized"):
        BeamSearchStrategy(checkpoints={2: {}}, beam_width=2).generate(model, batch)


@pytest.mark.parametrize(
    ("invalidate", "message"),
    [
        pytest.param(lambda state: [], "must contain", id="non-mapping-state"),
        pytest.param(lambda state: {"x": state["x"]}, "must contain", id="missing-step"),
        pytest.param(lambda state: {"step": 1}, "must contain", id="missing-tensors"),
        pytest.param(lambda state: {**state, "step": -1}, "step must be an integer", id="negative-step"),
        pytest.param(lambda state: {**state, "step": 2}, "step must be an integer", id="schedule-end"),
        pytest.param(lambda state: {**state, "step": True}, "step must be an integer", id="boolean-step"),
        pytest.param(lambda state: {**state, "step": 0.5}, "step must be an integer", id="fractional-step"),
        pytest.param(lambda state: {**state, "x": []}, "modalities must match", id="non-mapping-modalities"),
        pytest.param(
            lambda state: {**state, "x": {"bb_ca": state["x"]["bb_ca"]}},
            "modalities must match",
            id="missing-modality",
        ),
        pytest.param(
            lambda state: {**state, "x": {**state["x"], "unknown": state["x"]["bb_ca"]}},
            "modalities must match",
            id="extra-modality",
        ),
    ],
)
def test_beam_rejects_invalid_saved_state_before_reseeding(
    token_batch, simulation_probe, tmp_path, monkeypatch, invalidate, message
) -> None:
    _assert_saved_candidate_rejected(token_batch, simulation_probe, tmp_path, monkeypatch, invalidate, message)


@pytest.mark.parametrize(
    "invalidate_tensor",
    [
        pytest.param(lambda value: value[:1], id="batch-size"),
        pytest.param(lambda value: value[:, :-1], id="token-count"),
        pytest.param(lambda value: value[..., :-1], id="latent-dimension"),
        pytest.param(lambda value: value.to(torch.int64), id="integer-tensor"),
        pytest.param(lambda value: value.fill_(float("nan")), id="nan"),
        pytest.param(lambda value: value.fill_(float("inf")), id="infinity"),
        pytest.param(lambda value: value.tolist(), id="non-tensor"),
    ],
)
def test_beam_rejects_invalid_saved_tensors_before_reseeding(
    token_batch, simulation_probe, tmp_path, monkeypatch, invalidate_tensor
) -> None:
    def invalidate(state):
        state["x"]["local_latents"] = invalidate_tensor(state["x"]["local_latents"])
        return state

    _assert_saved_candidate_rejected(
        token_batch,
        simulation_probe,
        tmp_path,
        monkeypatch,
        invalidate,
        "local_latents must be a finite floating tensor",
    )


def _assert_saved_candidate_rejected(token_batch, simulation_probe, tmp_path, monkeypatch, invalidate, message):
    fm, _ = simulation_probe
    model = SimpleNamespace(
        fm=fm,
        inf_cfg=OmegaConf.create(
            {
                "args": {"nsteps": 2},
                "model": {mode: {"simulation_step_params": {}} for mode in fm.data_modes},
            }
        ),
        device=torch.device("cpu"),
        inference_seed=17,
        preprocess_batch_for_generation=lambda batch: batch,
        recycle_depths=lambda nsteps, full_nsteps=None: (1,) * nsteps,
    )
    batch = {**token_batch, "coords_nm": torch.zeros(2, len(TOKEN_NAMES), 37, 3)}
    state = {
        "step": 1,
        "x": {
            mode: torch.zeros(*batch["padding_mask"].shape, fm.base_flow_matchers[mode].dim) for mode in fm.data_modes
        },
    }
    saved_path = tmp_path / "invalid_candidate.pt"
    torch.save(invalidate(state), saved_path)
    reseed = Mock()
    monkeypatch.setattr(torch, "manual_seed", reseed)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match=message):
        BeamSearchStrategy(checkpoints={2: {}}, beam_width=2, resume_from=str(saved_path), resume_seed=42).generate(
            model, batch
        )

    reseed.assert_not_called()
    assert model.inference_seed == 17
    assert torch.equal(torch.get_rng_state(), rng_before)


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


def test_sequence_prediction_loss_uses_local_latent_supervision_mask(token_batch) -> None:
    """The sequence probe ignores context, padding, unsupported, and unresolved tokens."""
    n = len(TOKEN_NAMES)
    matcher = ProductSpaceFlowMatcher(
        OmegaConf.create(
            {
                "product_flowmatcher": {"bb_ca": {"dim": 3}, "local_latents": {"dim": 2}},
                "loss": {
                    "sequence_prediction": {
                        "enabled": True,
                        "weight": 1.0,
                        "t_source": "local_latents",
                        "t_min": 0.0,
                        "t_max": 1.0,
                    }
                },
            }
        )
    )
    rows = {key: value[:1].expand(n, *value.shape[1:]) for key, value in token_batch.items()}
    batch = {
        **rows,
        "residue_type": torch.zeros(n, n, dtype=torch.long),
        C_SEQ.full_name: torch.full((n, n), UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"], dtype=torch.long),
        "t": {"local_latents": torch.ones(n)},
    }
    baseline_logits = torch.zeros(n, n, N_RESIDUE_TYPES)
    changed_logits = baseline_logits.clone()
    diagonal = torch.arange(n)
    changed_logits[diagonal, diagonal, 0] = 10.0

    baseline = matcher._compute_sequence_prediction_loss(batch, {"sequence": {"logits": baseline_logits}})
    changed = matcher._compute_sequence_prediction_loss(batch, {"sequence": {"logits": changed_logits}})

    expected = torch.tensor(GRID["latent_supervised"])
    assert torch.equal(
        changed["sequence_prediction_ce_justlog"].ne(baseline["sequence_prediction_ce_justlog"]), expected
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
        "seq_logits": torch.zeros(n, n, N_RESIDUE_TYPES),
        "aatype_max": torch.zeros(n, n, dtype=torch.long),
    }
    batch = {**rows, "coords_nm": coords, "residue_type": torch.zeros(n, n, dtype=torch.long)}

    struct = AutoEncoder.compute_struct_rec_loss(output, batch, loss_mask=loss_mask)["struct_l2"]
    ae = SimpleNamespace()
    baseline_seq = AutoEncoder.compute_seq_rec_loss(ae, output, batch, loss_mask=loss_mask)["ce_seq"]
    output["seq_logits"][diagonal, diagonal, 0] = 10.0
    changed_seq = AutoEncoder.compute_seq_rec_loss(ae, output, batch, loss_mask=loss_mask)["ce_seq"]
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

    def decode(batch, required_atom_mask=None):
        decoded_batches.append(batch)
        _, emit = compute_ae_masks(batch)
        return {
            "coors_nm": torch.ones(2, len(TOKEN_NAMES), 37, 3) * emit[..., None, None],
            "residue_type": torch.ones(2, len(TOKEN_NAMES), dtype=torch.long) * emit,
            "emit_mask": emit,
            "atom_mask": emit[..., None].expand(-1, -1, 37),
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


def test_sequence_prediction_conditions_decoder_through_batch(token_batch, flow_matcher) -> None:
    """Predicted identities fill generated tokens without replacing exact conditions."""
    decoded_batches = []

    def decode(batch, required_atom_mask=None):
        decoded_batches.append(batch)
        _, emit = compute_ae_masks(batch)
        return {
            "coors_nm": torch.ones(2, len(TOKEN_NAMES), 37, 3) * emit[..., None, None],
            "residue_type": batch[C_SEQ.full_name] * emit,
            "emit_mask": emit,
            "atom_mask": emit[..., None].expand(-1, -1, 37),
        }

    batch = {
        **token_batch,
        C_SEQ.full_name: torch.full(
            (2, len(TOKEN_NAMES)), UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"], dtype=torch.long
        ),
        "token_type": torch.zeros(2, len(TOKEN_NAMES), dtype=torch.long),
    }
    batch[C_SEQ.full_name][0, 0] = 7
    batch["sequence_logits"] = torch.zeros(2, len(TOKEN_NAMES), N_RESIDUE_TYPES)
    batch["sequence_logits"][..., 4] = 1.0

    flow_state = flow_matcher.apply_flow_emit_masks(
        {
            "bb_ca": torch.ones(2, len(TOKEN_NAMES), 3),
            "local_latents": torch.ones(2, len(TOKEN_NAMES), 2),
        },
        batch,
    )
    model = SimpleNamespace(
        autoencoder=SimpleNamespace(decode=decode),
        inf_cfg=OmegaConf.create({"args": {"condition_decoder_on_sequence_prediction": True}}),
    )
    Proteina._format_sample_local_latents(
        model,
        x=flow_state,
        ret_mode="coors37_n_aatype",
        batch=batch,
    )

    conditioned_batch = decoded_batches[0]
    _, emit_mask = compute_ae_masks(batch)
    expected_values = torch.where(emit_mask, 4, batch[C_SEQ.full_name])
    expected_values[0, 0] = 7
    assert torch.equal(conditioned_batch[C_SEQ.full_name], expected_values)
