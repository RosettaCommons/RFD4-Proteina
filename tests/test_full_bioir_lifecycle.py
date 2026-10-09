"""CPU lifecycle contracts; full kernels/structures require the separate GPU gate."""

from types import SimpleNamespace

import pytest
import torch

from rfproteina.nn import bioir_full as bridge
from rfproteina.nn import execution


class Native(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = torch.nn.Linear(2, 2)
        self.autoencoder = torch.nn.ModuleDict(
            {
                "encoder": torch.nn.Linear(2, 2),
                "decoder": torch.nn.Linear(2, 2),
            }
        )
        self.is_inference = True
        self.device = torch.device("cpu")


def test_conversion_snapshots_selected_weights_without_aliases_or_checkpoint_reads(monkeypatch):
    native = Native()
    with torch.no_grad():
        native.nn.weight.fill_(7)  # Actual post-load/EMA-selected values.

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not reopen the checkpoint")

    monkeypatch.setattr(torch, "load", forbidden)
    states = bridge._copy_loaded_states(native)
    assert set(states["flow"]) == {"nn.weight", "nn.bias"}
    assert set(states["autoencoder"]) == set(native.autoencoder.state_dict())
    assert not states["flow"]["nn.weight"].requires_grad
    with torch.no_grad():
        native.nn.weight.fill_(9)
    torch.testing.assert_close(states["flow"]["nn.weight"], torch.full((2, 2), 7.0))


def test_full_policy_is_deferred_and_cleared_when_returning_to_native(monkeypatch):
    monkeypatch.setattr(execution, "configure_triangle_backends", lambda *args: None)
    monkeypatch.setattr(bridge, "require_bioir", lambda: None)
    native = Native()
    keys = tuple(native.state_dict())
    cfg = {
        "inference": dict.fromkeys(("flow", "encoder", "decoder"), "bioir"),
        "bioir": {"max_graphs": None, "attention_backend": None},
    }
    execution.configure_execution(native, cfg, "inference")
    assert native._full_bioir is None
    assert native._full_bioir_options == cfg["bioir"]
    assert all(m._execution.backend == "eager" for m in execution.execution_components(native).values())
    assert tuple(native.state_dict()) == keys
    cfg["inference"] = dict.fromkeys(("flow", "encoder", "decoder"), "eager")
    execution.configure_execution(native, cfg, "inference")
    assert native._full_bioir_options is None
    cfg["inference"]["flow"] = "bioir"
    with pytest.raises(ValueError, match="whole workflow"):
        execution.configure_execution(native, cfg, "inference")
    assert native._full_bioir_options is None
    assert tuple(native.state_dict()) == keys


def test_initializer_waits_for_final_device_and_keeps_runtime_out_of_state_dict(monkeypatch):
    native = Native()
    native._full_bioir_options = {}
    with pytest.raises(ValueError, match="final CUDA device"):
        bridge.initialize_full_bioir(native)
    calls = []

    # This test checks the lifecycle boundary only; it does not emulate CUDA kernels.
    def load(model, *, sampling_config, device, options):
        calls.append((model, device))
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(bridge, "load_full_model", load)
    monkeypatch.setattr(bridge, "sampling_from_native", lambda model: object())
    before = tuple(native.state_dict())
    adapter = bridge.initialize_full_bioir(native, "cuda:0")
    assert bridge.initialize_full_bioir(native, "cuda:0") is adapter
    assert calls == [(native, torch.device("cuda:0"))]
    assert tuple(native.state_dict()) == before
    assert "_full_bioir" not in native._modules
    assert adapter.native.nn is native.nn
    # An initialized full runtime offloaded native weights. Switching back must
    # restore both modules to the caller's active device before native execution.
    restored = []
    monkeypatch.setattr(adapter.model, "reset_graphs", lambda: restored.append("reset"), raising=False)
    for name in ("nn", "autoencoder"):
        module = getattr(native, name)
        original_to = module.to

        def restore(device, *, name=name, original_to=original_to):
            restored.append((name, device))
            return original_to(device)

        monkeypatch.setattr(module, "to", restore)
    monkeypatch.setattr(execution, "configure_triangle_backends", lambda *args: None)
    execution.configure_execution(
        native, {"inference": dict.fromkeys(("flow", "encoder", "decoder"), "eager")}, "inference"
    )
    assert restored == ["reset", ("nn", native.device), ("autoencoder", native.device)]
    assert native._full_bioir is None
    assert native._full_bioir_options is None
    assert tuple(native.state_dict()) == before


def test_cli_global_batch_stream_continues_after_bioir_fork():
    from bionemo_ir._torch.sampling.random import randn

    original = torch.get_rng_state().clone()
    try:
        torch.manual_seed(91)
        torch.randn(17)  # Already consumed by featurization.
        start = torch.get_rng_state().clone()
        expected = torch.randn((3, 4, 2))
        expected_end = torch.get_rng_state().clone()
        torch.set_rng_state(start)
        with (
            bridge.native_random_streams({"padding_mask": torch.ones(3, 4, dtype=torch.bool)}),
            torch.random.fork_rng(devices=[]),
        ):
            torch.manual_seed(999)  # BioIR's internal fork must not replace caller state.
            actual = randn((3, 4, 2), device="cpu", dtype=torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), expected_end)
    finally:
        torch.set_rng_state(original)


def test_graph_capacity_policy_delegates_only_admitted_growth():
    native = Native()
    model = torch.nn.Linear(2, 2)
    adapter = bridge.FullBioIRAdapter(native, model)
    adapter.options = {"max_graphs": 3}
    policy = SimpleNamespace(n_recycle_schedule=[(5, 1), (5, 2), (5, 4)])
    with pytest.raises(ValueError, match="at least 6 graph slots"):
        adapter.configure_graphs(policy)
    assert adapter.graph_capacity == 0
    calls = []
    model.configure_graphs = lambda **kwargs: calls.append(kwargs)
    adapter.options = {}
    adapter.configure_graphs(policy)
    adapter.configure_graphs(policy)
    adapter.configure_graphs(SimpleNamespace(n_recycle_schedule=[(5, 1)]))
    adapter.configure_graphs(SimpleNamespace(n_recycle_schedule=[(5, 1), (5, 2), (5, 4), (5, 6)]))
    assert calls == [
        {"capacity": 6, "verify_capture": True, "max_tokens": 511},
        {"capacity": 8, "verify_capture": True, "max_tokens": 511},
    ]


@pytest.mark.parametrize(
    ("options", "max_tokens"),
    [({}, 511), ({"cuda_graph_max_tokens": 1024}, 1024), ({"cuda_graph_max_tokens": None}, None)],
    ids=["default-under-512", "explicit", "bioir-limit"],
)
def test_graphs_cover_requests_up_to_the_token_limit(options, max_tokens):
    model = torch.nn.Linear(2, 2)
    calls = []
    model.configure_graphs = lambda **kwargs: calls.append(kwargs)
    adapter = bridge.FullBioIRAdapter(Native(), model)
    adapter.options = options
    adapter.configure_graphs(SimpleNamespace(n_recycle_schedule=[(5, 1)]))
    assert calls == [{"capacity": 2, "verify_capture": True, "max_tokens": max_tokens}]
    adapter = bridge.FullBioIRAdapter(Native(), model)
    adapter.options = {**options, "cuda_graph": False}
    adapter.configure_graphs(SimpleNamespace(n_recycle_schedule=[(5, 1)]))
    assert len(calls) == 1


def test_native_predict_dispatch_uses_shared_adapter(monkeypatch):
    from rfproteina.proteina import Proteina

    native = Native()
    native._full_bioir_options = {}
    batch = {"marker": object()}
    expected = [{"example_id": "a"}]
    calls = []

    def predict(model, value):
        calls.append((model, value))
        return expected

    monkeypatch.setattr(bridge, "predict_full_bioir", predict)
    assert Proteina.predict_step(native, batch, 0) is expected
    assert calls == [(native, batch)]


@pytest.mark.parametrize("with_mask", [False, True])
def test_actual_batch_formatter_preserves_per_row_decoder_occupancy(with_mask):
    from rfproteina.proteina import Proteina

    calls = []

    class Formatter:
        format_sample_prots_to_atom_arrays = Proteina.format_sample_prots_to_atom_arrays

        def _process_single_sample_to_atom_array(self, **kwargs):
            calls.append(kwargs)
            return {"generated_atom_array": object()}

    sample = {
        "coors": torch.full((2, 2, 37, 3), 10.0),
        "residue_type": torch.tensor([[1, 1], [2, 2]]),
        "padding_mask": torch.ones(2, 2, dtype=torch.bool),
    }
    if with_mask:
        sample["atom_mask"] = torch.zeros(2, 2, 37, dtype=torch.bool)
        sample["atom_mask"][0, :, :5] = True
        sample["atom_mask"][1, :, :4] = True
    output = Formatter().format_sample_prots_to_atom_arrays(
        sample, {"atom_array_batch": [object(), object()], "example_id_batch": ["a", "b"]}
    )
    assert [row["example_id"] for row in output] == ["a", "b"]
    for i, call in enumerate(calls):
        if with_mask:
            torch.testing.assert_close(call["atom_mask"], sample["atom_mask"][i])
        else:
            assert call["atom_mask"] is None


def test_bridge_passes_explicit_decoder_mask_and_native_backbone_emission():
    from rfproteina.model_masks import compute_flow_emit_masks

    captured = {}

    def format_native(sample, batch, **kwargs):
        captured.update(sample)
        return [{"example_id": "a"}]

    prediction = SimpleNamespace(
        coordinates=torch.ones(1, 3, 37, 3),
        residue_types=torch.tensor([[1, -1, 1]]),
        residue_type_offset=1,
        atom_mask=torch.zeros(1, 3, 37, dtype=torch.bool),
        partial_redesign=None,
        insertion_logits=None,
    )
    prediction.atom_mask[..., 1] = True
    batch = {
        "padding_mask": torch.ones(1, 3, dtype=torch.bool),
        "coords": torch.zeros(1, 3, 3),
        "is_in_context_conditioning": torch.tensor([[False, True, False]]),
        "supports_local_latent": torch.tensor([[True, False, False]]),
    }
    features = bridge._features(batch)
    torch.testing.assert_close(features["flow_backbone_mask"], compute_flow_emit_masks(batch)["bb_ca"])
    assert features["padding_mask"][0, 1]
    bridge.format_prediction(
        SimpleNamespace(format_sample_prots_to_atom_arrays=format_native), prediction, features, batch
    )
    assert captured["atom_mask"] is prediction.atom_mask


def test_old_runtime_rejected_before_native_weight_copy(monkeypatch):
    from bionemo_ir.models.rfproteina import RFProteina

    def forbidden(*args, **kwargs):
        raise AssertionError("Must fail before copying native weights")

    monkeypatch.setattr(bridge, "_copy_loaded_states", forbidden)
    for name in bridge.REQUIRED_MODEL_API:
        with monkeypatch.context() as patch:
            patch.setattr(RFProteina, name, None)
            with pytest.raises(RuntimeError, match=f"missing \\['{name}'\\]"):
                bridge.load_full_model(object(), sampling_config=None, device=torch.device("cpu"), options={})
    with monkeypatch.context() as patch:
        patch.setattr(RFProteina, "configure_graphs", lambda self, *, capacity=1, verify_capture=False: None)
        with pytest.raises(RuntimeError, match=r"missing \['configure_graphs\(max_tokens=\)'\]"):
            bridge.load_full_model(object(), sampling_config=None, device=torch.device("cpu"), options={})


def test_loader_passes_selected_weights_and_native_policy_to_public_factory(monkeypatch):
    from bionemo_ir.models.rfproteina import RFProteina

    native = Native()
    policy, converted = object(), object()
    captured = {}

    def build(flow, autoencoder, **kwargs):
        captured.update(flow=flow, autoencoder=autoencoder, **kwargs)
        return converted

    monkeypatch.setattr(RFProteina, "from_reference_state_dicts", build)
    assert (
        bridge.load_full_model(
            native, sampling_config=policy, device=torch.device("cpu"), options={"attention_backend": "SDPA"}
        )
        is converted
    )
    assert captured["sampling_config"] is policy
    assert captured["device"] == torch.device("cpu")
    assert captured["attention_backend"] == "SDPA"
    torch.testing.assert_close(captured["flow"]["nn.weight"], native.nn.weight)
    torch.testing.assert_close(captured["autoencoder"]["encoder.weight"], native.autoencoder["encoder"].weight)
