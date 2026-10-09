"""Backend selection preserves checkpoint weights and numerical kernel policies."""

import pytest
import torch
from torch import nn

from rfproteina.nn.execution import (
    BlockExecution,
    configure_execution,
    inference_padding_buckets,
    resolve_backends,
    run_trunk,
)
from rfproteina.nn.modules.openfold_modules import triangular_multiplicative_update as triangle


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, s, z, c, mask):
        return (s * self.weight + c) * mask[..., None]


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_layers = nn.ModuleList([Attention()])
        self.pair_update_layers = nn.ModuleList([])


def every_component(backend):
    return {"inference": {key: backend for key in ("flow", "encoder", "decoder")}}


@pytest.mark.parametrize(
    "public,internal", [("eager", "eager"), ("torch_compile", "torch"), ("torch", "torch"), ("bioir", "bioir")]
)
def test_resolve_selectors(public, internal):
    assert set(resolve_backends(every_component(public), "inference").values()) == {internal}


def test_invalid_and_padding_policies():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_backends({"inference": {"flow": "bogus"}}, "inference")
    with pytest.raises(ValueError, match="tri_mult"):
        resolve_backends({"inference": {"tri_mult": "bogus"}}, "inference")
    with pytest.raises(ValueError, match="multiples of 16"):
        inference_padding_buckets({"inference": {"length_buckets": [17]}})
    eager = every_component("eager")
    assert inference_padding_buckets(eager) is None
    assert inference_padding_buckets({"inference": {"flow": "torch_compile"}})[0] == 16
    # BioIR pads inside the trunk; bucketing it here would only enlarge the shape.
    assert inference_padding_buckets(every_component("bioir")) is None
    bucketed = {"inference": {**every_component("bioir")["inference"], "length_buckets": [448]}}
    assert inference_padding_buckets(bucketed) is None


def test_bioir_is_the_whole_workflow_or_nothing():
    for mixed in ({"flow": "bioir"}, {"flow": "bioir", "encoder": "bioir", "decoder": "eager"}):
        with pytest.raises(ValueError, match="whole workflow"):
            resolve_backends({"inference": mixed}, "inference")
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_backends({"training": every_component("bioir")["inference"]}, "training")


def test_guidance_is_refused_rather_than_quietly_downgraded():
    with pytest.raises(ValueError, match="guidance"):
        resolve_backends(every_component("bioir"), "inference", guidance=True)
    for backend in ("eager", "torch_compile", "torch"):
        policy = every_component(backend)
        assert resolve_backends(policy, "inference", guidance=True) == resolve_backends(policy, "inference")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_invalid_graph_capacity_is_rejected_before_conversion(value):
    with pytest.raises(ValueError, match="max_graphs"):
        resolve_backends({"bioir": {"max_graphs": value}}, "inference")
    assert resolve_backends({"bioir": {"max_graphs": None}}, "inference")


@pytest.mark.parametrize("option", ["triangle_backend", "compile_shared_triangles", "flow_prefix_masks"])
def test_options_bioir_decides_for_itself_are_rejected(option):
    with pytest.raises(ValueError, match="automatically"):
        resolve_backends({"bioir": {option: "shared"}}, "inference")


def test_eager_and_compile_preserve_weights(monkeypatch):
    trunk = Trunk()
    model = nn.Module()
    model.nn = trunk
    before = {k: v.clone() for k, v in model.state_dict().items()}
    identities = [id(p) for p in model.parameters()]
    calls = []
    monkeypatch.setattr(torch, "compile", lambda block, **kwargs: (calls.append(kwargs) or block))
    args = (torch.ones(1, 3, 1), None, torch.zeros(1, 3, 1), torch.tensor([[True, True, False]]))
    for backend in ("eager", "torch", "eager"):
        configure_execution(model, {"inference": {"flow": backend}}, "inference")
        out, _ = run_trunk(trunk, *args)
        torch.testing.assert_close(out, torch.tensor([[[2.0], [2.0], [0.0]]]))
        assert [id(p) for p in model.parameters()] == identities
        assert model.state_dict().keys() == before.keys()
        for key in before:
            torch.testing.assert_close(model.state_dict()[key], before[key])
    assert len(calls) == 1


def test_compile_inference_disables_cuda_graph_modes(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "compile", lambda block, **kwargs: (calls.append(kwargs) or block))
    runtime = BlockExecution(backend="torch", cuda_graphs=False, compile_kwargs={"mode": "reduce-overhead"})
    runtime.run(Trunk(), torch.ones(1, 2, 1), None, torch.zeros(1, 2, 1), torch.ones(1, 2, dtype=torch.bool))
    assert "mode" not in calls[0]


def test_checkpoint_triangle_policy_overridden(monkeypatch):
    monkeypatch.setattr(triangle, "_CUEQ_UNAVAILABLE", None)
    model = nn.Module()
    model.nn = Trunk()
    model.nn.triangle = triangle.TriangleMultiplicationOutgoing(4, 4, backend="torch")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    configure_execution(model, {"inference": {"flow": "eager"}}, "inference")
    assert model.nn.triangle.backend == "cuequivariance"
    configure_execution(model, {"inference": {"flow": "torch", "tri_mult": "torch"}}, "inference")
    assert model.nn.triangle.backend == "torch"
    for key in before:
        torch.testing.assert_close(model.state_dict()[key], before[key])


def test_missing_bioir_is_actionable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "bionemo_ir.models.rfproteina":
            raise ImportError("RFProteína model absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    model = nn.Module()
    model.is_inference = True
    with pytest.raises(ImportError, match="ductr/rfd4"):
        configure_execution(model, every_component("bioir"), "inference")


def test_bioir_selects_the_full_workflow_and_leaves_native_blocks_eager(monkeypatch):
    monkeypatch.setattr("rfproteina.nn.bioir_full.require_bioir", lambda: None)
    model = nn.Module()
    model.nn = Trunk()
    model.is_inference = True
    config = {**every_component("bioir"), "bioir": {"max_graphs": 3}}
    configure_execution(model, config, "inference")
    assert model._full_bioir_options == {"max_graphs": 3}
    assert model.nn._execution.backend == "eager"
    configure_execution(model, every_component("eager"), "inference")
    assert model._full_bioir_options is None


def test_bioir_outside_the_generation_entry_point_is_refused():
    model = nn.Module()
    model.nn = Trunk()
    with pytest.raises(ValueError, match="training-time validation"):
        configure_execution(model, every_component("bioir"), "inference")


def test_public_cueq_011_parameter_packing_and_gradients(monkeypatch):
    source = triangle.TriangleMultiplicationOutgoing(4, 3, bias=False, backend="torch")
    received = {}

    def operation(z, **kwargs):
        received.update(kwargs)
        return z + kwargs["p_in_weight"].sum() + kwargs["g_in_weight"].sum()

    monkeypatch.setattr(triangle, "_CUEQ_SPLIT", None)
    monkeypatch.setattr(triangle, "_CUEQ_PACKED", operation)
    z = torch.zeros(1, 4, 4, 4, requires_grad=True)
    out = source._cueq_forward(z, torch.ones(1, 4, 4), source._cueq_parameters())
    torch.testing.assert_close(received["p_in_weight"], torch.cat((source.linear_a_p.weight, source.linear_b_p.weight)))
    torch.testing.assert_close(received["g_in_weight"], torch.cat((source.linear_a_g.weight, source.linear_b_g.weight)))
    assert torch.count_nonzero(received["p_in_bias"]) == 0
    assert received["direction"] == "outgoing"
    out.sum().backward()
    assert source.linear_a_p.weight.grad is not None
    assert source.linear_b_p.weight.grad is not None


@pytest.mark.parametrize("backend", ["eager", "torch_compile", "bioir"])
def test_hydra_inference_backend_composition(backend):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    configs = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=configs, version_base=None):
        cfg = compose(config_name="inference", overrides=[f"inference_backend={backend}"])
    expected = "torch" if backend == "torch_compile" else backend
    assert resolve_backends(cfg.execution, "inference") == dict.fromkeys(("flow", "encoder", "decoder"), expected)
    assert cfg.execution.inference.tri_mult == "cuequivariance"
    # The default composition configures the compile buckets, which every backend but BioIR pads to.
    expected_buckets = None if backend == "bioir" else list(cfg.compile_lengths)
    assert inference_padding_buckets(cfg.execution) == expected_buckets
    with initialize_config_dir(config_dir=configs, version_base=None):
        mixed = compose(
            config_name="inference", overrides=[f"inference_backend={backend}", "execution.inference.decoder=eager"]
        )
    if backend == "bioir":
        with pytest.raises(ValueError, match="whole workflow"):
            resolve_backends(mixed.execution, "inference")
    else:
        assert resolve_backends(mixed.execution, "inference")["decoder"] == "eager"


def test_return_to_training_discards_inference_cache():
    from rfproteina.nn.local_latents_transformer import LocalLatentsTransformer
    from rfproteina.partial_autoencoder.decoder import DecoderTransformer
    from rfproteina.partial_autoencoder.encoder import EncoderTransformer

    for cls in (LocalLatentsTransformer, EncoderTransformer, DecoderTransformer):
        trunk = cls.__new__(cls)
        nn.Module.__init__(trunk)
        trunk._execution = BlockExecution(backend="torch")
        trunk.eval()
        assert trunk._execution.backend == "torch"
        trunk.train()
        assert "_execution" not in trunk.__dict__


def test_cueq_012_split_parameter_dispatch(monkeypatch):
    source = triangle.TriangleMultiplicationIncoming(4, 3, bias=True, backend="torch")
    called = []

    def split(z, direction, mask, *parameters, eps):
        called.append((direction, len(parameters), eps))
        return z + parameters[0].sum()

    monkeypatch.setattr(triangle, "_CUEQ_SPLIT", split)
    z = torch.zeros(1, 4, 4, 4)
    out = source._cueq_forward(z, torch.ones(1, 4, 4), source._cueq_parameters())
    assert called == [("incoming", 16, source.layer_norm_in.eps)]
    assert out.shape == z.shape


def test_whole_pass_owns_graph_output_and_keeps_parameters(monkeypatch):
    trunk = Trunk()
    model = nn.Module()
    model.nn = trunk
    original_keys = tuple(model.state_dict())
    storage = torch.empty(1, 2, 1)
    calls = []

    def compile_with_reused_storage(fn, **kwargs):
        calls.append(kwargs)

        def compiled(*args):
            out, _ = fn(*args)
            storage.copy_(out)
            return storage, None

        return compiled

    monkeypatch.setattr(torch, "compile", compile_with_reused_storage)
    configure_execution(model, {"inference": {"flow": "torch", "compile_scope": "whole_pass"}}, "inference")
    args = (torch.ones(1, 2, 1), None, torch.zeros(1, 2, 1), torch.ones(1, 2, dtype=torch.bool))
    with torch.no_grad():
        first, _ = run_trunk(trunk, *args)
        second, _ = run_trunk(trunk, args[0] * 3, *args[1:])
    torch.testing.assert_close(first, torch.full_like(first, 2))
    torch.testing.assert_close(second, torch.full_like(second, 6))
    assert first.data_ptr() != storage.data_ptr()
    assert len(calls) == 1 and calls[0]["mode"] == "reduce-overhead"
    assert tuple(model.state_dict()) == original_keys
    with pytest.raises(RuntimeError, match="disabled gradients"):
        run_trunk(trunk, *args)
    model.guidance_model = nn.Identity()
    model.decoder = Trunk()
    configure_execution(model, {"inference": {"flow": "torch", "compile_scope": "whole_pass"}}, "inference")
    assert model.decoder._execution.compile_scope == "blocks"
