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


@pytest.mark.parametrize(
    "public,internal", [("eager", "eager"), ("torch_compile", "torch"), ("torch", "torch"), ("bioir", "bioir")]
)
def test_resolve_selectors(public, internal):
    policy = {"inference": {key: public for key in ("flow", "encoder", "decoder")}}
    assert set(resolve_backends(policy, "inference").values()) == {internal}
    assert resolve_backends(policy, "inference", guidance=True)["decoder"] == (
        "torch" if public == "bioir" else internal
    )


def test_invalid_and_padding_policies():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_backends({"inference": {"flow": "bogus"}}, "inference")
    with pytest.raises(ValueError, match="tri_mult"):
        resolve_backends({"inference": {"tri_mult": "bogus"}}, "inference")
    with pytest.raises(ValueError, match="multiples of 16"):
        inference_padding_buckets({"inference": {"length_buckets": [17]}})
    eager = {"inference": {key: "eager" for key in ("flow", "encoder", "decoder")}}
    assert inference_padding_buckets(eager) is None
    assert inference_padding_buckets({"inference": {"flow": "torch_compile"}})[0] == 16


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
    from rfproteina.nn import bioir

    monkeypatch.setattr(bioir, "BIOIR_AVAILABLE", False)
    monkeypatch.setattr(bioir, "BIOIR_IMPORT_ERROR", ImportError("blocks absent"))
    with pytest.raises(ImportError, match="ductr/rfd4"):
        configure_execution(nn.Module(), {"inference": {"flow": "bioir"}}, "inference")


def test_bioir_cache_tracks_weight_changes(monkeypatch):
    from rfproteina.nn import bioir

    trunk = Trunk()
    runtime = BlockExecution(backend="bioir")
    copies = []

    def convert(attention, pair, apply, **kwargs):
        from copy import deepcopy

        copied = deepcopy(attention)
        copies.append(copied)
        return lambda s, z, c, mask: apply(s, z, c, mask, copied, pair)

    monkeypatch.setattr(bioir, "convert_trunk", convert)
    args = (torch.ones(1, 2, 1), None, torch.zeros(1, 2, 1), torch.ones(1, 2, dtype=torch.bool))
    with torch.no_grad():
        first, _ = runtime.run(trunk, *args)
        runtime.run(trunk, *args)
        trunk.transformer_layers[0].weight.add_(1)
        second, _ = runtime.run(trunk, *args)
    assert len(copies) == 2
    torch.testing.assert_close(second - first, torch.ones_like(first))


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


def test_shared_bioir_triangle_matches_native_and_copies_weights():
    from rfproteina.nn.bioir import _SharedTriangle

    source = triangle.TriangleMultiplicationOutgoing(4, 3, bias=False, backend="torch").eval()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(0, 0.1)
    shared = _SharedTriangle(source, torch.float32)
    z = torch.randn(1, 5, 5, 4)
    mask = torch.ones(1, 5, 5)
    mask[:, 2] = 0
    with torch.no_grad():
        torch.testing.assert_close(shared(z, mask, trimul_metadata=None), source(z, mask))
    assert not shared.can_fuse_residual(z)
    assert {id(p) for p in shared.parameters()}.isdisjoint(id(p) for p in source.parameters())


@pytest.mark.parametrize("backend", ["eager", "torch_compile", "bioir"])
def test_hydra_inference_backend_composition(backend):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "configs"), version_base=None):
        cfg = compose(
            config_name="inference", overrides=[f"inference_backend={backend}", "execution.inference.decoder=eager"]
        )
    expected = "torch" if backend == "torch_compile" else backend
    policy = resolve_backends(cfg.execution, "inference")
    assert policy == {"flow": expected, "encoder": expected, "decoder": "eager"}
    assert cfg.execution.inference.tri_mult == "cuequivariance"


def test_return_to_training_discards_inference_cache():
    from rfproteina.nn.local_latents_transformer import LocalLatentsTransformer
    from rfproteina.partial_autoencoder.decoder import DecoderTransformer
    from rfproteina.partial_autoencoder.encoder import EncoderTransformer

    for cls in (LocalLatentsTransformer, EncoderTransformer, DecoderTransformer):
        trunk = cls.__new__(cls)
        nn.Module.__init__(trunk)
        trunk._execution = BlockExecution(backend="bioir")
        trunk.eval()
        assert trunk._execution.backend == "bioir"
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


@pytest.mark.parametrize("fused", [False, True])
def test_attention_projection_layout_preserves_gate_bias(fused):
    from rfproteina.nn.bioir import _map_attention_projections, _Weights

    source = nn.Module()
    source.mhba = nn.Module()
    source.mhba.mha = nn.Module()
    source.mhba.mha.to_qkv = nn.Linear(4, 6)
    source.mhba.mha.to_g = nn.Linear(4, 2)
    target = nn.Module()
    target.mha = nn.Module()
    if fused:
        target.mha.in_proj = nn.Linear(4, 8)
    else:
        target.mha.proj_q = nn.Linear(4, 2)
        target.mha.proj_g = nn.Linear(4, 2)
        target.mha.proj_kv = nn.Linear(4, 4)
    weights = _Weights(source)
    _map_attention_projections(weights, target.mha)
    weights.load(target)
    x = torch.randn(2, 3, 4)
    q, k, v = source.mhba.mha.to_qkv(x).chunk(3, dim=-1)
    gate = source.mhba.mha.to_g(x)
    expected = torch.cat((q, gate, k, v), dim=-1)
    actual = (
        target.mha.in_proj(x)
        if fused
        else torch.cat((target.mha.proj_q(x), target.mha.proj_g(x), target.mha.proj_kv(x)), dim=-1)
    )
    torch.testing.assert_close(actual, expected)
    target_gate_bias = target.mha.in_proj.bias[2:4] if fused else target.mha.proj_g.bias
    torch.testing.assert_close(target_gate_bias, source.mhba.mha.to_g.bias, rtol=0, atol=0)


@pytest.mark.parametrize(
    "missing",
    [
        "bionemo_ir._torch",
        "bionemo_ir._torch.layers",
        "bionemo_ir._torch.layers.transformers",
        "bionemo_ir._torch.layers.transformers.blocks",
    ],
)
def test_legacy_bioir_import_handles_missing_namespace_ancestors(monkeypatch, missing):
    from rfproteina.nn import bioir

    legacy = object()
    calls = []

    def import_blocks(name):
        calls.append(name)
        if name == "bionemo_ir.blocks":
            return legacy
        raise ModuleNotFoundError("current layout absent", name=missing)

    monkeypatch.setattr(bioir, "import_module", import_blocks)
    assert bioir._load_bioir_blocks() is legacy
    assert calls == ["bionemo_ir._torch.layers.transformers.blocks", "bionemo_ir.blocks"]


def test_bioir_import_does_not_hide_broken_current_dependency(monkeypatch):
    from rfproteina.nn import bioir

    calls = []

    def import_blocks(name):
        calls.append(name)
        raise ModuleNotFoundError("native dependency absent", name="cutlass")

    monkeypatch.setattr(bioir, "import_module", import_blocks)
    with pytest.raises(ModuleNotFoundError, match="native dependency absent"):
        bioir._load_bioir_blocks()
    assert calls == ["bionemo_ir._torch.layers.transformers.blocks"]
