"""Bounded graph-cache configuration; actual replay needs GPU qualification."""

import pytest
import torch
from torch import nn

from rfproteina.nn import bioir, execution


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_invalid_graph_capacity_fails_before_conversion(value):
    with pytest.raises(ValueError, match="max_graphs"):
        execution.resolve_backends({"bioir": {"max_graphs": value}}, "inference")
    with pytest.raises(ValueError, match="max_graphs"):
        bioir.convert_trunk([], [], execution.apply_blocks, max_graphs=value)


@pytest.mark.parametrize("capacity", [None, 3])
@pytest.mark.parametrize("enabled", [False, True])
def test_graph_capacity_reaches_public_wrapper_only_when_enabled(monkeypatch, capacity, enabled):
    monkeypatch.setattr(bioir, "_require_bioir", lambda: None)
    captured = []

    def wrap(module, *, max_graphs):
        captured.append(max_graphs)
        return module

    monkeypatch.setattr(bioir, "with_cuda_graph", wrap)
    options = {} if capacity is None else {"max_graphs": capacity}
    converted = bioir.convert_trunk([], [], execution.apply_blocks, cuda_graph=enabled, **options)
    assert captured == ([1 if capacity is None else capacity] if enabled else [])
    with torch.no_grad():
        single = torch.ones(1, 2, 4)
        actual, pair = converted(single, None, single, torch.ones(1, 2, dtype=torch.bool))
    torch.testing.assert_close(actual, single)
    assert pair is None


@pytest.mark.parametrize("capacity", [None, 1, 3])
def test_capacity_routes_to_all_trunks_without_enabling_ae_kf(monkeypatch, capacity):
    trunks = {name: nn.Identity() for name in ("flow", "encoder", "decoder")}
    monkeypatch.setattr(execution, "execution_components", lambda model: trunks)
    monkeypatch.setattr(execution, "configure_triangle_backends", lambda model, backend: None)
    monkeypatch.setattr(bioir, "_require_bioir", lambda: None)
    config = {
        "inference": {name: "bioir" for name in trunks},
        "bioir": {},
    }
    if capacity is not None:
        config["bioir"]["max_graphs"] = capacity
    execution.configure_execution(nn.Identity(), config, "inference")
    for name, trunk in trunks.items():
        options = trunk._execution.bioir_options
        assert options.get("max_graphs", 1) == (1 if capacity is None else capacity)
        assert options["prefix_masks"] == (name == "flow")
        assert "triangle_backend" not in options
