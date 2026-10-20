"""CUDA-graph step boundaries for torch.compile'd trunks."""

from types import SimpleNamespace

import torch

from rfproteina.proteina import Proteina
from rfproteina.utils.torch_compile import (
    compiled_with_cudagraphs,
    maybe_mark_cudagraph_step_begin,
    torch_compile_kwargs,
)


def test_torch_compile_kwargs_passes_mode_only_when_set() -> None:
    assert "mode" not in torch_compile_kwargs({}, default_fullgraph=True)
    assert torch_compile_kwargs({"torch_compile_mode": "reduce-overhead"})["mode"] == "reduce-overhead"


def test_compiled_with_cudagraphs_reads_the_compile_mode() -> None:
    for mode, expected in [("reduce-overhead", True), ("max-autotune", True), ("default", False), (None, False)]:
        module = SimpleNamespace(_compile_kwargs={"mode": mode} if mode else {})
        assert compiled_with_cudagraphs(module) is expected
    assert compiled_with_cudagraphs(torch.nn.Identity()) is False


def test_step_begin_ends_the_tree_execution_only_when_idle(monkeypatch) -> None:
    calls: list[str] = []
    manager = SimpleNamespace(
        current_node=object(), in_recording=False, in_warmup=False, try_end_curr_execution=lambda: calls.append("end")
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: calls.append("mark"))
    monkeypatch.setattr(
        "torch._inductor.cudagraph_trees.get_container", lambda device: SimpleNamespace(tree_manager=manager)
    )

    maybe_mark_cudagraph_step_begin()
    assert calls == ["mark"]
    maybe_mark_cudagraph_step_begin(end_execution=True)
    assert calls == ["mark", "mark", "end"]
    manager.in_recording = True
    maybe_mark_cudagraph_step_begin(end_execution=True)
    assert calls == ["mark", "mark", "end", "mark"]


def test_call_nn_resets_variable_recycles_and_protects_copies_from_pool_overwrite(monkeypatch) -> None:
    marks: list[bool] = []
    recycles: list[int] = []
    graph_output = {"x": torch.zeros(2), "nested": {"y": torch.ones(1)}, "n": 3}

    class Trunk:
        _compile_kwargs = {"mode": "reduce-overhead"}
        _shapes_reserved = True

        def __call__(self, batch: dict) -> dict:
            recycles.append(batch["n_recycle"])
            return graph_output

    monkeypatch.setattr(
        "rfproteina.proteina.maybe_mark_cudagraph_step_begin",
        lambda *, end_execution=False: marks.append(end_execution),
    )
    module = SimpleNamespace(nn=Trunk(), training=False)
    first = Proteina.call_nn(module, {}, n_recycle=4)
    second = Proteina.call_nn(module, {}, n_recycle=1)
    assert marks == [True, True]
    assert recycles == [4, 1]
    assert first["x"] is not graph_output["x"] and second["nested"]["y"] is not graph_output["nested"]["y"]

    graph_output["x"].fill_(9)
    graph_output["nested"]["y"].fill_(8)
    assert torch.equal(first["x"], torch.zeros(2))
    assert torch.equal(second["nested"]["y"], torch.ones(1))
    assert first["n"] == 3

    Trunk._compile_kwargs = {}
    assert Proteina.call_nn(SimpleNamespace(nn=Trunk(), training=False), {}, n_recycle=4) is graph_output


def test_call_nn_keeps_graph_outputs_untouched_in_training() -> None:
    graph_output = {"x": torch.zeros(2)}

    class Trunk:
        _compile_kwargs = {"mode": "reduce-overhead"}
        _shapes_reserved = True

        def __call__(self, batch: dict) -> dict:
            return graph_output

    assert Proteina.call_nn(SimpleNamespace(nn=Trunk(), training=True), {}, n_recycle=4) is graph_output
