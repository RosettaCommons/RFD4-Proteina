"""Explicit graph policy overrides the installed runtime's factory defaults."""

from types import SimpleNamespace

import pytest

from rfproteina.nn.bioir_full import FullBioIRAdapter


@pytest.mark.parametrize("default_enabled", [False, True])
def test_explicit_graph_opt_out(default_enabled):
    class Model:
        enabled = default_enabled

        def reset_graphs(self):
            self.enabled = False

    model = Model()
    adapter = SimpleNamespace(model=model, options={"cuda_graph": False})
    FullBioIRAdapter.configure_graphs(adapter)
    assert not model.enabled
