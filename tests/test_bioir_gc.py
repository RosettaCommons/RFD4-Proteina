"""BioIR requests defer cyclic finalizers until CUDA graph capture has finished."""

import gc
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rfproteina.nn import bioir_full


@pytest.mark.parametrize("initially_enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_bioir_request_restores_gc_policy(initially_enabled: bool, fail: bool) -> None:
    original = gc.isenabled()
    finalized = []

    class Cycle:
        def __init__(self):
            self.cycle = self

        def __del__(self):
            finalized.append("collected")

    def predict(batch: dict) -> list:
        assert not gc.isenabled(), "Cyclic finalizers must not run inside CUDA graph capture"
        Cycle()
        assert not finalized
        if fail:
            raise RuntimeError("capture failed")
        return ["prediction"]

    adapter = SimpleNamespace(model=torch.nn.Linear(1, 1), predict_step=predict)
    try:
        gc.enable() if initially_enabled else gc.disable()
        with patch.object(bioir_full, "initialize_full_bioir", return_value=adapter):
            if fail:
                with pytest.raises(RuntimeError, match="capture failed"):
                    bioir_full.predict_full_bioir(None, {"padding_mask": torch.ones(1, 1)})
            else:
                assert bioir_full.predict_full_bioir(None, {"padding_mask": torch.ones(1, 1)}) == ["prediction"]
        assert gc.isenabled() == initially_enabled
        gc.collect()
        assert finalized == ["collected"]
    finally:
        gc.enable() if original else gc.disable()
