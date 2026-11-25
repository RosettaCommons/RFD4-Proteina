"""Runtime defaults must not replace a checkpoint's conditioning policy."""

import sys
from types import SimpleNamespace

import pytest
import torch

from rfproteina.nn import bioir_full


def _load(monkeypatch, factory, align=False, translate=False):
    monkeypatch.setitem(
        sys.modules,
        "bionemo_ir.models.rfproteina",
        SimpleNamespace(RFProteina=SimpleNamespace(from_reference_state_dicts=factory)),
    )
    monkeypatch.setattr(bioir_full, "require_bioir", lambda native: None)
    monkeypatch.setattr(bioir_full, "_copy_loaded_states", lambda native: {"flow": {}, "autoencoder": {}})
    native = SimpleNamespace(
        cfg_exp=SimpleNamespace(nn={"noise_align_chirality": align, "noise_align_chirality_translate": translate})
    )
    return bioir_full.load_full_model(native, sampling_config=None, device=torch.device("cpu"), options={})


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("translate", [False, True])
def test_checkpoint_alignment_overrides_runtime_defaults(monkeypatch, align, translate):
    def factory(flow, autoencoder, *, noise_align_chirality=True, noise_align_chirality_translate=True, **kwargs):
        return noise_align_chirality, noise_align_chirality_translate

    assert _load(monkeypatch, factory, align, translate) == (
        align,
        translate,
    )


def test_unaligned_checkpoint_accepts_legacy_factory(monkeypatch):
    def factory(flow, autoencoder, *, sampling_config, device, attention_backend):
        return "loaded"

    assert _load(monkeypatch, factory) == "loaded"
