"""``delete_failing_pre_oracle`` of ``rfd4-generate`` reaches the pre-oracle filter, not only the checkpoint's own config."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rfproteina.generate import load_ckpt_n_configure_inference
from rfproteina.proteina import Proteina
from tests.helpers import CONFIGS_DIR


class _StopError(Exception):
    """Raised by the stub filter so the epoch-end hook stops right after reading the flag."""


def _epoch_end_flag(monkeypatch, cfg_exp: dict, **inference) -> bool:
    """Run ``on_validation_epoch_end`` up to the pre-oracle filter and return its ``delete_failing`` argument."""
    seen = {}

    def apply_pre_oracle_filters(outputs, metric_manager, delete_failing=True):
        seen["delete_failing"] = delete_failing
        raise _StopError

    monkeypatch.setattr("rfproteina.proteina.apply_pre_oracle_filters", apply_pre_oracle_filters)
    model = Proteina.__new__(Proteina)
    model.cfg_exp = OmegaConf.create(cfg_exp)
    model.configure_inference(OmegaConf.create({"args": {}}), **inference)
    model._metrics_pool = SimpleNamespace(drain_generated_structure=lambda: {})
    model.pre_oracle_filter_metric_manager = object()
    model._oracle_mms_setup_this_epoch = True
    model.validation_output_generation = []
    with pytest.raises(_StopError):
        model.on_validation_epoch_end()
    return seen["delete_failing"]


@pytest.mark.parametrize("flag", [True, False])
def test_inference_flag_wins_over_the_checkpoint_config(monkeypatch, flag):
    assert _epoch_end_flag(monkeypatch, {"delete_failing_pre_oracle": not flag}, delete_failing_pre_oracle=flag) is flag


@pytest.mark.parametrize(("cfg_exp", "expected"), [({}, True), ({"delete_failing_pre_oracle": False}, False)])
def test_unset_flag_defers_to_the_checkpoint_config(monkeypatch, cfg_exp, expected):
    """Training-time validation never sets the flag, so it keeps reading the experiment config."""
    assert _epoch_end_flag(monkeypatch, cfg_exp) is expected


def test_generate_forwards_the_flag():
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base="1.3"):
        cfg = compose(
            config_name="inference",
            overrides=["inputs=unused", "delete_failing_pre_oracle=false", "autoencoder_ckpt_path=unused"],
        )
    model = MagicMock(spec=Proteina)
    model.cfg_exp = OmegaConf.create({"datasets": {}})

    load_ckpt_n_configure_inference(cfg, model=model)

    assert model.configure_inference.call_args.kwargs["delete_failing_pre_oracle"] is False
