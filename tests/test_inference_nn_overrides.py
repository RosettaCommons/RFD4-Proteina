"""Inference-time ``nn`` overrides merge into the checkpoint's experiment config before the model is built."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from rfproteina.generate import _inference_cfg_exp, load_ckpt_n_configure_inference
from rfproteina.nn.local_latents_transformer import LocalLatentsTransformer
from rfproteina.proteina import Proteina
from tests.helpers import CONFIGS_DIR


def _checkpoint(tmp_path, nn: dict) -> str:
    """A checkpoint stub whose hparams carry a struct experiment config, like a Lightning export."""
    cfg_exp = OmegaConf.create({"nn": nn, "datasets": {}})
    OmegaConf.set_struct(cfg_exp, True)
    path = tmp_path / "model-ema.ckpt"
    torch.save({"hyper_parameters": {"cfg_exp": cfg_exp}, "state_dict": {}}, path)
    return str(path)


def _inference_cfg(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base="1.3"):
        return compose(config_name="inference", overrides=["inputs=unused", *overrides])


def test_null_overrides_leave_the_checkpoint_unread(tmp_path):
    assert _inference_cfg_exp(str(tmp_path / "absent.ckpt"), {"torch_compile_mode": None}) is None


def test_overrides_merge_into_the_saved_nn_config(tmp_path):
    path = _checkpoint(tmp_path, {"use_torch_compile": True, "tri_mult_backend": "torch"})
    cfg_exp = _inference_cfg_exp(
        path, {"torch_compile_mode": "reduce-overhead", "tri_mult_backend": "cuequivariance", "attention_type": None}
    )
    assert isinstance(cfg_exp, DictConfig)
    assert cfg_exp.nn.torch_compile_mode == "reduce-overhead"  # added: the checkpoint saved no mode
    assert cfg_exp.nn.tri_mult_backend == "cuequivariance"
    assert cfg_exp.nn.use_torch_compile is True
    assert "attention_type" not in cfg_exp.nn


def test_unknown_override_is_rejected(tmp_path):
    path = _checkpoint(tmp_path, {"pair_repr_dim": 384})
    with pytest.raises(ValueError, match="nn.pair_repr_dim"):
        _inference_cfg_exp(path, {"pair_repr_dim": 256})


def test_inference_config_exposes_exactly_the_overridable_keys():
    cfg = _inference_cfg("nn.torch_compile_mode=reduce-overhead")
    assert set(cfg.nn.keys()) == set(LocalLatentsTransformer.RUNTIME_OPTIONS)
    assert cfg.nn.torch_compile_mode == "reduce-overhead"
    assert all(cfg.nn[key] is None for key in LocalLatentsTransformer.RUNTIME_OPTIONS if key != "torch_compile_mode")


@pytest.mark.parametrize(
    "overrides", [(), ("nn.torch_compile_mode=reduce-overhead", "nn.tri_mult_backend=cuequivariance")]
)
def test_loader_passes_the_merged_config_to_lightning(tmp_path, overrides):
    _checkpoint(tmp_path, {"use_torch_compile": True, "tri_mult_backend": "torch"})
    cfg = _inference_cfg(
        f"ckpt_path={tmp_path}",
        "hf_ckpt_path=null",
        "ckpt_name=model-ema.ckpt",
        "autoencoder_ckpt_path=unused",
        *overrides,
    )
    model = MagicMock(spec=Proteina)
    model.cfg_exp = OmegaConf.create({"datasets": {}})
    with patch.object(Proteina, "load_from_checkpoint", return_value=model) as load:
        assert load_ckpt_n_configure_inference(cfg) is model
    kwargs = load.call_args.kwargs
    if not overrides:
        assert "cfg_exp" not in kwargs  # the checkpoint's own hparams build the model
    else:
        assert kwargs["cfg_exp"].nn.torch_compile_mode == "reduce-overhead"
        assert kwargs["cfg_exp"].nn.tri_mult_backend == "cuequivariance"
        assert kwargs["cfg_exp"].nn.use_torch_compile is True


def test_hub_bundle_receives_the_overrides():
    cfg = _inference_cfg(
        "hf_ckpt_path=some/bundle", "nn.torch_compile_mode=reduce-overhead", "autoencoder_ckpt_path=unused"
    )
    model = MagicMock(spec=Proteina)
    model.cfg_exp = OmegaConf.create({"datasets": {}})
    with patch.object(Proteina, "from_pretrained", return_value=model) as load:
        assert load_ckpt_n_configure_inference(cfg) is model
    assert load.call_args.kwargs["nn_overrides"] == {"torch_compile_mode": "reduce-overhead"}


def test_from_pretrained_merges_overrides_into_the_bundle_nn_config(monkeypatch):
    built = {}

    class Stub(Proteina):
        def __init__(self, cfg_exp, **kwargs):
            built["nn"] = OmegaConf.to_container(cfg_exp.nn)
            torch.nn.Module.__init__(self)

        def load_state_dict(self, *args, **kwargs):
            return None

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    payload = {"config": {"nn": {"tri_mult_backend": "torch", "pair_repr_dim": 256}}}
    monkeypatch.setattr("rfproteina.utils.hub.load_pretrained_bundle", lambda *a, **k: (payload, {}))
    Stub.from_pretrained("bundle", nn_overrides={"tri_mult_backend": "cuequivariance"})
    assert built["nn"] == {"tri_mult_backend": "cuequivariance", "pair_repr_dim": 256}


def test_unknown_compile_mode_fails_before_anything_loads():
    with pytest.raises(ValueError, match="torch_compile_mode"):
        _inference_cfg_exp("/nonexistent.ckpt", {"torch_compile_mode": "reduce_overhead"})


def test_overrides_cannot_be_applied_to_a_supplied_model():
    cfg = _inference_cfg("nn.tri_mult_backend=torch")
    with pytest.raises(ValueError, match="supplied model"):
        load_ckpt_n_configure_inference(cfg, model=MagicMock(spec=Proteina))
