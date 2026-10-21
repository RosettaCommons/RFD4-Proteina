"""The quickstart preset composes the same config as the manual first-run incantation. CPU only."""

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.helpers import CONFIGS_DIR


@pytest.fixture(autouse=True)
def _data_path(monkeypatch):
    monkeypatch.setenv("DATA_PATH", "/unused")


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base="1.3"):
        return compose(config_name="inference", overrides=list(overrides))


def test_quickstart_matches_the_manual_incantation():
    preset = OmegaConf.to_container(_compose("inference_experiment=quickstart", "inputs=x"), resolve=True)
    manual = OmegaConf.to_container(_compose("inputs=x", "~metrics", "compile_lengths=null"), resolve=True)
    preset.pop("inference_experiment", None)
    manual.pop("inference_experiment", None)
    assert preset == manual


def test_quickstart_needs_no_checkpoint_arguments_and_keeps_the_pin():
    cfg = _compose("inference_experiment=quickstart", "inputs=x")
    base = _compose("inputs=x")
    assert "metrics" not in cfg
    assert cfg.compile_lengths is None
    assert cfg.hf_ckpt_path == "nvidia/RFD4-Proteina"
    assert cfg.hf_ckpt_revision == base.hf_ckpt_revision  # one source of truth: inference.yaml
    assert "hf_ckpt_revision" not in (CONFIGS_DIR / "inference_experiment" / "quickstart.yaml").read_text().replace(
        "`hf_ckpt_revision`", ""
    )


def test_quickstart_requires_inputs_and_accepts_overrides():
    with pytest.raises(Exception, match="inputs"):
        OmegaConf.to_container(_compose("inference_experiment=quickstart"), resolve=True)
    cfg = _compose("inference_experiment=quickstart", "inputs=x", "num_replicates=3", "hf_ckpt_path=null")
    assert cfg.generation.dataset.num_replicates == 3
    assert cfg.hf_ckpt_path is None
