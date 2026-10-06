"""AE inference precision and NumPy export contracts; no checkpoint or GPU needed."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


@pytest.mark.parametrize("entrypoint", ["inference", "latent_analysis"])
@pytest.mark.parametrize("precision", [None, "bf16-mixed", "32-true"])
def test_ae_entrypoint_uses_configured_precision(monkeypatch, tmp_path, entrypoint, precision):
    module = importlib.import_module(f"rfproteina.partial_autoencoder.{entrypoint}")
    cfg = OmegaConf.create({"ckpt_file": "ae/checkpoints/model.ckpt", "n_structs": 2, "bs": 1})
    if precision is not None:
        cfg.precision = precision
    monkeypatch.setattr(module, "load_dotenv", lambda: None)
    monkeypatch.setattr(module, "parse_args_and_cfg", lambda: (None, cfg, "inference_ae"))
    monkeypatch.setattr(module, "setup", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(module, "load_dataloader", lambda cfg: object())
    monkeypatch.setattr(module.AutoEncoder, "load_from_checkpoint", lambda *args, **kwargs: object())

    class TrainerReachedError(Exception):
        pass

    def trainer(**kwargs):
        assert kwargs["precision"] == (precision or "bf16-mixed")
        assert kwargs["limit_predict_batches"] == 2
        raise TrainerReachedError

    monkeypatch.setattr(module.L, "Trainer", trainer)
    with pytest.raises(TrainerReachedError):
        module.main()


def test_bf16_latents_export_to_numpy_without_changing_model_output():
    from rfproteina.partial_autoencoder.latent_analysis import extract_latents_from_predictions

    latents = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16, requires_grad=True)
    original = latents.detach().clone()
    inputs = {"residue_type": torch.tensor([[5, 6]]), "padding_mask": torch.tensor([[True, False]])}
    exported = extract_latents_from_predictions([(inputs, {"z_latent": latents})])
    np.testing.assert_array_equal(exported["latents"], [[1.0, 2.0]])
    np.testing.assert_array_equal(exported["residue_types"], [5])
    assert exported["latents"].dtype == np.float32
    assert latents.dtype == torch.bfloat16 and latents.requires_grad
    torch.testing.assert_close(latents.detach(), original)
    assert latents.grad is None


def test_ae_inference_config_defaults_to_bf16_mixed():
    config = Path(__file__).resolve().parents[2] / "configs" / "inference_ae.yaml"
    assert OmegaConf.load(config).precision == "bf16-mixed"
