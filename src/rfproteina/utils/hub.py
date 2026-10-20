"""Portable native model bundles and revision-aware Hugging Face downloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from rfproteina.utils.hydra import training_path_length_search_cutoffs

if TYPE_CHECKING:
    from rfproteina.proteina import Proteina

_MODEL_FIELDS = ("nn", "product_flowmatcher", "loss", "training", "opt")


def save_pretrained(model: Proteina, directory: str | Path) -> None:
    """Export loaded model/AE weights and architecture, without trainer state."""
    if OmegaConf.select(model.cfg_exp, "lora.r") or any("lora_" in key for key in model.state_dict()):
        raise ValueError("Model bundles require base weights. Export active adapters with the adapter API instead.")
    cfg = {
        key: OmegaConf.to_container(model.cfg_exp[key], resolve=True) for key in _MODEL_FIELDS if key in model.cfg_exp
    }
    cfg["nn"]["use_torch_compile"] = False
    cfg["autoencoder_config"] = None
    if model.autoencoder is not None:
        ae = OmegaConf.to_container(model.autoencoder.cfg_ae.nn_ae, resolve=True)
        ae["use_torch_compile"] = False
        cfg["autoencoder_config"] = {"nn_ae": ae}
    cutoffs = training_path_length_search_cutoffs(model.cfg_exp)
    if len(cutoffs) > 1:
        raise ValueError("Training datasets disagree on path_length_search_cutoff; select one before export.")
    cfg["path_length_search_cutoffs"] = list(cutoffs)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / "model.safetensors"
    save_file(
        {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()},
        weights_path,
        metadata={"format": "pt"},
    )
    payload = {
        "model_type": "rfd4_proteina",
        "format_version": 1,
        "config": cfg,
    }
    (directory / "config.json").write_text(json.dumps(payload, indent=2) + "\n")


def load_pretrained_bundle(
    model_id: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Read bundle configuration and CPU weights without constructing a model."""
    directory = Path(model_id).expanduser()
    if not directory.is_dir():
        if isinstance(model_id, Path) or str(model_id).startswith(("/", ".", "~")):
            raise FileNotFoundError(f"Model directory does not exist: {model_id}")
        directory = Path(
            snapshot_download(
                repo_id=str(model_id),
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                force_download=force_download,
                allow_patterns=["config.json", "model.safetensors"],
            )
        )
    if not all((directory / name).is_file() for name in ("config.json", "model.safetensors")):
        raise FileNotFoundError(
            f"{model_id} is not an RFD4 model bundle. Export config.json and model.safetensors "
            "with save_pretrained before loading it; native .ckpt files use load_from_checkpoint."
        )
    payload = json.loads((directory / "config.json").read_text())
    if payload.get("model_type") != "rfd4_proteina" or payload.get("format_version") != 1:
        raise ValueError("Unsupported RFD4 model bundle type or format version.")
    weights_path = directory / "model.safetensors"
    cfg = payload["config"]
    allowed = {*_MODEL_FIELDS, "autoencoder_config", "path_length_search_cutoffs"}
    if set(cfg) - allowed or "${" in json.dumps(cfg) or '"_target_"' in json.dumps(cfg):
        raise ValueError("Model bundles must contain resolved model configuration without runtime targets.")
    return payload, load_file(weights_path)
