"""PEFT adapters on the denoiser, with the outer Lightning model kept intact."""

from pathlib import Path

from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict
from torch import nn


def _inject(model: nn.Module, config: LoraConfig) -> None:
    """Freeze the complete model and inject one adapter into its denoiser."""
    denoiser = model.nn
    if getattr(denoiser, "attention_type", "torch") != "torch":
        raise ValueError("PEFT LoRA requires nn.attention_type=torch; fused attention bypasses adapter forwards.")
    if config.bias != "none" or config.modules_to_save:
        raise ValueError("PEFT LoRA supports adapter-only training: use bias=none and no modules_to_save.")
    if getattr(denoiser, "peft_config", None):
        raise ValueError("The denoiser already has a PEFT adapter.")
    model.requires_grad_(False)
    training = denoiser.training
    inject_adapter_in_model(config, denoiser)
    denoiser.train(training)


def add_lora_adapter(model: nn.Module, config: DictConfig) -> None:
    """Add one configured PEFT adapter to the denoiser after loading base weights."""
    if config.get("backend", "peft") != "peft" or not config.get("target_modules"):
        raise ValueError(
            "LoRA requires a PEFT configuration with explicit lora.target_modules. "
            "loralib checkpoints are incompatible: merge their adapters using the original "
            "implementation, then start a new PEFT fine-tune from the merged base checkpoint."
        )
    if config.get("exclude_keys"):
        raise ValueError("LoRA uses explicit lora.target_modules; remove lora.exclude_keys.")
    if not hasattr(model, "nn"):
        raise ValueError("LoRA adapts the RFD4-Proteína denoiser; autoencoder LoRA is not supported.")
    _inject(
        model,
        LoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.train_bias,
            target_modules=list(config.target_modules),
        ),
    )


def save_lora_adapter(model: nn.Module, directory: str | Path) -> None:
    """Save current denoiser adapter weights and configuration in standard PEFT format.

    Call on an unwrapped model on rank zero. The export uses the currently loaded
    weights; load an EMA checkpoint first to export EMA adapters. Base/AE weights
    and Lightning optimizer/trainer state are not included.
    """
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = model.nn.peft_config["default"]
    state = get_peft_model_state_dict(model.nn)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in state.items()},
        directory / "adapter_model.safetensors",
    )
    config.save_pretrained(directory)


def load_lora_adapter(model: nn.Module, directory: str | Path) -> None:
    """Load a local PEFT adapter onto its matching, already-loaded base/AE pair for inference.

    Requires the same base weights and denoiser configuration used for training.
    Full training resume uses Lightning checkpoints instead of this adapter export.
    """
    from safetensors.torch import load_file

    directory = Path(directory)
    config = LoraConfig.from_pretrained(directory)
    config.inference_mode = True
    state = load_file(directory / "adapter_model.safetensors")
    _inject(model, config)
    expected = get_peft_model_state_dict(model.nn)
    if state.keys() != expected.keys():
        raise ValueError(
            f"Adapter keys do not match the denoiser: missing={sorted(expected.keys() - state.keys())}, "
            f"unexpected={sorted(state.keys() - expected.keys())}"
        )
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"Adapter shape mismatch for {key}: {value.shape} != {expected[key].shape}")
    set_peft_model_state_dict(model.nn, state)
    model.requires_grad_(False)
