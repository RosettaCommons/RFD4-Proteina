import loralib as lora
from omegaconf import DictConfig
from torch import nn

from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def maybe_apply_lora(model: nn.Module, cfg_exp: DictConfig) -> bool:
    """Insert LoRA layers into ``model`` in place if ``cfg_exp.lora.r`` asks for them.

    Idempotent (skips if LoRA layers already present). Returns True if the model has
    LoRA layers on return, False if LoRA is off.
    """
    if not (cfg_exp.get("lora") and cfg_exp.lora.get("r")):
        return False

    if any(isinstance(m, lora.Linear | lora.Embedding) for m in model.modules()):
        logger.info("LoRA layers already present; skipping replacement.")
        return True

    lora_r = cfg_exp.lora.r
    lora_alpha = cfg_exp.lora.lora_alpha
    lora_dropout = cfg_exp.lora.lora_dropout
    logger.info(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    replace_lora_layers(
        model, lora_r, lora_alpha, lora_dropout, exclude_keys=tuple(cfg_exp.lora.get("exclude_keys", []))
    )
    lora.mark_only_lora_as_trainable(model, bias=cfg_exp.lora.train_bias)
    logger.info("LoRA layers successfully applied")
    return True


def replace_lora_layers(
    module: nn.Module,
    r: int,
    lora_alpha: float,
    lora_dropout: float,
    exclude_keys: tuple = (),
) -> None:
    """
    Recursively replace all nn.Linear and nn.Embedding layers in the module with
    the lora.Linear and lora.Embedding layers.

    Args:
        module (nn.Module): The module containing nn.Linear layers to be replaced.
        r (int): The rank for the pair of low-rank adaptation matrices.
        lora_alpha (float): Used for calculating lora scaling.
        lora_dropout (float): Dropout rate for the input before LoRA layers.
        exclude_keys (tuple): Keys to exclude from LoRA replacement (e.g., ("pair_update", "some_layer"))
    """
    for name, child in module.named_children():
        # Skip excluded layers
        if name == "pair_update" or any(exclude_key in name for exclude_key in exclude_keys):
            logger.info(f"Skipping LoRA replacement for '{name}' (excluded)")
            continue

        # Check if the child module is an instance of nn.Linear
        if isinstance(child, nn.Linear):
            # Skip frozen layers
            if not child.weight.requires_grad:
                logger.debug(f"Skipping frozen Linear layer '{name}'")
                continue

            # Replace the nn.Linear layer with a new one
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            # Turn off merge weights, as turning it on behaves strangely in the training mode of lightning trainer
            new_layer = lora.Linear(
                in_features,
                out_features,
                r,
                lora_alpha,
                lora_dropout,
                merge_weights=False,
                bias=bias,
            )
            setattr(module, name, new_layer)
        elif isinstance(child, nn.Embedding):
            # Skip frozen layers
            if not child.weight.requires_grad:
                logger.debug(f"Skipping frozen Embedding layer '{name}'")
                continue

            # Replace nn.Embedding layer
            num_embeddings = child.num_embeddings
            embedding_dim = child.embedding_dim
            padding_idx = child.padding_idx
            max_norm = child.max_norm
            norm_type = child.norm_type
            scale_grad_by_freq = child.scale_grad_by_freq
            sparse = child.sparse
            new_layer = lora.Embedding(
                num_embeddings,
                embedding_dim,
                r,
                lora_alpha,
                merge_weights=False,
                padding_idx=padding_idx,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                sparse=sparse,
            )
            setattr(module, name, new_layer)
        else:
            # Recursively replace layers in submodules
            replace_lora_layers(child, r, lora_alpha, lora_dropout, exclude_keys)
