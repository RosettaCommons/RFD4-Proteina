"""Optimizer construction for the Muon + Adam parameter split."""

import re
from collections.abc import Iterable

import torch
from omegaconf import DictConfig

from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.grouped_muon import GroupedMuon, supports_grouped_muon

logger = RankedLogger(__name__, rank_zero_only=True)

_LEGACY_PATTERNS_KEY = "muon_param_patterns"


def build_optimizers(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    cfg_opt: DictConfig,
) -> list[torch.optim.Optimizer]:
    """Build the optimizers for a model, splitting its parameters between Muon and Adam.

    Muon is opt-out: every trainable 2D parameter is routed to Muon unless its name matches one
    of ``cfg_opt.muon_exclude_param_patterns`` (the input embeddings and the output heads).
    ``torch.optim.Muon`` only accepts 2D parameters, so norms, biases and any other non-2D
    parameter always go to Adam regardless of the patterns.

    The patterns are the only control: exclude every parameter (e.g. ``['.*']``) and the model
    trains on Adam alone.

    ``muon_grouped`` defaults to true and batches ordinary FP32 CPU/CUDA matrices.
    Set it to false for native Muon. Unsupported parameter types use native Muon.

    Args:
      named_parameters: Model parameters, as from :py:meth:`torch.nn.Module.named_parameters`.
      cfg_opt: Optimizer config (``cfg_exp.opt`` / ``cfg_ae.opt``).

    Returns:
      ``[muon, adam]``, or ``[adam]`` if the patterns leave Muon with nothing to optimize. The
      manual optimization step in ``training_step`` steps whichever it gets.

    Raises:
      ValueError: If the removed ``muon_param_patterns`` allowlist key is still set.

    Examples:
      >>> optimizers = build_optimizers(model.named_parameters(), cfg_exp.opt)
    """
    if cfg_opt.get(_LEGACY_PATTERNS_KEY) is not None:
        raise ValueError(
            f"opt.{_LEGACY_PATTERNS_KEY} is no longer supported: Muon used to be opt-in via an "
            "allowlist of parameter names, and is now opt-out. List the parameters to keep on "
            "Adam in `opt.muon_exclude_param_patterns` instead."
        )

    lr = cfg_opt.lr
    exclude_patterns = [re.compile(p) for p in (cfg_opt.get("muon_exclude_param_patterns") or [])]

    muon_params, adam_params = [], []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        # Muon is 2D-only; excluded names (input embeddings, output heads) stay on Adam.
        if param.ndim != 2 or any(pat.search(name) for pat in exclude_patterns):
            adam_params.append(param)
        else:
            muon_params.append(param)

    logger.info(
        f"Muon: {len(muon_params)} params ({sum(p.numel() for p in muon_params) / 1e6:.1f}M), "
        f"Adam: {len(adam_params)} params ({sum(p.numel() for p in adam_params) / 1e6:.1f}M)"
    )

    adam = torch.optim.Adam(adam_params, lr=lr)
    if not muon_params:
        # Every parameter was excluded: train on Adam alone rather than handing Muon an empty
        # parameter list (which torch rejects).
        return [adam]
    muon_type = (
        GroupedMuon
        if cfg_opt.get("muon_grouped", True) and all(supports_grouped_muon(p) for p in muon_params)
        else torch.optim.Muon
    )
    return [
        muon_type(muon_params, lr=lr, adjust_lr_fn=cfg_opt.get("muon_adjust_lr_fn")),
        adam,
    ]
