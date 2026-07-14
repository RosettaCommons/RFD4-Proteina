"""Differentiable-metric gradient guidance for sampling.

:py:class:`GuidanceModel` is the differentiable twin of
:py:class:`~rfproteina.rewards.reward_model.RewardModel`. Each step it runs a
:py:class:`~rfproteina.metrics.metric.MetricManager` of differentiable metrics on the clean
prediction ``x_1_pred``, builds a weighted scalar objective, and backpropagates it to the
per-modality sampler state to produce an additive ``delta``.

Sign convention: guidance always *maximizes* the weighted objective. A schedule returning a
negative weight therefore minimizes that metric.
"""

from collections.abc import Callable
from typing import Any

import hydra
import torch
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from rfproteina.metrics.metric import MetricManager
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# A term maps a "metric_name.output_key" to a {modality: schedule} dict.
Term = dict[str, Callable[[float], float]]


def build_guidance_terms(terms_cfg: DictConfig) -> dict[str, Term]:
    """Instantiate per-term, per-modality schedule callables from a Hydra config.

    Args:
        terms_cfg: Mapping ``"metric_name.output_key" -> {modality: schedule_partial}``,
            where each ``schedule_partial`` is a ``_partial_`` Hydra target (see
            :py:mod:`rfproteina.guidance.schedules`).

    Returns:
        Mapping ``"metric_name.output_key" -> {modality: callable(t) -> weight}``.
    """
    return {
        key: {mode: hydra.utils.instantiate(sched_cfg) for mode, sched_cfg in modes_cfg.items()}
        for key, modes_cfg in terms_cfg.items()
    }


def _valid_mask(mode: str, extra_info: dict, ref: Tensor) -> Tensor | float:
    """Build a broadcastable validity mask for a modality's delta.

    ``bb_ca`` is masked by ``padding_mask``; ``local_latents`` additionally by
    ``can_be_encoded`` (only standard-AA tokens carry latents).
    """
    padding_mask = extra_info.get("padding_mask")
    if padding_mask is None:
        return 1.0
    mask = padding_mask
    if mode == "local_latents":
        can_be_encoded = extra_info.get("can_be_encoded")
        if can_be_encoded is not None:
            mask = padding_mask & can_be_encoded
    return mask[..., None].to(ref.dtype)


class GuidanceModel:
    """Aggregates differentiable metric outputs into a per-modality gradient ``delta``.

    Optional input namespaces (``decoded``, ``predicted_insertion_mapping``) are produced by
    *providers* and computed lazily: a provider runs only when some active term actually reads
    its namespace (detected via :py:attr:`Metric.required_kwargs`). New optional namespaces can
    be added by passing another provider — no change to the gating logic.

    Args:
        metric_manager: Manager of differentiable metrics, called inline each step with the
            ``state`` / ``extra_info`` namespace plus any provided optional namespaces (see
            :py:meth:`__call__`).
        terms: Mapping ``"metric_name.output_key" -> {modality: schedule}`` (see
            :py:func:`build_guidance_terms`).
        decode_fn: Optional ``(state, extra_info) -> decoded`` provider that decodes the
            grad-enabled state leaves to an atom37 namespace (e.g. ``model.decode_for_guidance``).
        insertion_fn: Optional ``(state, extra_info) -> predicted_insertion_mapping`` provider
            that resolves the insertion head's unindexed→generated token mapping
            (e.g. ``model.insertion_mapping_for_guidance``).

    Raises:
        ValueError: If a configured term references a non-differentiable metric.
    """

    def __init__(
        self,
        metric_manager: MetricManager,
        terms: dict[str, Term],
        decode_fn: Callable[[dict[str, Tensor], dict[str, Any]], Any] | None = None,
        insertion_fn: Callable[[dict[str, Tensor], dict[str, Any]], Any] | None = None,
    ):
        for key in terms:
            name = key.split(".")[0]
            metric = metric_manager.metrics[name]
            if not metric.differentiable:
                raise ValueError(
                    f"Guidance term '{key}' uses metric '{name}' "
                    f"({type(metric).__name__}) which is not differentiable."
                )
        self.metric_manager = metric_manager
        self.terms = terms
        # Lazy providers for optional input namespaces, keyed by the namespace they populate.
        self._providers: dict[str, Callable[[dict[str, Tensor], dict[str, Any]], Any]] = {}
        if decode_fn is not None:
            self._providers["decoded"] = decode_fn
        if insertion_fn is not None:
            self._providers["predicted_insertion_mapping"] = insertion_fn

    def _needs(self, active: set[str], namespace: str) -> bool:
        """Whether any active term reads from the given top-level input ``namespace``."""
        return any(
            req[0] == namespace
            for name in {key.split(".")[0] for key in active}
            for req in self.metric_manager.metrics[name].required_kwargs
        )

    def __call__(
        self,
        x_1_pred: dict[str, Tensor],
        extra_info: dict[str, Any],
    ) -> dict[str, Float[Tensor, "..."]]:
        """Compute the per-modality guidance delta for one sampling step.

        Args:
            x_1_pred: Clean prediction per modality (e.g. ``bb_ca``, ``local_latents``);
                the gradient is taken w.r.t. (detached copies of) these leaves.
            extra_info: Must contain ``t`` (per-modality time, tensor or float) and the
                masks ``padding_mask`` / ``can_be_encoded``.

        Returns:
            Mapping ``modality -> delta`` to add to the integrated state (empty if no term
            is active this step).
        """
        t = extra_info["t"]
        t_val = {m: (float(v.flatten()[0]) if torch.is_tensor(v) else float(v)) for m, v in t.items()}

        # Effective per-(term, modality) weights at this step.
        eff = {
            key: {mode: float(sched(t_val[mode])) for mode, sched in modes.items()} for key, modes in self.terms.items()
        }
        active = {key for key, w in eff.items() if any(v != 0.0 for v in w.values())}
        if not active:
            return {}

        # Atom arrays are routed through extra_info
        atom_array_batch = extra_info.get("atom_array_batch")

        with torch.inference_mode(False), torch.enable_grad():
            state = {m: x_1_pred[m].detach().clone().requires_grad_(True) for m in x_1_pred}

            # Lazily populate optional namespaces (decoded, predicted_insertion_mapping, ...)
            # only when an active term reads them.
            provided = {
                namespace: provider(state, extra_info)
                for namespace, provider in self._providers.items()
                if self._needs(active, namespace)
            }

            outputs = self.metric_manager(
                state=state,
                extra_info=extra_info,
                ground_truth_atom_array=atom_array_batch,
                **provided,
            )

            delta: dict[str, Tensor] = {}
            for key in active:
                if key not in outputs:
                    raise KeyError(f"Guidance term '{key}' not found in metric outputs {list(outputs)}.")
                modes = [m for m, w in eff[key].items() if w != 0.0]
                grads = torch.autograd.grad(
                    outputs[key], [state[m] for m in modes], retain_graph=True, allow_unused=True
                )
                for m, g in zip(modes, grads, strict=True):
                    if g is None:
                        continue
                    contrib = eff[key][m] * torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                    delta[m] = contrib if m not in delta else delta[m] + contrib

        # Mask outside the grad tape (grads are already detached).
        return {m: d * _valid_mask(m, extra_info, d) for m, d in delta.items()}
