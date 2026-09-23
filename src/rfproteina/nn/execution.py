"""Select runtime blocks without replacing checkpoint or optimizer parameters."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torch import nn

from rfproteina.nn.bioir import convert_blocks
from rfproteina.utils.torch_compile import setup_run_compile

logger = logging.getLogger(__name__)
Backend = Literal["eager", "torch", "bioir"]
BLOCK_ATTRIBUTES = ("transformer_layers", "pair_update_layers")


def native_blocks(trunk: nn.Module) -> tuple[Sequence, Sequence]:
    """Return registered attention and pair blocks, preserving optional pair layers."""
    return tuple(getattr(trunk, name, ()) for name in BLOCK_ATTRIBUTES)


@dataclass
class BlockExecution:
    """Off-tree runtime state; BioIR copies live only for one validation interval."""

    backend: Backend = "eager"
    compile_kwargs: dict = field(default_factory=dict)
    bioir_options: dict = field(default_factory=dict)
    compiled: tuple | None = None
    converted: tuple | None = None

    def blocks(self, trunk: nn.Module) -> tuple[Sequence, Sequence]:
        """Build blocks lazily on the final device, using the currently loaded weights."""
        if self.backend == "eager":
            return native_blocks(trunk)
        if self.backend == "torch":
            if self.compiled is None:
                self.compiled = tuple(
                    [torch.compile(block, **self.compile_kwargs) if block is not None else None for block in layers]
                    for layers in native_blocks(trunk)
                )
            return self.compiled
        if torch.is_grad_enabled():
            raise RuntimeError("BioIR blocks require disabled gradients; select execution.*.<component>=torch")
        if self.converted is None:
            self.converted = convert_blocks(trunk, **self.bioir_options)
        return self.converted


def execution_blocks(trunk: nn.Module, mask: torch.Tensor) -> tuple[Sequence, Sequence]:
    """Return active blocks; reject unaligned compiled backward shapes before launch."""
    runtime = getattr(trunk, "_execution", None)
    if runtime is None:
        return native_blocks(trunk)
    if runtime.backend == "torch" and mask.is_cuda and torch.is_grad_enabled() and mask.shape[-1] % 16:
        raise ValueError("Compiled backward requires 16-aligned token lengths; enable padding buckets")
    return runtime.blocks(trunk)


def execution_components(model: nn.Module) -> dict[str, nn.Module]:
    """Return the flow and/or AE trunks present in a flow or standalone AE model."""
    ae = getattr(model, "autoencoder", model)
    components = {name: getattr(ae, name) for name in ("encoder", "decoder") if hasattr(ae, name)}
    if hasattr(model, "nn"):
        components["flow"] = model.nn
    return components


def inference_buckets(config: Mapping) -> list[int] | None:
    """Validate the shared inference/validation padding buckets."""
    buckets = config.get("inference", {}).get("length_buckets")
    if buckets is None:
        return None
    if not buckets or any(type(n) is not int or n <= 0 or n % 16 for n in buckets):
        raise ValueError("execution.inference.length_buckets must be positive multiples of 16, or null")
    return sorted(set(buckets))


def resolve_backends(config: Mapping, phase: str, *, guidance: bool = False) -> dict[str, Backend]:
    """Resolve a phase policy, including the differentiable decoder override."""
    if phase not in ("training", "inference"):
        raise ValueError(f"Unknown execution phase: {phase}")
    for name in ("training", "inference"):
        policy = config.get(name, {})
        unknown = set(policy) - {"flow", "encoder", "decoder", *(("length_buckets",) if name == "inference" else ())}
        if unknown:
            raise ValueError(f"Unknown execution.{name} options: {sorted(unknown)}")
        for component in ("flow", "encoder", "decoder"):
            backend = policy.get(component, "eager")
            if backend not in ("eager", "torch", "bioir") or (name == "training" and backend == "bioir"):
                raise ValueError(f"Unsupported execution.{name}.{component}={backend!r}")
    inference_buckets(config)
    result = {name: config.get(phase, {}).get(name, "eager") for name in ("flow", "encoder", "decoder")}
    if guidance and result["decoder"] == "bioir":
        result["decoder"] = "torch"
        logger.info("Differentiable guidance selects execution.inference.decoder=torch instead of BioIR")
    return result


def configure_execution(model: nn.Module, config: Mapping, phase: str) -> None:
    """Select a policy after weight loading, preserving native modules and compile caches.

    BioIR copies are discarded on each phase boundary and rebuilt lazily from the
    active weights. Registered parameters and checkpoint keys never change.
    """
    backends = resolve_backends(config, phase, guidance=getattr(model, "guidance_model", None) is not None)
    components = execution_components(model)
    for name, trunk in components.items():
        runtime = getattr(trunk, "_execution", None)
        if runtime is None:
            runtime = BlockExecution(
                compile_kwargs=getattr(trunk, "_compile_kwargs", {"fullgraph": True, "dynamic": False})
            )
            trunk._execution = runtime
        runtime.backend = backends[name]
        runtime.bioir_options = dict(config.get("bioir", {}))
        runtime.converted = None
        trunk.bioir_enabled = runtime.backend == "bioir"
        logger.info("Execution %s.%s=%s", phase, name, runtime.backend)
    if any(backends[name] == "torch" for name in components):
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)


def setup_execution(config: Mapping, *, inference: bool = False, guidance: bool = False) -> None:
    """Initialize the optional compiler cache once per job, only when requested."""
    phases = ("inference",) if inference else ("training", "inference")
    if any("torch" in resolve_backends(config, phase, guidance=guidance).values() for phase in phases):
        setup_run_compile(inference=inference)


class ExecutionCallback(Callback):
    """Apply inference policies during validation and restore training on exit/errors."""

    def __init__(self, config: Mapping) -> None:
        self.config = config

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        configure_execution(pl_module, self.config, "training")

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        configure_execution(pl_module, self.config, "inference")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        configure_execution(pl_module, self.config, "training")

    def on_exception(self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException) -> None:
        configure_execution(pl_module, self.config, "training")


def enable_bioir(
    model: nn.Module,
    *,
    components: Sequence[str] = ("flow", "encoder", "decoder"),
    dtype: str = "bfloat16",
    attention_backend: Literal["SDPA", "CuTeDSL"] = "SDPA",
) -> None:
    """Select BioIR for direct inference callers; guidance selects a compiled decoder."""
    if isinstance(components, str) or not components or set(components) - {"flow", "encoder", "decoder"}:
        raise ValueError("BioIR components must be a nonempty sequence containing flow, encoder and/or decoder")
    configure_execution(
        model,
        {
            "inference": {name: "bioir" for name in components},
            "bioir": {"dtype": dtype, "attention_backend": attention_backend},
        },
        "inference",
    )
