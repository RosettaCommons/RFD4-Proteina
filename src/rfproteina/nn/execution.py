"""Select runtime blocks without replacing checkpoint or optimizer parameters."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from rfproteina.utils.torch_compile import setup_run_compile

logger = logging.getLogger(__name__)
#: Policy a component may select. ``bioir`` runs the whole workflow and replaces the native blocks outright.
Backend = Literal["eager", "torch", "bioir"]
#: Policy a native trunk's own blocks may run under.
BlockBackend = Literal["eager", "torch"]
BLOCK_ATTRIBUTES = ("transformer_layers", "pair_update_layers")
CUDA_GRAPH_MODES = frozenset({"reduce-overhead", "max-autotune"})


def native_blocks(trunk: nn.Module) -> tuple[Sequence, Sequence]:
    """Return registered attention and pair blocks, preserving optional pair layers."""
    return tuple(getattr(trunk, name, ()) for name in BLOCK_ATTRIBUTES)


def apply_blocks(
    s: torch.Tensor,
    z: torch.Tensor | None,
    c: torch.Tensor,
    mask: torch.Tensor,
    attention_blocks: Sequence,
    pair_blocks: Sequence,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run one trunk pass: each attention block, followed by its pair update when one exists."""
    for i, attention in enumerate(attention_blocks):
        s = attention(s, z, c, mask)  # [b, n, token_dim]
        if i < len(pair_blocks) and pair_blocks[i] is not None:
            z = pair_blocks[i](s, z, mask)  # [b, n, n, pair_dim]
    return s, z


@dataclass
class BlockExecution:
    """Off-tree runtime state built from the active checkpoint on its final device."""

    backend: BlockBackend = "eager"
    compile_kwargs: dict = field(default_factory=dict)
    compiled: dict = field(default_factory=dict)
    # Inference compiles without CUDA graphs: per-block graph trees overwrite outputs that later blocks still read.
    cuda_graphs: bool = True

    def run(
        self, trunk: nn.Module, s: torch.Tensor, z: torch.Tensor | None, c: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one trunk pass, building runtime blocks lazily on the final device from the loaded weights."""
        if self.backend == "eager":
            return apply_blocks(s, z, c, mask, *native_blocks(trunk))
        if self.cuda_graphs not in self.compiled:
            kwargs = dict(self.compile_kwargs)
            if not self.cuda_graphs and kwargs.get("mode") in CUDA_GRAPH_MODES:
                kwargs.pop("mode")
            self.compiled[self.cuda_graphs] = tuple(
                [torch.compile(block, **kwargs) if block is not None else None for block in layers]
                for layers in native_blocks(trunk)
            )
        return apply_blocks(s, z, c, mask, *self.compiled[self.cuda_graphs])


def run_trunk(
    trunk: nn.Module, s: torch.Tensor, z: torch.Tensor | None, c: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run one pass of a trunk's blocks with its execution policy; reject unaligned compiled backward shapes.

    Args:
        trunk: Flow or AE trunk owning ``transformer_layers`` and optional ``pair_update_layers``.
        s: Single representation ``[b, n, token_dim]``.
        z: Pair representation ``[b, n, n, pair_dim]``, or None for trunks without one.
        c: Conditioning ``[b, n, dim_cond]``.
        mask: Boolean valid-token mask ``[b, n]``.

    Returns:
        Updated single and pair representations in the input dtypes.
    """
    runtime = getattr(trunk, "_execution", None)
    if runtime is None:
        return apply_blocks(s, z, c, mask, *native_blocks(trunk))
    if runtime.backend == "torch" and mask.is_cuda and torch.is_grad_enabled() and mask.shape[-1] % 16:
        raise ValueError("Compiled backward requires 16-aligned token lengths; enable padding buckets")
    return runtime.run(trunk, s, z, c, mask)


def execution_components(model: nn.Module) -> dict[str, nn.Module]:
    """Return the flow and/or AE trunks present in a flow or standalone AE model."""
    ae = getattr(model, "autoencoder", model)
    components = {name: getattr(ae, name) for name in ("encoder", "decoder") if hasattr(ae, name)}
    if hasattr(model, "nn"):
        components["flow"] = model.nn
    return components


ALIGNED_LENGTHS = list(range(16, 16385, 16))


def inference_buckets(config: Mapping) -> list[int] | None:
    """Validate the shared inference/validation padding buckets."""
    buckets = config.get("inference", {}).get("length_buckets")
    if buckets is None:
        return None
    if not buckets or any(type(n) is not int or n <= 0 or n % 16 for n in buckets):
        raise ValueError("execution.inference.length_buckets must be positive multiples of 16, or null")
    return sorted(set(buckets))


def resolve_backends(config: Mapping, phase: str, *, guidance: bool = False) -> dict[str, Backend]:
    """Resolve a phase policy. ``bioir`` is the whole workflow, so every component selects it together."""
    if phase not in ("training", "inference"):
        raise ValueError(f"Unknown execution phase: {phase}")
    for name in ("training", "inference"):
        policy = config.get(name, {})
        unknown = set(policy) - {
            "flow",
            "encoder",
            "decoder",
            *(("length_buckets", "compile_lengths", "tri_mult") if name == "inference" else ("tri_mult",)),
        }
        if unknown:
            raise ValueError(f"Unknown execution.{name} options: {sorted(unknown)}")
        for component in ("flow", "encoder", "decoder"):
            backend = policy.get(component, "torch")
            backend = "torch" if backend == "torch_compile" else backend
            if backend not in ("eager", "torch", "bioir") or (name == "training" and backend == "bioir"):
                raise ValueError(f"Unsupported execution.{name}.{component}={backend!r}")
    for name in ("training", "inference"):
        if config.get(name, {}).get("tri_mult", "cuequivariance") not in ("torch", "cuequivariance"):
            raise ValueError(f"execution.{name}.tri_mult must be torch or cuequivariance")
    max_graphs = config.get("bioir", {}).get("max_graphs", 1)
    if max_graphs is not None and (type(max_graphs) is not int or max_graphs < 1):
        raise ValueError("execution.bioir.max_graphs must be null or a positive integer")
    obsolete = {"triangle_backend", "compile_shared_triangles", "flow_prefix_masks"} & set(config.get("bioir", {}))
    if obsolete:
        raise ValueError(f"BioIR selects KF, shared triangles and masking automatically; remove {sorted(obsolete)}")
    inference_buckets(config)
    result = {name: config.get(phase, {}).get(name, "torch") for name in ("flow", "encoder", "decoder")}
    result = {name: "torch" if backend == "torch_compile" else backend for name, backend in result.items()}
    selected = sorted(name for name, backend in result.items() if backend == "bioir")
    if selected and len(selected) != len(result):
        raise ValueError(
            f"execution.{phase}.*=bioir runs the whole workflow; select bioir for flow, encoder and decoder "
            f"together, not only {selected}"
        )
    if selected and guidance:
        raise ValueError(
            "Differentiable guidance is not qualified against the BioIR workflow; "
            "select inference_backend=torch_compile or eager, or remove the guidance terms"
        )
    return result


def inference_padding_buckets(config: Mapping, *, guidance: bool = False) -> list[int] | None:
    """Return explicit buckets, or the existing compile buckets for Torch alone.

    Explicit buckets apply to every backend. BioIR pads inside its trunk, so it
    should not inherit Torch's static-shape buckets.
    """
    buckets = inference_buckets(config)
    if buckets is not None:
        return buckets
    backends = set(resolve_backends(config, "inference", guidance=guidance).values())
    if "torch" not in backends:
        return None
    policy = config.get("inference", {})
    if "compile_lengths" in policy:
        return inference_buckets({"inference": {"length_buckets": policy["compile_lengths"]}})
    return ALIGNED_LENGTHS


def configure_triangle_backends(model: nn.Module, backend: str = "cuequivariance") -> None:
    """Apply the same triangle policy after loading checkpoint-selected architecture."""
    from rfproteina.nn.modules.openfold_modules.triangular_multiplicative_update import TriangleMultiplicativeUpdate

    for module in model.modules():
        if isinstance(module, TriangleMultiplicativeUpdate):
            module.set_backend(backend)


def configure_execution(model: nn.Module, config: Mapping, phase: str) -> None:
    """Select a policy after weight loading, preserving native modules and compile caches.

    ``bioir`` converts the model once at inference start and leaves the native blocks eager, so a
    phase boundary discards the converted copy and the native weights serve again. Registered
    parameters and checkpoint keys never change.
    """
    requested = resolve_backends(config, phase, guidance=getattr(model, "guidance_model", None) is not None)
    full = set(requested.values()) == {"bioir"}
    previous = getattr(model, "_full_bioir", None)
    if previous is not None:
        device = getattr(model, "device", next(previous.model.parameters()).device)
        previous.model.reset_graphs()
        model._full_bioir = None
        del previous  # Release converted GPU weights before restoring native ones.
        model.nn.to(device)
        model.autoencoder.to(device)
    model._full_bioir = None
    if full:
        if not getattr(model, "is_inference", False):
            raise ValueError(
                "execution.inference.*=bioir runs the whole workflow through the generation entry point; "
                "training-time validation must select torch_compile or eager"
            )
        from rfproteina.nn.bioir_full import require_bioir

        require_bioir()
    model._full_bioir_options = dict(config.get("bioir", {})) if full else None
    backends = dict.fromkeys(requested, "eager") if full else requested
    components = execution_components(model)
    # BioIR picks its own triangle kernels, so ``tri_mult`` has nothing to say
    # about a converted model: applying it would only reconfigure native blocks
    # that are about to be replaced and offloaded, and warn about a
    # cuEquivariance install the run never reaches.
    if not full:
        configure_triangle_backends(model, config.get(phase, {}).get("tri_mult", "cuequivariance"))
    for name, trunk in components.items():
        runtime = getattr(trunk, "_execution", None)
        if runtime is None:
            runtime = BlockExecution(
                compile_kwargs=getattr(trunk, "_compile_kwargs", {"fullgraph": True, "dynamic": False})
            )
            trunk._execution = runtime
        runtime.backend = backends[name]
        runtime.cuda_graphs = phase == "training"
        runtime.compiled.clear()
        logger.info("Execution %s.%s=%s", phase, name, requested[name])
    if any(backends[name] == "torch" for name in components):
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)


def setup_execution(config: Mapping, *, inference: bool = False, guidance: bool = False) -> None:
    """Initialize the optional compiler cache once per job, only when requested."""
    phases = ("inference",) if inference else ("training", "inference")
    if any("torch" in resolve_backends(config, phase, guidance=guidance).values() for phase in phases):
        setup_run_compile(inference=inference)
