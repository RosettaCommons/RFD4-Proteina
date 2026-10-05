"""Select runtime blocks without replacing checkpoint or optimizer parameters."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from rfproteina.utils.torch_compile import setup_run_compile

logger = logging.getLogger(__name__)
Backend = Literal["eager", "torch", "bioir"]
BLOCK_ATTRIBUTES = ("transformer_layers", "pair_update_layers")
CUDA_GRAPH_MODES = frozenset({"reduce-overhead", "max-autotune"})
BIOIR_TRIANGLE_BACKENDS = frozenset({"shared", "kf"})


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

    backend: Backend = "eager"
    compile_scope: Literal["blocks", "whole_pass"] = "blocks"
    compile_kwargs: dict = field(default_factory=dict)
    bioir_options: dict = field(default_factory=dict)
    compiled: dict = field(default_factory=dict)
    converted: nn.Module | None = None
    weight_signature: tuple = ()
    # Inference compiles without CUDA graphs: per-block graph trees overwrite outputs that later blocks still read.
    cuda_graphs: bool = True

    def run(
        self, trunk: nn.Module, s: torch.Tensor, z: torch.Tensor | None, c: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one trunk pass, building runtime blocks lazily on the final device from the loaded weights."""
        if self.backend == "eager":
            return apply_blocks(s, z, c, mask, *native_blocks(trunk))
        if self.backend == "torch":
            if self.compile_scope == "whole_pass":
                if torch.is_grad_enabled():
                    raise RuntimeError("Whole-pass CUDA graphs require disabled gradients; select compile_scope=blocks")
                if "whole_pass" not in self.compiled:
                    blocks = native_blocks(trunk)

                    def whole_pass(
                        s: torch.Tensor, z: torch.Tensor | None, c: torch.Tensor, mask: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor | None]:
                        return apply_blocks(s, z, c, mask, *blocks)

                    self.compiled["whole_pass"] = torch.compile(
                        whole_pass, fullgraph=True, dynamic=False, mode="reduce-overhead"
                    )
                result = self.compiled["whole_pass"](s, z, c, mask)
                # Own output storage before the next recycle/denoising step replays a graph.
                return tuple(value.clone() if value is not None else None for value in result)
            if self.cuda_graphs not in self.compiled:
                kwargs = dict(self.compile_kwargs)
                if not self.cuda_graphs and kwargs.get("mode") in CUDA_GRAPH_MODES:
                    kwargs.pop("mode")
                self.compiled[self.cuda_graphs] = tuple(
                    [torch.compile(block, **kwargs) if block is not None else None for block in layers]
                    for layers in native_blocks(trunk)
                )
            return apply_blocks(s, z, c, mask, *self.compiled[self.cuda_graphs])
        if torch.is_grad_enabled():
            raise RuntimeError("BioIR blocks require disabled gradients; select execution.*.<component>=torch")
        signature = tuple((p.data_ptr(), p._version, p.dtype, p.device) for p in trunk.parameters())
        if self.converted is None or signature != self.weight_signature:
            from rfproteina.nn.bioir import convert_trunk

            self.converted = convert_trunk(*native_blocks(trunk), apply_blocks, **self.bioir_options)
            self.weight_signature = signature
        return self.converted(s, z, c, mask)


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
# BioIR runs short inputs at their own length and longer ones at the next multiple of 16, where its kernels are faster.
BIOIR_UNPADDED_MAX_LENGTH = 250
BIOIR_LENGTHS = list(range(1, BIOIR_UNPADDED_MAX_LENGTH + 1)) + [
    n for n in ALIGNED_LENGTHS if n > BIOIR_UNPADDED_MAX_LENGTH
]


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
        unknown = set(policy) - {
            "flow",
            "encoder",
            "decoder",
            *(("length_buckets", "tri_mult", "compile_scope") if name == "inference" else ("tri_mult",)),
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
    if config.get("inference", {}).get("compile_scope", "blocks") not in ("blocks", "whole_pass"):
        raise ValueError("execution.inference.compile_scope must be blocks or whole_pass")
    triangle_backend = config.get("bioir", {}).get("triangle_backend", "shared")
    if triangle_backend not in BIOIR_TRIANGLE_BACKENDS:
        raise ValueError("execution.bioir.triangle_backend must be shared or kf")
    if triangle_backend == "kf" and not config.get("bioir", {}).get("flow_prefix_masks", False):
        raise ValueError("Experimental BioIR KF requires execution.bioir.flow_prefix_masks=true")
    inference_buckets(config)
    result = {name: config.get(phase, {}).get(name, "torch") for name in ("flow", "encoder", "decoder")}
    result = {name: "torch" if backend == "torch_compile" else backend for name, backend in result.items()}
    if guidance and result["decoder"] == "bioir":
        result["decoder"] = "torch"
        logger.info("Differentiable guidance selects execution.inference.decoder=torch instead of BioIR")
    return result


def inference_padding_buckets(config: Mapping, *, guidance: bool = False) -> list[int] | None:
    """Return generation padding sizes: configured buckets, else the sizes the selected backends need.

    Configured buckets apply to every backend. Without them, torch.compile components pad to the
    next multiple of 16, which bounds static-shape recompilation; BioIR pads inputs longer than 250
    tokens to the next multiple of 16, where its kernels run faster, and runs shorter ones at their
    own length; experimental flow KF always uses aligned buckets; eager does not pad.
    """
    buckets = inference_buckets(config)
    if buckets is not None:
        return buckets
    resolved = resolve_backends(config, "inference", guidance=guidance)
    if resolved["flow"] == "bioir" and config.get("bioir", {}).get("triangle_backend", "shared") == "kf":
        return ALIGNED_LENGTHS
    backends = set(resolved.values())
    if "torch" in backends:
        return ALIGNED_LENGTHS
    if "bioir" in backends:
        return BIOIR_LENGTHS
    return None


def configure_triangle_backends(model: nn.Module, backend: str = "cuequivariance") -> None:
    """Apply the same triangle policy after loading checkpoint-selected architecture."""
    from rfproteina.nn.modules.openfold_modules.triangular_multiplicative_update import TriangleMultiplicativeUpdate

    for module in model.modules():
        if isinstance(module, TriangleMultiplicativeUpdate):
            module.set_backend(backend)


def configure_execution(model: nn.Module, config: Mapping, phase: str) -> None:
    """Select a policy after weight loading, preserving native modules and compile caches.

    BioIR copies are discarded on each phase boundary and rebuilt lazily from the
    active weights. Registered parameters and checkpoint keys never change.
    """
    backends = resolve_backends(config, phase, guidance=getattr(model, "guidance_model", None) is not None)
    components = execution_components(model)
    configure_triangle_backends(model, config.get(phase, {}).get("tri_mult", "cuequivariance"))
    if "bioir" in backends.values():
        from rfproteina.nn.bioir import _require_bioir

        _require_bioir()
    for name, trunk in components.items():
        runtime = getattr(trunk, "_execution", None)
        if runtime is None:
            runtime = BlockExecution(
                compile_kwargs=getattr(trunk, "_compile_kwargs", {"fullgraph": True, "dynamic": False})
            )
            trunk._execution = runtime
        runtime.backend = backends[name]
        runtime.cuda_graphs = phase == "training"
        runtime.compile_scope = config.get(phase, {}).get("compile_scope", "blocks")
        if name == "decoder" and getattr(model, "guidance_model", None) is not None:
            runtime.compile_scope = "blocks"
        options = dict(config.get("bioir", {}))
        # Flow masks are collated padding masks (valid tokens first); AE masks also drop in-context tokens.
        flow_prefix_masks = options.pop("flow_prefix_masks", False)
        # KF is a flow-only experiment; AE masks may contain interior invalid tokens.
        if name != "flow":
            options["triangle_backend"] = "shared"
        runtime.bioir_options = {**options, "prefix_masks": name == "flow" and bool(flow_prefix_masks)}
        runtime.converted = None
        runtime.compiled.clear()
        trunk.bioir_enabled = runtime.backend == "bioir"
        logger.info("Execution %s.%s=%s", phase, name, runtime.backend)
    if any(backends[name] == "torch" for name in components):
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)


def setup_execution(config: Mapping, *, inference: bool = False, guidance: bool = False) -> None:
    """Initialize the optional compiler cache once per job, only when requested."""
    phases = ("inference",) if inference else ("training", "inference")
    if any("torch" in resolve_backends(config, phase, guidance=guidance).values() for phase in phases):
        setup_run_compile(inference=inference)


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
