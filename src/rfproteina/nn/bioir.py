"""Optional inference block conversion; model semantics remain in RFProteína."""

import logging
from collections.abc import Callable, Sequence
from copy import deepcopy
from importlib import import_module
from inspect import signature
from typing import Literal

import torch
from torch import nn

from rfproteina.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from rfproteina.nn.modules.pair_update import PairReprUpdate

logger = logging.getLogger(__name__)


def _load_bioir_blocks() -> object:
    """Prefer current blocks; support missing legacy namespace ancestors, not broken dependencies."""
    module = "bionemo_ir._torch.layers.transformers.blocks"
    try:
        return import_module(module)
    except ModuleNotFoundError as error:
        if error.name != module and not module.startswith(f"{error.name}."):
            raise
        return import_module("bionemo_ir.blocks")


# Reusable RFD4 blocks are optional and currently require the BioIR RFD4 source branch.
try:
    _blocks = _load_bioir_blocks()
    ConditionedTransformerBlock = _blocks.ConditionedTransformerBlock
    PairUpdateBlock = _blocks.PairUpdateBlock
    TriangleMultiplicationMetadata = _blocks.TriangleMultiplicationMetadata
    precompute_trimul_metadata = _blocks.precompute_trimul_metadata
    with_cuda_graph = _blocks.with_cuda_graph
    BIOIR_IMPORT_ERROR = None
except (ImportError, OSError, AttributeError) as error:
    ConditionedTransformerBlock = PairUpdateBlock = TriangleMultiplicationMetadata = None
    precompute_trimul_metadata = with_cuda_graph = None
    BIOIR_IMPORT_ERROR = error
BIOIR_AVAILABLE = BIOIR_IMPORT_ERROR is None


def _require_bioir() -> None:
    if not BIOIR_AVAILABLE:
        raise ImportError(
            "BioIR execution requires reusable RFD4 blocks from the TensorRT BioNeMo ductr/rfd4 branch "
            "(runtime MR545, qualified revision 4f97e9d). PyPI bionemo-ir 0.1.1 does not contain this API. "
            "Install that compatible BioIR runtime, or select inference_backend=torch_compile or eager."
        ) from BIOIR_IMPORT_ERROR


class _Weights:
    """Consume every source parameter exactly once when constructing a target state."""

    def __init__(self, module: nn.Module) -> None:
        self.source = dict(module.state_dict())
        self.target: dict[str, torch.Tensor] = {}

    def rename(self, source: str, target: str) -> None:
        for suffix in ("weight", "bias"):
            key = f"{source}.{suffix}"
            if key in self.source:
                self.target[f"{target}.{suffix}"] = self.source.pop(key)

    def adaln(self, source: str, target: str) -> None:
        self.rename(f"{source}.norm_cond", f"{target}.s_norm")
        self.target[f"{target}.fused_s_scale_s_bias.weight"] = torch.cat(
            [self.source.pop(f"{source}.to_gamma.0.weight"), self.source.pop(f"{source}.to_beta.weight")]
        )
        bias = self.source.pop(f"{source}.to_gamma.0.bias")
        self.target[f"{target}.fused_s_scale_s_bias.bias"] = torch.cat([bias, torch.zeros_like(bias)])

    def zero_biases(self, module: nn.Module, prefix: str) -> None:
        for name, parameter in module.named_parameters():
            if name.startswith(prefix) and name.endswith(".bias") and name not in self.target:
                self.target[name] = torch.zeros_like(parameter)

    def load(self, module: nn.Module) -> None:
        if self.source:
            raise ValueError(f"BioIR conversion did not consume parameters: {sorted(self.source)}")
        module.load_state_dict(self.target, strict=True)


def _map_attention_projections(weights: _Weights, target: nn.Module) -> None:
    """Preserve Q/K/V and learned gate biases across separate and fused BioIR layouts."""
    for suffix in ("weight", "bias"):
        q, k, v = weights.source.pop(f"mhba.mha.to_qkv.{suffix}").chunk(3)
        gate = weights.source.pop(f"mhba.mha.to_g.{suffix}")
        if hasattr(target, "in_proj"):
            weights.target[f"mha.in_proj.{suffix}"] = torch.cat((q, gate, k, v))
        else:
            weights.target[f"mha.proj_q.{suffix}"] = q
            weights.target[f"mha.proj_g.{suffix}"] = gate
            weights.target[f"mha.proj_kv.{suffix}"] = torch.cat((k, v))


def convert_attention_block(
    source: MultiheadAttnAndTransition,
    *,
    dtype: torch.dtype = torch.bfloat16,
    backend: Literal["SDPA", "CuTeDSL"] = "SDPA",
    sanitize_masked_inputs: bool = True,
) -> ConditionedTransformerBlock:
    """Convert a sequential conditioned block, preserving its trained parameters.

    The returned block is for inference and consumes the caller's attention mask. Callers that keep
    padded tokens and pairs at zero may disable ``sanitize_masked_inputs``, which otherwise re-masks
    the single, conditioning and pair inputs on every call.
    """
    _require_bioir()
    if source.parallel or not source.residual_mha or not source.residual_transition:
        raise ValueError("BioIR requires sequential attention/transition with both residuals enabled")
    attention = source.mhba.mha
    norm = source.mhba.adaln.norm_cond
    if not isinstance(norm, nn.LayerNorm | nn.RMSNorm):
        raise ValueError(f"Unsupported BioIR normalization: {type(norm).__name__}")
    if any("lora_" in key for key in source.state_dict()):
        raise ValueError("Merge LoRA weights before enabling BioIR")
    norm_type = "rms_norm" if isinstance(norm, nn.RMSNorm) else "layer_norm"
    transition = source.transition.transition
    target = ConditionedTransformerBlock(
        layer_idx=0,
        token_dim=attention.node_dim,
        pair_dim=attention.pair_dim or 1,
        condition_dim=norm.normalized_shape[0],
        num_heads=attention.heads,
        transition_expansion=transition.linear_out.in_features // attention.node_dim,
        backend=backend if source.use_attn_pair_bias else "SDPA",
        mask_inf=1e6,
        use_qk_norm=not isinstance(attention.q_layer_norm, nn.Identity),
        eps=norm.eps,
        dtype=dtype,
        norm_type=norm_type,
        cond_norm_bias=getattr(norm, "bias", None) is not None,
        use_pair_bias=source.use_attn_pair_bias,
        sanitize_masked_inputs=sanitize_masked_inputs,
    ).to(device=next(source.parameters()).device)
    weights = _Weights(source)
    weights.adaln("mhba.adaln", "adaln")
    for old, new in (
        ("node_norm", "norm_s"),
        ("q_layer_norm", "q_norm"),
        ("k_layer_norm", "k_norm"),
        ("to_out_node", "proj_o"),
        ("pair_norm", "proj_z.0"),
        ("to_bias", "proj_z.1"),
    ):
        weights.rename(f"mhba.mha.{old}", f"mha.{new}")
    _map_attention_projections(weights, target.mha)
    weights.rename("mhba.scale_output.to_adaln_zero_gamma.0", "to_adaln_zero_gamma.0")
    weights.adaln("transition.adaln", "transition.adaln")
    weights.rename("transition.transition.swish_linear.0", "transition.fused_swl_a_to_b")
    weights.rename("transition.transition.linear_out", "transition.b_to_a")
    weights.rename("transition.scale_output.to_adaln_zero_gamma.0", "transition.output_projection")
    weights.load(target)
    return target.eval()


class _SharedTriangle(nn.Module):
    """Run the checkpoint's triangle with the same kernel policy as native training."""

    def __init__(self, source: nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.native = deepcopy(source).to(dtype=dtype).eval()

    def can_fuse_residual(self, pair: torch.Tensor) -> bool:
        return False

    def forward(self, pair: torch.Tensor, mask: torch.Tensor, *, trimul_metadata: object) -> torch.Tensor:
        return self.native(pair, mask)


class _KFTriangle(nn.Module):
    """Experimental KF update with the training triangle as its only fallback.

    ``readiness`` describes construction, not executed CUDA graph dispatches.
    Both weight copies are retained; BioIR additionally caches folded weights.
    """

    def __init__(self, bioir: nn.Module, native: _SharedTriangle) -> None:
        super().__init__()
        self.bioir = bioir
        self.native = native
        self.readiness = "eligible_not_profiled"
        self._reported_fallback = False

    def can_fuse_residual(self, pair: torch.Tensor) -> bool:
        return False

    def forward(self, pair: torch.Tensor, mask: torch.Tensor, *, trimul_metadata: object) -> torch.Tensor:
        output = self.bioir.try_kf_forward(pair, mask, trimul_metadata, residual=False)
        if output is not None:
            return output
        if not self._reported_fallback:
            logger.warning("Experimental BioIR KF declined this input; using the shared native triangle")
            self._reported_fallback = True
        return self.native(pair, mask, trimul_metadata=trimul_metadata)


def _select_triangle(
    bioir: nn.Module, source: nn.Module, dtype: torch.dtype, triangle_backend: str, prefix_masks: bool
) -> nn.Module:
    """Select KF explicitly; missing APIs fail and unsupported kernels use the shared triangle."""
    native = _SharedTriangle(source, dtype)
    if triangle_backend == "shared":
        return native
    if triangle_backend != "kf":
        raise ValueError("BioIR triangle_backend must be shared or kf")
    if not prefix_masks:
        raise ValueError("Experimental BioIR KF requires flow_prefix_masks=true")
    attempt = getattr(bioir, "try_kf_forward", None)
    configure = getattr(bioir, "set_token_padding", None)
    if not callable(attempt) or not callable(configure):
        raise RuntimeError(
            "Experimental KF requires the BioIR MR229 try-KF API; the qualified MR545 runtime "
            "does not provide it. Use triangle_backend=shared until a compatible runtime is qualified."
        )
    try:
        signature(attempt).bind(None, None, None, residual=False)
    except TypeError as error:
        raise RuntimeError("Unsupported BioIR try-KF signature; use triangle_backend=shared") from error
    configure(True)
    if not bioir.kf_available:
        logger.warning(
            "Experimental BioIR KF has no supported kernels for this device/dtype/width; "
            "using the shared native triangle"
        )
        return native
    logger.info("Experimental BioIR KF is eligible; actual dispatch requires GPU trace verification")
    return _KFTriangle(bioir, native)


def convert_pair_block(
    source: PairReprUpdate,
    *,
    dtype: torch.dtype = torch.bfloat16,
    prefix_masks: bool = False,
    triangle_backend: Literal["shared", "kf"] = "shared",
) -> PairUpdateBlock:
    """Convert a pair update, including optional triangle multiplication.

    ``prefix_masks`` configures BioIR mask metadata. Triangle multiplication uses the
    same native cuEquivariance dispatcher as training by default. Experimental KF retains
    that dispatcher as its fallback when a supported kernel declines an input.
    """
    _require_bioir()
    target = PairUpdateBlock(
        layer_idx=0,
        token_dim=source.linear_x.in_features,
        pair_dim=source.linear_x.out_features // 2,
        trimul_hidden_dim=source.tri_mult_out.linear_a_p.out_features if source.use_tri_mult else 1,
        pair_transition_expansion=source.transition_out.n,
        pair_mask_left_aligned=prefix_masks,
        trimul_high_precision=dtype == torch.float32,
        eps=source.layer_norm_in.eps,
        dtype=dtype,
        pair_write=source.pair_write,
        use_tri_mult=source.use_tri_mult,
    ).to(device=next(source.parameters()).device)
    weights = _Weights(source)
    for name in ("layer_norm_in", "linear_x"):
        weights.rename(name, name)
    for triangle in ("tri_mult_out", "tri_mult_in") if source.use_tri_mult else ():
        for old, new in (
            ("layer_norm_in", "norm_in"),
            ("layer_norm_out", "norm_out"),
            ("linear_z", "p_out"),
            ("linear_g", "g_out"),
        ):
            weights.rename(f"{triangle}.{old}", f"{triangle}.{new}")
        for old, new in (("p", "p_in"), ("g", "g_in")):
            for suffix in ("weight", "bias"):
                parts = [
                    weights.source.pop(f"{triangle}.linear_{branch}_{old}.{suffix}", None) for branch in ("a", "b")
                ]
                if all(part is not None for part in parts):
                    weights.target[f"{triangle}.{new}.{suffix}"] = torch.cat(parts)
        # PairUpdateBlock always allocates triangle biases, and its cuEquivariance kernel runs only when all
        # are present; a bias-free source (nn.tri_mult_bias=false) is exactly a zero bias.
        weights.zero_biases(target, f"{triangle}.")
    for name in ("layer_norm", "linear_1", "linear_2"):
        weights.rename(f"transition_out.{name}", f"transition_out.{name}")
    weights.load(target)
    if source.use_tri_mult:
        for name in ("tri_mult_out", "tri_mult_in"):
            setattr(
                target,
                name,
                _select_triangle(getattr(target, name), getattr(source, name), dtype, triangle_backend, prefix_masks),
            )
    return target.eval()


class _PairUpdate(nn.Module):
    """Call a BioIR pair block with its trunk pass's triangle-multiplication metadata.

    Without prefix masks the metadata stays empty (no row lengths or token padding).
    """

    def __init__(self, block: PairUpdateBlock) -> None:
        super().__init__()
        self.block = block
        self.metadata = TriangleMultiplicationMetadata()

    def forward(self, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.block(s, z, mask, self.metadata)


class TrunkPass(nn.Module):
    """One inference pass over a trunk's converted attention and pair blocks.

    Activations cross the pass boundary once: inputs are cast to the block dtype with padded tokens
    and pairs zeroed, and outputs are cast back to the caller's dtypes. Blocks preserve zero padding,
    so they skip per-call input masking.
    """

    def __init__(
        self,
        apply_blocks: Callable,
        attention: Sequence[ConditionedTransformerBlock],
        pair: Sequence[_PairUpdate | None],
        dtype: torch.dtype,
        prefix_masks: bool = False,
    ) -> None:
        super().__init__()
        self.apply_blocks = apply_blocks
        self.attention = nn.ModuleList(attention)
        self.pair = list(pair)
        self.pair_modules = nn.ModuleList(block for block in pair if block is not None)
        self.dtype = dtype
        self.prefix_masks = prefix_masks

    def forward(
        self, s: torch.Tensor, z: torch.Tensor | None, c: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        s_dtype = s.dtype
        z_dtype = None if z is None else z.dtype
        with torch.autocast("cuda", enabled=False):
            valid = mask[..., None]  # [b, n, 1]
            s = s.to(self.dtype).masked_fill(~valid, 0)
            c = c.to(self.dtype).masked_fill(~valid, 0)
            if z is not None:
                z = z.to(self.dtype).masked_fill(~(valid[:, :, None] & valid[:, None]), 0)
                if self.prefix_masks:
                    torch._assert_async(
                        ~(mask[:, 1:] & ~mask[:, :-1]).any(), "BioIR prefix masks need valid tokens before padding"
                    )
                    lengths = (mask[:, :, None] & mask[:, None, :]).sum(dim=-1, dtype=torch.int32)  # [b, n]
                    metadata = precompute_trimul_metadata(z, lengths, lengths)
                    for block in self.pair_modules:
                        block.metadata = metadata
            s, z = self.apply_blocks(s, z, c, mask, self.attention, self.pair)
            return s.to(s_dtype), None if z is None else z.to(z_dtype)


def convert_trunk(
    attention: Sequence[MultiheadAttnAndTransition],
    pair: Sequence[PairReprUpdate | None],
    apply_blocks: Callable,
    *,
    dtype: str = "bfloat16",
    attention_backend: Literal["SDPA", "CuTeDSL"] = "SDPA",
    cuda_graph: bool = True,
    prefix_masks: bool = False,
    triangle_backend: Literal["shared", "kf"] = "shared",
) -> nn.Module:
    """Build one inference pass over a trunk's blocks without changing its registered parameters.

    With ``cuda_graph`` the whole pass is captured as one CUDA graph: its blocks share a single memory
    pool, and inputs are staged once per pass rather than once per block.

    Args:
        attention: Native attention blocks, in trunk order.
        pair: Native pair updates following each attention block; None entries are skipped.
        apply_blocks: Trunk loop shared with the native backends.
        dtype: Block activation dtype.
        attention_backend: BioIR attention kernel.
        triangle_backend: Shared training triangle, or experimental KF with shared fallback.
        cuda_graph: Capture and replay the pass as a CUDA graph (one shape at a time).
        prefix_masks: Masks list valid tokens before padding (checked on device), so triangle
            metadata is prepared once per pass. Shared native triangles retain their own masks.
    """
    _require_bioir()
    if dtype not in ("float32", "bfloat16") or attention_backend not in ("SDPA", "CuTeDSL"):
        raise ValueError("BioIR requires dtype=float32|bfloat16 and attention_backend=SDPA|CuTeDSL")
    if attention_backend == "CuTeDSL" and dtype != "bfloat16":
        raise ValueError("BioIR CuTeDSL attention requires dtype=bfloat16")
    if triangle_backend not in ("shared", "kf"):
        raise ValueError("BioIR triangle_backend must be shared or kf")
    if triangle_backend == "kf" and dtype != "bfloat16":
        raise ValueError("Experimental BioIR KF requires dtype=bfloat16")
    block_dtype = getattr(torch, dtype)
    # Block constructors randomly initialize weights on CPU before the trained weights are loaded.
    # Fork the CPU generator so converting (e.g. at each validation) leaves seeded sampling unchanged.
    with torch.random.fork_rng(devices=[]):
        converted_attention = [
            convert_attention_block(block, dtype=block_dtype, backend=attention_backend, sanitize_masked_inputs=False)
            for block in attention
        ]
        converted_pair = [
            None
            if block is None
            else _PairUpdate(
                convert_pair_block(
                    block, dtype=block_dtype, prefix_masks=prefix_masks, triangle_backend=triangle_backend
                )
            )
            for block in pair
        ]
    trunk_pass = TrunkPass(apply_blocks, converted_attention, converted_pair, block_dtype, prefix_masks)
    return with_cuda_graph(trunk_pass) if cuda_graph else trunk_pass
