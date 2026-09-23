"""Optional inference block conversion; model semantics remain in RFProteína."""

import logging
from typing import Literal

import torch
from bionemo_ir.blocks import ConditionedTransformerBlock, PairUpdateBlock, with_cuda_graph
from torch import nn

from rfproteina.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from rfproteina.nn.modules.pair_update import PairReprUpdate

logger = logging.getLogger(__name__)


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

    def load(self, module: nn.Module) -> None:
        if self.source:
            raise ValueError(f"BioIR conversion did not consume parameters: {sorted(self.source)}")
        module.load_state_dict(self.target, strict=True)


def convert_attention_block(
    source: MultiheadAttnAndTransition,
    *,
    dtype: torch.dtype = torch.bfloat16,
    backend: Literal["SDPA", "CuTeDSL"] = "SDPA",
) -> ConditionedTransformerBlock:
    """Convert a sequential conditioned block, preserving its trained parameters.

    The returned block is for inference and consumes the caller's attention mask.
    """
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
    ).to(device=next(source.parameters()).device)
    weights = _Weights(source)
    weights.adaln("mhba.adaln", "adaln")
    for old, new in (
        ("node_norm", "norm_s"),
        ("q_layer_norm", "q_norm"),
        ("k_layer_norm", "k_norm"),
        ("to_g", "proj_g"),
        ("to_out_node", "proj_o"),
        ("pair_norm", "proj_z.0"),
        ("to_bias", "proj_z.1"),
    ):
        weights.rename(f"mhba.mha.{old}", f"mha.{new}")
    for suffix in ("weight", "bias"):
        qkv = weights.source.pop(f"mhba.mha.to_qkv.{suffix}")
        q, kv = qkv.split([qkv.shape[0] // 3, 2 * qkv.shape[0] // 3])
        weights.target[f"mha.proj_q.{suffix}"] = q
        weights.target[f"mha.proj_kv.{suffix}"] = kv
    weights.rename("mhba.scale_output.to_adaln_zero_gamma.0", "to_adaln_zero_gamma.0")
    weights.adaln("transition.adaln", "transition.adaln")
    weights.rename("transition.transition.swish_linear.0", "transition.fused_swl_a_to_b")
    weights.rename("transition.transition.linear_out", "transition.b_to_a")
    weights.rename("transition.scale_output.to_adaln_zero_gamma.0", "transition.output_projection")
    weights.load(target)
    return target.eval()


def convert_pair_block(source: PairReprUpdate, *, dtype: torch.dtype = torch.bfloat16) -> PairUpdateBlock:
    """Convert a pair update, including optional triangle multiplication."""
    target = PairUpdateBlock(
        layer_idx=0,
        token_dim=source.linear_x.in_features,
        pair_dim=source.linear_x.out_features // 2,
        trimul_hidden_dim=source.tri_mult_out.linear_a_p.out_features if source.use_tri_mult else 1,
        pair_transition_expansion=source.transition_out.n,
        pair_mask_left_aligned=False,
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
                weights.target[f"{triangle}.{new}.{suffix}"] = torch.cat(
                    [weights.source.pop(f"{triangle}.linear_{branch}_{old}.{suffix}") for branch in ("a", "b")]
                )
    for name in ("layer_norm", "linear_1", "linear_2"):
        weights.rename(f"transition_out.{name}", f"transition_out.{name}")
    weights.load(target)
    return target.eval()


class _InferenceBlock(nn.Module):
    """Keep RFD4 activation dtypes at the boundary of a BioIR inference block."""

    def __init__(self, block: nn.Module, dtype: torch.dtype, output_index: int) -> None:
        super().__init__()
        self.block = block
        self.dtype = dtype
        self.output_index = output_index

    def forward(self, *args: torch.Tensor | None) -> torch.Tensor:
        output_dtype = args[self.output_index].dtype
        with torch.autocast("cuda", enabled=False):
            converted = tuple(
                arg.to(self.dtype) if arg is not None and arg.is_floating_point() else arg for arg in args
            )
            return self.block(*converted).to(output_dtype)


def convert_blocks(
    trunk: nn.Module,
    *,
    dtype: str = "bfloat16",
    attention_backend: Literal["SDPA", "CuTeDSL"] = "SDPA",
) -> tuple[list, list]:
    """Build graph-wrapped inference blocks without changing registered parameters."""
    if dtype not in ("float32", "bfloat16") or attention_backend not in ("SDPA", "CuTeDSL"):
        raise ValueError("BioIR requires dtype=float32|bfloat16 and attention_backend=SDPA|CuTeDSL")
    if attention_backend == "CuTeDSL" and dtype != "bfloat16":
        raise ValueError("BioIR CuTeDSL attention requires dtype=bfloat16")
    block_dtype = getattr(torch, dtype)

    def convert(block: nn.Module | None, output_index: int) -> nn.Module | None:
        if block is None:
            return None
        converted = (
            convert_pair_block(block, dtype=block_dtype)
            if output_index == 1
            else convert_attention_block(block, dtype=block_dtype, backend=attention_backend)
        )
        return _InferenceBlock(with_cuda_graph(converted), block_dtype, output_index)

    with torch.random.fork_rng(devices=[]):
        return tuple(
            [convert(block, index) for block in getattr(trunk, attribute, ())]
            for index, attribute in enumerate(("transformer_layers", "pair_update_layers"))
        )
