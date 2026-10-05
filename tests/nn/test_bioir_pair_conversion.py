"""BioIR pair conversion preserves biased and bias-free triangle updates."""

import pytest
import torch

from rfproteina.nn.modules.pair_update import PairReprUpdate


@pytest.mark.gpu
@pytest.mark.parametrize("use_tri_mult,bias", [(True, False), (True, True), (False, False)])
def test_bioir_pair_conversion_preserves_output(use_tri_mult: bool, bias: bool) -> None:
    from rfproteina.nn.bioir import BIOIR_AVAILABLE, TriangleMultiplicationMetadata, convert_pair_block

    if not BIOIR_AVAILABLE:
        pytest.skip("BioIR reusable RFD4 blocks are unavailable")

    torch.manual_seed(42)
    source = (
        PairReprUpdate(
            token_dim=32,
            pair_dim=16,
            tri_mult_c=16,
            use_tri_mult=use_tri_mult,
            tri_mult_bias=bias,
            checkpoint_tri_mult=False,
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            parameter.normal_(1.0 if name.endswith("weight") and parameter.ndim == 1 else 0.0, 0.1)
    original = {name: value.clone() for name, value in source.state_dict().items()}
    single = torch.randn(2, 16, 32, device="cuda")
    pair = torch.randn(2, 16, 16, 16, device="cuda")
    mask = torch.ones(2, 16, device="cuda", dtype=torch.bool)
    mask[0, 5] = False
    mask[1, -3:] = False

    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.inference_mode():
            converted = convert_pair_block(source, dtype=torch.float32)
            expected = source(single, pair, mask)
            observed = converted(single, pair, mask, TriangleMultiplicationMetadata())
    finally:
        torch.set_float32_matmul_precision(previous_precision)

    torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-4)
    assert source.state_dict().keys() == original.keys()
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
