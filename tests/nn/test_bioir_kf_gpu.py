"""Real SM90 KF dispatch and batched numerical parity; no mocked kernels."""

import os

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("length", [128, 250])
@pytest.mark.parametrize("bias", [False, True])
def test_kf_pair_batch_and_graph_parity(batch_size, length, bias, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        if os.environ.get("RFD4_REQUIRE_KF") == "1":
            pytest.fail("KF qualification requires an actual SM90 GPU")
        pytest.skip("KF kernels require SM90")
    from rfproteina.nn.bioir import _KFTriangle, convert_pair_block, precompute_trimul_metadata
    from rfproteina.nn.modules.pair_update import PairReprUpdate

    torch.manual_seed(731)
    padded = ((length + 15) // 16) * 16
    source = (
        PairReprUpdate(
            token_dim=768,
            pair_dim=384,
            tri_mult_c=256,
            use_tri_mult=True,
            tri_mult_bias=bias,
            checkpoint_tri_mult=False,
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            parameter.normal_(1.0 if name.endswith("weight") and parameter.ndim == 1 else 0.0, 0.03)
    for triangle in (source.tri_mult_in, source.tri_mult_out):
        triangle.set_backend("cuequivariance")
    converted = convert_pair_block(source, dtype=torch.bfloat16, prefix_masks=True, triangle_backend="kf")
    assert all(
        isinstance(node, _KFTriangle) for node in (converted.tri_mult_in, converted.tri_mult_out)
    ), "KF was requested but this runtime did not provide the actual kernels/configs"
    calls = []
    for node in (converted.tri_mult_in, converted.tri_mult_out):
        operation = node.bioir.try_kf_forward

        def record(*args, operation=operation, **kwargs):
            output = operation(*args, **kwargs)
            calls.append(output is not None)
            return output

        monkeypatch.setattr(node.bioir, "try_kf_forward", record)
    lengths = torch.tensor([length - 3 * index for index in range(batch_size)], device="cuda")
    mask = torch.arange(padded, device="cuda")[None] < lengths[:, None]
    pair_mask = mask[:, :, None] & mask[:, None, :]
    s = torch.randn(batch_size, padded, 768, device="cuda", dtype=torch.bfloat16).masked_fill(~mask[..., None], 0)
    z = torch.randn(batch_size, padded, padded, 384, device="cuda", dtype=torch.bfloat16)
    z.masked_fill_(~pair_mask[..., None], 0)
    counts = pair_mask.sum(-1, dtype=torch.int32)
    metadata = precompute_trimul_metadata(z, counts, counts)

    def check(actual, expected):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual.masked_select(~pair_mask[..., None])) == 0
        delta = actual.float()[pair_mask] - expected.float()[pair_mask]
        reference = expected.float()[pair_mask]
        relative_rmse = delta.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12)
        assert relative_rmse.item() < 0.05

    with torch.inference_mode():
        expected = source(s, z, mask)
        actual = converted(s, z, mask, metadata)
        assert calls == [True, True], "Both triangle directions must execute KF, not fallback"
        check(actual, expected)
        # Independent samples must agree with the same batched call.
        for index in range(batch_size):
            single_counts = counts[index : index + 1]
            single_metadata = precompute_trimul_metadata(z[index : index + 1], single_counts, single_counts)
            independent = converted(
                s[index : index + 1], z[index : index + 1], mask[index : index + 1], single_metadata
            )
            torch.testing.assert_close(actual[index : index + 1], independent, rtol=0.02, atol=0.06)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                converted(s, z, mask, metadata)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = converted(s, z, mask, metadata)
        graph.replay()
        check(captured, expected)
        s.mul_(0.9)
        z.mul_(1.1)
        graph.replay()
        replayed = captured.clone()
        check(replayed, source(s, z, mask))
        # Python counts above prove eager calls only; graph replay is validated numerically.
