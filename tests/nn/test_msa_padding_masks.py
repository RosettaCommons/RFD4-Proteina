"""Test that MSA module correctly masks padding positions (True = valid convention)."""

import torch

from proteinfoundation.nn.modules.msa_module import MSAModule


def test_msa_module_padding_does_not_leak():
    """Changing padding content should not affect output at valid positions."""
    torch.manual_seed(0)
    B, I, S = 2, 8, 6
    c_m, c_z = 64, 64  # pair_repr_dim must be multiple of 32 for triangle ops

    module = MSAModule(
        n_block=2,
        c_m=c_m,
        p_drop_msa=0.0,
        p_drop_pair=0.0,
        use_cuequivariance=False,
        msa_subsample_embedder={"num_sequences": S, "dim_raw_msa": 71, "c_s_inputs": c_z, "c_msa_embed": c_m},
        outer_product={"c_msa_embed": c_m, "c_outer_product": 16, "c_out": c_z},
        msa_pair_weighted_averaging={"n_heads": 4, "c_weighted_average": 16, "c_msa_embed": c_m, "c_z": c_z},
        msa_transition={"n": 2, "c": c_m},
        triangle_multiplication_outgoing={"d_pair": c_z, "d_hidden": c_z},
        triangle_multiplication_incoming={"d_pair": c_z, "d_hidden": c_z},
        triangle_attention_starting={"d_pair": c_z, "n_head": 4, "d_hidden": 16},
        triangle_attention_ending={"d_pair": c_z, "n_head": 4, "d_hidden": 16},
        pair_transition={"n": 2, "c": c_z},
    ).eval()

    # Masks: first 5 tokens valid, last 3 padding; first 4 MSA rows valid, last 2 padding
    mask_I = torch.zeros(B, I, dtype=torch.bool)
    mask_I[:, :5] = True
    mask_S = torch.zeros(B, S, dtype=torch.bool)
    mask_S[:, :4] = True

    msa = torch.randn(B, S, I, 71)
    Z_II = torch.randn(B, I, I, c_z)
    S_inputs = torch.randn(B, I, c_z)
    f = {"msa_final": msa}

    with torch.no_grad():
        out1 = module(f, Z_II.clone(), S_inputs, mask_I, mask_S)

        # Corrupt padding positions with large values — should not affect valid output
        msa2 = msa.clone()
        msa2[:, 4:, :, :] = 999.0  # corrupt padding MSA rows
        msa2[:, :, 5:, :] = 999.0  # corrupt padding token positions in MSA
        Z_II2 = Z_II.clone()
        Z_II2[:, 5:, :, :] = 999.0  # corrupt padding in pair repr
        Z_II2[:, :, 5:, :] = 999.0
        f2 = {"msa_final": msa2}

        out2 = module(f2, Z_II2, S_inputs, mask_I, mask_S)

    # Valid-position outputs must match regardless of padding content
    valid_out1 = out1[:, :5, :5, :]
    valid_out2 = out2[:, :5, :5, :]
    assert torch.allclose(valid_out1, valid_out2, atol=1e-5), (
        f"Padding leaked into valid positions! Max diff: {(valid_out1 - valid_out2).abs().max():.2e}"
    )
