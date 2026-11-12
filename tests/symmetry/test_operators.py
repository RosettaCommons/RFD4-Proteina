"""Unit tests for equivariant/invariant symmetry operators, noise, centering, and atom37 decode."""

from __future__ import annotations

import unittest

import torch

from tests.symmetry import assert_worktree_imports


class TestSymmetryOperators(unittest.TestCase):
    """Tests for expand_asu, reduce_to_asu, project_symmetric, noise, CoM, and atom37 decode."""

    def setUp(self) -> None:
        assert_worktree_imports()
        torch.manual_seed(1234)

    def test_expand_reduce_and_project_idempotence_and_optimality(self) -> None:
        from rfproteina.symmetry import expand_asu, parse_symmetry, project_symmetric, reduce_to_asu

        for g_name in ("C1", "C3", "D2", "D5", "T", "O", "I"):
            with self.subTest(group=g_name):
                spec = parse_symmetry(g_name)
                k = spec.order
                l_asu = 6
                x_asu = torch.randn(2, l_asu, 3, dtype=torch.float64)

                x_full = expand_asu(x_asu, spec)
                self.assertEqual(tuple(x_full.shape), (2, k * l_asu, 3))
                # Subunit 0 equals x_asu
                self.assertLess((x_full[:, :l_asu, :] - x_asu).abs().max().item(), 1e-13)

                # Each subunit g equals R_g @ x_asu
                r = spec.rotations(dtype=torch.float64)
                for g in range(k):
                    expected_g = torch.einsum("ij,blj->bli", r[g], x_asu)
                    actual_g = x_full[:, g * l_asu : (g + 1) * l_asu, :]
                    self.assertLess((actual_g - expected_g).abs().max().item(), 1e-13)

                # Exact round-trip via reduce_to_asu
                x_rec = reduce_to_asu(x_full, spec)
                self.assertLess((x_rec - x_asu).abs().max().item(), 1e-13)

                # Projecting an already-symmetric configuration is exact identity
                x_proj_clean = project_symmetric(x_full, spec)
                self.assertLess((x_proj_clean - x_full).abs().max().item(), 1e-13)

                # Projecting arbitrary asymmetric vector field: idempotence + orthogonal residual
                v_asym = torch.randn(2, k * l_asu, 3, dtype=torch.float64)
                v_sym = project_symmetric(v_asym, spec)
                v_sym_twice = project_symmetric(v_sym, spec)
                self.assertLess((v_sym_twice - v_sym).abs().max().item(), 1e-13)

                # Orthogonal projector check: <v_asym - v_sym, expand_asu(delta)> == 0
                residual = v_asym - v_sym
                delta_sym = expand_asu(torch.randn(2, l_asu, 3, dtype=torch.float64), spec)
                inner = (residual * delta_sym).sum().abs().item()
                self.assertLess(inner, 1e-11)

                # Strictly smaller L2 distance than perturbed symmetric candidate
                dist_opt = ((v_asym - v_sym) ** 2).sum().item()
                dist_pert = ((v_asym - (v_sym + 0.05 * delta_sym)) ** 2).sum().item()
                self.assertLess(dist_opt, dist_pert)

    def test_multi_atom_expand_and_project_shapes(self) -> None:
        from rfproteina.symmetry import expand_asu, parse_symmetry, project_symmetric

        spec = parse_symmetry("D3")
        k = spec.order  # 6
        l_asu = 5
        atoms_asu = torch.randn(2, l_asu, 37, 3, dtype=torch.float64)
        atoms_full = expand_asu(atoms_asu, spec)
        self.assertEqual(tuple(atoms_full.shape), (2, k * l_asu, 37, 3))

        r = spec.rotations(dtype=torch.float64)
        for g in range(k):
            expected_g = torch.einsum("ij,blaj->blai", r[g], atoms_asu)
            actual_g = atoms_full[:, g * l_asu : (g + 1) * l_asu]
            self.assertLess((actual_g - expected_g).abs().max().item(), 1e-13)

        noisy_atoms = atoms_full + 0.2 * torch.randn_like(atoms_full)
        proj_atoms = project_symmetric(noisy_atoms, spec)
        self.assertLess((project_symmetric(proj_atoms, spec) - proj_atoms).abs().max().item(), 1e-13)

    def test_equivariance_under_group_permutation(self) -> None:
        """Rotating all subunits by R_h and permuting subunit slots by left-multiplication commutes with P_G."""
        from rfproteina.symmetry import parse_symmetry, project_symmetric

        spec = parse_symmetry("T")
        k = spec.order
        l_asu = 4
        r = spec.rotations(dtype=torch.float64)
        x = torch.randn(1, k * l_asu, 3, dtype=torch.float64)
        x_grouped = x.view(1, k, l_asu, 3)

        # Pick non-identity group element h = 5
        h = 5
        r_h = r[h]
        # Under action of h: new subunit at index (h * g) receives R_h @ x_g
        products = torch.einsum("ij,gjk->gik", r_h, r)  # R_h R_g = R_{pi(g)}
        perm = torch.linalg.norm(products.unsqueeze(1) - r.unsqueeze(0), dim=(-2, -1)).argmin(dim=-1)  # [K]
        x_transformed_grouped = torch.zeros_like(x_grouped)
        for g in range(k):
            x_transformed_grouped[:, perm[g]] = torch.einsum("ij,blj->bli", r_h, x_grouped[:, g])
        x_transformed = x_transformed_grouped.reshape(1, k * l_asu, 3)

        proj_then_act = torch.zeros_like(x_grouped)
        p_x = project_symmetric(x, spec).view(1, k, l_asu, 3)
        for g in range(k):
            proj_then_act[:, perm[g]] = torch.einsum("ij,blj->bli", r_h, p_x[:, g])

        act_then_proj = project_symmetric(x_transformed, spec).view(1, k, l_asu, 3)
        self.assertLess((proj_then_act - act_then_proj).abs().max().item(), 1e-12)

    def test_invariant_operators_and_variance_preservation(self) -> None:
        from rfproteina.symmetry import (
            expand_invariant_asu,
            parse_symmetry,
            project_invariant_symmetric,
            project_symmetric,
        )

        spec = parse_symmetry("O")  # K = 24
        k = spec.order
        l_asu = 7
        logits_asu = torch.randn(3, l_asu, 147, dtype=torch.float32)
        logits_full = expand_invariant_asu(logits_asu, spec)
        self.assertEqual(tuple(logits_full.shape), (3, k * l_asu, 147))
        for g in range(k):
            self.assertTrue(torch.equal(logits_full[:, g * l_asu : (g + 1) * l_asu], logits_asu))

        noisy_logits = torch.randn(3, k * l_asu, 147, dtype=torch.float32)
        tied_logits = project_invariant_symmetric(noisy_logits, spec)
        for g in range(1, k):
            self.assertLess(
                (tied_logits[:, g * l_asu : (g + 1) * l_asu] - tied_logits[:, :l_asu]).abs().max().item(),
                1e-6,
            )

        # Variance preservation check (sqrt(K) scaling vs 1/K shrinkage)
        n_mc = 256
        xi_3d = torch.randn(n_mc, k * l_asu, 3, dtype=torch.float64)
        proj_unscaled = project_symmetric(xi_3d, spec, preserve_variance=False)
        proj_scaled = project_symmetric(xi_3d, spec, preserve_variance=True)
        self.assertAlmostEqual(proj_unscaled.var().item(), 1.0 / k, delta=0.01)
        self.assertAlmostEqual(proj_scaled.var().item(), 1.0, delta=0.05)

        xi_inv = torch.randn(n_mc, k * l_asu, 8, dtype=torch.float64)
        inv_scaled = project_invariant_symmetric(xi_inv, spec, preserve_variance=True)
        self.assertAlmostEqual(inv_scaled.var().item(), 1.0, delta=0.05)

    def test_global_com_and_symmetric_ca_noise_with_radial_offset(self) -> None:
        from rfproteina.symmetry import (
            compute_symmetry_rmsd,
            parse_symmetry,
            sample_symmetric_ca_noise,
            zero_global_com,
        )

        for g_name in ("C3", "D3", "T", "O", "I"):
            with self.subTest(group=g_name):
                spec = parse_symmetry(g_name)
                k = spec.order
                l_asu = 10
                n_tot = k * l_asu
                mask = torch.ones(4, n_tot, dtype=torch.bool)

                x0 = sample_symmetric_ca_noise(
                    shape=(4, n_tot),
                    group=spec,
                    mask=mask,
                    radius_nm=2.0,
                    zero_com=True,
                    dtype=torch.float64,
                )
                self.assertEqual(tuple(x0.shape), (4, n_tot, 3))
                # Global CoM is zero
                self.assertLess(x0.mean(dim=1).abs().max().item(), 1e-12)

                # Exact G-symmetry holds AFTER global CoM subtraction (even for C3 along z!)
                sym_rmsd = compute_symmetry_rmsd(x0, spec, mask=mask)
                if isinstance(sym_rmsd, torch.Tensor):
                    self.assertLess(sym_rmsd.max().item(), 1e-12)
                else:
                    self.assertLess(float(sym_rmsd), 1e-12)

                # Subunit CoMs are well separated (no origin clashing!)
                sub_coms = x0.view(4, k, l_asu, 3).mean(dim=2)  # [4, K, 3]
                dists = torch.cdist(sub_coms, sub_coms) + torch.eye(k, dtype=torch.float64).unsqueeze(0) * 100.0
                self.assertGreater(dists.min().item(), 0.25)

                # Applying zero_global_com again preserves exact G-symmetry
                x0_recentered = zero_global_com(x0, mask=mask)
                self.assertLess((x0_recentered - x0).abs().max().item(), 1e-12)

    def test_padding_mask_heterogeneous_batch_lengths(self) -> None:
        """Batch rows with different valid lengths K * L_asu_b + zero-padding do not corrupt each other."""
        from rfproteina.symmetry import (
            compute_symmetry_rmsd,
            parse_symmetry,
            project_invariant_symmetric,
            project_symmetric,
            sample_symmetric_ca_noise,
        )

        spec = parse_symmetry("C4")
        k = spec.order  # 4
        # Row 0: 4 * 5 = 20 valid tokens; Row 1: 4 * 8 = 32 valid tokens; padded to N_max = 36
        n_max = 36
        mask = torch.zeros(2, n_max, dtype=torch.bool)
        mask[0, :20] = True
        mask[1, :32] = True

        x_raw = torch.randn(2, n_max, 3, dtype=torch.float64)
        # Put large garbage in padded tail to catch any mask leak!
        x_raw[~mask] = 999.0

        x_sym = project_symmetric(x_raw, spec, mask=mask)
        # Padded region must be strictly zero
        self.assertEqual(x_sym[~mask].abs().max().item(), 0.0)
        # Row 0 must match standalone unpadded projection on [:20]
        x_row0_ref = project_symmetric(x_raw[0:1, :20], spec)
        self.assertLess((x_sym[0:1, :20] - x_row0_ref).abs().max().item(), 1e-12)
        # Row 1 must match standalone unpadded projection on [:32]
        x_row1_ref = project_symmetric(x_raw[1:2, :32], spec)
        self.assertLess((x_sym[1:2, :32] - x_row1_ref).abs().max().item(), 1e-12)

        # Invariant projection with heterogeneous padding
        logits_raw = torch.randn(2, n_max, 16, dtype=torch.float64)
        logits_raw[~mask] = -777.0
        logits_sym = project_invariant_symmetric(logits_raw, spec, mask=mask)
        self.assertEqual(logits_sym[~mask].abs().max().item(), 0.0)
        self.assertLess(
            (logits_sym[0:1, :20] - project_invariant_symmetric(logits_raw[0:1, :20], spec)).abs().max().item(),
            1e-12,
        )

        # Noise sampling with heterogeneous padding
        noise = sample_symmetric_ca_noise(
            shape=(2, n_max),
            group=spec,
            mask=mask,
            radius_nm=1.5,
            zero_com=True,
            dtype=torch.float64,
        )
        self.assertEqual(noise[~mask].abs().max().item(), 0.0)
        rmsd = compute_symmetry_rmsd(noise, spec, mask=mask)
        max_rmsd = rmsd.max().item() if isinstance(rmsd, torch.Tensor) else float(rmsd)
        self.assertLess(max_rmsd, 1e-12)

    def test_symmetrize_decoded_atom37_preserves_internal_geometry_and_exact_symmetry(self) -> None:
        """asu_medoid rejects outlier rotamer shrinkage and rigidly anchors CA without bond distortion."""
        from rfproteina.symmetry import expand_asu, parse_symmetry, symmetrize_decoded_atom37

        spec = parse_symmetry("C5")
        k = spec.order  # 5
        l_asu = 4
        n_tot = k * l_asu

        # Build realistic canonical ASU residue geometry in nm (N=0, CA=1, C=2, CB=3, O=4, CG=5)
        ca_asu = torch.tensor(
            [[0.8, 0.2, 0.0], [1.1, 0.4, 0.15], [0.9, 0.7, 0.30], [1.2, 0.9, 0.45]],
            dtype=torch.float64,
        ).unsqueeze(0)  # [1, 4, 3]
        ca_full = expand_asu(ca_asu, spec)  # [1, 20, 3]

        asu_rel = torch.zeros(1, l_asu, 37, 3, dtype=torch.float64)
        atom_mask_asu = torch.zeros(1, l_asu, 37, dtype=torch.bool)
        atom_mask_asu[:, :, :6] = True
        asu_rel[:, :, 0, :] = torch.tensor([-0.146, 0.0, 0.0], dtype=torch.float64)  # N
        asu_rel[:, :, 1, :] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)  # CA
        asu_rel[:, :, 2, :] = torch.tensor([0.05, 0.144, 0.0], dtype=torch.float64)  # C
        asu_rel[:, :, 3, :] = torch.tensor([0.05, -0.07, 0.126], dtype=torch.float64)  # CB
        asu_rel[:, :, 4, :] = torch.tensor([0.05, 0.267, 0.0], dtype=torch.float64)  # O
        asu_rel[:, :, 5, :] = torch.tensor([0.05, -0.07, 0.280], dtype=torch.float64)  # CG (chi1 rotamer +z)

        # Consensus conformer on subunits 0, 1, 2, 3 (with slight decoder CA drift of 0.005 nm)
        r = spec.rotations(dtype=torch.float64)
        coors_decoded = torch.zeros(1, n_tot, 37, 3, dtype=torch.float64)
        atom_mask_full = atom_mask_asu.repeat(1, k, 1)
        for g in range(k):
            conformer = asu_rel.clone()
            # Add small decoder CA drift to ensure rigid CA re-anchoring preserves internal bonds!
            conformer[:, :, :, :] += torch.tensor([0.004 * (g + 1), -0.003, 0.002], dtype=torch.float64)
            if g == 4:
                # Subunit 4 picks opposite sidechain rotamer (-z instead of +z for CG)!
                conformer[:, :, 5, :] = conformer[:, :, 1, :] + torch.tensor(
                    [0.05, -0.07, -0.280], dtype=torch.float64
                )
            asu_abs_g = ca_asu.unsqueeze(-2) + conformer
            coors_decoded[:, g * l_asu : (g + 1) * l_asu] = torch.einsum("ij,blaj->blai", r[g], asu_abs_g)

        mask = torch.ones(1, n_tot, dtype=torch.bool)
        sym_medoid = symmetrize_decoded_atom37(
            coors_decoded,
            spec,
            ca_coors_nm=ca_full,
            atom_mask=atom_mask_full,
            mask=mask,
            mode="asu_medoid",
        )

        # 1. Exact CA anchoring to ca_full
        self.assertLess((sym_medoid[:, :, 1, :] - ca_full).abs().max().item(), 1e-13)

        # 2. Exact all-atom G-symmetry across all 5 subunits and all 6 occupied atom slots
        sub0 = sym_medoid[:, :l_asu]
        for g in range(k):
            expected_g = torch.einsum("ij,blaj->blai", r[g], sub0)
            actual_g = sym_medoid[:, g * l_asu : (g + 1) * l_asu]
            diff = (actual_g - expected_g)[atom_mask_full[:, g * l_asu : (g + 1) * l_asu]]
            self.assertLess(diff.abs().max().item(), 1e-12)

        # 3. Internal bond lengths (CB-CG = 0.154 nm, CA-N = 0.146 nm) preserved WITHOUT rotamer shrinkage!
        cb_cg_len = torch.linalg.norm(sym_medoid[:, :, 5, :] - sym_medoid[:, :, 3, :], dim=-1)
        ca_n_len = torch.linalg.norm(sym_medoid[:, :, 0, :] - sym_medoid[:, :, 1, :], dim=-1)
        self.assertLess((cb_cg_len - 0.154).abs().max().item(), 1e-12)
        self.assertLess((ca_n_len - 0.146).abs().max().item(), 1e-12)

        # Contrast with chordal_mean: averaging +0.280 and -0.280 shrinks CB-CG bond!
        sym_mean = symmetrize_decoded_atom37(
            coors_decoded,
            spec,
            ca_coors_nm=ca_full,
            atom_mask=atom_mask_full,
            mask=mask,
            mode="chordal_mean",
        )
        cb_cg_mean_len = torch.linalg.norm(sym_mean[:, :, 5, :] - sym_mean[:, :, 3, :], dim=-1)
        self.assertLess(cb_cg_mean_len.mean().item(), 0.12)  # confirms test genuinely exercises rotamer conflict!

        # 4. Regression check (#22/#23): unbatched 3D [N, 37, 3], multi-batch [B=2, N, 37, 3], and autograd .backward()
        coors_3d = coors_decoded[0].clone().requires_grad_(True)
        out_3d = symmetrize_decoded_atom37(
            coors_3d, spec, ca_coors_nm=ca_full[0], atom_mask=atom_mask_full[0], mode="asu_medoid"
        )
        out_3d.sum().backward()
        self.assertIsNotNone(coors_3d.grad)
        self.assertLess((out_3d - sym_medoid[0]).abs().max().item(), 1e-12)

        coors_b2 = coors_decoded.repeat(2, 1, 1, 1).clone().requires_grad_(True)
        ca_b2 = ca_full.repeat(2, 1, 1)
        am_b2 = atom_mask_full.repeat(2, 1, 1)
        out_b2 = symmetrize_decoded_atom37(coors_b2, spec, ca_coors_nm=ca_b2, atom_mask=am_b2, mode="asu_medoid")
        out_b2.sum().backward()
        self.assertIsNotNone(coors_b2.grad)
        self.assertLess((out_b2[1] - sym_medoid[0]).abs().max().item(), 1e-12)

    def test_resolve_batch_symmetry_on_padded_batched_data_sample(self) -> None:
        """resolve_batch_symmetry must handle PaddedBatchedDataSample (not a collections.abc.Mapping) & dict-of-lists."""
        from rfproteina.datapipes.collate import collate_batch
        from rfproteina.symmetry import SymmetryConfig, resolve_batch_symmetry

        samples = [
            {
                "example_id": "s_c3",
                "feats": {"residue_type": torch.zeros(12, dtype=torch.long), "coords": torch.zeros(12, 37, 3)},
                "extra_info": {"symmetry": SymmetryConfig(group="C3", radius=11.0).to_dict()},
            },
            {
                "example_id": "s_d2",
                "feats": {"residue_type": torch.zeros(16, dtype=torch.long), "coords": torch.zeros(16, 37, 3)},
                "extra_info": {"symmetry": SymmetryConfig(group="D2", radius=14.0, decode_mode="asu_first").to_dict()},
            },
        ]
        collated = collate_batch(samples, schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")

        # Works both WITH and WITHOUT explicit batch_size
        resolved_auto = resolve_batch_symmetry(collated)
        self.assertEqual([c.group for c in resolved_auto], ["C3", "D2"])
        self.assertEqual([c.radius for c in resolved_auto], [11.0, 14.0])
        self.assertEqual(resolved_auto[1].decode_mode, "asu_first")

        resolved_sized = resolve_batch_symmetry(collated, batch_size=2)
        self.assertEqual([c.group for c in resolved_sized], ["C3", "D2"])

        # Mixed batch: row 0 has symmetry in extra_info, row 1 has empty extra_info -> falls back cleanly
        mixed = collate_batch(
            [samples[0], {"example_id": "s_c1", "feats": samples[1]["feats"], "extra_info": {}}],
            schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
        )
        resolved_mixed = resolve_batch_symmetry(mixed, batch_size=2)
        self.assertEqual([c.group for c in resolved_mixed], ["C3", None])


if __name__ == "__main__":
    unittest.main()
