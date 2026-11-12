"""Unit tests for 3D point-group definitions, parser, group axioms, and chain IDs."""

from __future__ import annotations

import inspect
import math
import unittest
from collections import Counter

import torch

from tests.symmetry import WORKTREE_SRC, assert_worktree_imports


def _cayley_table(rotations: torch.Tensor, tol: float = 1e-11) -> torch.Tensor:
    """Build the K x K Cayley table where entry (i, j) is the unique k with R_i @ R_j == R_k."""
    k = rotations.shape[0]
    products = torch.einsum("aij,bjk->abik", rotations, rotations)  # [K, K, 3, 3]
    diffs = torch.linalg.norm(products.unsqueeze(2) - rotations.unsqueeze(0).unsqueeze(0), dim=(-2, -1))  # [K, K, K]
    matches = diffs < tol
    match_counts = matches.sum(dim=-1)
    if not torch.all(match_counts == 1):
        bad = torch.nonzero(match_counts != 1, as_tuple=False)
        min_dists = diffs.min(dim=-1).values
        raise AssertionError(
            f"Closure failed at {bad.tolist()}: match_counts={match_counts.tolist()}, "
            f"max_min_dist={min_dists.max().item():.3e}"
        )
    return matches.float().argmax(dim=-1)


def _element_orders(rotations: torch.Tensor, max_order: int = 120, tol: float = 1e-10) -> Counter[int]:
    """Compute the exact multiplicative order of each rotation matrix in G."""
    eye = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    orders: Counter[int] = Counter()
    for g in range(rotations.shape[0]):
        power = eye.clone()
        found = None
        for m in range(1, max_order + 1):
            power = power @ rotations[g]
            if torch.linalg.norm(power - eye).item() < tol:
                found = m
                break
        if found is None:
            raise AssertionError(f"Element {g} did not return to identity within order {max_order}")
        orders[found] += 1
    return orders


class TestPointGroups(unittest.TestCase):
    """Comprehensive SO(3) point-group axiom and parser verification."""

    def test_worktree_import_isolation(self) -> None:
        assert_worktree_imports()
        import rfproteina.symmetry as sym

        self.assertTrue(sym.__file__ is not None and str(WORKTREE_SRC) in sym.__file__)

    def test_trivial_and_case_insensitive_parsing(self) -> None:
        from rfproteina.symmetry import parse_symmetry

        for trivial in (None, "", "none", "None", "NONE", "c1", "C1", " c1 "):
            spec = parse_symmetry(trivial)
            self.assertEqual(spec.name, "C1")
            self.assertEqual(spec.order, 1)
            self.assertFalse(spec.is_nontrivial)
            self.assertEqual(tuple(spec.rotations_f64.shape), (1, 3, 3))
            self.assertTrue(torch.allclose(spec.rotations_f64[0], torch.eye(3, dtype=torch.float64), atol=1e-15))

        cases = [
            ("c2", "C2", 2),
            ("C3", "C3", 3),
            ("c7", "C7", 7),
            ("d2", "D2", 4),
            ("D5", "D5", 10),
            ("t", "T", 12),
            ("T", "T", 12),
            ("o", "O", 24),
            ("O", "O", 24),
            ("i", "I", 60),
            ("I", "I", 60),
        ]
        for raw, canonical, expected_order in cases:
            spec = parse_symmetry(raw)
            self.assertEqual(spec.name, canonical)
            self.assertEqual(spec.order, expected_order)
            self.assertTrue(spec.is_nontrivial)

    def test_invalid_group_specs_rejected(self) -> None:
        from rfproteina.symmetry import parse_symmetry

        invalid_specs = ["C0", "C-1", "C-3", "D0", "D1", "D-2", "X3", "S4", "C2.5", "T2", "O4", "I60", "invalid"]
        for bad in invalid_specs:
            with self.subTest(spec=bad):
                with self.assertRaises(ValueError):
                    parse_symmetry(bad)

    def test_so3_group_axioms_all_families(self) -> None:
        from rfproteina.symmetry import parse_symmetry

        groups = [
            ("C1", 1),
            ("C2", 2),
            ("C3", 3),
            ("C4", 4),
            ("C5", 5),
            ("C6", 6),
            ("C7", 7),
            ("C8", 8),
            ("C12", 12),
            ("D2", 4),
            ("D3", 6),
            ("D4", 8),
            ("D5", 10),
            ("D6", 12),
            ("D8", 16),
            ("T", 12),
            ("O", 24),
            ("I", 60),
        ]
        eye64 = torch.eye(3, dtype=torch.float64)
        eye32 = torch.eye(3, dtype=torch.float32)

        for name, expected_k in groups:
            with self.subTest(group=name):
                spec = parse_symmetry(name)
                self.assertEqual(spec.order, expected_k)

                r64 = spec.rotations(dtype=torch.float64)
                r32 = spec.rotations(dtype=torch.float32)
                self.assertEqual(r64.dtype, torch.float64)
                self.assertEqual(r32.dtype, torch.float32)
                self.assertEqual(tuple(r64.shape), (expected_k, 3, 3))
                self.assertEqual(tuple(r32.shape), (expected_k, 3, 3))

                # 1. Canonical identity at index 0
                self.assertLess((r64[0] - eye64).abs().max().item(), 1e-14)
                self.assertLess((r32[0] - eye32).abs().max().item(), 1e-7)

                # 2. Orthonormality R^T R = I and R R^T = I
                rtr64 = torch.einsum("gji,gjk->gik", r64, r64)
                rrt64 = torch.einsum("gij,gkj->gik", r64, r64)
                self.assertLess((rtr64 - eye64).abs().max().item(), 1e-13)
                self.assertLess((rrt64 - eye64).abs().max().item(), 1e-13)
                rtr32 = torch.einsum("gji,gjk->gik", r32, r32)
                self.assertLess((rtr32 - eye32).abs().max().item(), 1e-6)

                # 3. Proper rotations det(R) = +1 (no improper reflections/inversions)
                dets64 = torch.linalg.det(r64)
                dets32 = torch.linalg.det(r32)
                self.assertLess((dets64 - 1.0).abs().max().item(), 1e-13)
                self.assertLess((dets32 - 1.0).abs().max().item(), 1e-6)

                # 4. Pairwise distinctness
                if expected_k > 1:
                    pairwise = torch.linalg.norm(r64.unsqueeze(1) - r64.unsqueeze(0), dim=(-2, -1))
                    off_diag = pairwise + torch.eye(expected_k, dtype=torch.float64) * 100.0
                    self.assertGreater(off_diag.min().item(), 0.1)

                # 5. Group closure + Latin-square Cayley table
                cayley = _cayley_table(r64)
                expected_perm = list(range(expected_k))
                for idx in range(expected_k):
                    self.assertEqual(sorted(cayley[idx].tolist()), expected_perm)
                    self.assertEqual(sorted(cayley[:, idx].tolist()), expected_perm)

                # 6. Inverse closure: R_g^T is in G and multiplies to R_0 = I
                inv_diffs = torch.linalg.norm(r64.transpose(-1, -2).unsqueeze(1) - r64.unsqueeze(0), dim=(-2, -1))
                inv_matches = inv_diffs < 1e-11
                self.assertTrue(torch.all(inv_matches.sum(dim=-1) == 1))
                inv_idx = inv_matches.float().argmax(dim=-1)
                for g in range(expected_k):
                    self.assertEqual(int(cayley[g, inv_idx[g]].item()), 0)
                    self.assertEqual(int(cayley[inv_idx[g], g].item()), 0)

    def test_polyhedral_and_dihedral_element_order_spectra(self) -> None:
        from rfproteina.symmetry import parse_symmetry

        # Tetrahedral T (order 12): 1 identity, 3 C2 (180 deg), 8 C3 (120 deg)
        self.assertEqual(_element_orders(parse_symmetry("T").rotations(dtype=torch.float64)), Counter({1: 1, 2: 3, 3: 8}))

        # Octahedral O (order 24): 1 identity, 9 C2, 8 C3, 6 C4
        self.assertEqual(
            _element_orders(parse_symmetry("O").rotations(dtype=torch.float64)),
            Counter({1: 1, 2: 9, 3: 8, 4: 6}),
        )

        # Icosahedral I (order 60): 1 identity, 15 C2, 20 C3, 24 C5
        self.assertEqual(
            _element_orders(parse_symmetry("I").rotations(dtype=torch.float64)),
            Counter({1: 1, 2: 15, 3: 20, 5: 24}),
        )

        # Dihedral D5 (order 10): second half are 5 distinct C2 involutions
        d5 = parse_symmetry("D5").rotations(dtype=torch.float64)
        self.assertEqual(_element_orders(d5), Counter({1: 1, 5: 4, 2: 5}))

    def test_reynolds_fixed_point_projector_and_collision_free_radial_direction(self) -> None:
        from rfproteina.symmetry import parse_symmetry

        # Fix(Cn) = span(e_z) for n >= 2
        expected_cn_proj = torch.diag(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        for n in (2, 3, 4, 5, 6):
            spec = parse_symmetry(f"C{n}")
            r = spec.rotations(dtype=torch.float64)
            reynolds = r.mean(dim=0)
            self.assertLess((reynolds - expected_cn_proj).abs().max().item(), 1e-13)

        # Fix(G) = {0} for Dn (n >= 2), T, O, I
        for g_name in ("D2", "D3", "D4", "D5", "T", "O", "I"):
            spec = parse_symmetry(g_name)
            r = spec.rotations(dtype=torch.float64)
            reynolds = r.mean(dim=0)
            self.assertLess(reynolds.abs().max().item(), 1e-13)

        # Canonical radial direction u_G has unit norm and avoids ALL non-identity rotation axes
        for g_name in ("C2", "C3", "C6", "D2", "D3", "D4", "D6", "T", "O", "I"):
            spec = parse_symmetry(g_name)
            u = spec.radial_dir(dtype=torch.float64)
            self.assertAlmostEqual(torch.linalg.norm(u).item(), 1.0, places=12)
            r = spec.rotations(dtype=torch.float64)
            orbit = torch.einsum("gij,j->gi", r, u)  # [K, 3]
            pairwise_orbit = torch.linalg.norm(orbit.unsqueeze(1) - orbit.unsqueeze(0), dim=-1)
            off_diag = pairwise_orbit + torch.eye(spec.order, dtype=torch.float64) * 100.0
            self.assertGreater(
                off_diag.min().item(),
                0.15,
                msg=f"Radial direction for {g_name} lies too close to a symmetry axis (orbit collision)!",
            )

    def test_polyhedral_invariant_vertex_sets(self) -> None:
        from rfproteina.symmetry import parse_symmetry

        # Tetrahedron vertices {(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)} invariant under T
        tet_verts = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]], dtype=torch.float64
        )
        r_t = parse_symmetry("T").rotations(dtype=torch.float64)
        for g in range(12):
            rotated = tet_verts @ r_t[g].T
            dists = torch.cdist(rotated, tet_verts)
            self.assertLess(dists.min(dim=-1).values.max().item(), 1e-13)

        # Octahedron vertices {+-e_x, +-e_y, +-e_z} invariant under O
        oct_verts = torch.cat([torch.eye(3, dtype=torch.float64), -torch.eye(3, dtype=torch.float64)], dim=0)
        r_o = parse_symmetry("O").rotations(dtype=torch.float64)
        for g in range(24):
            rotated = oct_verts @ r_o[g].T
            dists = torch.cdist(rotated, oct_verts)
            self.assertLess(dists.min(dim=-1).values.max().item(), 1e-13)

        # Golden-box icosahedron vertices cyc(0, +-1, +-phi) invariant under I
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        ico_list = []
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                ico_list.append([0.0, s1, s2 * phi])
                ico_list.append([s2 * phi, 0.0, s1])
                ico_list.append([s1, s2 * phi, 0.0])
        ico_verts = torch.tensor(ico_list, dtype=torch.float64)
        r_i = parse_symmetry("I").rotations(dtype=torch.float64)
        for g in range(60):
            rotated = ico_verts @ r_i[g].T
            dists = torch.cdist(rotated, ico_verts)
            self.assertLess(dists.min(dim=-1).values.max().item(), 1e-12)

    def test_symmetry_config_and_multi_letter_chain_ids(self) -> None:
        from rfproteina.symmetry import SymmetryConfig, get_symmetry_chain_ids

        cfg_inactive = SymmetryConfig.from_any(None)
        self.assertFalse(cfg_inactive.is_active)
        self.assertEqual(cfg_inactive.order, 1)

        cfg_c1 = SymmetryConfig.from_any("C1")
        self.assertFalse(cfg_c1.is_active)
        self.assertEqual(cfg_c1.order, 1)

        cfg_str = SymmetryConfig.from_any("d3")
        self.assertTrue(cfg_str.is_active)
        self.assertEqual(cfg_str.order, 6)
        self.assertEqual(cfg_str.spec.name, "D3")

        cfg_dict = SymmetryConfig.from_any(
            {"group": "I", "radius": 25.0, "input_is_asu": True, "tie_sequence": True, "decode_mode": "asu_medoid"}
        )
        self.assertTrue(cfg_dict.is_active)
        self.assertEqual(cfg_dict.order, 60)
        self.assertAlmostEqual(cfg_dict.radius_nm, 2.5, places=7)
        roundtrip = SymmetryConfig.from_any(cfg_dict.to_dict())
        self.assertEqual(roundtrip.order, 60)
        self.assertAlmostEqual(roundtrip.radius, 25.0, places=7)

        # Chain IDs up to 120 chains (e.g. Icosahedral hetero-dimer 60 * 2 = 120)
        sig = inspect.signature(get_symmetry_chain_ids)
        if len(sig.parameters) == 1:
            ids_60 = list(get_symmetry_chain_ids(60))
            ids_120 = list(get_symmetry_chain_ids(120))
        else:
            ids_60 = list(get_symmetry_chain_ids(60))
            ids_120 = [cid for sub in get_symmetry_chain_ids(60, 2) for cid in (sub if isinstance(sub, tuple | list) else [sub])]

        self.assertEqual(len(ids_60), 60)
        self.assertEqual(len(set(ids_60)), 60)
        self.assertEqual(ids_60[:3], ["A", "B", "C"])
        self.assertEqual(ids_60[25], "Z")
        self.assertEqual(ids_60[26], "AA")
        self.assertTrue(all(cid.isalpha() and cid.isupper() for cid in ids_60))
        self.assertEqual(len(ids_120), 120)
        self.assertEqual(len(set(ids_120)), 120)


if __name__ == "__main__":
    unittest.main()
