"""Tests for Slice B: symmetric flow matching, partial diffusion, guidance, decode tying, BeamSearch, BioIR."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from tests.symmetry import WORKTREE_ROOT, assert_worktree_imports, build_tiny_cpu_proteina

INPUT_1PRW_CIF = WORKTREE_ROOT / "data" / "benchmarks" / "motif_scaffolding" / "motif_data_aa" / "1prw_aa.cif.zst"


def _make_flow_cfg(latent_dim: int = 8) -> object:
    """Create a realistic ProductSpaceFlowMatcher config matching 2026_10_19_hero_pretrain.yaml."""
    return OmegaConf.create(
        {
            "loss": {"signed_volume_weight": 0.0},
            "product_flowmatcher": {
                "bb_ca": {
                    "zero_com_noise": True,
                    "guidance_enabled": True,
                    "dim": 3,
                    "stochastic_centering_mode": "x_0",
                    "stochastic_centering_scale": 1.5,
                    "stochastic_centering_scope": "per_group",
                    "stochastic_centering_group_keys": ["molecule_iid"],
                },
                "local_latents": {
                    "zero_com_noise": False,
                    "guidance_enabled": True,
                    "dim": latent_dim,
                    "stochastic_centering_mode": "none",
                    "stochastic_centering_scale": None,
                },
            },
            "t_distribution": {
                "bb_ca": {"name": "uniform", "p1": 0.0, "p2": 1.0},
                "local_latents": {"name": "uniform", "p1": 0.0, "p2": 1.0},
            },
        }
    )


def _make_sampling_model_args(sampling_mode: str = "vf_tsr") -> dict:
    gt_mode = "ode" if sampling_mode == "vf" else "1/t"
    return {
        "bb_ca": {
            "schedule": {"mode": "uniform", "p": 1.0},
            "gt": {"mode": gt_mode, "p": 1.0, "clamp_val": 4.0},
            "simulation_step_params": {
                "sampling_mode": sampling_mode,
                "sc_scale_noise": 0.4,
                "sc_scale_score": 1.0,
                "t_lim_ode": 0.98,
                "t_lim_ode_below": 0.02,
                "center_every_step": True,
                "tsr_k": 1.0,
                "tsr_sigma": 1.0,
            },
        },
        "local_latents": {
            "schedule": {"mode": "uniform", "p": 1.0},
            "gt": {"mode": "ode", "p": 1.0, "clamp_val": 4.0},
            "simulation_step_params": {
                "sampling_mode": "vf",
                "sc_scale_noise": 0.0,
                "sc_scale_score": 1.0,
                "t_lim_ode": 1.0,
                "t_lim_ode_below": 0.0,
                "center_every_step": False,
                "tsr_k": 1.0,
                "tsr_sigma": 1.0,
            },
        },
    }


def _make_synthetic_batch(b: int, k: int, l_asu: int, latent_dim: int = 8, n_pad_extra: int = 0) -> dict:
    """Create a synthetic flow-matcher batch with distinct molecule_iid per chain."""
    n_valid = k * l_asu
    n_total = n_valid + n_pad_extra
    padding_mask = torch.zeros(b, n_total, dtype=torch.bool)
    padding_mask[:, :n_valid] = True
    mol_iid = torch.full((b, n_total), -1, dtype=torch.long)
    for g in range(k):
        mol_iid[:, g * l_asu : (g + 1) * l_asu] = g
    return {
        "padding_mask": padding_mask,
        "is_in_context_conditioning": torch.zeros(b, n_total, dtype=torch.bool),
        "supports_local_latent": padding_mask.clone(),
        "is_resolved_mask": padding_mask.clone(),
        "coord_cond_mask": torch.zeros(b, n_total, dtype=torch.bool),
        "molecule_iid": mol_iid,
        "chain_iid": mol_iid.clone(),
        "x_1": {
            "bb_ca": torch.zeros(b, n_total, 3),
            "local_latents": torch.zeros(b, n_total, latent_dim),
        },
    }


def _asymmetric_mock_predictor(
    batch: dict, guidance_flag: bool = True, recycle_depth: int = 1, mode: str = "full", **kwargs: object
) -> dict:
    """Deliberately break symmetry in predicted velocities and sequence logits to test projection."""
    x_t_ca = batch["x_t"]["bb_ca"]
    x_t_lat = batch["x_t"]["local_latents"]
    b, n, _ = x_t_ca.shape
    token_idx = torch.arange(n, device=x_t_ca.device, dtype=x_t_ca.dtype).view(1, n, 1)
    # Asymmetric velocity depending explicitly on global token index (differs across subunits!)
    asym_ca_bias = torch.cat(
        [torch.sin(0.7 * token_idx + 0.3), torch.cos(1.1 * token_idx - 0.2), 0.5 * torch.sin(1.9 * token_idx)],
        dim=-1,
    )
    v_ca = -0.4 * x_t_ca + (0.6 if guidance_flag else 0.3) * asym_ca_bias

    lat_idx = torch.arange(x_t_lat.shape[-1], device=x_t_lat.device, dtype=x_t_lat.dtype).view(1, 1, -1)
    v_lat = -0.3 * x_t_lat + torch.sin(0.5 * token_idx + 0.3 * lat_idx)

    # Asymmetric sequence logits [b, n, 147]
    vocab = torch.arange(147, device=x_t_ca.device, dtype=x_t_ca.dtype).view(1, 1, 147)
    seq_logits = torch.cos(0.4 * token_idx * (1.0 + 0.05 * vocab)) + 0.2 * torch.randn(
        b, n, 147, device=x_t_ca.device
    )

    return {
        "bb_ca": {"v": v_ca},
        "local_latents": {"v": v_lat},
        "sequence": {"logits": seq_logits},
    }


class TestSymmetricFlowSampling(unittest.TestCase):
    """Tests for symmetric flow trajectory integration, decode tying, BeamSearch, and BioIR guard."""

    def setUp(self) -> None:
        assert_worktree_imports()
        torch.manual_seed(2026)
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="sym_sampling_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_full_simulation_symmetry_across_groups_and_modes(self) -> None:
        """Asymmetric predictor + per-molecule stochastic offsets STILL yield exact G-symmetric bb_ca."""
        from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
        from rfproteina.symmetry import SymmetryConfig, compute_symmetry_rmsd, parse_symmetry

        fm = ProductSpaceFlowMatcher(_make_flow_cfg(latent_dim=8))

        test_cases = [
            ("C3", "vf_tsr", 12.0),
            ("D2", "vf", 10.0),
            ("T", "sc", 15.0),
            ("O", "vf_tsr", 18.0),
            ("I", "vf", 22.0),
        ]
        for group_name, mode, radius_ang in test_cases:
            with self.subTest(group=group_name, mode=mode):
                spec = parse_symmetry(group_name)
                k = spec.order
                l_asu = 3 if k >= 24 else 5
                sym_cfg = SymmetryConfig(group=group_name, radius=radius_ang, tie_sequence=True)
                batch = _make_synthetic_batch(b=2, k=k, l_asu=l_asu, latent_dim=8, n_pad_extra=2)

                samples, info = fm.full_simulation(
                    batch=batch,
                    predict_for_sampling=_asymmetric_mock_predictor,
                    recycle_depths=[1, 1, 1, 1],
                    nsteps=4,
                    nsamples=2,
                    n=k * l_asu + 2,
                    sampling_model_args=_make_sampling_model_args(mode),
                    device=torch.device("cpu"),
                    save_trajectory_every=1,
                    dump_noisy_trajectories=True,
                    guidance_w=1.5,
                    symmetry=sym_cfg,
                )

                n_valid = k * l_asu
                bb_ca_valid = samples["bb_ca"][:, :n_valid, :]
                bb_ca_pad = samples["bb_ca"][:, n_valid:, :]

                # Padded tail stays strictly zero
                self.assertEqual(bb_ca_pad.abs().max().item(), 0.0)

                # Global CoM is zero on valid tokens
                self.assertLess(bb_ca_valid.mean(dim=1).abs().max().item(), 1e-5)

                # Exact G-symmetry on final bb_ca (< 1e-5 nm = 1e-4 Å)
                rmsd = compute_symmetry_rmsd(bb_ca_valid, spec)
                max_rmsd = rmsd.max().item() if torch.is_tensor(rmsd) else float(rmsd)
                self.assertLess(max_rmsd, 1e-5)

                # Check every intermediate snapshot (both x_1 and noisy x_t)
                self.assertEqual(len(info["trajectory_steps"]), 4)
                for snap in info["trajectory_steps"]:
                    snap_x1 = snap["x_1"]["bb_ca"][:, :n_valid, :]
                    snap_xt = snap["x_t"]["bb_ca"][:, :n_valid, :]
                    r1 = compute_symmetry_rmsd(snap_x1, spec)
                    rt = compute_symmetry_rmsd(snap_xt, spec)
                    self.assertLess(r1.max().item() if torch.is_tensor(r1) else float(r1), 1e-5)
                    self.assertLess(rt.max().item() if torch.is_tensor(rt) else float(rt), 1e-5)

                # CRITICAL CONTRACT (#9): local_latents are NOT averaged across subunits!
                lat_grouped = samples["local_latents"][:, :n_valid, :].view(2, k, l_asu, 8)
                lat_diff_01 = (lat_grouped[:, 0] - lat_grouped[:, 1]).abs().max().item()
                self.assertGreater(lat_diff_01, 1e-3, "local_latents must NOT be averaged across subunits!")

                # Sequence logits ARE tied across subunits when tie_sequence=True
                seq_logits = info["sequence_logits"][:, :n_valid, :].view(2, k, l_asu, 147)
                seq_diff = (seq_logits - seq_logits[:, :1]).abs().max().item()
                self.assertLess(seq_diff, 1e-5, "sequence_logits must be tied across subunits when tie_sequence=True")

    def test_untied_sequence_leaves_sequence_logits_distinct(self) -> None:
        from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
        from rfproteina.symmetry import SymmetryConfig, compute_symmetry_rmsd, parse_symmetry

        fm = ProductSpaceFlowMatcher(_make_flow_cfg(latent_dim=8))
        spec = parse_symmetry("C3")
        sym_cfg = SymmetryConfig(group="C3", radius=8.0, tie_sequence=False)
        batch = _make_synthetic_batch(b=1, k=3, l_asu=4, latent_dim=8)

        samples, info = fm.full_simulation(
            batch=batch,
            predict_for_sampling=_asymmetric_mock_predictor,
            recycle_depths=[1, 1, 1],
            nsteps=3,
            nsamples=1,
            n=12,
            sampling_model_args=_make_sampling_model_args("vf"),
            device=torch.device("cpu"),
            symmetry=sym_cfg,
        )
        rmsd = compute_symmetry_rmsd(samples["bb_ca"], spec)
        self.assertLess(rmsd.max().item() if torch.is_tensor(rmsd) else float(rmsd), 1e-5)

        seq_logits = info["sequence_logits"].view(1, 3, 4, 147)
        self.assertGreater((seq_logits[:, 0] - seq_logits[:, 1]).abs().max().item(), 1e-3)

    def test_external_guidance_asymmetric_delta_is_projected(self) -> None:
        """External differentiable guidance deltas on bb_ca are projected onto symmetric zero-CoM subspace."""
        from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
        from rfproteina.symmetry import SymmetryConfig, compute_symmetry_rmsd, parse_symmetry

        fm = ProductSpaceFlowMatcher(_make_flow_cfg(latent_dim=8))
        spec = parse_symmetry("D3")
        sym_cfg = SymmetryConfig(group="D3", radius=10.0, tie_sequence=True)
        batch = _make_synthetic_batch(b=1, k=6, l_asu=4, latent_dim=8)

        def _asym_guidance(x_1_pred: dict, extra_info: dict) -> dict:
            # Push only subunit 0 in +x/+z direction (violates D3 and shifts CoM!)
            delta_ca = torch.zeros_like(x_1_pred["bb_ca"])
            delta_ca[:, :4, 0] = 0.75
            delta_ca[:, :4, 2] = 0.50
            return {"bb_ca": delta_ca, "local_latents": 0.1 * torch.randn_like(x_1_pred["local_latents"])}

        samples, _ = fm.full_simulation(
            batch=batch,
            predict_for_sampling=_asymmetric_mock_predictor,
            recycle_depths=[1, 1, 1],
            nsteps=3,
            nsamples=1,
            n=24,
            sampling_model_args=_make_sampling_model_args("vf_tsr"),
            device=torch.device("cpu"),
            guidance_model=_asym_guidance,
            symmetry=sym_cfg,
        )
        rmsd = compute_symmetry_rmsd(samples["bb_ca"], spec)
        self.assertLess(rmsd.max().item() if torch.is_tensor(rmsd) else float(rmsd), 1e-5)
        self.assertLess(samples["bb_ca"].mean(dim=1).abs().max().item(), 1e-5)

    def test_partial_diffusion_corrupt_batch_and_partial_simulation(self) -> None:
        """corrupt_batch + partial_simulation preserve exact symmetry without per-chain offset breakage."""
        from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
        from rfproteina.search.vanilla import build_partial_schedule
        from rfproteina.symmetry import SymmetryConfig, compute_symmetry_rmsd, expand_asu, parse_symmetry

        fm = ProductSpaceFlowMatcher(_make_flow_cfg(latent_dim=8))
        spec = parse_symmetry("D2")
        sym_cfg = SymmetryConfig(group="D2", radius=10.0, tie_sequence=True)
        batch = _make_synthetic_batch(b=2, k=4, l_asu=5, latent_dim=8)
        clean_asu = torch.randn(2, 5, 3)
        clean_full = expand_asu(clean_asu, spec)
        clean_full = clean_full - clean_full.mean(dim=1, keepdim=True)
        batch["x_1"]["bb_ca"] = clean_full

        t_dict = {"bb_ca": torch.full((2,), 0.45), "local_latents": torch.full((2,), 0.45)}
        corrupted = fm.corrupt_batch(batch, t=t_dict, symmetry=sym_cfg)

        for key in ("x_0", "x_t"):
            r = compute_symmetry_rmsd(corrupted[key]["bb_ca"], spec)
            self.assertLess(r.max().item() if torch.is_tensor(r) else float(r), 1e-5)

        sampling_args = _make_sampling_model_args("vf_tsr")
        partial_ts, partial_gt = build_partial_schedule({"bb_ca": 0.45, "local_latents": 0.45}, 3, fm, sampling_args)
        sim_params = {dm: sampling_args[dm]["simulation_step_params"] for dm in fm.data_modes}

        x_out, x1_out, _ = fm.partial_simulation(
            batch=corrupted,
            x=corrupted["x_t"],
            predict_for_sampling=_asymmetric_mock_predictor,
            recycle_depths=[1, 1, 1],
            nsteps=3,
            ts=partial_ts,
            gt=partial_gt,
            simulation_step_params=sim_params,
            device=torch.device("cpu"),
            symmetry=sym_cfg,
        )
        for tsr in (x_out["bb_ca"], x1_out["bb_ca"]):
            r = compute_symmetry_rmsd(tsr, spec)
            self.assertLess(r.max().item() if torch.is_tensor(r) else float(r), 1e-5)

    def test_proteina_decode_tying_vanilla_and_beam_search_and_heterogeneous_batch(self) -> None:
        """Real tiny CPU Proteina + AutoEncoder: Vanilla, BeamSearch, decode modes, and mixed C3+D2 batch."""
        from rfproteina.datapipes.collate import collate_batch
        from rfproteina.datapipes.pipeline import build_design_validation_pipeline
        from rfproteina.inference_utils.datasets import ConditionSpecDataset
        from rfproteina.search.beam_search import BeamSearchStrategy
        from rfproteina.symmetry import compute_symmetry_rmsd, parse_symmetry
        from rfproteina.symmetry.transforms import ExpandSymmetricASUTransform

        model = build_tiny_cpu_proteina(self.tmp_dir / "tiny_store")
        inf_cfg = OmegaConf.merge(
            model.cfg_exp.generation,
            OmegaConf.create(
                {
                    "args": {
                        "nsteps": 3,
                        "condition_decoder_on_sequence_prediction": True,
                        "save_trajectory_every": 0,
                    }
                }
            ),
        )
        model.configure_inference(inf_cfg=inf_cfg, store_dir=str(self.tmp_dir / "out"), is_inference=True)

        spec_yaml = self.tmp_dir / "het_batch.yaml"
        spec_yaml.write_text(
            yaml.safe_dump(
                {
                    "ex_c3": {
                        "input": str(INPUT_1PRW_CIF),
                        "contig": "A16-19",
                        "symmetry": {"group": "C3", "radius": 10.0},
                    },
                    "ex_d2": {
                        "input": str(INPUT_1PRW_CIF),
                        "contig": "A16-18",
                        "symmetry": {"group": "D2", "radius": 10.0, "decode_mode": "asu_first"},
                    },
                }
            )
        )
        pipe = build_design_validation_pipeline(
            metrics_tags=None,
            sampling_transforms=[ExpandSymmetricASUTransform()],
        )
        ds = ConditionSpecDataset(data=spec_yaml, transform=pipe, name="het")

        # 1. Heterogeneous 2-row collated batch (Row 0: C3 len 12, Row 1: D2 len 12)
        collated = collate_batch([ds[0], ds[1]], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
        gen_list = model.predict_step(collated, batch_idx=0)
        self.assertEqual(len(gen_list), 2)

        for row_idx, g_name in enumerate(("C3", "D2")):
            spec = parse_symmetry(g_name)
            k = spec.order
            gen_aa = gen_list[row_idx]["generated_atom_array"]
            # Check all K subunits have identical atom counts, residue names, and pure-R_g coordinates!
            self.assertEqual(gen_aa.array_length() % k, 0)
            n_sub_atoms = gen_aa.array_length() // k
            res_0 = gen_aa.res_name[:n_sub_atoms]
            c0 = torch.from_numpy(gen_aa.coord[:n_sub_atoms].astype(np.float64))
            r_mats = spec.rotations(dtype=torch.float64)
            for g in range(k):
                sub_res = gen_aa.res_name[g * n_sub_atoms : (g + 1) * n_sub_atoms]
                np.testing.assert_array_equal(sub_res, res_0)
                cg = torch.from_numpy(gen_aa.coord[g * n_sub_atoms : (g + 1) * n_sub_atoms].astype(np.float64))
                expected_cg = c0 @ r_mats[g].T
                self.assertLess((cg - expected_cg).abs().max().item(), 1e-3)

        # 2. BeamSearchStrategy on C3 (branching at step 2 of 4, no external metrics required)
        beam_inf_cfg = OmegaConf.merge(inf_cfg, OmegaConf.create({"args": {"nsteps": 4}}))
        model.configure_inference(inf_cfg=beam_inf_cfg, store_dir=str(self.tmp_dir / "beam_out"), is_inference=True)
        beam = BeamSearchStrategy(checkpoints={2: {"rewards": {}}}, n_branch=2, beam_width=2)
        model.configure_search(beam)

        single_c3 = collate_batch([ds[0]], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
        beam_results = model.predict_step(single_c3, batch_idx=0)
        self.assertGreaterEqual(len(beam_results), 1)
        spec_c3 = parse_symmetry("C3")
        r_c3 = spec_c3.rotations(dtype=torch.float64)
        for cand in beam_results:
            aa = cand["generated_atom_array"]
            n_sub = aa.array_length() // 3
            c0 = torch.from_numpy(aa.coord[:n_sub].astype(np.float64))
            for g in range(3):
                cg = torch.from_numpy(aa.coord[g * n_sub : (g + 1) * n_sub].astype(np.float64))
                self.assertLess((cg - c0 @ r_c3[g].T).abs().max().item(), 1e-3)
                np.testing.assert_array_equal(aa.res_name[g * n_sub : (g + 1) * n_sub], aa.res_name[:n_sub])

    def test_configure_bioir_rejects_active_symmetry(self) -> None:
        """configure_bioir raises ValueError('Group symmetry requires inference_backend=native') when symmetry > C1."""
        from types import SimpleNamespace

        from rfproteina.nn.execution import configure_bioir

        dummy_model = SimpleNamespace(
            guidance_model=None,
            inf_cfg={"args": {"guidance_w": 1.0, "save_trajectory_every": 0, "dump_noisy_trajectories": False}},
        )

        # Top-level symmetry="C3"
        with self.assertRaisesRegex(ValueError, "Group symmetry requires inference_backend=native"):
            configure_bioir(dummy_model, {"symmetry": "C3"})

        # Nested generation.symmetry.group="O"
        with self.assertRaisesRegex(ValueError, "Group symmetry requires inference_backend=native"):
            configure_bioir(dummy_model, {"generation": {"symmetry": {"group": "O"}}})

        # Model inf_cfg symmetry="I"
        dummy_model_sym = SimpleNamespace(
            guidance_model=None,
            inf_cfg={
                "args": {"guidance_w": 1.0, "save_trajectory_every": 0, "dump_noisy_trajectories": False},
                "symmetry": {"group": "I"},
            },
        )
        with self.assertRaisesRegex(ValueError, "Group symmetry requires inference_backend=native"):
            configure_bioir(dummy_model_sym, {})


if __name__ == "__main__":
    unittest.main()
