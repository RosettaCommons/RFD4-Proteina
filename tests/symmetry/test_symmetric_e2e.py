"""End-to-end oligomer generation & mmCIF re-parse verification (CPU tiny model + GPU real checkpoint)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from atomworks.io.utils.io_utils import load_any
from biotite.structure import get_residue_starts

try:
    import pytest
except ImportError:  # pragma: no cover - O2 venv runs unittest without pytest installed
    pytest = None  # type: ignore[assignment]

from tests.symmetry import WORKTREE_ROOT, assert_worktree_imports, build_tiny_cpu_proteina


def _gpu_mark(cls: type) -> type:
    return pytest.mark.gpu(cls) if pytest is not None else cls

BENCHMARKS_DIR = WORKTREE_ROOT / "data" / "benchmarks"
MONOMER_SHORT_CIF = BENCHMARKS_DIR / "monomer" / "monomer-short.cif.zst"
INPUT_1PRW_CIF = BENCHMARKS_DIR / "motif_scaffolding" / "motif_data_aa" / "1prw_aa.cif.zst"


def _find_generated_cifs(out_dir: str | Path) -> list[Path]:
    from rfproteina.utils.structure_files import iter_structure_files

    return sorted(iter_structure_files(Path(out_dir) / "inference_samples", stem_suffix="_generated"))


def _verify_oligomer_cif_symmetry(
    cif_path: str | Path,
    group_name: str,
    expected_l_asu: int | None = None,
    atol_angstrom: float = 2e-3,
    rmsd_tol_angstrom: float = 5e-4,
    check_identical_sequence: bool = True,
) -> dict[str, float | int]:
    """Re-parse an exported mmCIF and verify exact 3D point-group symmetry across all K chains."""
    from rfproteina.symmetry import get_symmetry_chain_ids, parse_symmetry

    aa = load_any(str(cif_path), model=1)
    if hasattr(aa, "stack_depth"):
        aa = aa[0]
    spec = parse_symmetry(group_name)
    k = spec.order
    r_mats = spec.rotations(dtype=torch.float64).numpy()

    if aa.array_length() == 0:
        raise AssertionError(f"Parsed empty AtomArray from {cif_path}")
    if not np.isfinite(aa.coord).all():
        raise AssertionError(f"Non-finite coordinates found in {cif_path}")
    if aa.array_length() % k != 0:
        raise AssertionError(f"Total atom count {aa.array_length()} not divisible by |G|={k} for {group_name}")

    starts = get_residue_starts(aa)
    n_res_total = len(starts)
    if n_res_total % k != 0:
        raise AssertionError(f"Total residue count {n_res_total} not divisible by |G|={k} for {group_name}")
    l_asu = n_res_total // k
    if expected_l_asu is not None and l_asu != expected_l_asu:
        raise AssertionError(f"Expected L_asu={expected_l_asu}, got {l_asu} in {cif_path}")

    # Verify unique chain IDs (including multi-character base-26 IDs up to K=60 for I)
    chain_ids_per_res = aa.chain_id[starts]
    unique_chains = list(dict.fromkeys(chain_ids_per_res.tolist()))
    expected_chains = get_symmetry_chain_ids(k)
    if unique_chains != expected_chains:
        raise AssertionError(f"Chain IDs {unique_chains[:10]}... != expected {expected_chains[:10]}... (len {len(unique_chains)} vs {k})")

    n_atoms_sub = aa.array_length() // k
    coord_0 = aa.coord[:n_atoms_sub].astype(np.float64)
    res_names_0 = aa.res_name[starts[:l_asu]]
    atom_names_0 = aa.atom_name[:n_atoms_sub]

    max_abs_dev = 0.0
    max_sub_rmsd = 0.0
    for g in range(k):
        sub_slice = slice(g * n_atoms_sub, (g + 1) * n_atoms_sub)
        res_slice = slice(g * l_asu, (g + 1) * l_asu)

        if check_identical_sequence:
            if not np.array_equal(aa.res_name[starts[res_slice]], res_names_0):
                raise AssertionError(f"Subunit {g} residue sequence differs from ASU in {cif_path}")
            if not np.array_equal(aa.atom_name[sub_slice], atom_names_0):
                raise AssertionError(f"Subunit {g} atom names differ from ASU in {cif_path}")

        coord_g = aa.coord[sub_slice].astype(np.float64)
        # Note: mmCIF serializes Cartn_x/y/z to 3 decimal places (0.001 Å precision)
        expected_g = coord_0 @ r_mats[g].T
        diff = coord_g - expected_g
        sub_max = float(np.abs(diff).max())
        sub_rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))
        max_abs_dev = max(max_abs_dev, sub_max)
        max_sub_rmsd = max(max_sub_rmsd, sub_rmsd)

    if max_abs_dev > atol_angstrom:
        raise AssertionError(
            f"{group_name} CIF {cif_path}: max pure-R_g coordinate deviation {max_abs_dev:.6f} Å > {atol_angstrom} Å"
        )
    if max_sub_rmsd > rmsd_tol_angstrom:
        raise AssertionError(
            f"{group_name} CIF {cif_path}: max pure-R_g subunit RMSD {max_sub_rmsd:.6f} Å > {rmsd_tol_angstrom} Å"
        )

    return {
        "order": k,
        "l_asu": l_asu,
        "n_atoms_total": int(aa.array_length()),
        "max_abs_dev_ang": max_abs_dev,
        "max_rmsd_ang": max_sub_rmsd,
    }


def _find_real_checkpoint_overrides() -> list[str] | None:
    """Locate real RFD4 checkpoints from HF cache/bundle or native .ckpt paths and return Hydra overrides."""
    snapshot_candidates: list[Path] = []
    explicit_hf = os.environ.get("RFD4_HF_CKPT_PATH")
    if explicit_hf:
        snapshot_candidates.append(Path(explicit_hf))

    hf_homes: list[Path] = []
    if os.environ.get("HF_HOME"):
        hf_homes.append(Path(os.environ["HF_HOME"]))
    hf_homes.append(Path("/n/scratch/users/k/kg120/hf_home"))
    hf_homes.append(Path.home() / ".cache" / "huggingface")
    for hf_home in hf_homes:
        snaps_dir = hf_home / "hub" / "models--nvidia--RFD4-Proteina" / "snapshots"
        if snaps_dir.is_dir():
            snapshot_candidates.extend(sorted(p for p in snaps_dir.iterdir() if p.is_dir()))

    env_dir = os.environ.get("RFD4_CKPT_DIR") or os.environ.get("COMPLEXA_WEIGHTS_DIR")
    if env_dir:
        snapshot_candidates.append(Path(env_dir))
    snapshot_candidates.append(Path("/home/kettner/complexa_weights/proteina_complexa_v1.0"))

    for d in snapshot_candidates:
        if not d.is_dir():
            continue
        # 1. Portable HF safetensors bundle (config.json + model.safetensors)
        if (d / "config.json").is_file() and (d / "model.safetensors").is_file():
            return [f"hf_ckpt_path={d}"]
        # 2. Release/HF snapshot EMA .ckpt layout (flow_checkpoints/*.ckpt + ae_checkpoints/*.ckpt)
        flow_dir = d / "flow_checkpoints"
        ae_dir = d / "ae_checkpoints"
        if flow_dir.is_dir() and ae_dir.is_dir():
            flow_ckpts = sorted(flow_dir.glob("*.ckpt"))
            ae_ckpts = sorted(ae_dir.glob("*.ckpt"))
            if flow_ckpts and ae_ckpts:
                flow_ckpt = next((p for p in flow_ckpts if "rfd4_proteina_flow" in p.name), flow_ckpts[0])
                ae_ckpt = next((p for p in ae_ckpts if "rfd4_proteina_ae" in p.name), ae_ckpts[0])
                return [
                    "hf_ckpt_path=null",
                    f"ckpt_path={flow_ckpt.parent}",
                    f"ckpt_name={flow_ckpt.name}",
                    f"autoencoder_ckpt_path={ae_ckpt}",
                ]
        # 3. Local Complexa training checkpoint pair (complexa.ckpt + complexa_ae.ckpt)
        if (d / "complexa.ckpt").is_file() and (d / "complexa_ae.ckpt").is_file():
            return [
                "hf_ckpt_path=null",
                f"ckpt_path={d}",
                "ckpt_name=complexa.ckpt",
                f"autoencoder_ckpt_path={d / 'complexa_ae.ckpt'}",
            ]
    return None


class TestSymmetricEndToEndCPU(unittest.TestCase):
    """Fast CPU end-to-end generation + mmCIF re-parse tests across C3, D2, T, and 60-chain I."""

    @classmethod
    def setUpClass(cls) -> None:
        assert_worktree_imports()
        cls._shared_tmp = Path(tempfile.mkdtemp(prefix="sym_e2e_shared_"))
        cls.tiny_model = build_tiny_cpu_proteina(cls._shared_tmp / "tiny_model_store")
        cls.lyz_cif = INPUT_1PRW_CIF

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._shared_tmp, ignore_errors=True)

    def setUp(self) -> None:
        self.out_dir = Path(tempfile.mkdtemp(prefix="sym_e2e_run_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_yaml_condition_spec_e2e_c3_d2_t_and_icosahedral_60_chains(self) -> None:
        """Full run_generation via rfd4_proteina.generate on C3, D2, T, and 60-chain I from YAML condition_spec."""
        import rfd4_proteina

        spec_dict = {
            "global_args": {"input": str(self.lyz_cif)},
            "homomer_c3": {"contig": "A16-20", "symmetry": "C3", "sym_radius": 12.0},
            "homomer_d2": {"contig": "A16-19", "symmetry": {"group": "D2", "radius": 11.0}},
            "cage_t": {"contig": "A16-19", "symmetry": {"group": "T", "radius": 16.0}},
            "capsid_i60": {"contig": "A16-18", "symmetry": {"group": "I", "radius": 24.0}},
        }
        spec_yaml = self.out_dir / "sym_designs.yaml"
        spec_yaml.write_text(yaml.safe_dump(spec_dict, sort_keys=False))

        run_out = self.out_dir / "outputs_yaml"
        rfd4_proteina.generate(
            model=self.tiny_model,
            overrides=[
                "hf_ckpt_path=null",
                f"inputs={spec_yaml}",
                f"out_dir={run_out}",
                "hardware.accelerator=cpu",
                "precision=32-true",
                "compile_lengths=null",
                "~metrics",
                "metrics_tags=[]",
                "num_workers=0",
                "batch_size=1",
                "compress_outputs=false",
                "generation.args.nsteps=3",
                "generation.n_recycle_schedule=null",
                "generation.model.bb_ca.schedule.tail_steps=0",
                "generation.model.local_latents.schedule.tail_steps=0",
                "seed=101",
            ],
        )

        cifs = _find_generated_cifs(run_out)
        self.assertEqual(len(cifs), 4, f"Expected 4 generated CIFs, found {[str(p) for p in cifs]}")
        by_name = {p.parent.name: p for p in cifs}

        expected = {
            "homomer_c3_0": ("C3", 5),
            "homomer_d2_0": ("D2", 4),
            "cage_t_0": ("T", 4),
            "capsid_i60_0": ("I", 3),
        }
        for key, (grp, l_asu) in expected.items():
            # Match either exact `<id>_0` or `<id>` depending on replicate naming
            match = by_name.get(key) or next((v for k, v in by_name.items() if k.startswith(key.removesuffix("_0"))), None)
            self.assertIsNotNone(match, f"Missing generated CIF for {key}; got {list(by_name)}")
            stats = _verify_oligomer_cif_symmetry(
                match,
                group_name=grp,
                expected_l_asu=l_asu,
                # mmCIF coordinates have 0.001 Å fixed-point rounding; 2e-3 Å max / 1e-3 Å RMSD verifies exact R_g
                atol_angstrom=2.5e-3,
                rmsd_tol_angstrom=1.2e-3,
            )
            if grp == "I":
                self.assertEqual(stats["order"], 60)

    def test_cli_symmetry_override_and_symmetric_oligomer_preset(self) -> None:
        """Top-level symmetry=C4 sym_radius=14.0 and inference_experiment=symmetric_oligomer work end-to-end."""
        import rfd4_proteina

        # Single-example monomer YAML without per-example symmetry -> uses CLI override + preset!
        mono_yaml = self.out_dir / "mono_asu.yaml"
        mono_yaml.write_text(
            yaml.safe_dump({"c4_cli": {"input": str(self.lyz_cif), "contig": "A16-20"}}, sort_keys=False)
        )

        run_out = self.out_dir / "outputs_preset"
        rfd4_proteina.generate(
            model=self.tiny_model,
            overrides=[
                "hf_ckpt_path=null",
                "inference_experiment=symmetric_oligomer",
                f"inputs={mono_yaml}",
                f"out_dir={run_out}",
                "symmetry=C4",
                "sym_radius=14.0",
                "hardware.accelerator=cpu",
                "precision=32-true",
                "compile_lengths=null",
                "~metrics",
                "metrics_tags=[]",
                "num_workers=0",
                "compress_outputs=false",
                "generation.args.nsteps=3",
                "generation.n_recycle_schedule=null",
                "generation.model.bb_ca.schedule.tail_steps=0",
                "generation.model.local_latents.schedule.tail_steps=0",
                "seed=202",
            ],
        )

        cifs = _find_generated_cifs(run_out)
        self.assertEqual(len(cifs), 1)
        stats = _verify_oligomer_cif_symmetry(
            cifs[0],
            group_name="C4",
            expected_l_asu=5,
            atol_angstrom=2.5e-3,
            rmsd_tol_angstrom=1.2e-3,
        )
        self.assertEqual(stats["order"], 4)

    def test_asymmetric_monomer_regression_c1_default(self) -> None:
        """Default symmetry=null / C1 monomer generation works with zero regressions."""
        import rfd4_proteina

        mono_yaml = self.out_dir / "mono_c1.yaml"
        mono_yaml.write_text(
            yaml.safe_dump({"mono_c1": {"input": str(self.lyz_cif), "contig": "A16-21"}}, sort_keys=False)
        )
        run_out = self.out_dir / "outputs_c1"
        rfd4_proteina.generate(
            model=self.tiny_model,
            overrides=[
                "hf_ckpt_path=null",
                f"inputs={mono_yaml}",
                f"out_dir={run_out}",
                "hardware.accelerator=cpu",
                "precision=32-true",
                "compile_lengths=null",
                "~metrics",
                "metrics_tags=[]",
                "num_workers=0",
                "compress_outputs=false",
                "generation.args.nsteps=3",
                "generation.n_recycle_schedule=null",
                "generation.model.bb_ca.schedule.tail_steps=0",
                "generation.model.local_latents.schedule.tail_steps=0",
                "seed=303",
            ],
        )
        cifs = _find_generated_cifs(run_out)
        self.assertEqual(len(cifs), 1)
        stats = _verify_oligomer_cif_symmetry(cifs[0], group_name="C1", expected_l_asu=6)
        self.assertEqual(stats["order"], 1)


@_gpu_mark
@unittest.skipUnless(
    torch.cuda.is_available() and _find_real_checkpoint_overrides() is not None,
    "Requires CUDA GPU and real RFD4 checkpoints (run on O2 GPU node).",
)
class TestRealCheckpointSymmetricGPU(unittest.TestCase):
    """Real-checkpoint GPU end-to-end symmetry smoke tests (C3, D2, T, O, I) for Harvard O2 execution."""

    @classmethod
    def setUpClass(cls) -> None:
        assert_worktree_imports()
        found = _find_real_checkpoint_overrides()
        assert found is not None
        cls.ckpt_overrides = found
        cls._shared_tmp = Path(tempfile.mkdtemp(prefix="sym_real_gpu_"))
        cls.lyz_cif = INPUT_1PRW_CIF

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._shared_tmp, ignore_errors=True)

    def test_real_checkpoint_gpu_point_groups_c3_d2_t_o_i(self) -> None:
        import rfd4_proteina

        out_dir = self._shared_tmp / "real_gpu_runs"
        spec_yaml = self._shared_tmp / "real_gpu_sym.yaml"
        # Note: 1prw_aa.cif.zst has contiguous residues 16..35 on chain A.
        spec_yaml.write_text(
            yaml.safe_dump(
                {
                    "global_args": {"input": str(self.lyz_cif)},
                    "real_c3": {"contig": "A16-35", "symmetry": {"group": "C3", "radius": 14.0}},
                    "real_d2": {"contig": "A16-31", "symmetry": {"group": "D2", "radius": 15.0}},
                    "real_t": {"contig": "A16-27", "symmetry": {"group": "T", "radius": 18.0}},
                    "real_o": {"contig": "A16-25", "symmetry": {"group": "O", "radius": 22.0}},
                    "real_i": {"contig": "A16-21", "symmetry": {"group": "I", "radius": 26.0}},
                },
                sort_keys=False,
            )
        )

        rfd4_proteina.generate(
            overrides=[
                *self.ckpt_overrides,
                "inference_experiment=symmetric_oligomer",
                f"inputs={spec_yaml}",
                f"out_dir={out_dir}",
                "precision=32-true",
                "compile_lengths=null",
                "nn.attention_type=torch",
                "nn.tri_mult_backend=torch",
                "~metrics",
                "metrics_tags=[]",
                "num_workers=0",
                "batch_size=1",
                "compress_outputs=false",
                "generation.args.nsteps=20",
                "generation.n_recycle_schedule=null",
                "generation.model.bb_ca.schedule.tail_steps=0",
                "generation.model.local_latents.schedule.tail_steps=0",
                "seed=42",
            ],
        )

        cifs = _find_generated_cifs(out_dir)
        self.assertEqual(len(cifs), 5)
        by_name = {p.parent.name: p for p in cifs}
        for prefix, grp, l_asu in (
            ("real_c3", "C3", 20),
            ("real_d2", "D2", 16),
            ("real_t", "T", 12),
            ("real_o", "O", 10),
            ("real_i", "I", 6),
        ):
            match = next((v for k, v in by_name.items() if k.startswith(prefix)), None)
            self.assertIsNotNone(match, f"Missing CIF for {prefix}")
            _verify_oligomer_cif_symmetry(
                match,
                group_name=grp,
                expected_l_asu=l_asu,
                atol_angstrom=2.5e-3,
                rmsd_tol_angstrom=1.2e-3,
            )


if __name__ == "__main__":
    unittest.main()
