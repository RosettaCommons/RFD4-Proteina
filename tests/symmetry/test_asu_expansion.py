"""Integration tests for ASU expansion, entity-ID tying, condition_spec YAML, and CIF export frame."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from atomworks.io.utils.atom_array_plus import as_atom_array_plus, concatenate_atom_array_plus
from atomworks.io.utils.io_utils import load_any
from atomworks.ml.conditions import C_CRD
from atomworks.ml.utils.token import get_token_count, get_token_starts

from tests.symmetry import WORKTREE_ROOT, assert_worktree_imports

BENCHMARKS_DIR = WORKTREE_ROOT / "data" / "benchmarks"
MONOMER_SHORT_CIF = BENCHMARKS_DIR / "monomer" / "monomer-short.cif.zst"
INPUT_1PRW_CIF = BENCHMARKS_DIR / "motif_scaffolding" / "motif_data_aa" / "1prw_aa.cif.zst"
INPUT_1YCR_CIF = BENCHMARKS_DIR / "motif_scaffolding" / "motif_data_aa" / "1ycr_aa.cif.zst"


class TestASUExpansionAndPipeline(unittest.TestCase):
    """Tests for Slice C: ExpandSymmetricASUTransform, pipeline entity tying, condition_spec, export."""

    def setUp(self) -> None:
        assert_worktree_imports()
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="sym_asu_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_condition_spec_accepts_symmetry_and_sym_radius_not_legacy(self) -> None:
        """Trap Fix 2: YAML/JSON condition_spec with symmetry & sym_radius is NOT rejected as legacy."""
        from rfproteina.inference_utils.condition_spec import is_legacy_dialect, split_global_args

        spec_str_sym = {
            "global_args": {"input": str(INPUT_1PRW_CIF), "sym_radius": 12.5},
            "c3_homomer": {"contig": "2-4, A16-19", "symmetry": "C3"},
            "d2_mapping": {"contig": "A16-20", "symmetry": {"group": "D2", "radius": 10.0, "tie_sequence": True}},
        }
        self.assertFalse(is_legacy_dialect(spec_str_sym))
        merged = split_global_args(spec_str_sym)
        self.assertEqual(merged["c3_homomer"]["symmetry"], "C3")
        self.assertEqual(merged["c3_homomer"]["sym_radius"], 12.5)
        self.assertEqual(merged["d2_mapping"]["symmetry"]["group"], "D2")

        # True legacy dialect keys (e.g. select_fixed_atoms, unindex) must still be rejected
        legacy_spec = {
            "old_job": {"input": str(INPUT_1PRW_CIF), "contig": "A16-20", "select_fixed_atoms": "A16-18"},
        }
        self.assertTrue(is_legacy_dialect(legacy_spec))

    def test_expand_symmetric_asu_variable_contig_and_motif_rotation(self) -> None:
        """Variable-length contig realizes ONE L_asu, replicates K times, rotates coords/C_CRD, splits C_CRDGRP."""
        from rfproteina.conditions import C_CENTROID, C_CRDGRP
        from rfproteina.datapipes.pipeline import (
            AddSrcComponentAnnotation,
            RealizeExpandableSegments,
            build_design_validation_pipeline,
        )
        from rfproteina.inference_utils.datasets import ConditionSpecDataset
        from rfproteina.symmetry import SymmetryConfig, get_symmetry_chain_ids, parse_symmetry
        from rfproteina.symmetry.transforms import ExpandSymmetricASUTransform

        spec_dict = {
            "c3_var_motif": {
                "input": str(INPUT_1PRW_CIF),
                # Variable scaffold flanking a 4-residue motif on chain A (L_asu in [8, 16])
                "contig": "2-6, A16-19, 2-6",
                "symmetry": "C3",
                "conditions": {
                    "C_CRD": {True: [{"select": "res_id >= 16 and res_id <= 17"}]},
                    "C_SEQ": {True: [{"select": "res_id >= 16 and res_id <= 19"}]},
                },
            }
        }
        spec_yaml = self.tmp_dir / "c3_var_motif.yaml"
        spec_yaml.write_text(yaml.safe_dump(spec_dict))

        sym_cfg = SymmetryConfig(group="C3", radius=0.0, input_is_asu=True, tie_sequence=True)
        pipeline = build_design_validation_pipeline(
            metrics_tags=None,
            sampling_transforms=[ExpandSymmetricASUTransform(sym_cfg)],
            expandable_segment_rng=np.random.default_rng(7),
        )
        ds = ConditionSpecDataset(data=spec_yaml, transform=pipeline, name="sym_test")
        item = ds[0]

        aa = item["atom_array"]
        k = 3
        total_tokens = get_token_count(aa)
        self.assertEqual(total_tokens % k, 0)
        l_asu = total_tokens // k
        self.assertGreaterEqual(l_asu, 8)
        self.assertLessEqual(l_asu, 16)

        starts = get_token_starts(aa)
        chain_ids_per_token = aa.chain_id[starts]
        expected_chain_ids = get_symmetry_chain_ids(k)
        for g in range(k):
            block_chains = np.unique(chain_ids_per_token[g * l_asu : (g + 1) * l_asu])
            self.assertEqual(list(block_chains), [expected_chain_ids[g]])

        # Verify R_g rotation of CA / atom coordinates and finite C_CRD annotations across subunits
        spec = parse_symmetry("C3")
        r = spec.rotations(dtype=torch.float64).numpy()
        atoms_per_sub = aa.array_length() // k
        self.assertEqual(aa.array_length(), k * atoms_per_sub)

        coord_0 = aa.coord[:atoms_per_sub]
        crd_annot = C_CRD.annotation(aa, default="generate")
        crd_mask_0 = ~np.isnan(crd_annot[:atoms_per_sub]).any(axis=-1)
        self.assertTrue(crd_mask_0.any())

        for g in range(k):
            sub_slice = slice(g * atoms_per_sub, (g + 1) * atoms_per_sub)
            expected_coord_g = coord_0 @ r[g].T
            valid_c = ~np.isnan(coord_0).any(axis=-1)
            if valid_c.any():
                np.testing.assert_allclose(aa.coord[sub_slice][valid_c], expected_coord_g[valid_c], atol=1e-4)

            expected_crd_g = crd_annot[:atoms_per_sub][crd_mask_0] @ r[g].T
            np.testing.assert_allclose(crd_annot[sub_slice][crd_mask_0], expected_crd_g, atol=1e-4)

        # Verify extra_info["symmetry"] is recorded
        self.assertIn("symmetry", item.get("extra_info", {}))
        parsed_back = SymmetryConfig.from_any(item["extra_info"]["symmetry"])
        self.assertEqual(parsed_back.group, "C3")
        self.assertEqual(parsed_back.order, 3)

        # Also check C_CENTROID rotation and positive C_CRDGRP separation directly on ExpandSymmetricASUTransform
        raw_ds = ConditionSpecDataset(data=spec_yaml, transform=None, name="raw_sym")
        raw_item = raw_ds[0]
        raw_aa = as_atom_array_plus(raw_item["atom_array"].copy())
        crdgrp = C_CRDGRP.annotation(raw_aa, default="generate").copy()
        non_abs = np.where(crdgrp != C_CRDGRP.KNOWN_ABSOLUTE_COORDS)[0]
        crdgrp[non_abs[:5]] = 1
        C_CRDGRP.set_annotation(raw_aa, crdgrp)
        centroid = np.tile(np.array([[4.0, -2.0, 7.5]], dtype=np.float32), (raw_aa.array_length(), 1))
        C_CENTROID.set_annotation(raw_aa, centroid)

        staged = AddSrcComponentAnnotation()({"atom_array": raw_aa, "example_id": "test_c3", "extra_info": {}})
        staged = RealizeExpandableSegments(rng=np.random.default_rng(11))(staged)
        expanded = ExpandSymmetricASUTransform(sym_cfg)(staged)
        exp_aa = expanded["atom_array"]
        n_asu_atoms = exp_aa.array_length() // k

        exp_crdgrp = C_CRDGRP.annotation(exp_aa, default="generate")
        pos_groups_per_sub = [
            set(
                np.unique(
                    exp_crdgrp[g * n_asu_atoms : (g + 1) * n_asu_atoms][
                        exp_crdgrp[g * n_asu_atoms : (g + 1) * n_asu_atoms] > 0
                    ]
                ).tolist()
            )
            for g in range(k)
        ]
        for g in range(k):
            self.assertEqual(len(pos_groups_per_sub[g]), 1)
            for h in range(g + 1, k):
                self.assertTrue(pos_groups_per_sub[g].isdisjoint(pos_groups_per_sub[h]))

        exp_centroid = C_CENTROID.annotation(exp_aa, default="generate")
        c0 = exp_centroid[:n_asu_atoms]
        c0_valid = ~np.isnan(c0).any(axis=-1) & ~(c0 == 0).all(axis=-1)
        if c0_valid.any():
            for g in range(k):
                cg = exp_centroid[g * n_asu_atoms : (g + 1) * n_asu_atoms]
                np.testing.assert_allclose(cg[c0_valid], c0[c0_valid] @ r[g].T, atol=1e-4)

    def test_icosahedral_60_chain_ids_and_entity_tying_homomer_vs_hetero_asu(self) -> None:
        """Trap Fix 1: Icosahedral 60-chain base-26 IDs + molecule_entity_id tying for 1-chain and 2-chain ASUs."""
        from rfproteina.datapipes.pipeline import build_design_validation_pipeline
        from rfproteina.inference_utils.datasets import ConditionSpecDataset
        from rfproteina.symmetry import SymmetryConfig, get_symmetry_chain_ids

        # 1. Icosahedral (K=60) homomer with L_asu=4 -> 240 tokens, 60 unique base-26 chains A..BH
        ico_yaml = self.tmp_dir / "ico.yaml"
        ico_yaml.write_text(
            yaml.safe_dump({"ico_60": {"input": str(INPUT_1PRW_CIF), "contig": "A16-19", "symmetry": "I"}})
        )

        ico_cfg = SymmetryConfig(group="I", radius=22.0, tie_sequence=True)
        ico_pipe = build_design_validation_pipeline(metrics_tags=None, symmetry=ico_cfg)
        ico_item = ConditionSpecDataset(data=ico_yaml, transform=ico_pipe, name="ico")[0]
        ico_aa = ico_item["atom_array"]
        self.assertEqual(get_token_count(ico_aa), 60 * 4)
        starts = get_token_starts(ico_aa)
        unique_chains_in_order = list(dict.fromkeys(ico_aa.chain_id[starts].tolist()))
        self.assertEqual(unique_chains_in_order, get_symmetry_chain_ids(60))

        # Trap Fix 1 check on masked-sequence homomer: all 240 tokens share ONE molecule_entity_id
        mol_entity = ico_item["feats"]["molecule_entity_id"]
        mol_entity_np = mol_entity.numpy() if torch.is_tensor(mol_entity) else np.asarray(mol_entity)
        self.assertEqual(len(np.unique(mol_entity_np)), 1, "All 60 tied homomer chains must share molecule_entity_id")

        # When tie_sequence=False, masked-sequence chains get private molecule_entity_ids
        untied_cfg = SymmetryConfig(group="C3", tie_sequence=False)
        untied_yaml = self.tmp_dir / "c3_untied.yaml"
        untied_yaml.write_text(
            yaml.safe_dump(
                {
                    "c3_u": {
                        "input": str(INPUT_1PRW_CIF),
                        "contig": "A16-19",
                        "symmetry": {"group": "C3", "tie_sequence": False},
                    }
                }
            )
        )
        untied_pipe = build_design_validation_pipeline(metrics_tags=None, symmetry=untied_cfg)
        untied_item = ConditionSpecDataset(data=untied_yaml, transform=untied_pipe, name="c3_u")[0]
        untied_ent = untied_item["feats"]["molecule_entity_id"]
        untied_ent_np = untied_ent.numpy() if torch.is_tensor(untied_ent) else np.asarray(untied_ent)
        self.assertEqual(len(np.unique(untied_ent_np)), 3, "tie_sequence=False should keep distinct entity IDs")

        # 2. Two-chain hetero-ASU (1prw chain A len 3 + 1ycr chain B len 5 -> L_asu=8) under C3 (K=3 -> 24 tokens, 6 chains)
        hetero_yaml = self.tmp_dir / "hetero_c3.yaml"
        hetero_yaml.write_text(
            yaml.safe_dump(
                {
                    "hetero_c3": {
                        "input": [str(INPUT_1PRW_CIF), str(INPUT_1YCR_CIF)],
                        "contig": "A16-18, /0, B19-23",
                        "symmetry": "C3",
                    }
                },
                sort_keys=False,
            )
        )
        c3_cfg = SymmetryConfig(group="C3", tie_sequence=True)
        c3_pipe = build_design_validation_pipeline(metrics_tags=None, symmetry=c3_cfg)
        hetero_item = ConditionSpecDataset(data=hetero_yaml, transform=c3_pipe, name="hetero_c3")[0]
        hetero_aa = hetero_item["atom_array"]
        self.assertEqual(get_token_count(hetero_aa), 3 * 8)

        # Contiguous ASU block layout [g*8 : (g+1)*8]: first 3 tokens share entity A, next 5 share entity B
        h_ent = hetero_item["feats"]["molecule_entity_id"]
        h_ent_np = (h_ent.numpy() if torch.is_tensor(h_ent) else np.asarray(h_ent)).reshape(3, 8)
        np.testing.assert_array_equal(h_ent_np[0], h_ent_np[1])
        np.testing.assert_array_equal(h_ent_np[0], h_ent_np[2])
        self.assertEqual(len(np.unique(h_ent_np[0, :3])), 1)
        self.assertEqual(len(np.unique(h_ent_np[0, 3:])), 1)
        self.assertNotEqual(h_ent_np[0, 0], h_ent_np[0, 3])

    def test_input_is_asu_false_preexpanded_validation(self) -> None:
        """input_is_asu=False validates N % K == 0 without re-expanding, and rejects non-divisible N."""
        from rfproteina.datapipes.pipeline import AddSrcComponentAnnotation, RealizeExpandableSegments
        from rfproteina.inference_utils.datasets import ConditionSpecDataset
        from rfproteina.symmetry import SymmetryConfig
        from rfproteina.symmetry.transforms import ExpandSymmetricASUTransform

        # 12 tokens: divisible by K=3, NOT divisible by K=5
        spec_yaml = self.tmp_dir / "preexp.yaml"
        spec_yaml.write_text(yaml.safe_dump({"pre12": {"input": str(INPUT_1PRW_CIF), "contig": "A16-27"}}))
        raw = ConditionSpecDataset(data=spec_yaml, transform=None, name="pre12")[0]

        def _stage(raw_entry: dict) -> dict:
            d = {
                "atom_array": as_atom_array_plus(raw_entry["atom_array"].copy()),
                "example_id": "pre12",
                "extra_info": {},
            }
            d = AddSrcComponentAnnotation()(d)
            return RealizeExpandableSegments(rng=np.random.default_rng(0))(d)

        ok_out = ExpandSymmetricASUTransform(SymmetryConfig(group="C3", input_is_asu=False))(_stage(raw))
        self.assertEqual(_count_tokens := int(len(np.unique(ok_out["atom_array"].res_id))), 12)
        self.assertEqual(SymmetryConfig.from_any(ok_out["extra_info"]["symmetry"]).group, "C3")

        with self.assertRaises(ValueError):
            ExpandSymmetricASUTransform(SymmetryConfig(group="C5", input_is_asu=False))(_stage(raw))

    def test_heterogeneous_collation_preserves_per_example_symmetry(self) -> None:
        """Collate C3 + D2 (and C3 + asymmetric) via ProteinaDataSample and verify per-row metadata survives."""
        from rfproteina.datapipes.collate import collate_batch
        from rfproteina.datapipes.pipeline import build_design_validation_pipeline
        from rfproteina.inference_utils.datasets import ConditionSpecDataset
        from rfproteina.symmetry.transforms import resolve_batch_symmetry_configs

        spec_yaml = self.tmp_dir / "mixed_sym.yaml"
        spec_yaml.write_text(
            yaml.safe_dump(
                {
                    "row_c3": {"input": str(INPUT_1PRW_CIF), "contig": "A16-20", "symmetry": "C3"},
                    "row_d2": {
                        "input": str(INPUT_1PRW_CIF),
                        "contig": "A16-19",
                        "symmetry": {"group": "D2", "radius": 11.0},
                    },
                    "row_c1": {"input": str(INPUT_1PRW_CIF), "contig": "A16-21"},
                }
            )
        )
        pipe = build_design_validation_pipeline(metrics_tags=None, symmetry=None)
        # Note: build_design_validation_pipeline activates ExpandSymmetricASUTransform when per-example extra_info has symmetry!
        from rfproteina.symmetry.transforms import ExpandSymmetricASUTransform

        pipe_per_example = build_design_validation_pipeline(
            metrics_tags=None,
            sampling_transforms=[ExpandSymmetricASUTransform()],
        )
        ds = ConditionSpecDataset(data=spec_yaml, transform=pipe_per_example, name="mixed")
        s0 = ds[ds.id_to_idx("row_c3_0")]
        s1 = ds[ds.id_to_idx("row_d2_0")]
        s2 = ds[ds.id_to_idx("row_c1_0")]
        self.assertEqual(s0["feats"]["residue_type"].shape[0], 15)  # 3 * 5
        self.assertEqual(s1["feats"]["residue_type"].shape[0], 16)  # 4 * 4
        self.assertEqual(s2["feats"]["residue_type"].shape[0], 6)  # 1 * 6

        # Collate two symmetric rows (all rows have symmetry -> dict-of-lists in batch["extra_info"])
        b_sym = collate_batch([s0, s1], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
        self.assertEqual(tuple(b_sym["padding_mask"].shape), (2, 16))
        self.assertEqual(int(b_sym["padding_mask"][0].sum().item()), 15)
        self.assertEqual(int(b_sym["padding_mask"][1].sum().item()), 16)
        cfgs_sym = resolve_batch_symmetry_configs(b_sym)
        self.assertEqual([c.group for c in cfgs_sym], ["C3", "D2"])

        # Collate symmetric + asymmetric row
        b_mix = collate_batch([s0, s2], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
        self.assertEqual(tuple(b_mix["padding_mask"].shape), (2, 15))
        self.assertEqual(int(b_mix["padding_mask"][0].sum().item()), 15)
        self.assertEqual(int(b_mix["padding_mask"][1].sum().item()), 6)
        cfgs_mix = resolve_batch_symmetry_configs(b_mix)
        self.assertEqual([c.group for c in cfgs_mix], ["C3", None])

    def test_structure_export_preserves_canonical_symmetry_axes_under_intcrd(self) -> None:
        """Trap Fix 3: write_generated_structure_cif must NOT rotate symmetric C_INTCRD designs off canonical z-axis."""
        from atomworks.io.parser import parse
        from atomworks.io.utils.io_utils import load_any

        from rfproteina.conditions import C_CRDGRP, C_INTCRD
        from rfproteina.symmetry import parse_symmetry
        from rfproteina.utils.standard_annotations import SA_CWI
        from rfproteina.utils.structure_export import write_generated_structure_cif

        base = parse(str(INPUT_1PRW_CIF))["asym_unit"][0]
        resolved = np.isfinite(base.coord).all(axis=-1)
        asu = as_atom_array_plus(base[(base.res_id >= 16) & (base.res_id <= 20) & resolved].copy())
        asu.coord = asu.coord - asu.coord.mean(axis=0, keepdims=True) + np.array([[12.0, 2.0, -1.5]], dtype=np.float32)

        spec = parse_symmetry("C3")
        r = spec.rotations(dtype=torch.float64).numpy()
        chains = []
        for g, cid in enumerate(("A", "B", "C")):
            sub = asu.copy()
            sub.chain_id = np.full(sub.array_length(), cid, dtype="U4")
            sub.coord = (asu.coord.astype(np.float64) @ r[g].T).astype(np.float32)
            C_CRDGRP.set_annotation(sub, np.full(sub.array_length(), g + 1, dtype=np.int16))
            tilted_input = sub.coord @ np.array(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
            ) + np.array([[45.0, -30.0, 80.0]], dtype=np.float32)
            C_INTCRD.set_annotation(sub, tilted_input)
            SA_CWI.set_annotation(sub, tilted_input)
            sub.set_annotation("sym_entity_group", np.zeros(sub.array_length(), dtype=np.int32))
            sub.set_annotation("symmetry_group", np.full(sub.array_length(), "C3", dtype="<U8"))
            chains.append(sub)

        oligomer = concatenate_atom_array_plus(chains)
        out_cif = self.tmp_dir / "c3_intcrd_generated.cif"
        write_generated_structure_cif(oligomer.copy(), out_cif)

        reloaded = load_any(str(out_cif), model=1)
        n_asu = asu.array_length()
        c0 = reloaded.coord[:n_asu].astype(np.float64)
        for g in range(3):
            cg = reloaded.coord[g * n_asu : (g + 1) * n_asu].astype(np.float64)
            expected_cg = c0 @ r[g].T
            max_dev = np.abs(cg - expected_cg).max()
            self.assertLess(
                max_dev,
                5e-3,
                f"Exported CIF lost canonical C3 z-axis symmetry (max deviation {max_dev:.4f} Å)!",
            )


if __name__ == "__main__":
    unittest.main()
