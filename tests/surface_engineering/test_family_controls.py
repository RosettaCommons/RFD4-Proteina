"""Family-backed preferences preserve exact MSA profiles and unselected residues."""

import json
from pathlib import Path

import numpy as np
import pytest
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray

from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS
from scripts.add_surface_controls import extend
from scripts.run_surface_experiment import read_atoms


class TestFamilyConditioning:
    def test_family_profiles_are_tilted_only_at_eligible_residues(
        self, bond_valid_surface_atoms: AtomArray, campaign_manifest: dict[str, object], tmp_path: Path
    ) -> None:
        target = tmp_path / "fixture"
        target.mkdir()
        baseline = target / "baseline.cif"
        to_cif_file(bond_valid_surface_atoms, baseline)
        original = baseline.read_bytes()
        manifest = dict(campaign_manifest, arms=[], deferred={})
        (target / "manifest.json").write_text(json.dumps(manifest))
        profile = {"ASP": 0.2, "SER": 0.1, "GLU": 0.0}
        msa = tmp_path / "msa.json"
        msa.write_text(
            json.dumps(
                {
                    "target": "fixture",
                    "profiles": [{"key": r["key"], "probabilities": profile} for r in manifest["residues"]],
                }
            )
        )
        extend(target, msa, family_bias_only=True)
        atoms = read_atoms(target / "msa_xtr.cif")
        asp = next(c for c in SEQUENCE_PROBABILITY_CONDITIONS if c.token_name == "ASP")
        ser = next(c for c in SEQUENCE_PROBABILITY_CONDITIONS if c.token_name == "SER")
        values = np.asarray(asp.annotation(atoms, default="generate"))
        assert np.isnan(values[:4]).all() and np.isnan(values[8:]).all()
        assert values[4] / np.asarray(ser.annotation(atoms))[4] == pytest.approx(2 * np.exp(2), rel=1e-6)
        glu = next(c for c in SEQUENCE_PROBABILITY_CONDITIONS if c.token_name == "GLU")
        assert np.asarray(glu.annotation(atoms))[4] == 0
        assert (target / "msa_completed.cif").exists()
        assert baseline.read_bytes() == original
        assert (target / "msa_notion.cif").exists()
