"""Real CIF boundary and child-process lifecycle tests for the experiment runner."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray, BondList, concatenate
from biotite.structure.io.pdb import PDBFile

from scripts.prepare_surface_experiment import apply_ligand_formal_charges
from scripts.run_surface_experiment import analyse, audit_output, audit_retained_ligand_chemistry, execute, read_atoms


class TestCampaignAudit:
    @pytest.mark.parametrize("corruption", ["phosphate", "element", "duplicate", "missing"])
    def test_protected_histidine_heavy_atom_chemistry_is_preserved(
        self,
        protected_histidine_atoms: AtomArray,
        campaign_manifest: dict[str, object],
        tmp_path: Path,
        corruption: str,
    ) -> None:
        reference = protected_histidine_atoms
        campaign_manifest["residues"][0]["name"] = "HIS"
        generated = reference.copy()
        if corruption == "phosphate":
            phosphate = generated[-4:].copy()
            phosphate.atom_name[:] = ["P", "OP1", "OP2", "O3P"]
            phosphate.element[:] = ["P", "O", "O", "O"]
            generated = concatenate([generated, phosphate])
        elif corruption == "element":
            generated.element[generated.atom_name == "ND1"] = "C"
        elif corruption == "duplicate":
            generated = concatenate([generated, generated[-1:]])
        else:
            generated = generated[generated.atom_name != "ND1"]
        # Renaming the phosphonohistidine back to HIS does not repair chemistry.
        path = tmp_path / f"protected_histidine_{corruption}.pdb"
        output = PDBFile()
        output.set_structure(generated)
        output.write(path)
        with pytest.raises(ValueError, match="[Pp]rotected.*heavy.atom"):
            audit_output(path, campaign_manifest, reference)

    def test_eligible_mutation_may_change_heavy_atom_inventory(
        self, indexed_surface_atoms: AtomArray, campaign_manifest: dict[str, object], tmp_path: Path
    ) -> None:
        generated = indexed_surface_atoms.copy()
        generated.res_name[4:8] = "THR"
        sidechain = generated[4:7].copy()
        sidechain.atom_name[:] = ["CB", "OG1", "CG2"]
        sidechain.element[:] = ["C", "O", "C"]
        output = PDBFile()
        output.set_structure(concatenate([generated, sidechain]))
        path = tmp_path / "eligible_threonine.pdb"
        output.write(path)
        result = audit_output(path, campaign_manifest, indexed_surface_atoms)
        assert result["n_surface_mutations"] == 1

    def test_actual_cif_roundtrip_and_protected_identity_gate(
        self, indexed_surface_atoms: AtomArray, campaign_manifest: dict[str, object], tmp_path: Path
    ) -> None:
        changed = indexed_surface_atoms.copy()
        changed.res_name[4:8] = "THR"
        path = tmp_path / "generated.cif"
        to_cif_file(changed, path)
        result = audit_output(path, campaign_manifest, indexed_surface_atoms)
        assert result["n_surface_mutations"] == 1
        assert result["surface_fraction_THR"] == 1.0
        changed.res_name[8:] = "ALA"
        bad = tmp_path / "bad.cif"
        to_cif_file(changed, bad)
        with pytest.raises(ValueError, match="protected"):
            audit_output(bad, campaign_manifest, indexed_surface_atoms)

    def test_writer_insertion_code_loss_cannot_alias_two_residues(
        self, surface_atoms: AtomArray, campaign_manifest: dict[str, object], tmp_path: Path
    ) -> None:
        path = tmp_path / "unsafe.cif"
        to_cif_file(surface_atoms.copy(), path)
        with pytest.raises(ValueError, match="Mixed identities|mappings differ"):
            audit_output(path, campaign_manifest, surface_atoms)

    def test_timeout_kills_the_worker_and_its_descendant(self, tmp_path: Path) -> None:
        marker = tmp_path / "escaped.txt"
        grandchild = f"import time,pathlib; time.sleep(1); pathlib.Path({str(marker)!r}).write_text('escaped')"
        parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{grandchild!r}]); time.sleep(10)"
        code, error = execute([sys.executable, "-c", parent], tmp_path, tmp_path / "child.log", 0.2)
        assert code == 124 and error
        time.sleep(1.2)
        assert not marker.exists(), "A child process escaped the GPU-budget timeout"

    def test_missing_retained_ligand_is_rejected(
        self, indexed_surface_atoms: AtomArray, campaign_manifest: dict[str, object], tmp_path: Path
    ) -> None:
        ligand = AtomArray(1)
        ligand.chain_id[:] = "L"
        ligand.res_id[:] = 1
        ligand.res_name[:] = "ZMT"
        ligand.atom_name[:] = "CX"
        ligand.element[:] = "C"
        ligand.coord[:] = [10.0, 2.0, 1.0]
        reference = concatenate([indexed_surface_atoms, ligand])
        missing = tmp_path / "missing_ligand.cif"
        to_cif_file(indexed_surface_atoms, missing)
        with pytest.raises(ValueError, match="ligand.*inventory"):
            audit_output(missing, campaign_manifest, reference)

    def test_pdb_reference_comparator_preserves_the_same_audit_contract(
        self, indexed_surface_atoms: AtomArray, campaign_manifest: dict[str, object], tmp_path: Path
    ) -> None:
        path = tmp_path / "comparator.pdb"
        output = PDBFile()
        output.set_structure(indexed_surface_atoms)
        output.write(path)
        result = audit_output(path, campaign_manifest, indexed_surface_atoms)
        assert result["n_surface_mutations"] == 0
        assert result["passes_geometry_gate"] == 1

    def test_ligand_inventory_uses_heavy_atoms_across_protonation_formats(
        self, indexed_surface_atoms: AtomArray, campaign_manifest: dict[str, object], tmp_path: Path
    ) -> None:
        indexed_surface_atoms = indexed_surface_atoms.copy()
        # Three noncollinear CAs determine a unique alignment for ligand drift.
        indexed_surface_atoms.coord[9, 2] += 3.0
        water = AtomArray(3)
        water.chain_id[:] = "L"
        water.res_id[:] = 1
        water.res_name[:] = "HOH"
        water.hetero[:] = True
        water.atom_name[:] = ["O", "H1", "H2"]
        water.element[:] = ["O", "H", "H"]
        water.coord[:] = [[10, 2, 1], [10.7, 2.5, 1], [9.3, 2.5, 1]]
        reference = concatenate([indexed_surface_atoms, water[:1]])
        output = PDBFile()
        output.set_structure(concatenate([indexed_surface_atoms, water]))
        path = tmp_path / "hydrogenated.pdb"
        output.write(path)
        result = audit_output(path, campaign_manifest, reference)
        assert result["n_surface_mutations"] == 0
        assert result["ligand_all_atom_rmsd_A"] < 0.001

    def test_histidine_alias_is_normalized_before_pdb_chain_classification(
        self, indexed_surface_atoms: AtomArray, campaign_manifest: dict[str, object], tmp_path: Path
    ) -> None:
        reference = indexed_surface_atoms.copy()
        reference.res_name[:4] = "HIS"
        campaign_manifest["residues"][0]["name"] = "HIS"
        generated = reference.copy()
        generated.res_name[:4] = "HIP"
        generated.hetero[:4] = True
        output = PDBFile()
        output.set_structure(generated)
        path = tmp_path / "histidine_alias.pdb"
        output.write(path)
        result = audit_output(path, campaign_manifest, reference)
        assert result["n_surface_mutations"] == 0
        assert result["passes_geometry_gate"] == 1

    def test_formal_ligand_charge_is_written_to_atom_site_and_reloaded(self, tmp_path: Path) -> None:
        ligand = AtomArray(2)
        ligand.chain_id[:] = "L"
        ligand.res_id[:] = 1
        ligand.res_name[:] = "ZMT"
        ligand.hetero[:] = True
        ligand.atom_name[:] = ["CX", "OA"]
        ligand.element[:] = ["C", "O"]
        ligand.coord[:] = [[0, 0, 0], [1.4, 0, 0]]
        ligand.bonds = BondList(2, np.array([[0, 1, 1]], dtype=np.uint32))
        params = tmp_path / "ligand.params"
        params.write_text("CHARGE OA FORMAL -1\n")
        apply_ligand_formal_charges(ligand, params)
        output = tmp_path / "charged.cif"
        to_cif_file(ligand, output, ccd_entries={"ZMT": ligand})
        loaded = read_atoms(output)
        assert loaded.charge[loaded.atom_name == "OA"].tolist() == [-1]

    def test_changed_ligand_formal_charge_is_a_hard_failure(
        self, indexed_surface_atoms: AtomArray, campaign_manifest: dict[str, object], tmp_path: Path
    ) -> None:
        ligand = AtomArray(1)
        ligand.chain_id[:] = "L"
        ligand.res_id[:] = 1
        ligand.res_name[:] = "ZMT"
        ligand.atom_name[:] = "OA"
        ligand.element[:] = "O"
        ligand.coord[:] = [10, 2, 1]
        ligand.set_annotation("charge", np.array([-1]))
        indexed_surface_atoms = indexed_surface_atoms.copy()
        indexed_surface_atoms.set_annotation("charge", np.zeros(len(indexed_surface_atoms), dtype=int))
        reference = concatenate([indexed_surface_atoms, ligand])
        generated = reference.copy()
        generated.charge[:] = 0
        path = tmp_path / "lost_charge.cif"
        to_cif_file(generated, path, ccd_entries={"ZMT": generated[-1:]})
        with pytest.raises(ValueError, match="formal charge"):
            audit_output(path, campaign_manifest, reference)


class TestRetainedLigandChemistry:
    def test_missing_covalent_link_or_changed_element_cannot_pass(self, indexed_surface_atoms: AtomArray) -> None:
        ligand = AtomArray(2)
        ligand.chain_id[:] = "L"
        ligand.res_id[:] = 1
        ligand.res_name[:] = "ZMT"
        ligand.atom_name[:] = ["CX", "OA"]
        ligand.element[:] = ["C", "O"]
        reference = concatenate([indexed_surface_atoms, ligand])
        reference.bonds.add_bond(12, 13, 1)
        reference.bonds.add_bond(2, 12, 1)
        audit_retained_ligand_chemistry(reference, reference.copy())
        missing = reference.copy()
        missing.bonds.remove_bond(2, 12)
        with pytest.raises(ValueError, match="bond"):
            audit_retained_ligand_chemistry(reference, missing)
        changed = reference.copy()
        changed.element[13] = "C"
        with pytest.raises(ValueError, match="element"):
            audit_retained_ligand_chemistry(reference, changed)


class TestPartialGenerationAdmission:
    def test_one_saved_structure_does_not_complete_a_two_output_job(
        self, indexed_surface_atoms: AtomArray, campaign_manifest: dict[str, object], tmp_path: Path
    ) -> None:
        prepared = tmp_path / "prepared/fixture"
        prepared.mkdir(parents=True)
        (prepared / "manifest.json").write_text(json.dumps(campaign_manifest))
        to_cif_file(indexed_surface_atoms, prepared / "reference.cif")
        job = tmp_path / "output/fixture/inverse/baseline"
        samples = job / "inference_samples"
        samples.mkdir(parents=True)
        to_cif_file(indexed_surface_atoms, samples / "one_generated.cif")
        (job / "receipt.json").write_text(
            json.dumps(
                {"target": "fixture", "mode": "inverse", "arm": "baseline", "requested": 2, "exit_code": 0, "seed": 1}
            )
        )
        rows = analyse(prepared.parent, tmp_path / "output")
        assert len(rows) == 1 and rows[0]["status"] == "valid"
        assert rows[0]["generation_output_complete"] is False
        assert rows[0]["generation_output_count"] == 1
