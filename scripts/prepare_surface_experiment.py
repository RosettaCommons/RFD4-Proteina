"""Author retained-layout surface experiments from audited reference manifests.

The ligand bond types and covalent attachment come from the archived Rosetta
8MT parameters, not a CCD lookup for the colliding residue name. Coordinate
and sequence conditions are checked through a real annotated-CIF round trip.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from atomworks.io import parse
from atomworks.io._loaders import load_pdb
from atomworks.io.config import PrepareConfig
from atomworks.io.parser import parse_atom_array
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.ccd import custom_ccd_residues
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from atomworks.ml.conditions.base import ConditionBase
from biotite.structure import AtomArray, BondList, BondType
from loguru import logger

from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS
from rfproteina.conditions_reporting import run_pipeline_validation
from rfproteina.surface_engineering import (
    AA_NAMES,
    NOTION_BIASES,
    XTR_BIASES,
    annotate_probabilities,
    atom_residue_keys,
    residue_keys,
    reweight_probabilities,
)
from rfproteina.transforms.sequence_conditioning import CanonicalizeSequenceProbabilityConditions
from rfproteina.utils.parse_presets import ANNOTATED_CIF
from rfproteina.utils.sequence_sampling import _UNIPROT_AA_PERCENT


def apply_ligand_formal_charges(atoms: AtomArray, params: Path, ligand_name: str = "ZMT") -> None:
    """Apply params formal charges to atom sites, not just CCD metadata."""
    charges = (
        np.asarray(atoms.charge).copy()
        if "charge" in atoms.get_annotation_categories()
        else np.zeros(len(atoms), dtype=int)
    )
    for line in params.read_text().splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0] == "CHARGE" and fields[2] == "FORMAL":
            selected = (atoms.res_name == ligand_name) & (atoms.atom_name == fields[1])
            if int(selected.sum()) != 1:
                raise ValueError(f"Formal-charge atom must map uniquely: {ligand_name}/{fields[1]}")
            charges[selected] = int(fields[3])
    atoms.set_annotation("charge", charges)


def ligand_template(atoms: AtomArray, params: Path) -> AtomArray:
    """Use complete named heavy-atom inventory and explicit aromatic bond orders."""
    ligand = atoms[atoms.res_name == "8MT"].copy()
    names = {str(n): i for i, n in enumerate(ligand.atom_name)}
    if len(names) != 22:
        raise ValueError("Expected one 22-heavy-atom 8MT molecule")
    bonds = []
    for line in params.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in {"BOND", "BOND_TYPE"} and parts[1] in names and parts[2] in names:
            order = int(parts[3]) if parts[0] == "BOND_TYPE" else 1
            bonds.append((names[parts[1]], names[parts[2]], int(BondType.AROMATIC) if order == 4 else order))
    ligand.bonds = BondList(len(ligand), np.asarray(bonds, dtype=np.uint32))
    ligand.res_name[:] = "ZMT"
    ligand.set_annotation("chem_comp_type", np.full(len(ligand), "NON-POLYMER"))
    ligand.set_annotation("charge", np.array([-1 if n == "OA" else 0 for n in ligand.atom_name], dtype=int))
    return ligand


# Any is confined to heterogeneous archived JSON manifests, audited against
# parsed atom inventories and native condition validation before model use.
def load_reference(manifest: dict[str, Any], out_dir: Path) -> AtomArray:
    """Canonicalise histidine heavy-atom aliases and preserve authored ligand chemistry."""
    raw = as_atom_array_plus(load_pdb(manifest["source"], model=1)[0])
    if "ins_code" in raw.get_annotation_categories() and np.any(raw.ins_code != ""):
        raise ValueError("AtomWorks CIF IO clears insertion codes; provide an explicitly remapped reference first")
    raw = raw[~np.isin(raw.element, ["H", "D", "T"])]
    raw.res_name[np.isin(raw.res_name, ["HIP", "HID", "HIE"])] = "HIS"
    apply_ligand_formal_charges(raw, Path(manifest["params"]), "8MT")
    template = ligand_template(raw, Path(manifest["params"]))
    ligand_mask = raw.res_name == "8MT"
    raw.res_name[ligand_mask] = "ZMT"
    config = PrepareConfig(
        add_missing_atoms=False,
        remove_ccds=(),
        remove_waters=False,
        fix_ligands_at_symmetry_centers=False,
        fix_arginines=False,
        convert_mse_to_met=False,
        hydrogen_policy="remove",
        return_atom_array_plus=True,
        long_bond_policy="keep",
        struct_conn_distance_policy="keep",
    )
    with custom_ccd_residues({"ZMT": template}):
        prepared = as_atom_array_plus(parse_atom_array(raw, config=config)["assemblies"]["1"][0])
        apply_ligand_formal_charges(prepared, Path(manifest["params"]))
        # Explicit covalent TI attachment; original PDB CONECT may already contain it.
        ser = 146 if manifest["target"] == "tll" else 77
        left = np.flatnonzero((prepared.chain_id == "A") & (prepared.res_id == ser) & (prepared.atom_name == "OG"))
        right = np.flatnonzero((prepared.res_name == "ZMT") & (prepared.atom_name == "CX"))
        if len(left) != 1 or len(right) != 1:
            raise ValueError("Missing unique covalent attachment atoms")
        prepared.bonds.add_bond(int(left[0]), int(right[0]), BondType.SINGLE)
        to_cif_file(
            prepared, out_dir / "reference.cif", ccd_entries={"ZMT": template}, chain_disambiguation="chain_iid"
        )
    reloaded = parse(out_dir / "reference.cif", config=ANNOTATED_CIF)["assemblies"]["1"][0]
    return as_atom_array_plus(reloaded)


def prepare(manifest_path: Path, output: Path) -> None:
    """Write matched inputs and a mask/conditioning manifest; never overwrite inputs."""
    manifest = json.loads(manifest_path.read_text())
    target_dir = output / manifest["target"]
    target_dir.mkdir(parents=True, exist_ok=False)
    atoms = load_reference(manifest, target_dir)
    eligible = [tuple(r["key"]) for r in manifest["residues"] if r["eligible"]]
    protected = [tuple(r["key"]) for r in manifest["residues"] if not r["eligible"]]
    protein = np.isin(atoms.res_name, AA_NAMES)
    if {k for k, p in zip(atom_residue_keys(atoms), protein, strict=True) if p} != {
        tuple(r["key"]) for r in manifest["residues"]
    }:
        raise ValueError("PDB/CIF protein residue mapping changed")
    immutable = np.array([k in set(protected) for k in atom_residue_keys(atoms)]) | ~protein
    functional = {tuple(r["key"]) for r in manifest["residues"] if set(r["excluded_by"]) - {"not_surface"}}
    coordinate_mask = np.array([k in functional for k in atom_residue_keys(atoms)]) | ~protein
    # Registry exports erase the ConditionBase subclass in upstream typing.
    cast(type[ConditionBase], C_SEQ).set_annotation_from_ground_truth(atoms, immutable)
    cast(type[ConditionBase], C_CRD).set_annotation_from_ground_truth(atoms, coordinate_mask)
    C_IDX.set_annotation(atoms, np.ones(len(atoms), dtype=bool))
    base = {aa: _UNIPROT_AA_PERCENT[aa] / sum(_UNIPROT_AA_PERCENT.values()) for aa in AA_NAMES}
    profiles = {
        "baseline": None,
        "background": base,
        "xtr_weak": reweight_probabilities(base, XTR_BIASES, 0.5),
        "xtr_moderate": reweight_probabilities(base, XTR_BIASES, 1.0),
        "notion_weak": reweight_probabilities(base, NOTION_BIASES, 0.5),
        "notion_moderate": reweight_probabilities(base, NOTION_BIASES, 1.0),
        "native_050": None,
        "native_075": None,
    }
    indexed_keys = residue_keys(atoms)
    manifest["eligible_token_indices"] = [i for i, k in enumerate(indexed_keys) if k in set(eligible)]
    manifest["prepared_protected"] = [list(k) for k in protected]
    manifest["model_residue_names"] = {str(r["key"]): r["name"] for r in manifest["residues"]}
    manifest["arms"] = []
    for arm, profile in profiles.items():
        child = atoms.copy()
        if profile is not None:
            annotate_probabilities(child, eligible, profile)
        if arm.startswith("native_"):
            probability = int(arm.rsplit("_", 1)[1]) / 100
            for record in manifest["residues"]:
                if record["eligible"]:
                    name = "HIS" if record["name"] in {"HIP", "HID", "HIE"} else record["name"]
                    annotate_probabilities(child, [tuple(record["key"])], {name: probability})
        child = CanonicalizeSequenceProbabilityConditions()({"atom_array": child})["atom_array"]
        problems = run_pipeline_validation(child)
        if problems:
            raise ValueError(f"{arm}: {problems}")
        path = target_dir / f"{arm}.cif"
        to_cif_file(child, path, save_standard_annotations=True, chain_disambiguation="chain_iid")
        loaded = parse(path, config=ANNOTATED_CIF)["assemblies"]["1"][0]
        for condition in [C_SEQ, C_CRD, *SEQUENCE_PROBABILITY_CONDITIONS]:
            left = np.asarray(condition.annotation(child, default="generate"))
            right = np.asarray(condition.annotation(loaded, default="generate"))
            if left.dtype.kind in "fci":
                np.testing.assert_allclose(left, right, equal_nan=True)
            else:
                np.testing.assert_array_equal(left, right)
        manifest["arms"].append(
            {
                "name": arm,
                "input": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "probabilities": profile,
            }
        )
        logger.info("Validated {} / {}: {} mutable surface residues", manifest["target"], arm, len(eligible))
    manifest["deferred"] = {
        "msa_native": "No archived aligned MSA was found; do not substitute a fabricated family profile"
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Prepare one or more audited reference manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for manifest in args.manifests:
        prepare(manifest, args.output)


if __name__ == "__main__":
    main()
