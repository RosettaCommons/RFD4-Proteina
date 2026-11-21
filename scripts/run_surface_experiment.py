"""Run and audit matched surface-engineering arms under a wall-clock GPU cap.

Each child has its own immutable output directory, resolved command and log.
Failures stay in the ledger and cost denominator. A protection/mapping violation
stops the screen. Use --analyse-only to regenerate tables from saved structures.
"""

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io import parse
from biotite.structure import AtomArray, superimpose
from biotite.structure.io.pdb import PDBFile
from loguru import logger

from rfproteina.surface_engineering import AA_NAMES, XTR_BIASES, atom_residue_keys, logit_bias_override, surface_metrics
from rfproteina.utils.parse_presets import ANNOTATED_CIF

# Any is restricted to versioned heterogeneous JSON manifests/receipts. Structure
# identities, selections and numeric outputs are validated before admitting a run.
Manifest = dict[str, Any]


def protein_map(atoms: AtomArray) -> dict[tuple[str, int, str], str]:
    """Read one canonical identity per exact residue, rejecting inconsistent atoms."""
    result: dict[tuple[str, int, str], str] = {}
    for key, name in zip(atom_residue_keys(atoms), atoms.res_name, strict=True):
        if name not in AA_NAMES:
            continue
        if key in result and result[key] != str(name):
            raise ValueError(f"Mixed identities for {key}")
        result[key] = str(name)
    return result


def read_atoms(path: Path) -> AtomArray:
    """Read an annotated structure with the registered RFD4 conditions."""
    atoms: AtomArray
    if path.suffix.lower() == ".pdb":
        # Parse raw PDB atom records before chemistry classification: Rosetta
        # writes HIP as HETATM, which otherwise gets split into another chain.
        atoms = PDBFile.read(path).get_structure(model=1, altloc="first")
    else:
        atoms = parse(path, config=ANNOTATED_CIF)["assemblies"]["1"][0]
    atoms = atoms[~np.isin(atoms.element, ["H", "D", "T"])]
    atoms.res_name[np.isin(atoms.res_name, ["HIP", "HID", "HIE"])] = "HIS"
    return atoms


def audit_retained_ligand_chemistry(reference: AtomArray, atoms: AtomArray) -> None:
    """Preserve named ligand elements and all bonds touching retained nonprotein atoms."""

    def identity(array: AtomArray) -> list[tuple[tuple[str, int, str], str]]:
        return [(key, str(name)) for key, name in zip(atom_residue_keys(array), array.atom_name, strict=True)]

    ref_ids, out_ids = identity(reference), identity(atoms)
    ligand = {key for key, name in zip(ref_ids, reference.res_name, strict=True) if name not in AA_NAMES}
    ref_elements = dict(zip(ref_ids, reference.element, strict=True))
    out_elements = dict(zip(out_ids, atoms.element, strict=True))
    if any(out_elements.get(key) != ref_elements[key] for key in ligand):
        raise ValueError("Retained ligand element inventory differs from reference")

    def edges(
        array: AtomArray, identifiers: list[tuple[tuple[str, int, str], str]]
    ) -> set[tuple[tuple[tuple[str, int, str], str], tuple[tuple[str, int, str], str], int]]:
        result: set[tuple[tuple[tuple[str, int, str], str], tuple[tuple[str, int, str], str], int]] = set()
        if array.bonds is None:
            return result
        for left, right, order in array.bonds.as_array():
            a, b = identifiers[int(left)], identifiers[int(right)]
            if a in ligand or b in ligand:
                a, b = sorted((a, b))
                result.add((a, b, int(order)))
        return result

    if edges(reference, ref_ids) != edges(atoms, out_ids):
        raise ValueError("Retained ligand bond graph or covalent attachment differs from reference")


def audit_output(path: Path, manifest: Manifest, reference: AtomArray) -> dict[str, float | int]:
    """Measure fixed-mask composition, protection and aligned backbone displacement."""
    atoms = read_atoms(path)
    reference = reference[~np.isin(reference.element, ["H", "D", "T"])]
    expected = {
        tuple(r["key"]): "HIS" if r["name"] in {"HIP", "HID", "HIE"} else r["name"] for r in manifest["residues"]
    }
    eligible = [tuple(r["key"]) for r in manifest["residues"] if r["eligible"]]
    protected = [tuple(r["key"]) for r in manifest["residues"] if not r["eligible"]]
    result: dict[str, float | int] = surface_metrics(expected, protein_map(atoms), eligible, protected)
    if not np.isfinite(atoms.coord).all():
        raise ValueError("Non-finite generated coordinates")
    # Names alone cannot distinguish canonical His from a renamed modified
    # residue. Protect the complete heavy-atom multiset, including elements.
    protected_keys = set(protected)
    ref_inventory = Counter(
        (key, str(name), str(element))
        for key, name, element in zip(atom_residue_keys(reference), reference.atom_name, reference.element, strict=True)
        if key in protected_keys
    )
    out_inventory = Counter(
        (key, str(name), str(element))
        for key, name, element in zip(atom_residue_keys(atoms), atoms.atom_name, atoms.element, strict=True)
        if key in protected_keys
    )
    if ref_inventory != out_inventory:
        raise ValueError("Protected protein heavy-atom inventory/elements differ from reference")
    ref_ca = {
        k: c
        for k, n, c in zip(atom_residue_keys(reference), reference.atom_name, reference.coord, strict=True)
        if n == "CA"
    }
    out_ca = {k: c for k, n, c in zip(atom_residue_keys(atoms), atoms.atom_name, atoms.coord, strict=True) if n == "CA"}
    keys = list(expected)
    if not set(keys) <= ref_ca.keys() or not set(keys) <= out_ca.keys():
        raise ValueError("Missing backbone CA mapping")
    aligned, alignment = superimpose(np.array([ref_ca[k] for k in keys]), np.array([out_ca[k] for k in keys]))
    displacements = np.linalg.norm(aligned - np.array([ref_ca[k] for k in keys]), axis=-1)
    result["ca_rmsd_A"] = float(np.sqrt(np.mean(displacements**2)))
    result["max_ca_displacement_A"] = float(displacements.max())
    # Coordinate-conditioned functional heavy atoms must retain their relative
    # geometry; translation/alignment does not change pairwise distances.
    functional = {tuple(r["key"]) for r in manifest["residues"] if set(r["excluded_by"]) - {"not_surface"}}

    def atom_lookup(array: AtomArray) -> dict[tuple[tuple[str, int, str], str], np.ndarray]:
        return {(k, str(n)): c for k, n, c in zip(atom_residue_keys(array), array.atom_name, array.coord, strict=True)}

    ref_atoms, out_atoms = atom_lookup(reference), atom_lookup(atoms)
    reference_names = dict(zip(atom_residue_keys(reference), reference.res_name, strict=True))
    output_names = dict(zip(atom_residue_keys(atoms), atoms.res_name, strict=True))
    ligand_atoms = {key for key in ref_atoms if reference_names[key[0]] not in AA_NAMES}
    emitted_ligand = {key for key in out_atoms if output_names[key[0]] not in AA_NAMES}
    if ligand_atoms != emitted_ligand:
        raise ValueError("Retained ligand atom inventory differs from reference")
    if ligand_atoms:
        audit_retained_ligand_chemistry(reference, atoms)
        ref_charge = {
            key: int(charge)
            for key, charge in zip(
                [(key, str(name)) for key, name in zip(atom_residue_keys(reference), reference.atom_name, strict=True)],
                reference.charge if "charge" in reference.get_annotation_categories() else np.zeros(len(reference)),
                strict=True,
            )
        }
        out_charge = {
            key: int(charge)
            for key, charge in zip(
                [(key, str(name)) for key, name in zip(atom_residue_keys(atoms), atoms.atom_name, strict=True)],
                atoms.charge if "charge" in atoms.get_annotation_categories() else np.zeros(len(atoms)),
                strict=True,
            )
        }
        if any(ref_charge[key] != out_charge[key] for key in ligand_atoms):
            raise ValueError("Retained ligand formal charges differ from reference")
        result["ligand_formal_charge"] = sum(out_charge[key] for key in ligand_atoms)
        ordered_ligand = sorted(ligand_atoms)
        ligand_ref = np.array([ref_atoms[key] for key in ordered_ligand])
        ligand_out = np.array([out_atoms[key] for key in ordered_ligand])
        result["ligand_all_atom_rmsd_A"] = float(
            np.sqrt(np.mean(np.sum((ligand_ref - alignment.apply(ligand_out)) ** 2, axis=-1)))
        )
        result["ligand_max_internal_distance_error_A"] = float(
            np.max(
                np.abs(
                    np.linalg.norm(ligand_ref[:, None] - ligand_ref[None, :], axis=-1)
                    - np.linalg.norm(ligand_out[:, None] - ligand_out[None, :], axis=-1)
                )
            )
        )
        # The pilot has one covalent tetrahedral intermediate attached to Ser OG.
        # Record the actual link distance, independent of global alignment.
        ser = 146 if manifest.get("target") == "tll" else 77
        ser_key = (("A", ser, ""), "OG")
        cx = [key for key in ordered_ligand if key[1] == "CX"]
        if len(cx) == 1 and ser_key in out_atoms and ser_key in ref_atoms:
            result["covalent_link_distance_A"] = float(np.linalg.norm(out_atoms[cx[0]] - out_atoms[ser_key]))
            result["covalent_link_distance_error_A"] = abs(
                result["covalent_link_distance_A"] - float(np.linalg.norm(ref_atoms[cx[0]] - ref_atoms[ser_key]))
            )
    functional_atoms = [k for k in ref_atoms if k[0] in functional]
    if any(k not in out_atoms for k in functional_atoms):
        raise ValueError("Missing protected functional atoms")
    r = np.array([ref_atoms[k] for k in functional_atoms])
    g = np.array([out_atoms[k] for k in functional_atoms])
    max_distance_error = float(
        np.max(
            np.abs(np.linalg.norm(r[:, None] - r[None, :], axis=-1) - np.linalg.norm(g[:, None] - g[None, :], axis=-1))
        )
    )
    result["functional_max_distance_error_A"] = max_distance_error
    # C_CRD influences native generation, but the AE reconstruction is not an
    # exact coordinate copy. Report that drift; apply the campaign's existing
    # backbone-aligned all-atom 2 A gate, not a fabricated exact-copy tolerance.
    catalytic = {tuple(r["key"]) for r in manifest["residues"] if "functional" in r["excluded_by"]}
    catalytic_atoms = [k for k in functional_atoms if k[0] in catalytic]
    cat_ref = np.array([ref_atoms[k] for k in catalytic_atoms])
    cat_out = alignment.apply(np.array([out_atoms[k] for k in catalytic_atoms]))
    result["functional_all_atom_rmsd_A"] = float(np.sqrt(np.mean(np.sum((cat_ref - cat_out) ** 2, axis=-1))))
    result["passes_geometry_gate"] = int(result["functional_all_atom_rmsd_A"] < 2.0 and result["ca_rmsd_A"] < 2.0)
    return result


def analyse(prepared: Path, output: Path) -> list[Manifest]:
    """Write all outputs and failures, not just favourable surface designs."""
    rows: list[Manifest] = []
    for receipt_file in sorted(output.glob("*/*/*/receipt.json")):
        receipt = json.loads(receipt_file.read_text())
        manifest = json.loads((prepared / receipt["target"] / "manifest.json").read_text())
        reference = read_atoms(prepared / receipt["target"] / "reference.cif")
        paths = sorted(receipt_file.parent.glob("inference_samples/**/*_generated.cif*"))
        if not paths:
            rows.append(
                {
                    **{k: receipt[k] for k in ("target", "mode", "arm")},
                    "status": "no_output",
                    "error": receipt.get("error", ""),
                }
            )
        for path in paths:
            row = {k: receipt[k] for k in ("target", "mode", "arm")}
            row["file"] = str(path)
            row["seed"] = receipt.get("seed")
            row["candidate_id"] = path.parent.name
            row["generation_exit_code"] = receipt.get("exit_code")
            row["generation_output_count"] = len(paths)
            row["generation_output_complete"] = len(paths) == receipt["requested"]
            if "guidance_audit_passed" in receipt:
                row["guidance_audit_passed"] = receipt["guidance_audit_passed"]
            try:
                row.update(audit_output(path, manifest, reference))
                row["status"] = "valid"
            except (ValueError, AssertionError) as exc:
                row.update(status="invalid", error=str(exc))
            rows.append(row)
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = sorted({key for row in rows for key in row})
    if fields:
        with (output / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def execute(command: list[str], cwd: Path, log: Path, remaining: float) -> tuple[int, str | None]:
    """Terminate the process group on deadline, including dataloader children."""
    with log.open("w") as handle:
        process = subprocess.Popen(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=remaining), None
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124, "GPU wall-clock deadline reached"


def recover_completed(job: Path, requested: int) -> bool:
    """Resume only jobs with verified count, completion log, and existing receipt."""
    receipt_file = job / "receipt.json"
    log = job / "generation.log"
    if not receipt_file.exists() or not log.exists():
        return False
    receipt = json.loads(receipt_file.read_text())
    outputs = list(job.glob("inference_samples/**/*_generated.cif*"))
    if len(outputs) != requested or "Generation complete!" not in log.read_text():
        return False
    if receipt.get("exit_code") not in (None, 0):
        return False
    if receipt.get("exit_code") is None:
        receipt.update(exit_code=0, recovered_from_completed_output=True, finished_epoch=log.stat().st_mtime)
        receipt["elapsed_seconds"] = receipt["finished_epoch"] - receipt["started_epoch"]
        receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    return True


def main() -> None:
    """Run validated candidate inputs serially, preserving the eight-hour ceiling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=["tll", "bacillus"])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["inverse", "partial035", "partial045", "partial055"],
        default=["inverse", "partial035"],
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "baseline",
            "background",
            "xtr_weak",
            "xtr_moderate",
            "notion_weak",
            "notion_moderate",
            "native_050",
            "native_075",
            "logit_weak",
            "logit_moderate",
        ],
    )
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.prepared = args.prepared.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.analyse_only:
        analyse(args.prepared, args.output)
        return
    source = Path(__file__).resolve().parents[1]
    for mode in args.modes:
        for arm in args.arms:
            for target in args.targets:
                remaining = args.deadline_epoch - time.time()
                if remaining < 120:
                    logger.warning("Stopping with {} seconds left in the allocation budget", remaining)
                    analyse(args.prepared, args.output)
                    return
                job = args.output / target / mode / arm
                if args.resume and job.exists():
                    if recover_completed(job, args.replicates):
                        logger.info("Preserving completed {} / {} / {}", target, mode, arm)
                        continue
                    raise RuntimeError(f"Refusing to overwrite incomplete job {job}; inspect its process and receipt")
                job.mkdir(parents=True, exist_ok=False)
                manifest = json.loads((args.prepared / target / "manifest.json").read_text())
                input_arm = "baseline" if arm.startswith("logit_") else arm
                command = [
                    str(source / ".venv/bin/rfd4-generate"),
                    f"inputs={args.prepared / target / (input_arm + '.cif')}",
                    f"out_dir={job}",
                    f"num_replicates={args.replicates}",
                    f"seed={args.seed}",
                    "precision=16-mixed",
                    "compile_lengths=null",
                    "num_workers=0",
                    "batch_size=1",
                    "~metrics",
                    "recycle_strategy=medium/midpeak",
                ]
                command += (
                    ["generation=inverse_folding"]
                    if mode == "inverse"
                    else [f"generation.args.partial_trajectory_fraction_steps={int(mode.removeprefix("partial"))/100}"]
                )
                if arm.startswith("logit_"):
                    scale = {"logit_zero": 0.0, "logit_weak": 0.5, "logit_moderate": 1.0}[arm]
                    bias = {key: value * scale for key, value in XTR_BIASES.items()}
                    command += [
                        logit_bias_override(bias),
                        "generation.args.sequence_bias_token_indices="
                        + json.dumps(manifest["eligible_token_indices"], separators=(",", ":")),
                    ]
                receipt = {
                    "target": target,
                    "mode": mode,
                    "arm": arm,
                    "command": command,
                    "seed": args.seed,
                    "requested": args.replicates,
                    "started_epoch": time.time(),
                    "deadline_epoch": args.deadline_epoch,
                }
                receipt["source_sha256"] = {
                    str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [
                        source / "src/rfproteina/proteina.py",
                        source / "src/rfproteina/surface_engineering.py",
                        Path(__file__),
                    ]
                }
                (job / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                logger.info("Starting {} / {} / {}", target, mode, arm)
                code, error = execute(command, source, job / "generation.log", remaining)
                receipt.update(exit_code=code, error=error, finished_epoch=time.time())
                receipt["elapsed_seconds"] = receipt["finished_epoch"] - receipt["started_epoch"]
                (job / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                rows = analyse(args.prepared, args.output)
                if any(row["status"] == "invalid" for row in rows):
                    raise RuntimeError(
                        "A generated output failed the mapping/protection/geometry audit; screen stopped"
                    )
                if code:
                    logger.error(
                        "{} / {} / {} failed with {}; inspect {}", target, mode, arm, code, job / "generation.log"
                    )
                logger.info("Completed {} / {} / {} in {:.1f}s", target, mode, arm, receipt["elapsed_seconds"])
    (args.output / "COMPLETE").write_text(f"Completed planned arms at {time.time()}\n")


if __name__ == "__main__":
    main()
