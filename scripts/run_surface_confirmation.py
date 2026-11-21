"""Confirm two eligible-surface hits with independent seeds and matched controls.

Selection is exploratory on the screen; confirmation uses new base seeds. No
second scaffold used during screening is described as a held-out target.
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from rfproteina.surface_engineering import XTR_BIASES, select_surface_pool
from scripts.run_surface_experiment import execute, read_atoms

# JSON ledgers combine validated scalars, metadata and error records; Any is
# restricted to that serialization boundary. Selection reads named numeric fields.
Row = dict[str, Any]
STRATA = [(target, mode) for target in ["tll", "bacillus"] for mode in ["inverse", "partial035"]]


def select_finalists(rows: list[Row]) -> list[str]:
    """Require complete geometry-safe blocks and no added hydrophobic exposure."""
    grouped: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in rows:
        if row.get("physics_status") == "valid" and row.get("scientific_admitted", True):
            grouped[row["target"], row["mode"], row["arm"]].append(row)
    baselines = {stratum: grouped[*stratum, "baseline"] for stratum in STRATA}
    if any(not values for values in baselines.values()):
        raise ValueError("Every target/task needs its measured baseline before finalist selection")
    candidates = sorted({key[2] for key in grouped} - {"baseline", "background"})
    ranked: list[tuple[float, str]] = []
    for arm in candidates:
        shifts, excess_mutations, hydrophobic_ratios = [], [], []
        for stratum in STRATA:
            values = grouped[*stratum, arm]
            base = baselines[stratum]
            if len(values) < len(base) or any(not r["passes_geometry_gate"] for r in values):
                break
            shifts.append(
                statistics.mean(float(r["surface_fraction_DETA"]) for r in values)
                - statistics.mean(float(r["surface_fraction_DETA"]) for r in base)
            )
            excess_mutations.append(
                max(
                    0.0,
                    statistics.mean(float(r["n_surface_mutations"]) / float(r["n_surface"]) for r in values)
                    - statistics.mean(float(r["n_surface_mutations"]) / float(r["n_surface"]) for r in base),
                )
            )
            hydrophobic_ratios.append(
                statistics.mean(float(r["hydrophobic_sasa_A2"]) for r in values)
                / statistics.mean(float(r["hydrophobic_sasa_A2"]) for r in base)
            )
        if len(shifts) != len(STRATA) or min(shifts) <= 0 or statistics.mean(hydrophobic_ratios) > 1:
            continue
        ranked.append((statistics.mean(shifts) / (1 + statistics.mean(excess_mutations)), arm))
    chosen, mechanisms = [], set()
    for _, arm in sorted(ranked, reverse=True):
        mechanism = arm.split("_", 1)[0] if arm.startswith(("xtr_", "notion_", "logit_", "native_")) else arm
        if arm == "msa_xtr":
            mechanism = "xtr"
        elif arm == "msa_notion":
            mechanism = "notion"
        if mechanism not in mechanisms:
            chosen.append(arm)
            mechanisms.add(mechanism)
        if len(chosen) == 2:
            break
    return chosen


def validate_zero_controls(rows: list[Row]) -> None:
    """Require one successful identical native zero-control in every stratum."""
    if len(rows) != len(STRATA) or {(row["target"], row["mode"]) for row in rows} != set(STRATA):
        raise RuntimeError("Native zero-steering audit must contain all four unique strata")
    if any(
        not row["same_identities_and_atoms"]
        or row["max_coordinate_delta_A"] is None
        or row["max_coordinate_delta_A"] > 1e-5
        for row in rows
    ):
        raise RuntimeError("Native zero-steering control differs from its matched baseline")


def main() -> None:
    """Wait for screen completion, freeze selection, then spend at most two hours."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20261129)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--screen-output", type=Path)
    parser.add_argument("--advanced-output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    prepared = (args.prepared or root / "inputs").resolve()
    screen_output = (args.screen_output or root / "screen").resolve()
    advanced_output = (args.advanced_output or root / "advanced").resolve()
    if not 1 <= args.n_seeds <= 8:
        parser.error("n-seeds must be in [1,8]")
    source = Path(__file__).resolve().parents[1]
    while not (advanced_output / "COMPLETE").exists():
        time.sleep(30)
    started = time.time()
    # Preserve 90 min independent validation plus 15 min packaging/shutdown.
    deadline = min(started + 7200, args.deadline_epoch - 6300)
    output = root / "confirmation"
    output.mkdir(exist_ok=False)
    os.environ["PYTHONPATH"] = str(source / "src")
    command = [
        str(source / ".venv/bin/python"),
        str(root / "analyse_surface_physics.py"),
        "--root",
        str(root),
        "--prepared-dir",
        str(prepared),
        "--stages",
        screen_output.name,
        "--exclude-cpu",
        "--rfd4-source",
        str(source),
    ]
    code, error = execute(command, source, output / "selection_analysis.log", max(1, deadline - time.time()))
    if code:
        raise RuntimeError(f"Selection analysis failed: {code}: {error}")
    rows = json.loads((root / "analysis/physics.json").read_text())
    frozen_physics = output / "selection_physics.json"
    frozen_ledger = output / "screen_results_at_selection.json"
    shutil.copyfile(root / "analysis/physics.json", frozen_physics)
    shutil.copyfile(screen_output / "results.json", frozen_ledger)
    finalists = select_finalists(rows)
    selection = {
        "finalists": finalists,
        "confirmation_seed_start": args.seed,
        "n_confirmation_seeds": args.n_seeds,
        "selected_epoch": time.time(),
        "deadline_epoch": deadline,
        "rule": (
            "positive DETA shift in every target/task, all geometry gates, "
            "mean hydrophobic SASA no higher than baseline; "
            "rank shift/mutation excess; distinct method families"
        ),
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physics_sha256": hashlib.sha256(frozen_physics.read_bytes()).hexdigest(),
        "physics_file": str(frozen_physics.relative_to(root)),
        "screen_ledger_file": str(frozen_ledger.relative_to(root)),
        "screen_ledger_sha256": hashlib.sha256(frozen_ledger.read_bytes()).hexdigest(),
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    logger.info("Frozen confirmation finalists: {}", finalists)
    jobs = [
        (seed, arm, ["inverse", "partial035"])
        for seed in range(args.seed, args.seed + args.n_seeds)
        for arm in ["baseline", "background", *finalists]
    ]
    jobs.insert(1, (args.seed, "logit_zero", ["inverse", "partial035"]))
    jobs += [
        (seed, "baseline", ["inverse", "partial035"])
        for seed in [args.seed + args.n_seeds, args.seed + args.n_seeds + 1]
    ]
    jobs += [(args.seed + 10, arm, ["partial045", "partial055"]) for arm in finalists]
    receipts = []
    for seed, arm, modes in jobs:
        remaining = deadline - time.time()
        if remaining < 300:
            logger.warning("Confirmation deadline reached; preserve independent validation")
            break
        destination = output / f"seed{seed}"
        command = [
            str(source / ".venv/bin/python"),
            str(source / "scripts/run_surface_experiment.py"),
            "--prepared",
            str(prepared),
            "--output",
            str(destination),
            "--seed",
            str(seed),
            "--replicates",
            "1",
            "--deadline-epoch",
            str(deadline),
            "--resume",
            "--arms",
            arm,
            "--modes",
            *modes,
        ]
        begin = time.time()
        code, error = execute(command, source, output / f"{seed}_{arm}.log", max(1, remaining))
        receipts.append({"command": command, "exit_code": code, "error": error, "elapsed_seconds": time.time() - begin})
        (output / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
        ledger = destination / "results.json"
        if ledger.exists() and any(row["status"] == "invalid" for row in json.loads(ledger.read_text())):
            raise RuntimeError("Confirmation stopped on a protected-identity/mapping violation")
    all_rows = []
    for ledger in sorted(output.glob("seed*/results.json")):
        all_rows.extend(json.loads(ledger.read_text()))
    (output / "results.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    zero_checks: list[Row] = []
    for target, mode in STRATA:
        paired = {
            arm: [
                row
                for row in all_rows
                if row["target"] == target
                and row["mode"] == mode
                and row["arm"] == arm
                and row.get("seed") == args.seed
                and row["status"] == "valid"
                and row.get("generation_exit_code") == 0
            ]
            for arm in ["baseline", "logit_zero"]
        }
        if not all(len(values) == 1 for values in paired.values()):
            continue
        baseline = read_atoms(Path(paired["baseline"][0]["file"]))
        zero = read_atoms(Path(paired["logit_zero"][0]["file"]))
        same_atoms = np.array_equal(baseline.atom_name, zero.atom_name) and np.array_equal(
            baseline.res_name, zero.res_name
        )
        max_delta = float(np.max(np.abs(baseline.coord - zero.coord))) if same_atoms else None
        zero_checks.append(
            {
                "target": target,
                "mode": mode,
                "same_identities_and_atoms": same_atoms,
                "max_coordinate_delta_A": max_delta,
            }
        )
    (output / "zero_control_audit.json").write_text(json.dumps(zero_checks, indent=2) + "\n")
    validate_zero_controls(zero_checks)
    posthoc = []
    basic_rows = json.loads((screen_output / "results.json").read_text())
    for target, mode in STRATA:
        generated = [
            row
            for row in [*basic_rows, *all_rows]
            if row["target"] == target and row["mode"] == mode and row["arm"] == "baseline"
        ]
        # The four-candidate view matches terminal beam proposals, not its
        # additional intermediate clean-prediction score evaluations. Freeze
        # the generated pool before quality filtering; failures retain cost.
        for budget, candidates in [("terminal4", generated[:4]), ("full_pool", generated)]:
            pool = [row for row in candidates if row["status"] == "valid" and row.get("generation_exit_code") == 0]
            scores = {
                row["file"]: sum(float(row[f"surface_fraction_{aa}"]) * bias for aa, bias in XTR_BIASES.items())
                for row in pool
            }
            geometry_scores = {row["file"]: scores[row["file"]] - 0.5 * float(row["ca_rmsd_A"]) for row in pool}
            if len(scores) >= 2:
                posthoc.append(
                    {
                        "target": target,
                        "mode": mode,
                        "budget": budget,
                        "n_requested_pool": len(candidates),
                        "n_completed_pool": sum(row["status"] == "valid" for row in candidates),
                        "n_pool": len(pool),
                        "scores": scores,
                        "ranked": select_surface_pool(scores, 2),
                        "geometry_ranked": select_surface_pool(geometry_scores, 2),
                        "geometry_scores": geometry_scores,
                        "geometry_policy": (
                            "Common downstream gates; final ranking uses chemistry minus 0.5 CA RMSD. "
                            "Beam trajectory scores are not reused."
                        ),
                        "random": select_surface_pool(scores, 2, random_seed=args.seed + 99),
                    }
                )
    (output / "posthoc.json").write_text(json.dumps(posthoc, indent=2) + "\n")
    (output / "COMPLETE").write_text(f"Confirmation phase finished at {time.time()}\n")


if __name__ == "__main__":
    main()
