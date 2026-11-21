"""Run AE guidance and equal-budget beam controls after the basic screen finishes.

Full-generation AE comparisons are deliberately a separate track. Their baseline
uses the same decoder path as guided runs. Search arms share branches/checkpoints.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from loguru import logger

from scripts.run_surface_experiment import analyse, execute


def main() -> None:
    """Wait for the serial GPU owner, then run advanced methods within the cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    os.environ["PYTHONPATH"] = str(source / "src")
    while not args.after.exists():
        if time.time() > args.deadline_epoch - 300:
            raise RuntimeError("No advanced-method budget remains after the basic screen")
        time.sleep(30)
    args.output.mkdir(parents=True, exist_ok=True)
    for arm in [
        "ae_baseline",
        "gradient_005",
        "beam_random",
        "beam_surface",
        "gradient_entropy",
        "beam_geometry",
        "gradient_100",
    ]:
        for target in ["bacillus", "tll"]:
            remaining = args.deadline_epoch - time.time()
            if remaining < 300:
                logger.warning("Preserving validation reserve; advanced screen stopped")
                analyse(args.prepared, args.output)
                (args.output / "BUDGET_EXHAUSTED").write_text(f"Deferred remaining advanced cells at {time.time()}\n")
                (args.output / "COMPLETE").write_text(f"Advanced phase finished with deferred cells at {time.time()}\n")
                return
            manifest = json.loads((args.prepared / target / "manifest.json").read_text())
            is_beam = arm.startswith("beam_")
            mode = "partial035_search" if is_beam else "full_ae"
            job = args.output / target / mode / arm
            job.mkdir(parents=True, exist_ok=False)
            command = [
                str(source / ".venv/bin/rfd4-generate"),
                f"inputs={args.prepared / target / 'baseline.cif'}",
                f"out_dir={job}",
                "seed=20261120",
                "batch_size=1",
                "precision=16-mixed",
                "compile_lengths=null",
                "num_workers=0",
                "recycle_strategy=medium/midpeak",
            ]
            if is_beam:
                selected = [r["key"] for r in manifest["residues"] if r["eligible"]]
                command += [
                    "inference_experiment=surface_search",
                    "num_replicates=1",
                    "surface_selected_keys=" + json.dumps(selected, separators=(",", ":")),
                    f"surface_reference_file={args.prepared / target / 'reference.cif'}",
                ]
                if arm == "beam_random":
                    command += ["surface_cost_weight=0.0", "surface_geometry_weight=0.0", "surface_random_weight=1.0"]
                elif arm == "beam_geometry":
                    command += ["surface_cost_weight=0.0"]
            else:
                command += [
                    "num_replicates=2",
                    "~metrics",
                    "generation.args.condition_decoder_on_sequence_prediction=false",
                ]
                if arm.startswith("gradient_"):
                    indices = json.dumps(manifest["eligible_token_indices"], separators=(",", ":"))
                    weight = -0.05 if arm == "gradient_005" else -1.0
                    command += [
                        "+guidance=surface_preference",
                        "generation.args.sequence_guidance_token_indices=" + indices,
                        f"guidance.weight={weight}",
                    ]
                    if arm == "gradient_entropy":
                        command.append("guidance.objective=sidechain_entropy")
            if arm.startswith("gradient_"):
                command = [
                    str(source / ".venv/bin/python"),
                    str(source / "scripts/surface_guidance_probe.py"),
                    *command[1:],
                ]
            receipt = {
                "target": target,
                "mode": mode,
                "arm": arm,
                "command": command,
                "seed": 20261120,
                "requested": 2,
                "requested_scored_evaluations": 8 if is_beam else None,
                "requested_terminal_candidates": 4 if is_beam else 2,
                "requested_intermediate_evaluations": 4 if is_beam else 0,
                "retention": "final_beam_only" if is_beam else "all_replicates",
                "started_epoch": time.time(),
                "deadline_epoch": args.deadline_epoch,
                "source_sha256": {
                    str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [
                        source / "src/rfproteina/proteina.py",
                        source / "src/rfproteina/partial_autoencoder/autoencoder.py",
                        source / "src/rfproteina/metrics/surface_preference.py",
                        source / "src/rfproteina/surface_engineering.py",
                        source / "configs/guidance/surface_preference.yaml",
                        source / "configs/metrics/surface_preference.yaml",
                        source / "configs/inference_experiment/surface_search.yaml",
                        source / "scripts/run_surface_experiment.py",
                        source / "scripts/surface_guidance_probe.py",
                        Path(__file__),
                    ]
                },
            }
            (job / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            logger.info("Starting advanced {} / {}", target, arm)
            code, error = execute(command, source, job / "generation.log", min(remaining, 1800))
            receipt.update(exit_code=code, error=error, finished_epoch=time.time())
            if arm.startswith("gradient_"):
                audit_file = job / "guidance_audit.json"
                audit_data = json.loads(audit_file.read_text()) if audit_file.exists() else {}
                receipt["guidance_audit_passed"] = bool(audit_data.get("passed", False))
                for key in ["raw_grad_calls", "nonzero_calls", "max_outside_force"]:
                    receipt["guidance_" + key] = audit_data.get(key)
            receipt["elapsed_seconds"] = receipt["finished_epoch"] - receipt["started_epoch"]
            emitted = len(list(job.glob("inference_samples/**/*_generated.cif*")))
            receipt["terminal_generated_candidates"] = (
                (4 if code == 0 and emitted == 2 else None) if is_beam else emitted
            )
            receipt["scored_evaluations"] = 8 if is_beam and code == 0 and emitted == 2 else None
            receipt["interior_prediction_evaluations"] = (4 if code == 0 and emitted == 2 else None) if is_beam else 0
            receipt["all_structure_files"] = len(list(job.rglob("*.cif")))
            (job / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            rows = analyse(args.prepared, args.output)
            if any(r["status"] == "invalid" for r in rows):
                raise RuntimeError("Advanced output failed protected-site/mapping checks")
            logger.info("Finished {} / {} with exit {} in {:.1f}s", target, arm, code, receipt["elapsed_seconds"])
    (args.output / "COMPLETE").write_text(f"Advanced screen complete at {time.time()}\n")


if __name__ == "__main__":
    main()
