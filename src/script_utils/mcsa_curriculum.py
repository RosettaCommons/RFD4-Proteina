"""Build and summarize the versioned M-CSA motif curriculum without launching jobs."""

import argparse
import csv
import hashlib
import json
import math
import shlex
from pathlib import Path


def load_tasks(path: Path) -> dict:
    """Validate the manifest and pinned experimental structure before creating inputs."""
    spec = json.loads(path.read_text())
    if spec["schema_version"] != 1:
        raise ValueError("Unsupported curriculum schema_version")
    reference = path.parent / spec["reference"]["file"]
    if hashlib.sha256(reference.read_bytes()).hexdigest() != spec["reference"]["sha256"]:
        raise ValueError("Reference checksum mismatch")
    ids = set()
    for task in spec["tasks"]:
        if task["id"] in ids:
            raise ValueError(f"Duplicate task: {task['id']}")
        ids.add(task["id"])
        previous = 0
        size = 0
        for start, end in task["segments"]:
            if start <= previous or end < start:
                raise ValueError(f"Overlapping, unordered, or invalid segments: {task['id']}")
            previous = end + 1  # Require at least one omitted residue between segments.
            size += end - start + 1
        if not size or size != task["motif_size"]:
            raise ValueError(f"Incorrect motif size: {task['id']}")
    if not ids:
        raise ValueError("No curriculum tasks")
    return spec


def build_curriculum(manifest: Path, out_dir: Path, length: int, replicates: int, seed: int) -> Path:
    """Write condition specs, the full attempt ledger, and a reviewable generation command."""
    spec = load_tasks(manifest)
    if replicates < 1:
        raise ValueError("replicates must be positive")
    reference = (manifest.parent / spec["reference"]["file"]).resolve()
    chain = spec["reference"]["chain"]
    conditions = {}
    attempts = []
    for task in spec["tasks"]:
        blocks = task["segments"]
        remaining = length - task["motif_size"]
        if remaining < len(blocks) + 1:
            raise ValueError("length must leave at least one residue in each linker and both termini")
        gap, extra = divmod(remaining, len(blocks) + 1)
        gaps = [gap + int(i < extra) for i in range(len(blocks) + 1)]
        contig = [f"L0:{gaps[0]}"]
        for i, (start, end) in enumerate(blocks, 1):
            contig.extend([f"{chain}{start}-{end}", f"L{i}:{gaps[i]}"])
        fixed = [r for r in spec["reference"]["catalytic_residues"] if any(a <= r <= b for a, b in blocks)]
        catalytic = " or ".join(f"sel('{chain}/*/{r}')" for r in fixed)
        if not catalytic:
            raise ValueError(f"Task has no catalytic anchor: {task['id']}")
        conditions[task["id"]] = {
            "input": str(reference),
            # Load observed atoms unchanged: do not add missing experimental atoms.
            "input_as_annotated_cif": True,
            "contig": ", ".join(contig),
            "conditions": {
                "C_CRD": {"true": [{"select": f"is_protein_backbone or ({catalytic})"}]},
                "C_SEQ": {"true": [{"select": catalytic}]},
            },
        }
        attempts.extend({"example_id": f"{task['id']}_{i}", "task": task["id"]} for i in range(replicates))
    # A new directory prevents accidental replacement of a prior run's ledger.
    out_dir.mkdir(parents=True, exist_ok=False)
    inputs = out_dir.resolve() / "inputs.json"
    inputs.write_text(json.dumps(conditions, indent=2) + "\n")
    run_manifest = {
        "schema_version": 1,
        "curriculum_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "reference": spec["reference"],
        "length": length,
        "replicates": replicates,
        "seed": seed,
        "tasks": spec["tasks"],
        "attempts": attempts,
        "protocol": "indexed_apo_direct_sequence_v1",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    command = [
        "rfd4-generate",
        f"inputs={inputs}",
        f"out_dir={out_dir.resolve() / 'generation'}",
        f"num_replicates={replicates}",
        "batch_size=1",
        f"seed={seed}",
        "recycle_strategy=medium/midpeak",
        "metrics/refolding_metrics=mcsa_curriculum",
        "metrics/pass_criteria=mcsa_curriculum",
        "metrics_tags=[design,motif_scaffold_refolding,mcsa_curriculum]",
        "save_chain_of_interest=true",
    ]
    (out_dir / "generate.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n")
    return inputs


def _number(row: dict, key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except (ValueError, TypeError):
        return math.inf
    return value if math.isfinite(value) else math.inf


def summarize(run_manifest: Path, metrics_files: list[Path], output: Path) -> None:
    """Left-join metrics onto planned attempts; missing or nonfinite metrics never pass."""
    manifest = json.loads(run_manifest.read_text())
    expected = {a["example_id"]: a["task"] for a in manifest["attempts"]}
    if len(expected) != len(manifest["attempts"]):
        raise ValueError("Duplicate example IDs in attempt ledger")
    rows = {}
    for path in metrics_files:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if "example_id" not in (reader.fieldnames or []):
                raise ValueError(f"Missing example_id column: {path}")
            for row in reader:
                example_id = row["example_id"]
                if example_id not in expected:
                    raise ValueError(f"Unplanned example_id {example_id!r}; use metrics from this run only")
                if example_id in rows:
                    raise ValueError(f"Duplicate metrics for {example_id}")
                rows[example_id] = row
    results = []
    for task in manifest["tasks"]:
        ids = [name for name, task_id in expected.items() if task_id == task["id"]]
        n_evaluated = n_backbone = n_site = 0
        clusters = set()
        for name in ids:
            row = rows.get(name, {})
            scaffold = _number(row, "curriculum_geometry.scaffold_ca_rmsd")
            motif = _number(row, "curriculum_geometry.motif_bb_rmsd")
            catalytic = _number(row, "curriculum_geometry.catalytic_rmsd")
            n_evaluated += int(
                all(
                    math.isfinite(_number(row, f"curriculum_geometry.{key}"))
                    for key in (
                        "scaffold_ca_rmsd",
                        "motif_bb_rmsd",
                        "catalytic_rmsd",
                        "max_catalytic_displacement",
                        "max_catalytic_distance_error",
                        "worst_segment_bb_rmsd",
                    )
                )
            )
            backbone = scaffold <= 2 and motif <= 1
            # These are declared curriculum gates, not the published AME definition.
            site = (
                backbone
                and catalytic <= 1.5
                and _number(row, "curriculum_geometry.max_catalytic_displacement") <= 2
                and _number(row, "curriculum_geometry.max_catalytic_distance_error") <= 1
                and _number(row, "curriculum_geometry.worst_segment_bb_rmsd") <= 1.5
                and _number(row, "sequence_adherence.n_violations") == 0
            )
            n_backbone += int(backbone)
            n_site += int(site)
            cluster = row.get("foldseek_cluster_id", "")
            if site and cluster and cluster.lower() not in {"nan", "none"}:
                clusters.add(cluster)
        results.append(
            {
                "task": task["id"],
                "series": task["series"],
                "level": task["level"],
                "motif_size": task["motif_size"],
                "segments": len(task["segments"]),
                "attempted": len(ids),
                "metric_rows": sum(name in rows for name in ids),
                "evaluated": n_evaluated,
                "backbone_successes": n_backbone,
                "site_geometry_successes": n_site,
                "backbone_success_rate": n_backbone / len(ids),
                "site_geometry_success_rate": n_site / len(ids),
                "unique_site_solutions": len(clusters) if clusters else "",
            }
        )
    with output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build", help="Prepare inputs and a generation command; does not run the model")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--length", type=int, default=200)
    build.add_argument("--replicates", type=int, default=100)
    build.add_argument("--seed", type=int, default=0)
    report = sub.add_parser("summarize")
    report.add_argument("--run-manifest", type=Path, required=True)
    report.add_argument("--metrics", type=Path, nargs="+", required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "build":
        print(build_curriculum(args.manifest, args.out_dir, args.length, args.replicates, args.seed))
    else:
        summarize(args.run_manifest, args.metrics, args.output)


if __name__ == "__main__":
    main()
