#!/usr/bin/env python3
"""Compare inference backends across every design workflow the model serves.

Runs one generation per (workflow, backend) pair and reports the rollout time
beside the total, because the two answer different questions: a run spends
tens of seconds loading the released checkpoint pair before it samples, and
that floor is identical on every backend. Comparing totals therefore
understates whichever backend is faster.

Each workflow names one benchmark input. Pointing ``inputs`` at a directory
would design every structure in it, and ``subset_to_examples`` selects by
example ID rather than by position, so the script resolves a single file.

Examples:
    python src/script_utils/benchmark_backends.py
    python src/script_utils/benchmark_backends.py --workflows monomer,ppi --backends bioir
    python src/script_utils/benchmark_backends.py --out-dir runs/backends --recycle-strategy thinking_low
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BENCHMARKS = PROJECT / "data" / "benchmarks"

#: workflow -> (benchmark input, extra generation overrides)
#:
#: Partial redesign shares the unconditional input; what separates them is
#: where on the trajectory sampling starts.
WORKFLOWS: dict[str, tuple[str, tuple[str, ...]]] = {
    "unconditional": ("monomer", ()),
    "adherence": ("adherence/monomer_helical", ()),
    "motif_scaffolding": ("motif_scaffolding/motif_indexed", ()),
    "ppi_binder": ("ppi/ppi_bb_only", ()),
    "small_molecule_binder": ("small_molecule_binder/sm_binder", ()),
    "nucleic_acid_binder": ("dna_binder/dna_binder_denovo_val", ()),
    "rna_binder": ("rna/pseudoknot", ()),
    "nanobody": ("nanobody", ()),
    "enzyme": ("mcsa/mcsa_37", ()),
    "structure_prediction": ("structure_prediction/de_novo_proteins", ()),
    "partial_redesign": ("monomer", ("generation.args.partial_trajectory_fraction_steps=0.3",)),
}

BACKENDS = ("bioir", "torch_compile", "eager")

#: Lightning's progress bar reports elapsed sampling time as ``H:MM:SS • ``.
#: Parsing it is how the rollout is separated from checkpoint loading without
#: instrumenting the generation entry point; an unreadable bar reports None
#: and the row falls back to the total.
_ELAPSED = re.compile(r"(\d+):(\d\d):(\d\d) • ")


def one_input(relative: str) -> Path:
    """Resolve a workflow's single benchmark structure."""
    path = BENCHMARKS / relative
    if path.is_file():
        return path
    files = sorted(child for child in path.iterdir() if child.is_file() and ".cif" in child.suffixes)
    if not files:
        raise SystemExit(f"no structure input under {path}")
    return files[0]


def rollout_seconds(log: Path) -> int | None:
    """Longest elapsed time the progress bar reported, or None if it printed none."""
    if not log.is_file():
        return None
    spans = [
        int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        for hours, minutes, seconds in _ELAPSED.findall(log.read_text(errors="ignore"))
    ]
    return max(spans) if spans else None


def run_one(workflow: str, backend: str, options: argparse.Namespace) -> dict:
    """Generate one design and time it."""
    relative, overrides = WORKFLOWS[workflow]
    out_dir = options.out_dir / f"{workflow}__{backend}"
    log = options.out_dir / f"{workflow}__{backend}.log"
    command = [
        sys.executable,
        "-m",
        "rfproteina.generate",
        f"inputs={one_input(relative)}",
        f"inference_backend={backend}",
        f"recycle_strategy={options.recycle_strategy}",
        f"seed={options.seed}",
        "num_replicates=1",
        "num_workers=0",
        # A rerun into a populated directory returns the previous designs in
        # seconds, which would read as an extraordinary speedup.
        "skip_existing=false",
        f"out_dir={out_dir}",
        *(f"~metrics.inline_metrics.{name}" for name in options.drop_metrics),
        *overrides,
    ]
    started = time.perf_counter()
    with log.open("w") as handle:
        completed = subprocess.run(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    total = time.perf_counter() - started
    designs = [path for path in out_dir.rglob("*.cif*") if "input" not in path.name]
    return {
        "workflow": workflow,
        "backend": backend,
        "returncode": completed.returncode,
        "total_seconds": round(total, 2),
        "rollout_seconds": rollout_seconds(log),
        "designs": len(designs),
        "log": str(log),
    }


def report(results: list[dict], backends: tuple[str, ...]) -> None:
    """Print one row per workflow, rollout time first."""
    by_workflow: dict[str, dict[str, dict]] = {}
    for result in results:
        by_workflow.setdefault(result["workflow"], {})[result["backend"]] = result

    baseline = backends[-1]
    columns = "".join(f"{backend:>16}" for backend in backends)
    print(f"\n{'workflow':<24}{columns}{'speedup':>10}")
    speedups = []
    for workflow, runs in by_workflow.items():
        cells = ""
        for backend in backends:
            run = runs.get(backend)
            if run is None:
                cells += f"{'-':>16}"
            elif run["returncode"] != 0:
                cells += f"{'failed':>16}"
            else:
                rollout = run["rollout_seconds"]
                shown = f"{rollout}s" if rollout is not None else f"~{run['total_seconds']:.0f}s"
                cells += f"{shown:>16}"
        fastest, slowest = runs.get(backends[0]), runs.get(baseline)
        speedup = None
        if (
            fastest
            and slowest
            and fastest["returncode"] == 0
            and slowest["returncode"] == 0
            and fastest["rollout_seconds"]
            and slowest["rollout_seconds"]
        ):
            speedup = slowest["rollout_seconds"] / fastest["rollout_seconds"]
            speedups.append(speedup)
        print(f"{workflow:<24}{cells}{(f'{speedup:.2f}x' if speedup else '-'):>10}")

    failed = [f"{r['workflow']}/{r['backend']}" for r in results if r["returncode"] != 0]
    if failed:
        print(f"\nfailed before producing a design: {', '.join(failed)}")
    if speedups:
        ordered = sorted(speedups)
        print(
            f"\n{backends[0]} against {baseline}: median {ordered[len(ordered) // 2]:.2f}x, "
            f"range {ordered[0]:.2f}x-{ordered[-1]:.2f}x over {len(ordered)} workflows"
        )
    print("\nRollout time excludes checkpoint loading; the progress bar reports whole seconds.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workflows", default=",".join(WORKFLOWS), help="comma-separated subset to run")
    parser.add_argument("--backends", default="bioir,torch_compile", help="comma-separated, fastest expected first")
    parser.add_argument("--recycle-strategy", default="thinking_medium", help="sampling schedule for every run")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "runs" / "benchmark_backends")
    parser.add_argument(
        "--drop-metrics",
        default="",
        help="comma-separated inline metrics to remove, for environments missing their dependencies",
    )
    options = parser.parse_args()

    workflows = [name.strip() for name in options.workflows.split(",") if name.strip()]
    backends = tuple(name.strip() for name in options.backends.split(",") if name.strip())
    options.drop_metrics = [name.strip() for name in options.drop_metrics.split(",") if name.strip()]
    unknown = sorted(set(workflows) - set(WORKFLOWS))
    if unknown:
        raise SystemExit(f"unknown workflows {unknown}; choose from {sorted(WORKFLOWS)}")
    unsupported = sorted(set(backends) - set(BACKENDS))
    if unsupported:
        raise SystemExit(f"unknown backends {unsupported}; choose from {sorted(BACKENDS)}")

    options.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = options.out_dir / "benchmark_backends.json"
    results: list[dict] = []
    for workflow in workflows:
        for backend in backends:
            print(f"==> {workflow} / {backend}", flush=True)
            result = run_one(workflow, backend, options)
            results.append(result)
            report_path.write_text(json.dumps(results, indent=2) + "\n")
            status = "ok" if result["returncode"] == 0 else f"exit {result['returncode']}"
            print(f"    {status}, {result['total_seconds']}s total, {result['designs']} design(s)", flush=True)

    report(results, backends)
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
