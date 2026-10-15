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

``--lengths`` sweeps unconditional design over chain lengths instead, writing
one de-novo input per length. ``--batch-sizes`` sweeps the batch beside it.
The discarded run warms persistent compiler caches. Every measured run starts
a fresh process and repeats graph capture; rollout is whole-second validation
progress, including formatting and output staging.

Each run initialises the CUDA context and the BLAS/DNN handles before the timed
region, because that cost is per process rather than per shape and is identical
on every backend. ``--no-device-warmup`` leaves it in.

Examples:
    python src/script_utils/benchmark_backends.py
    python src/script_utils/benchmark_backends.py --workflows unconditional --backends bioir
    python src/script_utils/benchmark_backends.py --lengths 128,256,400,600 --batch-sizes 1,2,4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

PROJECT = Path(__file__).resolve().parents[2]
BENCHMARKS = PROJECT / "data" / "benchmarks"

#: Argument that re-enters this file as the generation subprocess. The warmup and
#: the per-design timing live here rather than in a helper module so that the whole
#: measurement -- what is warmed, what is timed, and where the seam is -- is one file.
CHILD_FLAG = "--run-generation"

#: Set to "0" by the parent when ``--no-device-warmup`` is passed.
WARMUP_ENV = "BENCH_DEVICE_WARMUP"


def warm_device() -> float:
    """Initialise the CUDA context, the BLAS/DNN handles, and their workspaces."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    started = time.perf_counter()
    device = torch.device("cuda")
    torch.zeros(1, device=device)
    for dtype in (torch.bfloat16, torch.float32):
        operand = torch.randn(512, 512, device=device, dtype=dtype)
        (operand @ operand).sum().item()
    query = torch.randn(1, 8, 256, 64, device=device, dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(query, query, query).sum().item()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def install_design_timing() -> None:
    """Print one line per design: end to end, and the model forward inside it.

    ``Proteina.predict_step`` is the seam every backend crosses -- BioIR reaches
    it through ``predict_full_bioir`` -- so it is the end-to-end number, carrying
    featurisation, the model, and formatting into atom arrays.

    The forward is the model call inside that seam, scoped to match across
    backends: BioIR runs trunk, sampler and decoder in one call, while the native
    path integrates the flow and decodes separately, so both of its halves are
    accumulated. Dividing one backend's forward into the other's is only fair
    because of that.

    The example id is printed rather than the token count because the padding
    backends report a padded shape, and the id names the input either way.
    """
    import torch

    from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
    from rfproteina.proteina import Proteina

    forward = {"seconds": 0.0}

    def accumulate(fn):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            forward["seconds"] += time.perf_counter() - started
            return out

        return wrapper

    ProductSpaceFlowMatcher.full_simulation = accumulate(ProductSpaceFlowMatcher.full_simulation)
    Proteina.sample_formatting = accumulate(Proteina.sample_formatting)
    try:
        from bionemo_ir.models.rfproteina.modeling import RFProteina

        RFProteina.forward = accumulate(RFProteina.forward)
    except ImportError:
        pass

    inner = Proteina.predict_step
    index = {"n": 0}

    def predict_step(self, batch, batch_idx):
        forward["seconds"] = 0.0
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = inner(self, batch, batch_idx)
        torch.cuda.synchronize()
        index["n"] += 1
        mask = batch["padding_mask"]
        ids = batch.get("example_id_batch") or ["?"]
        print(
            f"BENCH design={index['n']} example={ids[0]} tokens={mask.shape[-1]} "
            f"rows={mask.shape[0]} e2e={time.perf_counter() - started:.4f} "
            f"forward={forward['seconds']:.4f}",
            flush=True,
        )
        return out

    Proteina.predict_step = predict_step


def run_generation(argv: list[str]) -> int:
    """Subprocess entry: warm the device, install timing, then run generation."""
    if os.environ.get(WARMUP_ENV, "1") != "0":
        print(f"device warmup: {warm_device():.3f}s before generation", flush=True)
    install_design_timing()
    sys.argv = ["rfproteina.generate", *argv]
    runpy.run_module("rfproteina.generate", run_name="__main__")
    return 0


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


#: The monomer benchmark inputs are de-novo length specifications rather than
#: structures: an expandable segment of ``<min>-<max>`` residues with no
#: coordinates. Rewriting that span is how a sweep asks for an exact length.
_DENOVO_TEMPLATE = BENCHMARKS / "monomer" / "monomer-short.cif"
_TEMPLATE_SPAN = "100-100"
_EXPSEG_PREFIX = "_atom_site.annotation_expseg"


def write_denovo_input(length: int, directory: Path) -> Path:
    """Write an unconditional input asking for a chain of exactly ``length`` residues."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"denovo-{length}.cif"
    span = f"{length}-{length}"
    lines = []
    for line in _DENOVO_TEMPLATE.read_text().splitlines():
        if _TEMPLATE_SPAN in line:
            line = line.replace(_TEMPLATE_SPAN, span)
        elif line.startswith(_EXPSEG_PREFIX):
            field, _, _ = line.partition(" ")
            line = f"{field}{' ' * max(1, 45 - len(field))}{length}"
        elif line.startswith("data_") or line.startswith("_entry.id"):
            line = line.replace("monomer-short", f"denovo-{length}")
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    return path


def one_input(relative: str) -> Path:
    """Resolve a workflow's single benchmark structure."""
    path = BENCHMARKS / relative
    if path.is_file():
        return path
    files = sorted(child for child in path.iterdir() if child.is_file() and ".cif" in child.suffixes)
    if not files:
        raise SystemExit(f"no structure input under {path}")
    return files[0]


#: Length the in-process warm design runs at, so that everything shape-independent
#: -- CUDA context, BLAS/DNN handles, kernel modules, featurisation, formatting --
#: is already paid when the measured shape starts, while that shape's own
#: compilation and graph capture are not.
WARM_LENGTH = 192
WARM_ALTERNATE = 224


def warm_and_measure(length: int, options: argparse.Namespace) -> Path:
    """One de-novo input for one length.

    Lengths are deliberately **not** pooled into a single process. Generation
    collates across inputs, so a batch drawn from a directory holding 128 and 400
    residues pads to 400 and reports a 128-residue design that never ran at 128.
    That is safe only at batch 1, which is too narrow a guarantee to build on, so
    each length gets its own process and the batches stay homogeneous.
    """
    return write_denovo_input(length, options.out_dir / "inputs")


def rollout_seconds(log: Path) -> int | None:
    """Longest elapsed time the progress bar reported, or None if it printed none."""
    if not log.is_file():
        return None
    spans = [
        int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        for hours, minutes, seconds in _ELAPSED.findall(log.read_text(errors="ignore"))
    ]
    return max(spans) if spans else None


_DESIGN = re.compile(
    r"BENCH design=(\d+) example=(\S+) tokens=(\d+) rows=(\d+) e2e=([0-9.]+) forward=([0-9.]+)"
)


#: ``1-measure-128_rep0`` -> 128. Replicates append their own suffix.
_LENGTH_IN_ID = re.compile(r"-(\d+)(?:_rep\d+)?$")


def design_rows(log: Path) -> list[dict]:
    """Every design the wrapper timed, with its length taken from the example id.

    The id is used rather than the token count because a padding backend reports
    the padded shape; ``2-measure-400`` is 400 residues whichever bucket it ran in.
    """
    if not log.is_file():
        return []
    rows = []
    for _, example, tokens, batch, e2e, forward in _DESIGN.findall(log.read_text(errors="ignore")):
        found = _LENGTH_IN_ID.search(example)
        if found is None:
            continue
        rows.append(
            {
                "example": example,
                "length": int(_LENGTH_IN_ID.search(example).group(1)),
                "tokens": int(tokens),
                "rows": int(batch),
                "e2e": float(e2e),
                "forward": float(forward),
            }
        )
    return rows


def cold_and_warm(values: list[float]) -> tuple[float | None, float | None]:
    """First design beside the median of the rest.

    Design 1 carries what the warmup could not: this shape's compilation or graph
    capture. Their ratio is the check that the warmup sufficed -- near 1.00x when
    only shape-independent setup was left to do.
    """
    if not values:
        return None, None
    if len(values) == 1:
        return round(values[0], 4), round(values[0], 4)
    return round(values[0], 4), round(median(values[1:]), 4)


def generate(
    inputs: Path,
    backend: str,
    batch: int,
    out_dir: Path,
    log: Path,
    options: argparse.Namespace,
    overrides: tuple[str, ...],
) -> tuple[int, float]:
    """Run one generation, returning its exit status and wall time."""
    entry = [str(Path(__file__).resolve()), CHILD_FLAG]
    command = [
        sys.executable,
        *entry,
        f"inputs={inputs}",
        f"inference_backend={backend}",
        f"recycle_strategy={options.recycle_strategy}",
        f"seed={options.seed}",
        f"num_replicates={batch * options.designs}",
        f"batch_size={batch}",
        "num_workers=0",
        # A rerun into a populated directory returns the previous designs in
        # seconds, which would read as an extraordinary speedup.
        "skip_existing=false",
        f"out_dir={out_dir}",
        *(f"~metrics.inline_metrics.{name}" for name in options.drop_metrics),
        *overrides,
    ]
    environment = {**os.environ, WARMUP_ENV: "1" if options.device_warmup else "0"}
    started = time.perf_counter()
    with log.open("w") as handle:
        completed = subprocess.run(
            command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT, check=False, env=environment
        )
    return completed.returncode, time.perf_counter() - started


def run_one(case: dict, backend: str, options: argparse.Namespace) -> dict:
    """Warm one shape, then measure it.

    The discarded process warms persistent compiler caches. The measured
    process reloads the model and repeats graph capture.
    """
    workflow, inputs, batch = case["workflow"], Path(case["inputs"]), case["batch"]
    overrides = WORKFLOWS[workflow][1]
    stem = f"{case['slug']}__{backend}"
    out_dir = options.out_dir / stem
    log = options.out_dir / f"{stem}.log"

    warmup_status = None
    if options.warmup:
        warmup_status, _ = generate(
            inputs,
            backend,
            batch,
            options.out_dir / f"{stem}__warmup",
            options.out_dir / f"{stem}__warmup.log",
            options,
            overrides,
        )
    status, total = generate(inputs, backend, batch, out_dir, log, options, overrides)
    designs = [path for path in out_dir.rglob("*.cif*") if "input" not in path.name]
    measured = [row for row in design_rows(log) if row["length"] != case["warm_length"]]
    return {
        "designs_timed": measured,
        "workflow": workflow,
        "label": case["label"],
        "lengths": case.get("lengths", []),
        "batch": batch,
        "backend": backend,
        "returncode": status,
        "warmup_returncode": warmup_status,
        "total_seconds": round(total, 2),
        "rollout_seconds": rollout_seconds(log),
        "designs": len(designs),
        "log": str(log),
    }


def _per_shape(results: list[dict]) -> dict[tuple, dict[str, dict]]:
    """``(length, batch) -> backend -> {metric: (design 1, steady)}``."""
    table: dict[tuple, dict[str, dict]] = {}
    for run in results:
        if run["returncode"] != 0:
            continue
        by_length: dict[int, list[dict]] = {}
        for row in run.get("designs_timed", []):
            by_length.setdefault(row["length"], []).append(row)
        for length, rows in by_length.items():
            cell = table.setdefault((length, run["batch"]), {})
            cell[run["backend"]] = {
                metric: cold_and_warm([row[metric] for row in rows]) for metric in ("forward", "e2e")
            }
    return table


def report(results: list[dict], backends: tuple[str, ...]) -> None:
    """Steady seconds per batch, model forward beside end to end."""
    table = _per_shape(results)
    if not table:
        print("\nno timed designs parsed; see the logs")
        return
    baseline = backends[-1]
    header = "".join(f"{b + ' fwd':>14}{b + ' e2e':>14}" for b in backends)
    print(f"\n{'shape':<16}{header}{'fwd':>9}{'e2e':>9}")
    gains: dict[str, list[float]] = {"forward": [], "e2e": []}
    for (length, batch) in sorted(table):
        cells, speed = "", {}
        for backend in backends:
            cell = table[(length, batch)].get(backend)
            for metric in ("forward", "e2e"):
                steady = cell[metric][1] if cell else None
                cells += f"{steady:>13.3f}s" if steady else f"{'-':>14}"
        for metric in ("forward", "e2e"):
            quick = table[(length, batch)].get(backends[0], {}).get(metric, (None, None))[1]
            slow = table[(length, batch)].get(baseline, {}).get(metric, (None, None))[1]
            speed[metric] = slow / quick if quick and slow else None
            if speed[metric]:
                gains[metric].append(speed[metric])
        trailing = "".join(f"{speed[m]:>8.2f}x" if speed[m] else f"{'-':>9}" for m in ("forward", "e2e"))
        print(f"{f'N={length} bs={batch}':<16}{cells}{trailing}")

    for metric in ("forward", "e2e"):
        values = sorted(gains[metric])
        if values:
            print(
                f"\n{backends[0]} against {baseline} on {metric}: "
                f"median {values[len(values) // 2]:.2f}x, range {values[0]:.2f}x-{values[-1]:.2f}x "
                f"over {len(values)} shapes"
            )

    print(f"\n{'shape':<16}{'backend':<16}{'design 1':>12}{'steady':>12}{'ratio':>9}  (end to end)")
    for (length, batch) in sorted(table):
        for backend in backends:
            cell = table[(length, batch)].get(backend)
            if not cell:
                continue
            cold, steady = cell["e2e"]
            if cold and steady:
                print(
                    f"{f'N={length} bs={batch}':<16}{backend:<16}"
                    f"{cold:>11.3f}s{steady:>11.3f}s{cold / steady:>8.2f}x"
                )

    failed = [f"{r['label']}/{r['backend']}" for r in results if r["returncode"] != 0]
    if failed:
        print(f"\nfailed before producing a design: {', '.join(failed)}")
    print("\nSeconds are per batch -- one predict_step over all its rows, so divide by")
    print("the batch size for per-design cost. The CUDA context and BLAS/DNN handles are")
    print("warmed")
    print("before the timed region. Steady is the median of designs 2+; design 1 is")
    print("above it, and what separates them is first-touch in the model plus this")
    print("shape's compilation or graph capture. One process per length and batch:")
    print("generation collates across inputs, so pooling lengths would pad a small")
    print("shape up to a large one at batch > 1.")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == CHILD_FLAG:
        return run_generation(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workflows", default=",".join(WORKFLOWS), help="comma-separated subset to run")
    parser.add_argument("--backends", default="bioir,torch_compile", help="comma-separated, fastest expected first")
    parser.add_argument("--recycle-strategy", default="thinking_medium", help="sampling schedule for every run")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "runs" / "benchmark_backends")
    parser.add_argument(
        "--lengths",
        default="",
        help="comma-separated chain lengths to sweep unconditionally, instead of the benchmark inputs",
    )
    parser.add_argument("--batch-sizes", default="1", help="comma-separated batch sizes; one batch per run")
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="measure the first run of each shape, including compilation and graph capture",
    )
    parser.add_argument(
        "--designs",
        type=int,
        default=3,
        help="designs per shape in one process; design 1 is reported apart from the median of the rest",
    )
    parser.add_argument(
        "--no-device-warmup",
        dest="device_warmup",
        action="store_false",
        help="leave one-time CUDA context and cuBLAS/cuDNN setup inside the timed rollout",
    )
    parser.add_argument(
        "--drop-metrics",
        default="",
        help="comma-separated inline metrics to remove, for environments missing their dependencies",
    )
    options = parser.parse_args()

    workflows = [name.strip() for name in options.workflows.split(",") if name.strip()]
    backends = tuple(name.strip() for name in options.backends.split(",") if name.strip())
    lengths = [int(value) for value in options.lengths.split(",") if value.strip()]
    batches = [int(value) for value in options.batch_sizes.split(",") if value.strip()]
    options.drop_metrics = [name.strip() for name in options.drop_metrics.split(",") if name.strip()]
    unknown = sorted(set(workflows) - set(WORKFLOWS))
    if unknown:
        raise SystemExit(f"unknown workflows {unknown}; choose from {sorted(WORKFLOWS)}")
    unsupported = sorted(set(backends) - set(BACKENDS))
    if unsupported:
        raise SystemExit(f"unknown backends {unsupported}; choose from {sorted(BACKENDS)}")
    if lengths and workflows != ["unconditional"]:
        raise SystemExit("--lengths sweeps unconditional design; pass --workflows unconditional")

    options.out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for batch in batches:
        if lengths:
            for length in lengths:
                cases.append(
                    {
                        "workflow": "unconditional",
                        "inputs": str(warm_and_measure(length, options)),
                        "lengths": [length],
                        "warm_length": -1,
                        "batch": batch,
                        "label": f"N={length} bs={batch}",
                        "slug": f"n{length}_bs{batch}",
                    }
                )
        else:
            for workflow in workflows:
                inputs = one_input(WORKFLOWS[workflow][0])
                label = workflow if batch == 1 else f"{workflow} bs={batch}"
                cases.append(
                    {
                        "workflow": workflow,
                        "inputs": str(inputs),
                        "lengths": [],
                        "warm_length": -1,
                        "batch": batch,
                        "label": label,
                        "slug": f"{workflow}_bs{batch}",
                    }
                )

    report_path = options.out_dir / "benchmark_backends.json"
    results: list[dict] = []
    for case in cases:
        for backend in backends:
            print(f"==> {case['label']} / {backend}", flush=True)
            result = run_one(case, backend, options)
            results.append(result)
            report_path.write_text(json.dumps(results, indent=2) + "\n")
            status = "ok" if result["returncode"] == 0 else f"exit {result['returncode']}"
            rollout = result["rollout_seconds"]
            print(
                f"    {status}, rollout {rollout}s, {result['total_seconds']}s total, "
                f"{result['designs']} design(s)",
                flush=True,
            )

    report(results, backends)
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
