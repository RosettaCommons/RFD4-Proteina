"""Refresh the website from the manuscript's reviewed per-target benchmark tables.

The paper root may be a read-only export of the specified commit. No model runs
or database writes are performed. Keep every original attempt in the denominator.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--snapshot-date", required=True)
    args = parser.parse_args()
    site = Path(__file__).resolve().parents[1]
    output = site / "benchmarks"
    existing = json.loads((output / "data.json").read_text())
    sources = {
        "source_data.csv": "figure_sources/supplementary/benchmark_targets/data/source_data.csv",
        "scatter.csv": "figure_sources/figure_02/data/01_scatter.csv",
        "coverage.csv": "figure_sources/figure_02/data/02_coverage.csv",
        "target-labels.csv": "figure_sources/supplementary/benchmark_targets/data/target-labels.csv",
    }
    tables = {name: read_csv(args.paper_root / path) for name, path in sources.items()}
    hashes = {
        name: {"source_path": path, "source_sha256": hashlib.sha256((args.paper_root / path).read_bytes()).hexdigest()}
        for name, path in sources.items()
    }
    integers = {
        "attempts",
        "scored",
        "passes",
        "clusters",
        "disco_passes",
        "disco_clusters",
        "generation_timed_attempts",
        "generation_unknown_attempts",
        "af3_timed_attempts",
        "generation_proxy_attempts",
        "publication_timed_attempts",
    }
    decimals = {
        "generation_gpu_s",
        "af3_gpu_s",
        "pass_percent",
        "clustered_percent",
        "successes_per_gpu_hour",
        "unique_per_gpu_hour",
        "known_generation_gpu_s",
        "known_af3_gpu_s",
        "measured_generation_gpu_s",
        "generation_proxy_gpu_s",
    }
    strings = {
        "run_id",
        "model",
        "protocol",
        "dataset",
        "target",
        "timing_basis",
        "source_sha256",
        "criterion",
        "catalog_model",
        "catalog_export_sha256",
        "score_receipt_sha256",
        "raw_timing_basis",
        "publication_generation_cost_basis",
    }
    # Publish measurement provenance, without machine-local receipt paths.
    fields = [
        k
        for k in tables["source_data.csv"][0]
        if k in integers | decimals | strings or k == "publication_timing_complete"
    ]
    paper_arms = {
        (r["model"].split("__")[0], r["protocol"], r["benchmark"])
        for r in tables["scatter.csv"]
        if r["displayed"] == "True"
    }
    rows, identities = [], set()
    for raw in tables["source_data.csv"]:
        row = {}
        for key in fields:
            value = raw[key]
            if not value:
                row[key] = None
            elif key in integers:
                number = float(value)
                assert number.is_integer(), (key, value)
                row[key] = int(number)
            elif key in decimals:
                row[key] = float(value)
                assert math.isfinite(row[key]), (key, value)
            elif key == "publication_timing_complete":
                assert value in ("True", "False")
                row[key] = value == "True"
            else:
                row[key] = value
        identity = tuple(row[k] for k in ("model", "protocol", "dataset", "target", "criterion"))
        assert identity not in identities, identity
        identities.add(identity)
        assert 0 <= row["clusters"] <= row["passes"] <= row["scored"] <= row["attempts"] == 200, identity
        for field, count in [("pass_percent", "passes"), ("clustered_percent", "clusters")]:
            assert math.isclose(row[field], 100 * row[count] / row["attempts"], abs_tol=1e-8), identity
        row["figure2"] = (row["model"], row["protocol"], row["dataset"]) in paper_arms
        rows.append(row)
    # Check every plotted arm against the independently supplied plotting table.
    for plot in tables["scatter.csv"]:
        arm = [
            r
            for r in rows
            if r["model"] == plot["model"].split("__")[0]
            and r["protocol"] == plot["protocol"]
            and r["dataset"] == plot["benchmark"]
            and r["criterion"] != "disco"
        ]
        assert len(arm) == int(plot["targets"]), plot["model"]
        assert math.isclose(
            sum(r["clustered_percent"] for r in arm) / len(arm), float(plot["clustered_percent"]), abs_tol=1e-8
        ), plot["model"]
        assert sum(r["passes"] > 0 for r in arm) == int(plot["solved"]), plot["model"]
        if plot["designs_per_gpu_hour"]:
            speed = 3600 * sum(r["attempts"] for r in arm) / sum(r["generation_gpu_s"] for r in arm)
            assert math.isclose(speed, float(plot["designs_per_gpu_hour"]), rel_tol=1e-8), plot["model"]
    models = {name: existing["models"][name] for name in sorted({r["model"] for r in rows})}
    for name, model in models.items():
        if name.startswith("rfd4"):
            model["method"]["text"] = (
                "Joint sequence/structure generation with the recorded low-, medium-, or high-thinking schedule. "
                "The native config and source cohort identify the checkpoint and conditioning. "
                "Search is a separate workflow: AF2-Multimer for PPI and OF3 for the other benchmarks, "
                "with all search work timed. Unmatched required catalytic motifs fail."
            )
    labels = existing["labels"] | {r["target"]: r["label"] for r in tables["target-labels.csv"]}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=[*fields, "figure2"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (output / "source_data.csv").write_text(buffer.getvalue())
    for name in ("scatter.csv", "coverage.csv", "target-labels.csv"):
        (output / name).write_text((args.paper_root / sources[name]).read_text())
    for name in hashes:
        hashes[name]["sha256"] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    payload = {
        "meta": {
            "snapshot": args.snapshot_date,
            "title": "Reviewed manuscript benchmark tables",
            "source_repository": "kierandidi/RFD4-Proteina-paper",
            "source_revision": args.source_commit,
            "source_rows": len(rows),
            "unique_cohorts": len({tuple(r[k] for k in ("model", "protocol", "dataset", "target")) for r in rows}),
            "live_database": False,
            "files": hashes,
        },
        "models": models,
        "labels": labels,
        "rows": rows,
    }
    (output / "data.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(
        f"Synced {len(rows)} rows; validated {len(tables['scatter.csv'])} plotted arms "
        "against source counts, rates, coverage, and timings."
    )


if __name__ == "__main__":
    main()
