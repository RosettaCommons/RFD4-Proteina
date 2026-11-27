"""Run and inventory a generation shard without changing the model's sampling code."""

import base64
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

OVERRIDE_KEYS = {
    "recycle_strategy",
    "compile_lengths",
    "precision",
    "compress_outputs",
    "num_workers",
    "save_chain_of_interest",
    "prevalidate_inputs",
}
OVERRIDE_PREFIXES = ("nn.", "generation.args.", "generation.model.", "generation.n_recycle_schedule")
CIF_SUFFIXES = (".cif", ".cif.gz", ".cif.zst")
GPU_PROBE = """
import json, torch
if not torch.cuda.is_available():
    raise RuntimeError("RFD4 generation requires a CUDA GPU; check the task accelerator and container profile")
print(json.dumps({
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "visible_gpus": torch.cuda.device_count()
}))
"""


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    """Validate integers without accepting booleans or truncating floats."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def generated_inventory(directory: Path) -> list[dict[str, Any]]:
    """Inventory native generated files, including partial outputs after a CLI failure."""
    return [
        {"path": str(path.relative_to(directory)), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted((directory / "inference_samples").rglob("*"))
        if path.is_file() and any(path.name.endswith(f"_generated{suffix}") for suffix in CIF_SUFFIXES)
    ]


def check_native_seed(directory: Path, record: dict[str, Any]) -> int:
    """Check the effective seed reported by the model, independently of the command plan."""
    path = directory / "inference_metrics" / f"per_example_metrics_job{record['shard_id']}.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != record["expected_designs"]:
        raise ValueError("Native inventory row count does not match the shard's expected designs")
    if any(float(row["seed"]) != record["effective_seed"] for row in rows):
        raise ValueError("Native inventory seed does not match the shard's effective seed")
    observed = sorted(Path(row["designed_structure_path"]).name for row in rows)
    expected = sorted(Path(item["path"]).name for item in record["generated_files"])
    if observed != expected:
        raise ValueError("Native inventory does not match the generated filenames")
    return record["effective_seed"]


def check_shard_files(directory: Path, record: dict[str, Any]) -> None:
    """Verify transferred files rather than trusting counts in a producer's manifest."""
    if record["status"] == "STUB":
        if record["generated_designs"] != 0 or record["generated_files"]:
            raise ValueError("A STUB manifest must not claim generated designs")
        return
    if record["generated_designs"] != record["expected_designs"]:
        raise ValueError("Shard generated count does not match its expected designs")
    listed = record["generated_files"]
    if len(listed) != record["expected_designs"]:
        raise ValueError("Generated file inventory count is inconsistent")
    for item in listed:
        relative = Path(item["path"])
        path = directory / relative
        if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Generated file path escapes its shard directory")
        if not path.is_file() or item["bytes"] <= 0 or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Missing, empty or resized generated file: {item['path']}")
    observed = generated_inventory(directory)
    if sorted(listed, key=lambda item: item["path"]) != observed:
        raise ValueError("Generated file inventory or SHA-256 mismatch after staging")
    record["native_seed"] = check_native_seed(directory, record)


def prepare(request: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Validate staged inputs and preserve the CLI's replicate and seed semantics."""
    input_cif = Path(request["input_cif"]).resolve()
    bundle = Path(request["model_bundle"]).resolve()
    settings = json.loads(Path(request["settings"]).read_text())
    if not isinstance(settings, dict):
        raise ValueError("settings must be a JSON object")
    if not input_cif.is_file() or not input_cif.name.endswith(CIF_SUFFIXES):
        raise ValueError("input_cif must be one existing annotated .cif, .cif.gz or .cif.zst file")
    for name in ("config.json", "model.safetensors"):
        if not (bundle / name).is_file():
            raise ValueError(f"model_bundle is missing {name}")
    replicates = positive_integer(settings.get("num_replicates"), "num_replicates")
    batch_size = positive_integer(settings.get("batch_size"), "batch_size")
    seed = positive_integer(settings.get("seed"), "seed", minimum=0)
    shard_id = positive_integer(request["shard_id"], "shard_id", minimum=0)
    shards = positive_integer(request["num_shards"], "num_shards")
    if not shard_id < shards <= replicates:
        raise ValueError("Require 0 <= shard_id < num_shards <= num_replicates")
    if seed + shards - 1 >= 2**32:
        raise ValueError("seed + num_shards - 1 must be less than 2**32")
    overrides = settings.get("overrides", [])
    if not isinstance(overrides, list) or any(not isinstance(item, str) for item in overrides):
        raise ValueError("overrides must be a JSON list of Hydra key=value strings")
    for item in overrides:
        key, separator, _ = item.partition("=")
        if not separator or (key not in OVERRIDE_KEYS and not key.startswith(OVERRIDE_PREFIXES)):
            raise ValueError(
                f"Unsupported override {key!r}; workflow inputs, metrics and search are managed separately"
            )

    out_dir = Path(f"shard-{shard_id}").resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory {out_dir}")
    out_dir.mkdir()
    command = [
        "rfd4-generate",
        f"inputs={json.dumps(str(input_cif), ensure_ascii=False)}",
        f"hf_ckpt_path={json.dumps(str(bundle), ensure_ascii=False)}",
        f"out_dir={json.dumps(str(out_dir), ensure_ascii=False)}",
        f"seed={seed}",
        f"num_replicates={replicates}",
        f"batch_size={batch_size}",
        f"job_id={shard_id}",
        f"gen_njobs={shards}",
        "hardware.ngpus_per_node_=1",
        "hardware.nnodes_=1",
        "skip_existing=false",
        "delete_failing_pre_oracle=false",
        "~metrics",
        f"hydra.run.dir={json.dumps(str(out_dir / 'hydra'), ensure_ascii=False)}",
        *overrides,
    ]
    try:
        version = importlib.metadata.version("rfd4-proteina")
    except importlib.metadata.PackageNotFoundError:
        version = None
    record = {
        "status": "PENDING",
        "shard_id": shard_id,
        "num_shards": shards,
        "requested_replicates": replicates,
        "expected_designs": replicates // shards + int(shard_id < replicates % shards),
        "generated_designs": 0,
        "base_seed": seed,
        "effective_seed": seed + shard_id,
        "settings": settings,
        "input_name": input_cif.name,
        "input_sha256": sha256(input_cif),
        "model_files_sha256": {name: sha256(bundle / name) for name in ("config.json", "model.safetensors")},
        "container_image": request["container_image"],
        "package_version": version,
        "runner_sha256": sha256(Path(__file__)),
        "healthomics_run_id": os.environ.get("AWS_WORKFLOW_RUN"),
        "command": command,
        "generated_files": [],
    }
    return out_dir, record


def generate(request: dict[str, Any], *, plan_only: bool = False) -> None:
    """Run generation, saving provenance even when the subprocess or output checks fail."""
    out_dir, record = prepare(request)
    started = time.monotonic()
    try:
        if plan_only:
            record["status"] = "STUB"
            return
        command = record["command"]
        environment = os.environ.copy()
        # HealthOmics shares /tmp across tasks; keep Inductor writes in this task's work directory.
        environment.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path.cwd() / ".torchinductor"))
        probe = subprocess.run(
            [sys.executable, "-c", GPU_PROBE], env=environment, capture_output=True, text=True, check=True
        )
        record["runtime"] = json.loads(probe.stdout)
        with (out_dir / "effective_config.yaml").open("w") as config_file:
            subprocess.run([*command, "--cfg", "job", "--resolve"], stdout=config_file, env=environment, check=True)
        subprocess.run(command, env=environment, check=True)
        record["generated_files"] = generated_inventory(out_dir)
        record["generated_designs"] = len(record["generated_files"])
        if record["generated_designs"] != record["expected_designs"] or any(
            item["bytes"] == 0 for item in record["generated_files"]
        ):
            raise RuntimeError(
                f"Expected {record['expected_designs']} nonempty designs, found {record['generated_designs']}"
            )
        record["native_seed"] = check_native_seed(out_dir, record)
        record["status"] = "COMPLETED"
    except Exception as error:
        record["status"] = "FAILED"
        record["error"] = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            record["error"] += "\n" + error.stderr
        raise
    finally:
        if record["status"] == "FAILED":
            try:
                record["generated_files"] = generated_inventory(out_dir)
                record["generated_designs"] = len(record["generated_files"])
            except OSError as inventory_error:
                record["inventory_error"] = str(inventory_error)
        record["elapsed_seconds"] = time.monotonic() - started
        if record["status"] == "FAILED":
            # Failed Nextflow tasks do not publish their output directories. Preserve the
            # manifest in task stderr so HealthOmics CloudWatch logs retain the evidence.
            print("RFD4_SHARD_FAILURE " + json.dumps(record), file=sys.stderr, flush=True)
        (out_dir / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


def summarize(directory: Path, expected_shards: int) -> None:
    """Aggregate verified shard inventories without treating stub runs as real generation."""
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("shard-*/provenance.json"))]
    ids = sorted(record["shard_id"] for record in records)
    if ids != list(range(expected_shards)):
        raise ValueError("Missing or duplicate shard manifests")
    if records[0]["num_shards"] != expected_shards:
        raise ValueError("Shard manifest count disagrees with the requested workflow size")
    states = {record["status"] for record in records}
    if states not in ({"COMPLETED"}, {"STUB"}):
        raise ValueError(f"Cannot summarize incomplete or mixed runs: {states}")
    for key in (
        "input_sha256",
        "model_files_sha256",
        "settings",
        "num_shards",
        "requested_replicates",
        "container_image",
        "runner_sha256",
        "package_version",
    ):
        if any(record[key] != records[0][key] for record in records[1:]):
            raise ValueError(f"Shard manifests disagree on {key}")
    for record in records:
        if record["requested_replicates"] != record["settings"]["num_replicates"]:
            raise ValueError("Shard replicate count disagrees with settings")
        expected = record["requested_replicates"] // expected_shards + int(
            record["shard_id"] < record["requested_replicates"] % expected_shards
        )
        if record["expected_designs"] != expected:
            raise ValueError("Shard expected count disagrees with native replicate splitting")
        if record["base_seed"] != record["settings"]["seed"] or record["effective_seed"] != (
            record["base_seed"] + record["shard_id"]
        ):
            raise ValueError("Shard seed metadata is inconsistent")
        check_shard_files(directory / f"shard-{record['shard_id']}", record)
    total = sum(record["generated_designs"] for record in records)
    if states == {"COMPLETED"} and total != records[0]["requested_replicates"]:
        raise ValueError("Generated count does not match requested replicates")
    out_dir = Path("summary")
    out_dir.mkdir(exist_ok=False)
    summary = {
        "status": next(iter(states)),
        "requested_replicates": records[0]["requested_replicates"],
        "generated_designs": total,
        "shards": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ["shard_id", "status", "effective_seed", "expected_designs", "generated_designs", "elapsed_seconds"]
    with (out_dir / "shards.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda record: record["shard_id"]))


def main() -> None:
    """Dispatch the Nextflow task helper."""
    if len(sys.argv) == 3 and sys.argv[1] in {"generate", "plan"}:
        request = json.loads(base64.b64decode(sys.argv[2], validate=True))
        generate(request, plan_only=sys.argv[1] == "plan")
    elif len(sys.argv) == 4 and sys.argv[1] == "summarize":
        summarize(Path(sys.argv[2]), positive_integer(int(sys.argv[3]), "expected_shards"))
    else:
        raise SystemExit("Usage: workflow.py {generate|plan} REQUEST_BASE64 | summarize DIRECTORY NUM_SHARDS")


if __name__ == "__main__":
    main()
