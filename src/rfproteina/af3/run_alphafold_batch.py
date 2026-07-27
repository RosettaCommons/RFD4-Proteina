"""Batch fold wrapper for AF3-compatible weights, executed inside af3-open.sif.

Manifest format (JSON list)::

    [{"name": "...", "output_dir": "/abs/path/<example>",
      "cif": "<inline CIF string or path>",
      "template"?: {...}, "user_ccd"?: "...",
      "paired_msa"?: {...}, "unpaired_msa"?: {...}}, ...]

Spawned by :mod:`rfproteina.af3.af3_daemon`. Two run modes:

- One-shot: pass ``--manifest=...`` and the script loads AF3, folds the
  designs, and exits.
- ``--serve``: load AF3 once, then loop reading manifest paths
  (newline-delimited) on stdin; for each, run the fold and print a
  ``{"status": ...}`` JSON line to stdout. Sending the literal line
  ``shutdown`` exits cleanly.
"""

import dataclasses
import datetime
import json
import os
import pathlib
import queue
import sys
import threading
import time
from typing import Any

import af3_runtime as _runtime
import jax
from absl import app, flags
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation as _featurisation

_MANIFEST = flags.DEFINE_string("manifest", None, "JSON manifest with a list of design payloads.")
_SERVE = flags.DEFINE_bool(
    "serve",
    False,
    "If True, load AF3 once and loop reading manifest paths (one per line) on stdin. "
    "Each request runs a fold and prints a JSON status line to stdout. The literal "
    "line 'shutdown' exits cleanly. When --serve is set, --manifest is ignored.",
)
_MODEL_DIR = flags.DEFINE_string(
    "model_dir",
    None,
    "Directory containing exactly one AF3-format parameter set.",
)
_JAX_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    "jax_compilation_cache_dir",
    os.environ.get("AF3_JAX_CACHE_DIR"),
    "Persistent JAX compilation cache directory.",
)
_GPU_DEVICE = flags.DEFINE_integer("gpu_device", 0, "Visible JAX GPU index.")
_BUCKETS = flags.DEFINE_list(
    "buckets",
    ["64", "128", "192", "256", "384", "512", "768", "1024", "1536", "2048", "2560", "3072"],
    "Token buckets used to control JAX recompilation.",
)
_FLASH_ATTENTION_IMPLEMENTATION = flags.DEFINE_enum(
    "flash_attention_implementation",
    "triton",
    ["triton", "cudnn", "xla"],
    "AF3 attention backend.",
)
_NUM_RECYCLES = flags.DEFINE_integer("num_recycles", 5, "AF3 recycle count.", lower_bound=1)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer("num_diffusion_samples", 1, "Number of diffusion samples.", lower_bound=1)
_MAX_TEMPLATE_DATE = flags.DEFINE_string("max_template_date", "2021-09-30", "Maximum template date.")
_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    "conformer_max_iterations", None, "Optional RDKit conformer iteration limit.", lower_bound=0
)
_RESOLVE_MSA_OVERLAPS = flags.DEFINE_bool("resolve_msa_overlaps", True, "Deduplicate paired and unpaired MSA rows.")
_SAVE_FULL_OUTPUTS = flags.DEFINE_bool(
    "save_full_outputs", False, "Retain full confidences and per-sample output files."
)


def _build_fold_input(payload: dict) -> Any:
    cif_value = payload["cif"]
    # Inline CIF strings are far longer than any filesystem limit, so guard the
    # exists() check to avoid ENAMETOOLONG. A CIF always contains a newline; a path never does.
    if "\n" not in cif_value and os.path.exists(cif_value):
        cif_path = pathlib.Path(cif_value)
        cif_string, cif_name = cif_path.read_text(), cif_path.stem
    else:
        cif_string, cif_name = cif_value, payload["name"]

    fold_input = _runtime.build_fold_input_from_cif(
        cif_string=cif_string,
        cif_name=cif_name,
        paired_msa=payload.get("paired_msa"),
        unpaired_msa=payload.get("unpaired_msa"),
        template=payload.get("template"),
        user_ccd=payload.get("user_ccd"),
    )

    # from_mmcif samples a fresh random seed per call; a manifest ``seed`` pins it for reproducibility.
    seed = payload.get("seed")
    if seed is None:
        return fold_input
    return dataclasses.replace(fold_input, rng_seeds=(int(seed),))


def _featurise(
    fold_input: Any,
    max_template_date: datetime.date,
    conformer_max_iters: int | None,
    resolve_msa_overlaps: bool,
) -> Any:
    buckets = tuple(int(b) for b in _BUCKETS.value)
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    return _featurisation.featurise_input(
        fold_input=fold_input,
        buckets=buckets,
        ccd=ccd,
        verbose=False,
        ref_max_modified_date=max_template_date,
        conformer_max_iterations=conformer_max_iters,
        resolve_msa_overlaps=resolve_msa_overlaps,
    )


def _infer_and_extract(fold_input: Any, featurised_examples: Any, model_runner: Any) -> list[Any]:
    """Mirror of `predict_structure`'s per-seed loop body."""
    results = []
    for seed, example in zip(fold_input.rng_seeds, featurised_examples, strict=True):
        rng_key = jax.random.PRNGKey(seed)
        result = model_runner.run_inference(example, rng_key)
        inference_results = model_runner.extract_inference_results(
            batch=example, result=result, target_name=fold_input.name
        )
        results.append(
            _runtime.ResultsForSeed(
                seed=seed,
                inference_results=inference_results,
            )
        )
    return results


def _run(
    designs: list[dict],
    model_runner: Any,
    max_template_date: datetime.date,
    save_full: bool,
) -> int:
    """Producer thread featurises design N+1 while main thread runs inference on N."""
    q: queue.Queue = queue.Queue(maxsize=1)

    def producer() -> None:
        for i, payload in enumerate(designs):
            try:
                fold_input = _build_fold_input(payload)
                feats = _featurise(
                    fold_input,
                    max_template_date,
                    _CONFORMER_MAX_ITERATIONS.value,
                    _RESOLVE_MSA_OVERLAPS.value,
                )
                q.put(("ok", i, payload, fold_input, feats))
            except Exception as exc:
                q.put(("err", i, payload, exc))
        q.put(("done",))

    t = threading.Thread(target=producer, name="af3-feat-producer", daemon=True)
    t.start()

    n_ok = 0
    while True:
        msg = q.get()
        if msg[0] == "done":
            break
        if msg[0] == "err":
            _, i, payload, exc = msg
            print(
                f"Batch wrapper: ({i + 1}/{len(designs)}) {payload['name']} FAILED in featurise: {exc}", file=sys.stderr
            )
            continue
        _, i, payload, fold_input, feats = msg
        try:
            os.makedirs(payload["output_dir"], exist_ok=True)
            results = _infer_and_extract(fold_input, feats, model_runner)
            _runtime.write_outputs(
                all_inference_results=results,
                output_dir=payload["output_dir"],
                job_name=fold_input.name.replace(" ", "_"),
                compress_large_output_files=False,
                save_full_outputs=save_full,
            )
            print(
                f"Batch wrapper: ({i + 1}/{len(designs)}) {payload['name']} OK",
                file=sys.stderr,
                flush=True,
            )
            n_ok += 1
        except Exception as exc:
            print(
                f"Batch wrapper: ({i + 1}/{len(designs)}) {payload['name']} FAILED in infer/write: {exc}",
                file=sys.stderr,
            )
    t.join()
    return n_ok


def _setup_runner() -> tuple[Any, datetime.date]:
    """Load AF3 once and return ``(model_runner, max_template_date)``."""
    if _JAX_COMPILATION_CACHE_DIR.value is not None:
        jax.config.update("jax_compilation_cache_dir", _JAX_COMPILATION_CACHE_DIR.value)

    max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)

    devices = jax.local_devices(backend="gpu")
    gpu_device = _GPU_DEVICE.value
    model_dir = _MODEL_DIR.value or os.environ.get("AF3_MODEL_DIR")
    if model_dir is None:
        raise ValueError("--model_dir or AF3_MODEL_DIR is required")
    model_runner = _runtime.ModelRunner(
        config=_runtime.make_model_config(
            flash_attention_implementation=_FLASH_ATTENTION_IMPLEMENTATION.value,
            num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
            num_recycles=_NUM_RECYCLES.value,
        ),
        device=devices[gpu_device],
        model_dir=pathlib.Path(model_dir),
        cache_dir=(pathlib.Path(_JAX_COMPILATION_CACHE_DIR.value) if _JAX_COMPILATION_CACHE_DIR.value else None),
    )
    _ = model_runner.model_params  # force weights load
    return model_runner, max_template_date


def _serve_loop(model_runner: Any, max_template_date: datetime.date) -> None:
    """Read manifest paths from stdin, fold each, print one JSON status line per request.

    Each input line is either an absolute manifest path or the literal line
    ``shutdown`` to exit cleanly.
    """
    print(json.dumps({"status": "ready"}), flush=True)
    save_full = _SAVE_FULL_OUTPUTS.value
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "shutdown":
            print(json.dumps({"status": "ok", "shutdown": True}), flush=True)
            return
        manifest_path = pathlib.Path(line)
        try:
            designs = json.loads(manifest_path.read_text())
            if not designs:
                print(json.dumps({"status": "error", "message": "empty manifest"}), flush=True)
                continue
            t0 = time.perf_counter()
            n_ok = _run(designs, model_runner, max_template_date, save_full)
            wall_s = time.perf_counter() - t0
            print(
                f"AF3 batch wrapper: {n_ok}/{len(designs)} succeeded in {wall_s:.1f}s "
                f"({wall_s / max(len(designs), 1):.2f}s/example)",
                file=sys.stderr,
                flush=True,
            )
            response = {"status": "ok", "n_ok": n_ok, "n_total": len(designs)}
            if n_ok != len(designs):
                response.update(
                    status="error",
                    message=f"{len(designs) - n_ok}/{len(designs)} designs failed",
                )
            print(json.dumps(response), flush=True)
        except Exception as exc:
            print(json.dumps({"status": "error", "message": str(exc)}), flush=True)


def main(_argv: list[str]) -> None:
    model_runner, max_template_date = _setup_runner()

    if _SERVE.value:
        _serve_loop(model_runner, max_template_date)
        return

    if _MANIFEST.value is None:
        raise SystemExit("--manifest is required when --serve is not set.")
    designs = json.loads(pathlib.Path(_MANIFEST.value).read_text())
    if not designs:
        raise SystemExit("Manifest is empty.")
    t0 = time.perf_counter()
    n_ok = _run(designs, model_runner, max_template_date, _SAVE_FULL_OUTPUTS.value)
    wall_s = time.perf_counter() - t0
    print(
        f"AF3 batch wrapper: {n_ok}/{len(designs)} succeeded in {wall_s:.1f}s "
        f"({wall_s / max(len(designs), 1):.2f}s/example)",
        file=sys.stderr,
    )
    if n_ok != len(designs):
        raise SystemExit(1)


if __name__ == "__main__":
    app.run(main)
