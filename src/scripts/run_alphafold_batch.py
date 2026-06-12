"""Batch fold wrapper for AF3, executed inside af3.sif.

Manifest format (JSON list)::

    [{"name": "...", "output_dir": "/abs/path/<example>",
      "cif": "<inline CIF string or path>",
      "template"?: {...}, "user_ccd"?: "...",
      "paired_msa"?: {...}, "unpaired_msa"?: {...}}, ...]

Spawned by :py:mod:`scripts.af3_daemon`. Two run modes:

- One-shot: pass ``--manifest=...`` and the script loads AF3, folds the
  designs, and exits.
- ``--serve``: load AF3 once, then loop reading manifest paths
  (newline-delimited) on stdin; for each, run the fold and print a
  ``{"status": ...}`` JSON line to stdout. Sending the literal line
  ``shutdown`` exits cleanly.
"""

import datetime
import importlib.util
import json
import os
import pathlib
import queue
import sys
import threading
import time
from typing import Any

import jax
from absl import app, flags
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation as _featurisation


def _load_raf3() -> Any:
    """Load AF3's upstream ``run_alphafold_from_cif.py`` from the mount."""
    scripts_dir = os.environ.get("AF3_SCRIPTS_DIR", "/path/to/data")
    raf3_path = os.path.join(scripts_dir, "run_alphafold_from_cif.py")
    spec = importlib.util.spec_from_file_location("run_alphafold_from_cif", raf3_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_raf3 = _load_raf3()

_MANIFEST = flags.DEFINE_string("manifest", None, "JSON manifest with a list of design payloads.")
_SERVE = flags.DEFINE_bool(
    "serve",
    False,
    "If True, load AF3 once and loop reading manifest paths (one per line) on stdin. "
    "Each request runs a fold and prints a JSON status line to stdout. The literal "
    "line 'shutdown' exits cleanly. When --serve is set, --manifest is ignored.",
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

    return _raf3._build_fold_input_from_cif(
        cif_string=cif_string,
        cif_name=cif_name,
        paired_msa=payload.get("paired_msa"),
        unpaired_msa=payload.get("unpaired_msa"),
        template=payload.get("template"),
        user_ccd=payload.get("user_ccd"),
    )


def _featurise(
    fold_input: Any,
    max_template_date: datetime.date,
    conformer_max_iters: int | None,
    resolve_msa_overlaps: bool,
) -> Any:
    buckets = tuple(int(b) for b in _raf3._BUCKETS.value)
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
    for seed, example in zip(fold_input.rng_seeds, featurised_examples, strict=False):
        rng_key = jax.random.PRNGKey(seed)
        result = model_runner.run_inference(example, rng_key)
        inference_results = model_runner.extract_inference_results(
            batch=example, result=result, target_name=fold_input.name
        )
        results.append(
            _raf3.ResultsForSeed(
                seed=seed,
                inference_results=inference_results,
                full_fold_input=fold_input,
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
                    _raf3._CONFORMER_MAX_ITERATIONS.value,
                    _raf3._RESOLVE_MSA_OVERLAPS.value,
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
            _raf3.write_outputs(
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
    if _raf3._JAX_COMPILATION_CACHE_DIR.value is not None:
        jax.config.update("jax_compilation_cache_dir", _raf3._JAX_COMPILATION_CACHE_DIR.value)

    max_template_date = datetime.date.fromisoformat(_raf3._MAX_TEMPLATE_DATE.value)

    devices = jax.local_devices(backend="gpu")
    gpu_device = _raf3._GPU_DEVICE.value
    model_runner = _raf3.ModelRunner(
        config=_raf3.make_model_config(
            flash_attention_implementation=_raf3._FLASH_ATTENTION_IMPLEMENTATION.value,
            num_diffusion_samples=_raf3._NUM_DIFFUSION_SAMPLES.value,
            num_recycles=_raf3._NUM_RECYCLES.value,
        ),
        device=devices[gpu_device],
        model_dir=pathlib.Path(_raf3.MODEL_DIR.value),
    )
    _ = model_runner.model_params  # force weights load
    return model_runner, max_template_date


def _serve_loop(model_runner: Any, max_template_date: datetime.date) -> None:
    """Read manifest paths from stdin, fold each, print one JSON status line per request.

    Each input line is either an absolute manifest path or the literal line
    ``shutdown`` to exit cleanly.
    """
    print(json.dumps({"status": "ready"}), flush=True)
    save_full = _raf3._SAVE_FULL_OUTPUTS.value
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
            print(
                json.dumps({"status": "ok", "n_ok": n_ok, "n_total": len(designs)}),
                flush=True,
            )
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
    n_ok = _run(designs, model_runner, max_template_date, _raf3._SAVE_FULL_OUTPUTS.value)
    wall_s = time.perf_counter() - t0
    print(
        f"AF3 batch wrapper: {n_ok}/{len(designs)} succeeded in {wall_s:.1f}s "
        f"({wall_s / max(len(designs), 1):.2f}s/example)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    app.run(main)
