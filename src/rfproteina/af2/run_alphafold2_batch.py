"""Batch fold wrapper for AF2, executed inside the AF2 apptainer image.

In one-shot mode ``--manifest`` is processed and the script exits. In
``--serve`` mode AF2 models remain resident while manifest paths are read from
stdin. Each request produces one line-delimited JSON status response.
"""

import json
import os
import pathlib
import queue
import sys
import threading
import time
import traceback
from itertools import groupby
from typing import Any

import af2_runtime as _runtime
from absl import app, flags


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


_MANIFEST = flags.DEFINE_string("manifest", None, "JSON manifest containing design payloads.")
_SERVE = flags.DEFINE_bool(
    "serve",
    False,
    "Load AF2 once and process newline-delimited manifest paths from stdin.",
)
_MODEL_DIR = flags.DEFINE_string("model_dir", None, "Directory containing AF2 model parameters.")
_JAX_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    "jax_compilation_cache_dir",
    os.environ.get("AF2_JAX_CACHE_DIR"),
    "Persistent JAX compilation cache directory.",
)
_GPU_DEVICE = flags.DEFINE_integer("gpu_device", 0, "Visible JAX GPU index.")
_BUCKETS = flags.DEFINE_list(
    "buckets",
    ["64", "128", "192", "256", "384", "512", "768", "1024", "1536", "2048", "2560", "3072"],
    "Sequence-length buckets used to control JAX recompilation.",
)
_NUM_RECYCLES = flags.DEFINE_integer("num_recycles", 3, "AF2 recycle count.", lower_bound=1)
_RESIDENT_MODELS = flags.DEFINE_list(
    "resident_models",
    os.environ.get(
        "AF2_RESIDENT_MODELS",
        f"{_runtime.DEFAULT_MONOMER_MODEL},{_runtime.DEFAULT_MULTIMER_MODEL}",
    ),
    "Comma-separated AF2 model names loaded by the resident worker.",
)
_USE_PALLAS = flags.DEFINE_bool(
    "use_pallas",
    _env_bool("AF2_USE_PALLAS", True),
    "Use ColabFold Pallas/Triton Evoformer kernels (requires bfloat16 and Ampere+).",
)
_COMPILE_MODE = flags.DEFINE_enum(
    "compile_mode",
    os.environ.get("AF2_COMPILE_MODE", "tuned"),
    ("fast", "tuned", "full"),
    "XLA GEMM compile mode.",
)


def _chain_count(payload: dict) -> int:
    """Return the payload's chain count, delegating CIF parsing to the runtime."""
    if "chain_count" in payload:
        return int(payload["chain_count"])
    return int(_runtime.chain_count(payload))


def _model_name(payload: dict) -> str:
    model_name = payload.get("model_name")
    if model_name:
        return str(model_name)
    return "model_1_multimer_v3" if _chain_count(payload) >= 2 else "model_1_ptm"


def _residue_count(payload: dict) -> int:
    if "num_res" in payload:
        return int(payload["num_res"])
    return int(_runtime.residue_count(payload))


def _order_designs_by_bucket(
    designs: list[dict],
    buckets: tuple[int, ...],
) -> tuple[list[tuple[int, dict]], list[tuple[str, int, int]]]:
    """Stable-sort designs by resident model and padded compile shape."""
    decorated = []
    for index, payload in enumerate(designs):
        try:
            key = (_model_name(payload), _runtime.nearest_bucket(_residue_count(payload), buckets), index)
        except Exception:
            # Keep malformed/oversized designs in the work queue so the
            # producer reports them individually without aborting the manifest.
            key = ("\uffff", -1, index)
        decorated.append((key, payload))
    decorated.sort(key=lambda item: item[0])
    ordered = [(key[2], payload) for key, payload in decorated]
    groups = [
        ("unbucketed" if model_name == "\uffff" else model_name, bucket, sum(1 for _ in members))
        for (model_name, bucket), members in groupby(decorated, key=lambda item: item[0][:2])
    ]
    return ordered, groups


def _run(designs: list[dict], model_runners: dict[str, Any]) -> int:
    """Prepare design N+1 while the main thread predicts design N."""
    work: queue.Queue = queue.Queue(maxsize=1)
    buckets = tuple(int(bucket) for bucket in _BUCKETS.value)
    ordered_designs, groups = _order_designs_by_bucket(designs, buckets)
    group_summary = ", ".join(f"{model_name}:{bucket}x{count}" for model_name, bucket, count in groups)
    print(f"AF2 batch wrapper: bucket groups [{group_summary}]", file=sys.stderr, flush=True)

    def producer() -> None:
        for index, payload in ordered_designs:
            try:
                model_name = _model_name(payload)
                if model_name not in model_runners:
                    raise ValueError(
                        f"model {model_name!r} is not resident; available models: "
                        f"{', '.join(sorted(model_runners))}"
                    )
                prepared = _runtime.prepare_input(
                    payload=payload,
                    model_name=model_name,
                    buckets=buckets,
                )
                work.put(("ok", index, payload, model_name, prepared))
            except Exception as exc:
                work.put(("err", index, payload, exc))
        work.put(("done",))

    thread = threading.Thread(target=producer, name="af2-input-producer", daemon=True)
    thread.start()

    n_ok = 0
    while True:
        message = work.get()
        if message[0] == "done":
            break
        if message[0] == "err":
            _, index, payload, exc = message
            print(
                f"AF2 batch wrapper: ({index + 1}/{len(designs)}) "
                f"{payload.get('name', '<unnamed>')} FAILED in prepare: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exception(exc, file=sys.stderr)
            continue

        _, index, payload, model_name, prepared = message
        try:
            output_dir = pathlib.Path(payload["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            _runtime.predict_and_write(
                payload=payload,
                prepared_input=prepared,
                model_name=model_name,
                model_runner=model_runners[model_name],
                output_dir=output_dir,
            )
            print(
                f"AF2 batch wrapper: ({index + 1}/{len(designs)}) " f"{payload.get('name', '<unnamed>')} OK",
                file=sys.stderr,
                flush=True,
            )
            n_ok += 1
        except Exception as exc:
            print(
                f"AF2 batch wrapper: ({index + 1}/{len(designs)}) "
                f"{payload.get('name', '<unnamed>')} FAILED in infer/write: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exception(exc, file=sys.stderr)
    thread.join()
    return n_ok


def _setup_runners() -> dict[str, Any]:
    model_dir = _MODEL_DIR.value or os.environ.get("AF2_MODEL_DIR")
    if model_dir is None:
        raise ValueError("--model_dir or AF2_MODEL_DIR is required")
    resident_models = tuple(model for model in _RESIDENT_MODELS.value if model)
    if not resident_models:
        raise ValueError("--resident_models must contain at least one model")
    return _runtime.load_model_runners(
        model_dir=pathlib.Path(model_dir),
        model_names=resident_models,
        num_recycles=_NUM_RECYCLES.value,
        gpu_device=_GPU_DEVICE.value,
        buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
        jax_compilation_cache_dir=(
            pathlib.Path(_JAX_COMPILATION_CACHE_DIR.value) if _JAX_COMPILATION_CACHE_DIR.value else None
        ),
        use_pallas=_USE_PALLAS.value,
        compile_mode=_COMPILE_MODE.value,
    )


def _process_manifest(manifest_path: pathlib.Path, model_runners: dict[str, Any]) -> dict:
    designs = json.loads(manifest_path.read_text())
    if not designs:
        return {"status": "error", "message": "empty manifest"}
    t0 = time.perf_counter()
    n_ok = _run(designs, model_runners)
    wall_s = time.perf_counter() - t0
    print(
        f"AF2 batch wrapper: {n_ok}/{len(designs)} succeeded in {wall_s:.1f}s "
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
    return response


def _serve_loop(model_runners: dict[str, Any]) -> None:
    print(json.dumps({"status": "ready"}), flush=True)
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "shutdown":
            print(json.dumps({"status": "ok", "shutdown": True}), flush=True)
            return
        try:
            response = _process_manifest(pathlib.Path(line), model_runners)
        except Exception as exc:
            response = {"status": "error", "message": str(exc)}
        print(json.dumps(response), flush=True)


def main(_argv: list[str]) -> None:
    model_runners = _setup_runners()
    if _SERVE.value:
        _serve_loop(model_runners)
        return
    if _MANIFEST.value is None:
        raise SystemExit("--manifest is required when --serve is not set.")
    response = _process_manifest(pathlib.Path(_MANIFEST.value), model_runners)
    if response["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    app.run(main)
