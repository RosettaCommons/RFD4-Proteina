"""Single-replica asynchronous design API with durable disk artifacts."""

import io
import json
import logging
import os
import queue
import re
import secrets
import shutil
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from biotite import InvalidFileError
from biotite.structure.io.pdbx import CIFFile
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from app.errors import STATUS_FOR_CODE, ApiError, ErrorCode, ErrorResponse, error_body
from app.models import (
    MAX_SEED,
    TASKS,
    UNSUPPORTED_TASKS,
    DesignRequest,
    Info,
    JobStatus,
    Live,
    Ready,
    Results,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
work = Path(os.environ.get("WORK_PATH", "/workspace/requests"))


# Service limits are reported by /v1/info; the token default matches the compile buckets.
def _limit(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


MAX_REQUEST_BYTES = _limit("RFD4_MAX_REQUEST_BYTES", 32 * 1024 * 1024)
MAX_TOKENS = _limit("RFD4_MAX_TOKENS", 1024)
MAX_REPLICATES = _limit("RFD4_MAX_REPLICATES", 4)
QUEUE_DEPTH = _limit("RFD4_QUEUE_DEPTH", 8)

pending: queue.Queue[str] = queue.Queue(maxsize=QUEUE_DEPTH)
admission_lock = threading.Lock()
state: dict[str, Any] = {
    "ready": False,
    "error": None,
    "runner": None,
    "stop": False,
}
instance_id = uuid.uuid4().hex
store: Path | None = None
state_lock = threading.Lock()


def snapshot() -> dict[str, Any]:
    with state_lock:
        return state.copy()


def update_state(**changes: Any) -> None:
    with state_lock:
        state.update(changes)


def request_dir(request_id: str) -> Path:
    if store is None:
        raise ApiError(ErrorCode.MODEL_UNAVAILABLE, "Storage is not initialized yet.")
    if not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise ApiError(ErrorCode.NOT_FOUND, "Request not found.")
    return store / "requests" / request_id


def save_status(request_id: str, data: dict[str, Any]) -> None:
    """Atomically persist a request status with its last update time."""
    data["updated_at"] = time.time()
    path = request_dir(request_id) / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data))
    temp.replace(path)


def read_status(request_id: str) -> dict[str, Any]:
    """Read a persisted request, rejecting invalid identifiers."""
    path = request_dir(request_id) / "status.json"
    if not path.is_file():
        raise ApiError(ErrorCode.NOT_FOUND, "Request not found.")
    data = json.loads(path.read_text())
    # Old deployments persisted untyped error strings. Keep their jobs readable
    # without guessing a category from the wording or rewriting historical files.
    if isinstance(data.get("error"), str):
        data["error"] = ApiError(ErrorCode.INTERNAL_ERROR, data["error"]).body()["error"]
    return data


def worker() -> None:
    """Load one GPU model and execute queued requests serially."""
    try:
        # Defer GPU dependencies until startup; HTTP tests use a fake runner.
        from app.runner import Runner

        runner = Runner()
        update_state(runner=runner)
        # Scan historical status files before readiness. Startup is O(request history);
        # operators should prune old requests according to their retention policy.
        for path in store.glob("requests/*/status.json"):
            try:
                request_id = path.parent.name
                request_dir(request_id)  # Validate the directory ID, not just the JSON.
                data = json.loads(path.read_text())
                if not isinstance(data, dict) or data.get("id") != request_id:
                    raise ValueError("Status ID must match its request directory")
                if data.get("status") not in ("queued", "running", "succeeded", "failed"):
                    raise ValueError("Missing or invalid request status")
                if data["status"] in ("queued", "running"):
                    data.update(
                        status="failed",
                        error=error_body(
                            ApiError(ErrorCode.MODEL_UNAVAILABLE, "Server restarted; submit a new request.")
                        ),
                    )
                    save_status(request_id, data)
            except (OSError, ValueError, ApiError) as exc:
                log.warning("Skipping invalid request status %s: %s", path, exc)
        update_state(ready=True)
        log.info("Model ready on %s; checkpoint loads=%s", runner.device, runner.loads)
        while not snapshot()["stop"]:
            try:
                request_id = pending.get(timeout=1)
            except queue.Empty:
                continue
            data = {"id": request_id}
            try:
                data = read_status(request_id)
                data.update(status="running", started_at=time.time())
                save_status(request_id, data)
                input_path = request_dir(request_id) / f"{data.get('example_id', 'input')}.cif"
                output_dir = work / request_id / "outputs"
                runner.generate(input_path, output_dir, data["settings"], MAX_TOKENS, MAX_REPLICATES)
                artifacts = []
                for path in sorted(output_dir.rglob("*")):
                    if path.is_file():
                        name = path.relative_to(output_dir).as_posix()
                        target = request_dir(request_id) / "outputs" / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
                        artifacts.append(name)
                data.update(
                    status="succeeded",
                    artifacts=artifacts,
                    duration_seconds=round(time.time() - data["started_at"], 2),
                    checkpoint_loads=runner.loads,
                    server_instance_id=instance_id,
                )
                save_status(request_id, data)
            except Exception as exc:
                log.exception("Request %s failed", request_id)
                data.update(status="failed", error=error_body(exc))
                save_status(request_id, data)
            finally:
                shutil.rmtree(work / request_id, ignore_errors=True)
                pending.task_done()
    except Exception as exc:
        log.exception("GPU worker stopped")
        update_state(ready=False, error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop the single owner of the GPU and request queue."""
    global store
    update_state(ready=False, error=None, runner=None, stop=False)
    store = Path(os.environ.get("STORE_PATH", "/workspace/store"))
    store.mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        update_state(ready=False, stop=True)
        thread.join(timeout=2)


app = FastAPI(
    title="RFD4-Proteína",
    version="1.0.0",
    lifespan=lifespan,
    responses={int(status): {"model": ErrorResponse} for status in {*STATUS_FOR_CODE.values(), 405}},
)


class RequestLimit:
    """Bound the JSON body, including chunked requests, before schema parsing."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/v1/designs":
            await self.app(scope, receive, send)
            return
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                error = ApiError(ErrorCode.REQUEST_TOO_LARGE, f"Request exceeds {MAX_REQUEST_BYTES} bytes.")
                await JSONResponse(status_code=error.status, content=error.body())(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def buffered_receive() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, buffered_receive, send)


app.add_middleware(RequestLimit)


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content=exc.body())


@app.exception_handler(Exception)
async def _internal_error(_: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled HTTP error", exc_info=exc)
    return JSONResponse(status_code=500, content={"error": error_body(exc)})


@app.exception_handler(RequestValidationError)
async def _schema_violation(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Render framework validation failures in the declared error body."""
    details = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
    error = ApiError(ErrorCode.SCHEMA_VIOLATION, "Request does not match the schema.", details)
    return JSONResponse(status_code=error.status, content=error.body())


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Unknown route or wrong method: one error shape for those too."""
    code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.INVALID_REQUEST
    return JSONResponse(status_code=exc.status_code, content=ApiError(code, str(exc.detail)).body())


@app.get("/v1/info", response_model=Info)
def info() -> dict[str, Any]:
    """Report the task vocabulary and the limits, before the model is ready."""
    return {
        "tasks": {
            "supported": sorted(set(TASKS) - UNSUPPORTED_TASKS),
            "unsupported": sorted(UNSUPPORTED_TASKS),
        },
        "limits": {
            "max_request_bytes": MAX_REQUEST_BYTES,
            "max_tokens": MAX_TOKENS,
            "max_replicates": MAX_REPLICATES,
            "queue_depth": QUEUE_DEPTH,
        },
        "submission": "asynchronous",
    }


@app.get("/health/live", response_model=Live)
def live() -> dict[str, str]:
    """Report whether the GPU worker has encountered a fatal error."""
    if snapshot()["error"]:
        raise ApiError(ErrorCode.MODEL_UNAVAILABLE, "The GPU worker stopped.")
    return {"status": "alive"}


@app.get("/health/ready", response_model=Ready)
def ready() -> dict[str, Any]:
    """Report readiness after checkpoint loading and restart recovery."""
    current = snapshot()
    if not current["ready"]:
        message = "The GPU worker stopped." if current["error"] else "The model is still loading."
        raise ApiError(ErrorCode.MODEL_UNAVAILABLE, message)
    runner = current["runner"]
    return {
        "status": "ready",
        "gpu": runner.device,
        "checkpoint_loads": runner.loads,
        "server_instance_id": instance_id,
    }


@app.post("/v1/designs", status_code=202, response_model=JobStatus, response_model_exclude_unset=True)
def submit(request: DesignRequest) -> dict[str, Any]:
    """Queue an annotated mmCIF design and return its polling identifier."""
    if not snapshot()["ready"]:
        raise ApiError(ErrorCode.MODEL_UNAVAILABLE, "The model is not ready.")
    # Advisory fast path: avoid CIF parsing when already overloaded.
    # The worker may change the queue while we parse; the locked check remains authoritative.
    if pending.full():
        raise ApiError(ErrorCode.SERVICE_BUSY, "No free worker; route elsewhere or retry later.")
    if request.task in UNSUPPORTED_TASKS:
        raise ApiError(
            ErrorCode.INVALID_REQUEST, f"Task {request.task!r} is not supported by this deployment.", ["task"]
        )
    if request.sampling.num_replicates > MAX_REPLICATES:
        raise ApiError(
            ErrorCode.REQUEST_TOO_LARGE,
            f"This API accepts at most {MAX_REPLICATES} design samples per request.",
            ["sampling.num_replicates"],
        )
    # FastAPI runs this synchronous handler (including CIF parsing and disk IO) in its thread pool.
    return admit_request(request)


def admit_request(request: DesignRequest) -> dict[str, Any]:
    """Parse and persist an upload in a thread, serializing only queue admission."""
    # Validate CIF syntax and required coordinate columns, not all design semantics.
    # Upstream performs semantic checks and transformed-token admission in the worker.
    try:
        cif = CIFFile.read(io.StringIO(request.input.file.content))
        atoms = cif.block["atom_site"]
        required = {"label_atom_id", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z"}
        if atoms.row_count == 0 or not required.issubset(atoms):
            raise ValueError("Missing atom coordinates")
    except (InvalidFileError, ValueError, KeyError) as exc:
        raise ApiError(ErrorCode.INVALID_INPUT, "Upload a parseable mmCIF with atom coordinates.") from exc
    request_id = uuid.uuid4().hex
    with admission_lock:
        if not snapshot()["ready"]:
            raise ApiError(ErrorCode.MODEL_UNAVAILABLE, "The model is not ready.")
        if pending.full():
            raise ApiError(ErrorCode.SERVICE_BUSY, "No free worker; route elsewhere or retry later.")
        data = {
            "id": request_id,
            "status": "queued",
            "created_at": time.time(),
            "task": request.task,
            "example_id": request.input.example_id,
            "settings": {
                "num_replicates": request.sampling.num_replicates,
                "seed": request.sampling.seed if request.sampling.seed is not None else secrets.randbelow(MAX_SEED + 1),
            },
        }
        directory = request_dir(request_id)
        directory.mkdir(parents=True)
        (directory / f"{request.input.example_id}.cif").write_text(request.input.file.content, encoding="utf-8")
        save_status(request_id, data)
        pending.put_nowait(request_id)
    return data


@app.get("/v1/designs/{request_id}", response_model=JobStatus, response_model_exclude_unset=True)
def status(request_id: str) -> dict[str, Any]:
    """Return the current status of an accepted design request."""
    return read_status(request_id)


@app.get("/v1/designs/{request_id}/results", response_model=Results)
def results(request_id: str) -> dict[str, Any]:
    """List downloadable artifacts for a completed design."""
    data = read_status(request_id)
    if data["status"] != "succeeded":
        raise ApiError(ErrorCode.CONFLICT, "Results are available after successful generation.")
    return {
        "id": request_id,
        "artifacts": [
            {"name": name, "download": f"/v1/designs/{request_id}/artifacts/{name}"} for name in data["artifacts"]
        ],
    }


@app.get(
    "/v1/designs/{request_id}/artifacts/{name:path}",
    response_class=Response,
    responses={200: {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}},
)
def artifact(request_id: str, name: str) -> Response:
    """Download a file listed in the completed request manifest."""
    data = read_status(request_id)
    if data["status"] != "succeeded" or name not in data.get("artifacts", []):
        raise ApiError(ErrorCode.NOT_FOUND, "Artifact not found.")
    root = (request_dir(request_id) / "outputs").resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ApiError(ErrorCode.NOT_FOUND, "Artifact not found.")
    return Response(
        path.read_bytes(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{Path(name).name}"'},
    )
