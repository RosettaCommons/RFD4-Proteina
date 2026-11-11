"""Single-worker design jobs persisted on a mounted volume."""

import io
import json
import logging
import os
import queue
import re
import secrets
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
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Receive, Scope, Send

from .errors import STATUS_FOR_CODE, ApiError, ErrorCode, ErrorResponse, error_body
from .models import MAX_SEED, TASKS, UNSUPPORTED_TASKS, DesignRequest, Info, JobStatus, Live, Ready, Results

log = logging.getLogger(__name__)


def limit(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


MAX_REQUEST_BYTES = limit("RFD4_MAX_REQUEST_BYTES", 32 * 1024 * 1024)
MAX_TOKENS = limit("RFD4_MAX_TOKENS", 1024)  # Largest upstream compile bucket.
MAX_REPLICATES = limit("RFD4_MAX_REPLICATES", 4)
QUEUE_DEPTH = limit("RFD4_QUEUE_DEPTH", 8)
pending: queue.Queue[str] = queue.Queue(maxsize=QUEUE_DEPTH)
admission_lock = threading.Lock()
state_lock = threading.Lock()
state = {"ready": False, "error": None, "runner": None, "stop": False}
store: Path | None = None
instance_id = uuid.uuid4().hex


def snapshot() -> dict[str, Any]:
    with state_lock:
        return state.copy()


def update_state(**changes: Any) -> None:
    with state_lock:
        state.update(changes)


def request_dir(request_id: str) -> Path:
    if store is None:
        raise ApiError(ErrorCode.MODEL_UNAVAILABLE, "Storage is not initialized.")
    if not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise ApiError(ErrorCode.NOT_FOUND, "Request not found.")
    return store / "requests" / request_id


def save_status(request_id: str, data: dict[str, Any]) -> None:
    data["updated_at"] = time.time()
    path = request_dir(request_id) / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data))
    temp.replace(path)


def read_status(request_id: str) -> dict[str, Any]:
    path = request_dir(request_id) / "status.json"
    if not path.is_file():
        raise ApiError(ErrorCode.NOT_FOUND, "Request not found.")
    return json.loads(path.read_text())


def worker() -> None:
    try:
        from .runner import Runner

        runner = Runner()
        update_state(runner=runner)
        # Recovery scans history; operators manage retention. One corrupt file cannot block startup.
        for path in store.glob("requests/*/status.json"):
            try:
                job = JobStatus.model_validate_json(path.read_text())
                if request_dir(job.id) != path.parent:
                    raise ValueError("Status ID does not match directory")
                if job.status in ("queued", "running"):
                    data = job.model_dump(exclude_unset=True)
                    data.update(
                        status="failed",
                        error=error_body(ApiError(ErrorCode.MODEL_UNAVAILABLE, "Server restarted; resubmit.")),
                    )
                    save_status(job.id, data)
            except (OSError, ValueError, ApiError) as exc:
                log.warning("Skipping invalid request status %s: %s", path, exc)
        update_state(ready=True)
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
                directory = request_dir(request_id)
                output = directory / "outputs"
                runner.generate(
                    directory / f"{data['example_id']}.cif", output, data["settings"], MAX_TOKENS, MAX_REPLICATES
                )
                data.update(
                    status="succeeded",
                    artifacts=[p.relative_to(output).as_posix() for p in sorted(output.rglob("*")) if p.is_file()],
                    duration_seconds=round(time.time() - data["started_at"], 2),
                    checkpoint_loads=runner.loads,
                    server_instance_id=instance_id,
                )
            except Exception as exc:
                log.exception("Request %s failed", request_id)
                data.update(status="failed", error=error_body(exc))
            finally:
                try:
                    save_status(request_id, data)
                finally:
                    pending.task_done()
    except Exception as exc:
        log.exception("GPU worker stopped")
        update_state(ready=False, error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global store
    store = Path(os.environ.get("STORE_PATH", "/workspace/store"))
    store.mkdir(parents=True, exist_ok=True)
    update_state(ready=False, error=None, runner=None, stop=False)
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
    """Bound the complete JSON body before parsing, including chunked uploads."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/v1/designs":
            return await self.app(scope, receive, send)
        body = bytearray()
        try:
            async for chunk in Request(scope, receive).stream():
                if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                    exc = ApiError(ErrorCode.REQUEST_TOO_LARGE, f"Request exceeds {MAX_REQUEST_BYTES} bytes.")
                    return await JSONResponse(exc.body(), status_code=exc.status)(scope, receive, send)
                body.extend(chunk)
        except ClientDisconnect:
            return
        delivered = False

        async def buffered_receive() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, buffered_receive, send)


app.add_middleware(RequestLimit)


@app.exception_handler(Exception)
@app.exception_handler(ApiError)
@app.exception_handler(RequestValidationError)
@app.exception_handler(HTTPException)
async def handle_error(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, RequestValidationError):
        details = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        exc = ApiError(ErrorCode.SCHEMA_VIOLATION, "Request does not match the schema.", details)
    if isinstance(exc, ApiError):
        return JSONResponse(exc.body(), status_code=exc.status)
    if isinstance(exc, HTTPException):
        code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.INVALID_REQUEST
        return JSONResponse(ApiError(code, str(exc.detail)).body(), status_code=exc.status_code)
    log.error("Unhandled HTTP error", exc_info=exc)
    return JSONResponse({"error": error_body(exc)}, status_code=500)


@app.get("/v1/info", response_model=Info)
def info() -> dict[str, Any]:
    return {
        "tasks": {"supported": sorted(set(TASKS) - UNSUPPORTED_TASKS), "unsupported": sorted(UNSUPPORTED_TASKS)},
        "limits": {
            "max_request_bytes": MAX_REQUEST_BYTES,
            "max_tokens": MAX_TOKENS,
            "max_replicates": MAX_REPLICATES,
            "queue_depth": QUEUE_DEPTH,
        },
    }


@app.get("/health/live", response_model=Live)
def live() -> dict[str, str]:
    if snapshot()["error"]:
        raise ApiError(ErrorCode.MODEL_UNAVAILABLE, "The GPU worker stopped.")
    return {"status": "alive"}


@app.get("/health/ready", response_model=Ready)
def ready() -> dict[str, Any]:
    current = snapshot()
    if not current["ready"]:
        raise ApiError(ErrorCode.MODEL_UNAVAILABLE, "The model is not ready.")
    return {
        "status": "ready",
        "gpu": current["runner"].device,
        "checkpoint_loads": current["runner"].loads,
        "server_instance_id": instance_id,
    }


def check_capacity() -> None:
    if not snapshot()["ready"]:
        raise ApiError(ErrorCode.MODEL_UNAVAILABLE, "The model is not ready.")
    if pending.full():
        raise ApiError(ErrorCode.SERVICE_BUSY, "No free worker; retry later.")


@app.post("/v1/designs", status_code=202, response_model=JobStatus, response_model_exclude_unset=True)
def submit(request: DesignRequest) -> dict[str, Any]:
    # A synchronous handler keeps CIF parsing and disk IO in FastAPI's thread pool.
    check_capacity()  # Early rejection; the locked check below is authoritative.
    if request.task in UNSUPPORTED_TASKS:
        raise ApiError(ErrorCode.INVALID_REQUEST, f"Unsupported task: {request.task}", ["task"])
    if request.sampling.num_replicates > MAX_REPLICATES:
        raise ApiError(ErrorCode.REQUEST_TOO_LARGE, f"At most {MAX_REPLICATES} samples per request.")
    try:
        atoms = CIFFile.read(io.StringIO(request.input.file.content)).block["atom_site"]
        if not atoms.row_count or not {"label_atom_id", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z"}.issubset(
            atoms
        ):
            raise ValueError("Missing coordinates")
    except (InvalidFileError, ValueError, KeyError) as exc:
        raise ApiError(ErrorCode.INVALID_INPUT, "Upload a parseable mmCIF with atom coordinates.") from exc
    with admission_lock:
        check_capacity()
        request_id = uuid.uuid4().hex
        settings = request.sampling.model_dump()
        if settings["seed"] is None:
            settings["seed"] = secrets.randbelow(MAX_SEED + 1)
        data = {
            "id": request_id,
            "status": "queued",
            "created_at": time.time(),
            "task": request.task,
            "example_id": request.input.example_id,
            "settings": settings,
        }
        directory = request_dir(request_id)
        directory.mkdir(parents=True)
        (directory / f"{request.input.example_id}.cif").write_text(request.input.file.content, encoding="utf-8")
        save_status(request_id, data)
        pending.put_nowait(request_id)
    return data


@app.get("/v1/designs/{request_id}", response_model=JobStatus, response_model_exclude_unset=True)
def status(request_id: str) -> dict[str, Any]:
    return read_status(request_id)


@app.get("/v1/designs/{request_id}/results", response_model=Results)
def results(request_id: str) -> dict[str, Any]:
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
    response_class=FileResponse,
    responses={200: {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}},
)
def artifact(request_id: str, name: str) -> FileResponse:
    data = read_status(request_id)
    root = (request_dir(request_id) / "outputs").resolve()
    path = (root / name).resolve()
    if data["status"] != "succeeded" or name not in data.get("artifacts", []) or not path.is_relative_to(root):
        raise ApiError(ErrorCode.NOT_FOUND, "Artifact not found.")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)
