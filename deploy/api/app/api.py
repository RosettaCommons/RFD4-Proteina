"""Single-replica asynchronous design API with durable disk artifacts."""

import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.storage import Store

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
work = Path(os.environ.get("WORK_PATH", "/workspace/requests"))
pending: queue.Queue[str] = queue.Queue(maxsize=8)
admission_lock = threading.Lock()
state: dict[str, Any] = {"ready": False, "error": None, "runner": None, "stop": False, "started": time.time()}
instance_id = uuid.uuid4().hex
bucket: Store | None = None


def save_status(request_id: str, data: dict[str, Any]) -> None:
    """Atomically persist a request status with its last update time."""
    data["updated_at"] = time.time()
    bucket.blob(f"requests/{request_id}/status.json").upload_from_string(
        json.dumps(data), content_type="application/json"
    )


def read_status(request_id: str) -> dict[str, Any]:
    """Read a persisted request, rejecting invalid identifiers."""
    if not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise HTTPException(404, "Request not found")
    blob = bucket.blob(f"requests/{request_id}/status.json")
    if not blob.exists():
        raise HTTPException(404, "Request not found")
    return json.loads(blob.download_as_text())


def worker() -> None:
    """Load one GPU model and execute queued requests serially."""
    try:
        # Defer GPU dependencies until startup; HTTP tests use a fake runner.
        from app.runner import Runner

        runner = Runner(bucket)
        state["runner"] = runner
        # Interrupted requests are explicit failures; clients can resubmit them.
        for blob in bucket.list_blobs(prefix="requests/"):
            if blob.name.endswith("/status.json"):
                data = json.loads(blob.download_as_text())
                if data["status"] in ("queued", "running"):
                    data.update(status="failed", error="Server restarted; submit a new request")
                    save_status(data["id"], data)
        state["ready"] = True
        log.info("Model ready on %s; checkpoint loads=%s", runner.device, runner.loads)
        while not state["stop"]:
            try:
                request_id = pending.get(timeout=1)
            except queue.Empty:
                continue
            data = {"id": request_id}
            try:
                data = read_status(request_id)
                data.update(status="running", started_at=time.time())
                save_status(request_id, data)
                request_dir = work / request_id
                request_dir.mkdir(parents=True, exist_ok=True)
                input_path = request_dir / "input.cif"
                bucket.blob(f"requests/{request_id}/input.cif").download_to_filename(str(input_path))
                output_dir = request_dir / "outputs"
                runner.generate(input_path, output_dir, data["settings"])
                artifacts = []
                for path in sorted(output_dir.rglob("*")):
                    if path.is_file():
                        name = path.relative_to(output_dir).as_posix()
                        bucket.blob(f"requests/{request_id}/outputs/{name}").upload_from_filename(str(path))
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
                data.update(status="failed", error=str(exc))
                save_status(request_id, data)
            finally:
                shutil.rmtree(work / request_id, ignore_errors=True)
                pending.task_done()
    except Exception as exc:
        log.exception("GPU worker stopped")
        state.update(ready=False, error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop the single owner of the GPU and request queue."""
    global bucket
    state.update(ready=False, error=None, runner=None, stop=False)
    bucket = Store()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    yield
    state.update(ready=False, stop=True)
    thread.join(timeout=2)


app = FastAPI(title="RFD4-Proteína", lifespan=lifespan)


@app.get("/health/live")
def live() -> dict[str, str]:
    """Report whether the GPU worker has encountered a fatal error."""
    if state["error"]:
        raise HTTPException(503, "GPU worker stopped")
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict[str, Any]:
    """Report readiness after checkpoint loading and restart recovery."""
    if not state["ready"]:
        raise HTTPException(503, state["error"] or "Model loading")
    runner = state["runner"]
    return {
        "status": "ready",
        "gpu": runner.device,
        "checkpoint_loads": runner.loads,
        "server_instance_id": instance_id,
    }


@app.post("/v1/designs", status_code=202)
async def submit(
    file: Annotated[UploadFile, File()],
    num_replicates: Annotated[int, Form(ge=1, le=4)] = 1,
    seed: Annotated[int, Form(ge=0, le=2147483647)] = 42,
) -> dict[str, Any]:
    """Queue an annotated mmCIF design and return its polling identifier."""
    if not state["ready"]:
        raise HTTPException(503, "Model not ready")
    content = await file.read(4 * 1024 * 1024 + 1)
    if len(content) > 4 * 1024 * 1024:
        raise HTTPException(413, "Input limit is 4 MiB")
    if b"_atom_site." not in content or not content.lstrip().startswith(b"data_"):
        raise HTTPException(422, "Upload an annotated mmCIF file")
    request_id = uuid.uuid4().hex
    with admission_lock:
        if pending.full():
            raise HTTPException(429, "Queue full; retry later")
        data = {
            "id": request_id,
            "status": "queued",
            "created_at": time.time(),
            "settings": {"num_replicates": num_replicates, "seed": seed},
        }
        bucket.blob(f"requests/{request_id}/input.cif").upload_from_string(content)
        save_status(request_id, data)
        pending.put_nowait(request_id)
    return data


@app.get("/v1/designs/{request_id}")
def status(request_id: str) -> dict[str, Any]:
    """Return the current status of an accepted design request."""
    return read_status(request_id)


@app.get("/v1/designs/{request_id}/results")
def results(request_id: str) -> dict[str, Any]:
    """List downloadable artifacts for a completed design."""
    data = read_status(request_id)
    if data["status"] != "succeeded":
        raise HTTPException(409, "Results are available after successful generation")
    return {
        "id": request_id,
        "artifacts": [
            {"name": name, "download": f"/v1/designs/{request_id}/artifacts/{name}"} for name in data["artifacts"]
        ],
    }


@app.get("/v1/designs/{request_id}/artifacts/{name:path}")
def artifact(request_id: str, name: str) -> Response:
    """Download a file listed in the completed request manifest."""
    data = read_status(request_id)
    if data["status"] != "succeeded" or name not in data.get("artifacts", []):
        raise HTTPException(404, "Artifact not found")
    blob = bucket.blob(f"requests/{request_id}/outputs/{name}")
    return Response(
        blob.download_as_bytes(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{Path(name).name}"'},
    )
