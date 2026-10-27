"""Validate serialization, persistence, and failures without model weights."""

import asyncio
import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from app import api, runner
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

EXAMPLE = (Path(__file__).resolve().parents[1] / "examples/monomer-short.cif").read_bytes()


@pytest.fixture(autouse=True)
def isolated_service(tmp_path: Path, monkeypatch: MonkeyPatch) -> Iterator[None]:
    """Restore all service globals after every test, including lifespan assignments."""
    monkeypatch.setenv("STORE_PATH", str(tmp_path / "store"))
    monkeypatch.setattr(api, "store", None)
    monkeypatch.setattr(api, "work", tmp_path / "work")
    monkeypatch.setattr(api, "pending", queue.Queue(maxsize=8))
    monkeypatch.setattr(api, "state", {"ready": False, "error": None, "runner": None, "stop": False})
    yield


def wait_for(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out")


def test_resident_worker_serializes_requests_and_persists_results(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    class FakeRunner:
        loads = 1
        device = "fake GPU"
        instances = 0
        active = 0
        maximum = 0

        def __init__(self):
            type(self).instances += 1

        def generate(self, input_path: Path, output_dir: Path, settings: dict[str, int]) -> None:
            cls = type(self)
            cls.active += 1
            cls.maximum = max(cls.maximum, cls.active)
            try:
                time.sleep(0.05)
                if settings["seed"] == 99:
                    raise ValueError("Bad design input")
                output_dir.mkdir(parents=True)
                (output_dir / "test_generated.cif").write_bytes(input_path.read_bytes())
            finally:
                cls.active -= 1

    monkeypatch.setattr(runner, "Runner", FakeRunner)
    with TestClient(api.app) as client:
        wait_for(lambda: client.get("/health/ready").status_code == 200)
        assert client.post("/v1/designs", files={"file": ("bad.cif", b"bad")}).status_code == 422
        assert client.post("/v1/designs", files={"file": ("big.cif", b"x" * (4 * 1024 * 1024 + 1))}).status_code == 413
        assert (
            client.post(
                "/v1/designs",
                files={"file": ("input.cif", EXAMPLE)},
                data={"num_replicates": 5},
            ).status_code
            == 422
        )
        assert client.get("/v1/designs/not-an-id").status_code == 404
        ids = []
        for seed in (42, 43, 99, 44):
            response = client.post(
                "/v1/designs",
                files={"file": ("input.cif", EXAMPLE)},
                data={"seed": seed},
            )
            assert response.status_code == 202
            ids.append(response.json()["id"])
        wait_for(
            lambda: all(client.get(f"/v1/designs/{rid}").json()["status"] in ("succeeded", "failed") for rid in ids)
        )
        assert FakeRunner.instances == 1 and FakeRunner.maximum == 1
        assert client.get(f"/v1/designs/{ids[2]}").json()["status"] == "failed"
        assert client.get(f"/v1/designs/{ids[3]}").json()["status"] == "succeeded"
        artifact = client.get(f"/v1/designs/{ids[0]}/results").json()["artifacts"][0]
        assert client.get(artifact["download"]).content == EXAMPLE
        assert client.get(f"/v1/designs/{ids[0]}/artifacts/private.txt").status_code == 404
    wait_for(lambda: not api.state["ready"])
    interrupted_id = "a" * 32
    api.save_status(interrupted_id, {"id": interrupted_id, "status": "running"})
    with TestClient(api.app) as client:
        wait_for(lambda: client.get("/health/ready").status_code == 200)
        assert client.get(f"/v1/designs/{ids[0]}").json()["status"] == "succeeded"
        assert client.get(artifact["download"]).status_code == 200
        assert client.get(f"/v1/designs/{interrupted_id}").json()["status"] == "failed"


def test_startup_failure_is_unready_and_not_live(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    def fail_startup() -> None:
        raise RuntimeError("Checkpoint unavailable")

    monkeypatch.setenv("STORE_PATH", str(tmp_path / "store"))
    monkeypatch.setattr(runner, "Runner", fail_startup)
    with TestClient(api.app) as client:
        wait_for(lambda: api.state["error"] is not None)
        assert client.get("/health/ready").status_code == 503
        assert client.get("/health/live").status_code == 503
        assert (
            client.post(
                "/v1/designs",
                files={"file": ("input.cif", EXAMPLE)},
            ).status_code
            == 503
        )


def test_queue_applies_backpressure(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    import threading

    started = threading.Event()
    release = threading.Event()

    class BlockingRunner:
        loads = 1
        device = "fake GPU"

        def __init__(self):
            pass

        def generate(self, input_path: Path, output_dir: Path, settings: dict[str, int]) -> None:
            started.set()
            assert release.wait(timeout=5)
            output_dir.mkdir(parents=True)
            (output_dir / "test_generated.cif").write_bytes(input_path.read_bytes())

    monkeypatch.setenv("STORE_PATH", str(tmp_path / "store"))
    monkeypatch.setattr(api, "work", tmp_path / "work")
    monkeypatch.setattr(api, "pending", queue.Queue(maxsize=8))
    monkeypatch.setattr(runner, "Runner", BlockingRunner)
    with TestClient(api.app) as client:
        wait_for(lambda: client.get("/health/ready").status_code == 200)
        files = {"file": ("input.cif", EXAMPLE)}
        first = client.post("/v1/designs", files=files)
        assert first.status_code == 202
        assert started.wait(timeout=2)
        assert client.get(f"/v1/designs/{first.json()['id']}/results").status_code == 409
        try:
            accepted = [client.post("/v1/designs", files=files) for _ in range(8)]
            assert all(response.status_code == 202 for response in accepted)
            assert client.post("/v1/designs", files=files).status_code == 429
        finally:
            release.set()
        ids = [first.json()["id"], *[response.json()["id"] for response in accepted]]
        wait_for(lambda: all(client.get(f"/v1/designs/{rid}").json()["status"] == "succeeded" for rid in ids))


def test_uninitialized_storage_returns_unavailable() -> None:
    # Without the context manager lifespan has not initialized storage.
    client = TestClient(api.app)
    assert client.get("/v1/designs/" + "a" * 32).status_code == 503


def test_format_checks_reject_header_only_cif(monkeypatch: MonkeyPatch) -> None:
    api.update_state(ready=True)
    client = TestClient(api.app)
    for content in (
        b"data_bad\n_atom_site.id 1",
        b"data_bad\nloop_\n_atom_site.id\n1 2 3",
        b"\xff",
        b"_atom_site.id 1",
    ):
        assert client.post("/v1/designs", files={"file": ("bad.cif", content)}).status_code == 422


def test_artifact_path_escape_is_rejected(tmp_path: Path) -> None:
    api.store = tmp_path
    request_id = "b" * 32
    api.save_status(request_id, {"id": request_id, "status": "succeeded", "artifacts": ["../private.txt"]})
    with pytest.raises(HTTPException) as exc:
        api.artifact(request_id, "../private.txt")
    assert exc.value.status_code == 404


def test_slow_large_cif_admission_does_not_block_health(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Hold parsing open and verify another request completes before parsing ends."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_read = api.CIFFile.read

    def slow_read(file: object) -> object:
        started.set()
        assert release.wait(timeout=10), "Test did not release the parser"
        try:
            return original_read(file)
        finally:
            finished.set()

    monkeypatch.setattr(api.CIFFile, "read", slow_read)
    api.store = tmp_path / "store"
    api.update_state(ready=True)
    content = EXAMPLE + b"\n#" + b"x" * (3 * 1024 * 1024) + b"\n"

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            submission = asyncio.create_task(client.post("/v1/designs", files={"file": ("large.cif", content)}))
            try:
                assert await asyncio.to_thread(started.wait, 3)
                assert not finished.is_set()
                response = await asyncio.wait_for(client.get("/health/live"), timeout=1)
                assert response.status_code == 200
                assert not finished.is_set(), "Health must respond while parsing is still blocked"
            finally:
                release.set()
                response = await asyncio.wait_for(submission, timeout=10)
            assert response.status_code == 202

    asyncio.run(scenario())


def test_recovery_skips_corrupt_and_mismatched_statuses(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(runner, "Runner", lambda: SimpleNamespace(device="fake GPU", loads=1))
    root = tmp_path / "store" / "requests"
    corrupt = {
        "a" * 32: '{"id":',
        "b" * 32: json.dumps({"id": "b" * 32}),
        "c" * 32: json.dumps({"id": "../invalid", "status": "running"}),
        "d" * 32: json.dumps({"id": "e" * 32, "status": "running"}),
        "f" * 32: "[]",
        "invalid-directory": json.dumps({"id": "invalid-directory", "status": "queued"}),
        "3" * 32: json.dumps({"status": "running"}),
        "4" * 32: json.dumps({"id": "4" * 32, "status": "unknown"}),
        "5" * 32: json.dumps({"id": "5" * 32, "status": []}),
    }
    for request_id, text in corrupt.items():
        path = root / request_id / "status.json"
        path.parent.mkdir(parents=True)
        path.write_text(text)
    completed_id = "e" * 32
    healthy = {"1" * 32: "queued", "2" * 32: "running", completed_id: "succeeded"}
    for request_id, status in healthy.items():
        path = root / request_id / "status.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"id": request_id, "status": status}))
    with TestClient(api.app) as client:
        wait_for(lambda: client.get("/health/ready").status_code == 200)
        for request_id in ("1" * 32, "2" * 32):
            assert client.get(f"/v1/designs/{request_id}").json()["status"] == "failed"
        assert client.get(f"/v1/designs/{completed_id}").json()["status"] == "succeeded"
        assert client.get("/health/live").status_code == 200
    assert sum("Skipping invalid request status" in record.message for record in caplog.records) == len(corrupt)
    for request_id, text in corrupt.items():
        assert (root / request_id / "status.json").read_text() == text
