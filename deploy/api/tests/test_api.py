"""HTTP contract, worker lifecycle and admission regressions without model weights."""

import asyncio
import json
import re
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from app import api, runner
from app.errors import STATUS_FOR_CODE, ApiError, ErrorCode, ErrorResponse
from conftest import EXAMPLE, design_request, wait_for
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from pytest import MonkeyPatch

pytestmark = pytest.mark.usefixtures("isolated_service")


def validate_response(response: httpx.Response, path: str | None, method: str = "get") -> None:
    if path is None:
        Draft202012Validator(ErrorResponse.model_json_schema()).validate(response.json())
        return
    document = api.app.openapi()
    schema = document["paths"][path][method]["responses"][str(response.status_code)]["content"]["application/json"][
        "schema"
    ]
    Draft202012Validator({**schema, "components": document["components"]}).validate(response.json())


def test_worker_backpressure_errors_and_persistence(monkeypatch: MonkeyPatch) -> None:
    started, release = threading.Event(), threading.Event()
    failures = {
        1: ApiError(ErrorCode.REQUEST_TOO_LARGE, "Too many tokens.", ["input"]),
        2: ApiError(ErrorCode.OUT_OF_MEMORY, "Too large for GPU.", ["input"]),
        3: ValueError("private diagnostic"),
    }

    class FakeRunner:
        loads = 1
        device = "fake GPU"
        instances = 0
        active = 0
        maximum = 0

        def __init__(self):
            type(self).instances += 1

        def generate(self, input_path: Path, output_dir: Path, settings: dict[str, int], *limits: int) -> None:
            cls = type(self)
            cls.active += 1
            cls.maximum = max(cls.maximum, cls.active)
            try:
                started.set()
                assert release.wait(timeout=5)
                time.sleep(0.05)
                if settings["seed"] in (1, 2, 3):
                    raise failures[settings["seed"]]
                output_dir.mkdir(parents=True)
                (output_dir / "test_generated.cif").write_bytes(input_path.read_bytes())
            finally:
                cls.active -= 1

    monkeypatch.setattr(runner, "Runner", FakeRunner)
    with TestClient(api.app) as client:
        wait_for(lambda: client.get("/health/ready").status_code == 200)
        ids = []
        seeds = (42, 43, 1, 2, 3, 44, 45, 46, 47)  # One active request plus eight queued.
        try:
            for seed in seeds:
                response = client.post("/v1/designs", json=design_request(seed=seed))
                assert response.status_code == 202
                ids.append(response.json()["id"])
                assert started.wait(timeout=2)
            assert client.get(f"/v1/designs/{ids[0]}/results").status_code == 409
            busy = client.post("/v1/designs", json=design_request())
            assert busy.status_code == 503 and busy.json()["error"]["code"] == "service_busy"
        finally:
            release.set()
        wait_for(
            lambda: all(client.get(f"/v1/designs/{rid}").json()["status"] in ("succeeded", "failed") for rid in ids)
        )
        assert FakeRunner.instances == 1 and FakeRunner.maximum == 1
        codes = {1: "request_too_large", 2: "out_of_memory", 3: "internal_error"}
        for request_id, seed in zip(ids, seeds, strict=True):
            code = codes.get(seed)
            url = f"/v1/designs/{request_id}"
            response = client.get(url)
            assert response.status_code == 200
            validate_response(response, "/v1/designs/{request_id}")
            assert response.json()["status"] == ("failed" if code else "succeeded")
            results = client.get(url + "/results")
            validate_response(results, "/v1/designs/{request_id}/results")
            if code:
                assert response.json()["error"]["code"] == code
                assert "private diagnostic" not in response.text
                assert results.status_code == 409
            else:
                artifact = results.json()["artifacts"][0]
                assert client.get(artifact["download"]).content == EXAMPLE
                assert client.get(url + "/artifacts/private.txt").status_code == 404
        assert client.get("/health/ready").status_code == 200
    wait_for(lambda: not api.state["ready"])
    with TestClient(api.app) as client:
        wait_for(lambda: client.get("/health/ready").status_code == 200)
        assert client.get(f"/v1/designs/{ids[0]}").json()["status"] == "succeeded"
        assert client.get(artifact["download"]).status_code == 200


def test_startup_failure_is_unready_and_not_live(monkeypatch: MonkeyPatch) -> None:
    def fail_startup() -> None:
        raise RuntimeError("Checkpoint unavailable")

    monkeypatch.setattr(runner, "Runner", fail_startup)
    with TestClient(api.app) as client:
        wait_for(lambda: api.state["error"] is not None)
        assert client.get("/health/ready").status_code == 503
        assert client.get("/health/live").status_code == 503
        assert client.post("/v1/designs", json=design_request()).status_code == 503


def test_uninitialized_storage_returns_unavailable() -> None:
    # Without the context manager lifespan has not initialized storage.
    client = TestClient(api.app)
    assert client.get("/v1/designs/" + "a" * 32).status_code == 503


def test_format_checks_reject_header_only_cif() -> None:
    api.update_state(ready=True)
    client = TestClient(api.app)
    for content in (
        b"data_bad\n_atom_site.id 1",
        b"data_bad\nloop_\n_atom_site.id\n1 2 3",
        b"\xff",
        b"_atom_site.id 1",
    ):
        response = client.post(
            "/v1/designs", content=json.dumps(design_request(content)), headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


def test_artifact_path_escape_is_rejected(tmp_path: Path) -> None:
    api.store = tmp_path
    request_id = "b" * 32
    api.save_status(request_id, {"id": request_id, "status": "succeeded", "artifacts": ["../private.txt"]})
    with pytest.raises(ApiError) as exc:
        api.artifact(request_id, "../private.txt")
    assert exc.value.status == 404 and exc.value.code == ErrorCode.NOT_FOUND


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
            submission = asyncio.create_task(client.post("/v1/designs", json=design_request(content)))
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
    api.store = tmp_path / "store"
    old_id = "6" * 32
    api.save_status(old_id, {"id": old_id, "status": "failed", "error": "Old failure"})
    old_path = api.request_dir(old_id) / "status.json"
    old_bytes = old_path.read_bytes()
    with TestClient(api.app) as client:
        wait_for(lambda: client.get("/health/ready").status_code == 200)
        for request_id in ("1" * 32, "2" * 32, old_id):
            response = client.get(f"/v1/designs/{request_id}")
            validate_response(response, "/v1/designs/{request_id}")
            assert response.json()["status"] == "failed"
            assert response.json()["error"]["code"] == (
                "internal_error" if request_id == old_id else "model_unavailable"
            )
        assert old_path.read_bytes() == old_bytes
        assert client.get(f"/v1/designs/{completed_id}").json()["status"] == "succeeded"
        assert client.get("/health/live").status_code == 200
    assert sum("Skipping invalid request status" in record.message for record in caplog.records) == len(corrupt)
    for request_id, text in corrupt.items():
        assert (root / request_id / "status.json").read_text() == text


@pytest.mark.parametrize("during_parse", [False, True])
def test_queue_admission_races(tmp_path: Path, monkeypatch: MonkeyPatch, during_parse: bool) -> None:
    api.update_state(ready=True)
    api.store = tmp_path / "store"
    original_read = api.CIFFile.read

    def fill_queue() -> None:
        for index in range(api.pending.maxsize):
            api.pending.put_nowait(f"queued-{index}")

    def parse_and_fill(file: object) -> object:
        fill_queue()
        return original_read(file)

    parse = Mock(side_effect=parse_and_fill)
    admit = Mock(wraps=api.admit_request)
    monkeypatch.setattr(api.CIFFile, "read", parse)
    monkeypatch.setattr(api, "admit_request", admit)
    if not during_parse:
        fill_queue()
    busy = TestClient(api.app).post("/v1/designs", json=design_request())
    assert busy.status_code == 503 and busy.json()["error"]["code"] == "service_busy"
    assert parse.call_count == admit.call_count == int(during_parse)
    assert not api.store.exists(), "Rejection must precede persisting the request"


def test_status_for_code_is_total_over_error_code() -> None:
    """A handler never has to guess a status, and the two cannot drift apart."""
    assert set(STATUS_FOR_CODE) == set(ErrorCode)
    assert STATUS_FOR_CODE[ErrorCode.SERVICE_BUSY] == 503, "a full queue is not client rate limiting"
    assert STATUS_FOR_CODE[ErrorCode.OUT_OF_MEMORY] == 413, "retrying the same job exhausts the same GPU"


def test_max_tokens_default_matches_the_repository_compile_buckets() -> None:
    """The token ceiling is the repository's own: configs/inference.yaml tops out at 1024."""
    config = (Path(__file__).resolve().parents[3] / "configs/inference.yaml").read_text()
    buckets = [int(n) for n in re.search(r"compile_lengths: \[([^\]]*)\]", config).group(1).split(",")]
    assert max(buckets) == api.MAX_TOKENS


@pytest.mark.parametrize(
    "task,status,code",
    [
        ("generation", 202, None),
        ("nonsense", 422, "schema_violation"),
        # Changes the sampler call rather than the conditioning, so it cannot be
        # silently downgraded to an ordinary trajectory.
        ("partial_diffusion", 400, "invalid_request"),
        ("inverse_folding", 400, "invalid_request"),
    ],
)
def test_task_is_validated_and_unsupported_modes_are_refused(
    tmp_path: Path, task: str, status: int, code: str | None
) -> None:
    api.store = tmp_path
    api.update_state(ready=True)
    client = TestClient(api.app)
    response = client.post("/v1/designs", json=design_request(task=task))
    assert response.status_code == status, response.json()
    if code is None:
        assert client.get(f"/v1/designs/{response.json()['id']}").json()["task"] == task
    else:
        assert response.json()["error"]["code"] == code


@pytest.mark.parametrize(
    "path,value",
    [
        (("partial",), {"t_backbone": 0.5, "t_latent": 0.5}),
        (("search",), {}),
        (("metrics",), {}),
        (("output",), {"include_sequence": False}),
        (("seed",), 123),
        (("sampling", "sead"), 123),
        (("sampling", "num_steps"), 10),
        (("sampling", "batch_size"), 2),
        (("input", "files"), []),
        (("input", "file", "url"), "ignored"),
    ],
)
def test_unsupported_fields_are_rejected_before_admission(
    monkeypatch: pytest.MonkeyPatch, path: tuple[str, ...], value: Any
) -> None:
    api.update_state(ready=True)
    admit = Mock(side_effect=AssertionError("Invalid request must never be admitted"))
    monkeypatch.setattr(api, "admit_request", admit)
    request = design_request()
    target = request
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    response = TestClient(api.app).post("/v1/designs", json=request)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "schema_violation"
    assert any(".".join(path) in detail for detail in response.json()["error"]["details"])
    validate_response(response, "/v1/designs", "post")
    admit.assert_not_called()


def test_supported_json_fields_and_random_seed_are_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api.store = tmp_path
    api.update_state(ready=True)
    random_seed = Mock(side_effect=[17, 18])
    monkeypatch.setattr(api.secrets, "randbelow", random_seed)
    client = TestClient(api.app)
    for requested, expected in ((0, 0), (None, 17), (None, 18)):
        request = design_request(seed=requested, num_replicates=2)
        request["input"]["example_id"] = "monomer"
        response = client.post("/v1/designs", json=request)
        assert response.status_code == 202, response.text
        validate_response(response, "/v1/designs", "post")
        job = response.json()
        assert job["settings"] == {"seed": expected, "num_replicates": 2}
        assert job["task"] == "generation" and job["example_id"] == "monomer"
        assert (api.request_dir(job["id"]) / "monomer.cif").read_bytes() == EXAMPLE
        polled = client.get(f"/v1/designs/{job['id']}")
        validate_response(polled, "/v1/designs/{request_id}")
        assert polled.json() == job
    assert random_seed.call_count == 2


@pytest.mark.parametrize("body", ["json", "raw", "chunked"])
def test_request_size_limit_applies_before_json_parsing(monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", 32)
    admit = Mock(side_effect=AssertionError("Oversized request must never be admitted"))
    monkeypatch.setattr(api, "admit_request", admit)

    async def scenario() -> None:
        async def chunks() -> AsyncIterator[bytes]:
            yield b" " * 20
            yield b" " * 20

        content = {"chunked": chunks(), "json": json.dumps({"content": " " * 40}).encode(), "raw": b" " * 40}[body]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            response = await client.post("/v1/designs", content=content, headers={"Content-Type": "application/json"})
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"
        validate_response(response, "/v1/designs", "post")

    asyncio.run(scenario())
    admit.assert_not_called()


def test_errors_and_health_match_openapi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api.store = tmp_path
    client = TestClient(api.app, raise_server_exceptions=False)
    responses = [
        (client.get("/v1/info"), "/v1/info", "get", 200),
        (client.get("/health/ready"), "/health/ready", "get", 503),
        (client.get("/health/live"), "/health/live", "get", 200),
        (client.post("/v1/designs", json=design_request()), "/v1/designs", "post", 503),
        (client.post("/v1/designs", json={}), "/v1/designs", "post", 422),
        (client.get("/no/such/route"), None, "get", 404),
        (client.get("/v1/designs/not-an-id"), "/v1/designs/{request_id}", "get", 404),
        (client.post("/v1/designs", json=design_request(b"data_x", num_replicates=0)), "/v1/designs", "post", 422),
        (client.get("/v1/designs/" + "a" * 32), "/v1/designs/{request_id}", "get", 404),
    ]
    body = client.get("/v1/info").raise_for_status().json()
    assert body["limits"] == {
        "max_request_bytes": api.MAX_REQUEST_BYTES,
        "max_tokens": api.MAX_TOKENS,
        "max_replicates": api.MAX_REPLICATES,
        "queue_depth": api.QUEUE_DEPTH,
    }
    assert set(body["tasks"]["supported"]) | set(body["tasks"]["unsupported"]) == set(api.TASKS)
    assert set(body["tasks"]["unsupported"]) == api.UNSUPPORTED_TASKS

    api.update_state(ready=True, runner=SimpleNamespace(device="fake GPU", loads=1))
    responses.extend(
        [
            (client.get("/health/ready"), "/health/ready", "get", 200),
            (client.post("/v1/designs", json=design_request(b"bad")), "/v1/designs", "post", 422),
            (client.post("/v1/designs", json=design_request(task="inverse_folding")), "/v1/designs", "post", 400),
            (
                client.post("/v1/designs", json=design_request(num_replicates=api.MAX_REPLICATES + 1)),
                "/v1/designs",
                "post",
                413,
            ),
        ]
    )
    monkeypatch.setattr(api, "read_status", Mock(side_effect=OSError("private diagnostic")))
    response = client.get("/v1/designs/" + "a" * 32)
    responses.append((response, "/v1/designs/{request_id}", "get", 500))
    assert response.json()["error"] == {"code": "internal_error", "message": "Internal error.", "details": []}
    for response, path, method, expected in responses:
        assert response.status_code == expected, response.text
        validate_response(response, path, method)
        if expected >= 400:
            assert set(response.json()["error"]) == {"code", "message", "details"}
    document = api.app.openapi()
    request_schema = document["paths"]["/v1/designs"]["post"]["requestBody"]["content"]
    assert set(request_schema) == {"application/json"}
    assert document["components"]["schemas"]["TaskSupport"]["properties"]["supported"]["items"]["enum"] == list(
        api.TASKS
    )
