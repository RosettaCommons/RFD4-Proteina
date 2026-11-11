"""Exercise the public contract against HTTP responses and the generated schema."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from app import api, runner
from app.errors import ApiError, ErrorCode
from conftest import EXAMPLE, design_request, wait_for
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

pytestmark = pytest.mark.usefixtures("isolated_service")


def validate_response(response: httpx.Response, path: str, method: str = "get") -> None:
    document = api.app.openapi()
    schema = document["paths"][path][method]["responses"][str(response.status_code)]["content"]["application/json"][
        "schema"
    ]
    Draft202012Validator({**schema, "components": document["components"]}).validate(response.json())


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


@pytest.mark.parametrize("chunked", [False, True])
def test_request_size_limit_applies_before_json_parsing(monkeypatch: pytest.MonkeyPatch, chunked: bool) -> None:
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", 32)
    admit = Mock(side_effect=AssertionError("Oversized request must never be admitted"))
    monkeypatch.setattr(api, "admit_request", admit)

    async def scenario() -> None:
        async def chunks() -> AsyncIterator[bytes]:
            yield b" " * 20
            yield b" " * 20

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            response = await client.post(
                "/v1/designs", content=chunks() if chunked else b" " * 40, headers={"Content-Type": "application/json"}
            )
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
        (client.get("/v1/designs/" + "a" * 32), "/v1/designs/{request_id}", "get", 404),
    ]
    api.update_state(ready=True, runner=SimpleNamespace(device="fake GPU", loads=1))
    responses.extend(
        [
            (client.get("/health/ready"), "/health/ready", "get", 200),
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
    document = api.app.openapi()
    request_schema = document["paths"]["/v1/designs"]["post"]["requestBody"]["content"]
    assert set(request_schema) == {"application/json"}
    assert document["components"]["schemas"]["TaskSupport"]["properties"]["supported"]["items"]["enum"] == list(
        api.TASKS
    )


def test_worker_errors_are_typed_and_later_requests_still_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeRunner:
        loads, device = 1, "fake GPU"

        def generate(self, input_path: Path, output_dir: Path, settings: dict[str, int], *limits: int) -> None:
            if settings["seed"] == 1:
                raise ApiError(ErrorCode.REQUEST_TOO_LARGE, "Too many tokens.", ["input"])
            if settings["seed"] == 2:
                raise ApiError(ErrorCode.OUT_OF_MEMORY, "Too large for GPU.", ["input"])
            if settings["seed"] == 3:
                raise RuntimeError("private diagnostic")
            output_dir.mkdir(parents=True)
            (output_dir / "design_generated.cif").write_bytes(input_path.read_bytes())

    monkeypatch.setattr(runner, "Runner", FakeRunner)
    with TestClient(api.app) as client:
        wait_for(lambda: api.snapshot()["ready"])
        for seed, code in [(1, "request_too_large"), (2, "out_of_memory"), (3, "internal_error"), (4, None)]:
            job = client.post("/v1/designs", json=design_request(seed=seed)).json()
            url = f"/v1/designs/{job['id']}"
            wait_for(lambda url=url: client.get(url).json()["status"] in ("succeeded", "failed"))
            response = client.get(url)
            validate_response(response, "/v1/designs/{request_id}")
            if code:
                assert response.status_code == 200 and response.json()["status"] == "failed"
                assert response.json()["error"]["code"] == code
                assert "private diagnostic" not in response.text
                validate_response(client.get(url + "/results"), "/v1/designs/{request_id}/results")
            else:
                assert response.json()["status"] == "succeeded"
                results = client.get(url + "/results")
                validate_response(results, "/v1/designs/{request_id}/results")
                assert client.get(results.json()["artifacts"][0]["download"]).content == EXAMPLE
        assert client.get("/health/ready").status_code == 200


def test_legacy_failure_and_restart_errors_remain_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api.store = tmp_path / "store"
    old_id, interrupted_id = "a" * 32, "b" * 32
    api.save_status(old_id, {"id": old_id, "status": "failed", "error": "Old failure"})
    old_path = api.request_dir(old_id) / "status.json"
    old_bytes = old_path.read_bytes()
    api.save_status(interrupted_id, {"id": interrupted_id, "status": "running"})
    monkeypatch.setattr(runner, "Runner", lambda: SimpleNamespace(device="fake GPU", loads=1))
    with TestClient(api.app) as client:
        wait_for(lambda: api.snapshot()["ready"])
        for request_id, code in [(old_id, "internal_error"), (interrupted_id, "model_unavailable")]:
            response = client.get(f"/v1/designs/{request_id}")
            validate_response(response, "/v1/designs/{request_id}")
            assert response.json()["error"]["code"] == code
        assert old_path.read_bytes() == old_bytes
