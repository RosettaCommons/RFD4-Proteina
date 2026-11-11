"""Essential API contract, persistence and upstream delegation checks."""

import asyncio
import os
import queue
import time
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import zstandard
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from deploy.api.app import api, runner
from deploy.api.app.errors import ApiError, ErrorCode

ROOT = Path(__file__).resolve().parents[2]
CIF = (
    zstandard.ZstdDecompressor()
    .decompress((ROOT / "data/benchmarks/monomer/monomer-short.cif.zst").read_bytes())
    .decode()
)


def design(seed=42, content=CIF):
    return {
        "task": "generation",
        "input": {"format": "annotated_cif", "file": {"content": content}},
        "sampling": {"seed": seed, "num_replicates": 1},
    }


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result := predicate():
            return result
        time.sleep(0.02)
    raise AssertionError("Timed out")


@pytest.fixture(autouse=True)
def isolated_service(tmp_path, monkeypatch):
    class FakeRunner:
        loads, device = 1, "fake GPU"

        def generate(self, source, output, settings, *limits):
            if settings["seed"] == 1:
                raise ApiError(ErrorCode.OUT_OF_MEMORY, "Reduce input size.")
            if settings["seed"] == 2:
                raise RuntimeError("private diagnostic")
            output.mkdir(parents=True)
            (output / "design_generated.cif").write_bytes(source.read_bytes())

    monkeypatch.setenv("STORE_PATH", str(tmp_path))
    monkeypatch.setattr(api, "store", None)
    monkeypatch.setattr(api, "pending", queue.Queue(maxsize=2))
    monkeypatch.setattr(api, "state", {"ready": False, "error": None, "runner": None, "stop": False})
    monkeypatch.setattr(runner, "Runner", FakeRunner)


def test_jobs_errors_downloads_and_restart(tmp_path):
    with TestClient(api.app) as client:
        wait_for(lambda: client.get("/health/ready").status_code == 200)
        for seed, error in [(42, None), (1, "out_of_memory"), (2, "internal_error"), (43, None)]:
            submitted = client.post("/v1/designs", json=design(seed))
            assert submitted.status_code == 202 and submitted.json()["settings"]["seed"] == seed
            url = "/v1/designs/" + submitted.json()["id"]
            wait_for(lambda url=url: client.get(url).json()["status"] in ("succeeded", "failed"))
            status = client.get(url).json()
            assert status["status"] == ("failed" if error else "succeeded")
            if error:
                assert status["error"]["code"] == error and "private diagnostic" not in str(status)
                assert client.get(url + "/results").status_code == 409
            else:
                download = client.get(url + "/results").json()["artifacts"][0]["download"]
                assert client.get(download).text == CIF
        assert client.get(url + "/artifacts/private.txt").status_code == 404
    api.save_status("a" * 32, {"id": "a" * 32, "status": "running"})
    (tmp_path / "requests" / "broken").mkdir()
    (tmp_path / "requests" / "broken/status.json").write_text("{")
    with TestClient(api.app) as client:
        wait_for(lambda: client.get("/health/ready").status_code == 200)
        assert client.get(download).text == CIF
        assert client.get("/v1/designs/" + "a" * 32).json()["error"]["code"] == "model_unavailable"


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("task", "partial_diffusion", "invalid_request"),
        ("task", "inverse_folding", "invalid_request"),
        ("task", "unknown", "schema_violation"),
        ("sampling", {"num_steps": 10}, "schema_violation"),
        ("partial", {}, "schema_violation"),
        ("input", {"format": "annotated_cif", "file": {"content": "bad"}}, "invalid_input"),
        ("sampling", {"num_replicates": 5}, "request_too_large"),
    ],
)
def test_request_contract(field, value, code):
    api.update_state(ready=True)
    request = design()
    request[field] = value
    response = TestClient(api.app).post("/v1/designs", json=request)
    assert response.status_code == api.STATUS_FOR_CODE[code]
    assert response.json()["error"]["code"] == code
    assert api.pending.empty()


def test_capacity_paths_and_chunked_limit(tmp_path, monkeypatch):
    client = TestClient(api.app)
    assert client.get("/health/ready").status_code == 503
    assert client.get("/v1/info").json()["submission"] == "asynchronous"
    api.store = tmp_path
    api.update_state(ready=True)
    for _ in range(api.pending.maxsize):
        api.pending.put_nowait("queued")
    parse = Mock(side_effect=AssertionError("Full queue must not parse"))
    monkeypatch.setattr(api.CIFFile, "read", parse)
    assert client.post("/v1/designs", json=design()).status_code == 503
    parse.assert_not_called()
    api.save_status("b" * 32, {"id": "b" * 32, "status": "succeeded", "artifacts": ["../private"]})
    with pytest.raises(ApiError):
        api.artifact("b" * 32, "../private")
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", 32)

    async def scenario():
        async def chunks():
            yield b" " * 20
            yield b" " * 20

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as http:
            response = await http.post("/v1/designs", content=chunks())
        assert response.status_code == 413 and response.json()["error"]["code"] == "request_too_large"

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [None, "tokens", "samples", "oom"])
def test_upstream_delegation(tmp_path, monkeypatch, failure):
    import torch

    import rfproteina.generate as upstream

    # The autouse fixture replaces the worker constructor; exercise the actual class here.
    model_runner = ORIGINAL_RUNNER.__new__(ORIGINAL_RUNNER)
    model_runner.base = OmegaConf.create({"inputs": None, "out_dir": None, "seed": None, "num_replicates": 1})
    model_runner.model = object()
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "test_generated.cif").write_text(CIF)
    errors = {
        None: None,
        "tokens": upstream.TokenLimitError(1),
        "samples": upstream.SampleLimitError(1),
        "oom": torch.OutOfMemoryError(),
    }
    generate = Mock(side_effect=errors[failure])
    monkeypatch.setattr(upstream, "run_generation", generate)
    if failure:
        with pytest.raises(ApiError) as exc:
            model_runner.generate(tmp_path / "input.cif", output, {"seed": 42, "num_replicates": 1}, 1024, 4)
        assert exc.value.status == 413
    else:
        assert model_runner.generate(tmp_path / "input.cif", output, {"seed": 42, "num_replicates": 1}, 1024, 4)
        assert generate.call_args.kwargs == {"model": model_runner.model, "max_tokens": 1024, "max_samples": 4}
        assert generate.call_args.args[0].seed == 42 and model_runner.base.seed is None


ORIGINAL_RUNNER = runner.Runner


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_API_TESTS") != "1", reason="Requires a live GPU API")
def test_live_generation():
    with httpx.Client(base_url=os.environ.get("API_URL", "http://localhost:8000"), timeout=60) as client:
        ready = client.get("/health/ready").raise_for_status().json()
        limit = client.get("/v1/info").json()["limits"]["max_tokens"]
        for length, expected in [(min(100, limit), "succeeded"), (limit + 1, "failed"), (min(100, limit), "succeeded")]:
            job = (
                client.post("/v1/designs", json=design(content=CIF.replace("100", str(length))))
                .raise_for_status()
                .json()
            )
            url = "/v1/designs/" + job["id"]
            wait_for(lambda url=url: client.get(url).json()["status"] in ("succeeded", "failed"), timeout=1800)
            result = client.get(url).json()
            assert result["status"] == expected
            if expected == "failed":
                assert result["error"]["code"] == "request_too_large"
            else:
                assert result["checkpoint_loads"] == 1 and result["server_instance_id"] == ready["server_instance_id"]
                files = client.get(url + "/results").json()["artifacts"]
                cif = next(a for a in files if a["name"].endswith("_generated.cif"))
                assert "_atom_site." in client.get(cif["download"]).text
