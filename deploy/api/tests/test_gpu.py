"""Opt-in end-to-end check against a running GPU API.

The fake-runner tests in `test_api.py` cover the HTTP surface without weights.
This one needs a real server and a GPU:

    RUN_GPU_API_TESTS=1 API_URL=http://localhost:8000 \
        PYTHONPATH=deploy/api pytest deploy/api/tests/test_gpu.py -m gpu
"""

import json
import os
import time
from pathlib import Path

import httpx
import pytest
from conftest import EXAMPLE, design_request


def expansion(length: int) -> bytes:
    """This example contains one variable-length segment, with no fixed residues."""
    return EXAMPLE.replace(b"100", str(length).encode())


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_API_TESTS") != "1", reason="Requires a running GPU API")
def test_generation_reuses_resident_model_and_recovers(tmp_path: Path) -> None:
    """Real generation, resident-weight reuse, token rejection, and recovery after one."""
    base = os.environ.get("API_URL", "http://localhost:8000")
    with httpx.Client(base_url=base, timeout=60) as client:
        limits = client.get("/v1/info").raise_for_status().json()["limits"]
        ready = client.get("/health/ready").raise_for_status().json()
        assert ready["checkpoint_loads"] == 1, ready
        example = expansion(min(100, limits["max_tokens"]))
        oversized = expansion(limits["max_tokens"] + 1)

        # The rejection sits between two real designs: the server must keep serving after it.
        for seed, payload, expected in (
            (42, example, "succeeded"),
            (44, oversized, "failed"),
            (43, example, "succeeded"),
        ):
            submitted = client.post("/v1/designs", json=design_request(payload, seed=seed)).raise_for_status().json()
            request_id = submitted["id"]

            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                status = client.get(f"/v1/designs/{request_id}").raise_for_status().json()
                if status["status"] in ("succeeded", "failed"):
                    break
                time.sleep(5)
            else:
                raise TimeoutError(f"{request_id} never reached a terminal status")

            assert status["status"] == expected, status
            (tmp_path / f"{request_id}.json").write_text(json.dumps(status, indent=2))
            if expected == "failed":
                assert status["error"]["code"] == "request_too_large", status
                assert f"{limits['max_tokens']} tokens" in status["error"]["message"], status
                continue

            # Same loaded model and server process; this does not prove graph reuse or GPU residency.
            assert status["checkpoint_loads"] == 1, status
            assert status["server_instance_id"] == ready["server_instance_id"], status

            listing = client.get(f"/v1/designs/{request_id}/results").raise_for_status().json()
            generated = [a for a in listing["artifacts"] if a["name"].endswith("_generated.cif")]
            assert generated, listing
            for index, artifact in enumerate(generated):
                cif = client.get(artifact["download"]).raise_for_status().content
                assert b"_atom_site." in cif and b"data_" in cif, f"Invalid generated CIF: {artifact['name']}"
                (tmp_path / f"{request_id}-{index}.cif").write_bytes(cif)

        final = client.get("/health/ready").raise_for_status().json()
        assert final["checkpoint_loads"] == 1, final
        assert final["server_instance_id"] == ready["server_instance_id"], final
