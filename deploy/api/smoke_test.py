"""Exercise a running GPU API; also exposed as an opt-in GPU pytest case."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import httpx


def run_smoke(output: Path) -> None:
    """Verify real generation, resident-weight reuse, and recovery from a token rejection."""
    output.mkdir(parents=True, exist_ok=True)
    content = (Path(__file__).resolve().parent / "examples/monomer-short.cif").read_bytes()
    with httpx.Client(base_url=os.environ.get("API_URL", "http://localhost:8000"), timeout=60) as client:
        ready = client.get("/health/ready").raise_for_status().json()
        assert ready["checkpoint_loads"] == 1, ready
        for seed, payload, expected in (
            (42, content, "succeeded"),
            (44, content.replace(b"100", b"513"), "failed"),
            (43, content, "succeeded"),
        ):
            data = (
                client.post("/v1/designs", files={"file": ("input.cif", payload)}, data={"seed": seed})
                .raise_for_status()
                .json()
            )
            request_id = data["id"]
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                status = client.get(f"/v1/designs/{request_id}").raise_for_status().json()
                if status["status"] in ("succeeded", "failed"):
                    break
                time.sleep(5)
            else:
                raise TimeoutError(request_id)
            assert status["status"] == expected, status
            (output / f"{request_id}.json").write_text(json.dumps(status, indent=2))
            if expected == "failed":
                assert "512 tokens" in status["error"], status
                continue
            assert status["checkpoint_loads"] == 1, status
            assert status["server_instance_id"] == ready["server_instance_id"], status
            listing = client.get(f"/v1/designs/{request_id}/results").raise_for_status().json()
            generated = [a for a in listing["artifacts"] if a["name"].endswith("_generated.cif")]
            assert generated, listing
            for index, artifact in enumerate(generated):
                cif = client.get(artifact["download"]).raise_for_status().content
                assert b"_atom_site." in cif and b"data_" in cif, "Invalid generated CIF"
                (output / f"{request_id}-{index}.cif").write_bytes(cif)
            print(f"Succeeded {request_id} in {status['duration_seconds']} seconds", flush=True)
        final = client.get("/health/ready").raise_for_status().json()
        assert final["checkpoint_loads"] == 1 and final["server_instance_id"] == ready["server_instance_id"]
    print(f"PASS: two designs, downloads, token rejection/recovery; outputs: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Download directory (default: a new temporary directory)")
    args = parser.parse_args()
    run_smoke(args.out or Path(tempfile.mkdtemp(prefix="rfd4-smoke-")))
