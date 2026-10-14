"""Exercise HTTP generation twice and verify reuse of the loaded model."""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

base = os.environ.get("API_URL", "http://localhost:8000")


def request(path: str, payload: bytes | None = None, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(base + path, data=payload, headers=headers or {})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def submit(content: bytes, seed: int) -> dict[str, Any]:
    boundary = "rfd4smokeform"
    payload = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="input.cif"\r\n'
        "Content-Type: chemical/x-cif\r\n\r\n"
    ).encode() + content
    payload += (
        f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="seed"\r\n\r\n{seed}' f"\r\n--{boundary}--\r\n"
    ).encode()
    return json.loads(request("/v1/designs", payload, {"Content-Type": f"multipart/form-data; boundary={boundary}"}))


ready = json.loads(request("/health/ready"))
assert ready["checkpoint_loads"] == 1, ready
assert request("/docs")
try:
    submit(b"not a CIF", 42)
    raise AssertionError("Invalid input was accepted")
except urllib.error.HTTPError as exc:
    assert exc.code == 422, exc.code
source = Path(os.environ.get("RFD4_SOURCE", str(Path(__file__).resolve().parents[2])))
content = (source / "data/benchmarks/monomer/monomer-short.cif").read_bytes()
Path("results").mkdir(exist_ok=True)
for seed in (42, 43):
    data = submit(content, seed)
    request_id = data["id"]
    print(f"Submitted {request_id}", flush=True)
    deadline = time.time() + 1800
    while time.time() < deadline:
        status = json.loads(request(f"/v1/designs/{request_id}"))
        if status["status"] == "failed":
            raise RuntimeError(status)
        if status["status"] == "succeeded":
            break
        time.sleep(5)
    else:
        raise TimeoutError(request_id)
    listing = json.loads(request(f"/v1/designs/{request_id}/results"))
    generated = [a for a in listing["artifacts"] if a["name"].endswith("_generated.cif")]
    assert generated, listing
    cif = request(generated[0]["download"])
    assert b"_atom_site." in cif and b"data_" in cif, "Invalid generated CIF"
    Path(f"results/{request_id}.cif").write_bytes(cif)
    Path(f"results/{request_id}.json").write_text(json.dumps(status, indent=2))
    assert status["checkpoint_loads"] == 1, status
    assert status["server_instance_id"] == ready["server_instance_id"], status
    print(f"Succeeded in {status['duration_seconds']} seconds; downloaded {len(cif)} bytes", flush=True)
    if seed == 42:
        # Preserve the input grammar while requesting an over-limit expansion.
        oversized = submit(content.replace(b"100", b"513"), 44)
        failure_deadline = time.time() + 180
        while time.time() < failure_deadline:
            invalid = json.loads(request(f"/v1/designs/{oversized['id']}"))
            if invalid["status"] in ("failed", "succeeded"):
                break
            time.sleep(2)
        else:
            raise TimeoutError(oversized["id"])
        assert invalid["status"] == "failed" and "512 tokens" in invalid["error"], invalid
        Path("results/over-limit.json").write_text(json.dumps(invalid, indent=2))
        print("Rejected 513-token design; checking recovery with the next request", flush=True)
final = json.loads(request("/health/ready"))
assert final["checkpoint_loads"] == 1 and final["server_instance_id"] == ready["server_instance_id"]
print("PASS: two designs, artifact downloads, input/token limits, recovery, resident model, HTTP docs")
