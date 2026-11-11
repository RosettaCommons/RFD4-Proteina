# Run the protein-design API

A single-replica deployment sample for Linux x86_64 with a [tested NVIDIA GPU](../docs/installation.rst#supported-hardware),
a compatible driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Keep **one replica and one Uvicorn worker**; the queue and disk store have one owner.
There is no built-in authentication: keep access internal or provide authenticated HTTPS.

## Docker

From the repository root, with BuildKit available:

```bash
docker build --platform linux/amd64 --target api -t rfd4-api:local .
docker volume create rfd4-data
docker run --rm --name rfd4-api --gpus all --shm-size=2g \
  -p 127.0.0.1:8000:8000 -v rfd4-data:/workspace rfd4-api:local
```

The default Docker target retains the CLI; select `api` explicitly. CUDA 13 is the default;
add `--build-arg RFD4_ACCELERATOR=cu12` for CUDA 12. CPU and macOS Docker cannot serve this model.
Startup downloads the model/autoencoder pinned in `configs/inference.yaml` into
`/workspace/cache/huggingface`. Wait for `/health/ready` to return 200.
Persist `/workspace` for weights, results and compile caches; `HF_HOME`, `STORE_PATH` and
`WORK_PATH` override cache, artifact and scratch locations. Prepopulate the cache and set
`HF_HUB_OFFLINE=1` for offline operation.

## Kubernetes

Provide GPU nodes with the NVIDIA device plugin, a readable image registry and a default
StorageClass or matching 100 GiB ReadWriteOnce PV. Customize node selectors, tolerations,
RuntimeClass and imagePullSecrets for your cluster; volume topology may restrict scheduling.

```bash
export IMAGE=YOUR_REGISTRY/rfd4-api:YOUR_TAG
docker tag rfd4-api:local "$IMAGE"
docker push "$IMAGE"
kubectl kustomize deploy/kubernetes > /tmp/rfd4-api.yaml
python3 - <<'PYIMAGE'
import os
from pathlib import Path
p = Path('/tmp/rfd4-api.yaml')
p.write_text(p.read_text().replace('image: rfd4-api:local', 'image: ' + os.environ['IMAGE']))
PYIMAGE
kubectl apply -f /tmp/rfd4-api.yaml
kubectl rollout status deployment/api -n rfd4-proteina --timeout=1800s
kubectl port-forward -n rfd4-proteina service/api 8000:8000
```

Prefer an immutable image digest. The base creates a namespace, Deployment, internal Service
and PVC. Recreate rollouts release the GPU/volume before replacement. Diagnose failures with
`kubectl logs -n rfd4-proteina deployment/api` and `kubectl describe pod -n rfd4-proteina -l app=rfd4-api`.

## Request contract

`/docs` and `/openapi.json` describe the schemas; `/v1/info` reports tasks and limits during loading.
This accepts a strict subset of the [NIM JSON request](https://docs.nvidia.com/nim/bionemo/rfd4-proteina/api-reference/generate-rfd-4-proteina-designs):
`task`, annotated-CIF `input`, and `sampling.seed` / `sampling.num_replicates`.
Unknown fields are rejected at every level. Optional `file.filename` is a label; `example_id`
names the staged CIF and outputs. Omitted/null seeds are randomized and recorded in job settings;
explicit seeds span 0–2147483647. Earlier multipart clients must migrate to JSON.

```bash
python3 - <<'PY' > /tmp/rfd4-request.json
import json
from pathlib import Path
print(json.dumps({
    "task": "generation",
    "input": {"format": "annotated_cif", "example_id": "monomer",
              "file": {"content": Path("deploy/api/examples/monomer-short.cif").read_text()}},
    "sampling": {"seed": 42, "num_replicates": 1},
}))
PY
curl -fsS http://localhost:8000/v1/designs \
  -H 'Content-Type: application/json' --data-binary @/tmp/rfd4-request.json
export REQUEST_ID=RETURNED_ID
curl -fsS "http://localhost:8000/v1/designs/$REQUEST_ID"
curl -fsS "http://localhost:8000/v1/designs/$REQUEST_ID/results"
```

Submission returns 202. Poll until `succeeded` or `failed`; `/results` lists artifact download paths.
**This is not a drop-in NIM endpoint:** NIM returns designs, sequences and metadata synchronously.
This service uses the upstream medium preset/checkpoint, without promising identical outputs or
per-design seed/batching semantics. CIF conditioning determines what is designed. Condition specs,
search, metrics, output/batching controls, partial diffusion and inverse folding are not exposed.
The latter two task names return 400; unsupported fields fail schema validation first with 422.

Defaults: `RFD4_MAX_REQUEST_BYTES=33554432` (whole JSON body, including chunked transfers),
`RFD4_MAX_TOKENS=1024` (largest compile bucket), `RFD4_MAX_REPLICATES=4`, `RFD4_QUEUE_DEPTH=8`.
These are configurable service policies, not model limits or memory guarantees. Admission checks
CIF syntax/coordinate columns; semantic and token checks run asynchronously before DataLoader creation.

HTTP errors use `{"error": {"code": "…", "message": "…", "details": []}}`. Failed jobs poll
with HTTP 200 and the same object under `error`. Codes/statuses are:

| Code | HTTP status |
|---|---|
| `schema_violation`, `invalid_input` | 422 |
| `invalid_request` | 400 |
| `request_too_large`, `out_of_memory` | 413 |
| `model_unavailable`, `service_busy` | 503 |
| `not_found`, `conflict` | 404, 409 |
| `internal_error` | 500 |

A full queue is unavailable capacity (503); OOM asks the caller to shrink the job (413).
Unexpected diagnostics stay in logs. Restart marks interrupted jobs failed with `model_unavailable`;
completed results persist. Recovery scans history, logs/skips corrupt statuses and preserves them
for inspection. Historical string errors remain readable. Operators must manage disk retention.

## Validation and cleanup

Readiness covers weight loading, not compilation. Upstream compiles buckets on demand: allow at
least 1800 seconds for a cold design, longer on slower GPUs. The image sets
`TORCH_COMPILE_CACHE_BLOB=/workspace/cache/torch/compile`; reuse depends on hardware/runtime compatibility.
Replay requests with the same seed, software, checkpoint and hardware.

```bash
python3 -m venv /tmp/rfd4-api-tests
/tmp/rfd4-api-tests/bin/pip install fastapi uvicorn 'biotite==1.6.0' hydra-core pytest httpx jsonschema
PYTHONPATH=deploy/api /tmp/rfd4-api-tests/bin/pytest deploy/api/tests -q
# Optional: exercise a running GPU server (saves artifacts/status under pytest's tmp_path).
RUN_GPU_API_TESTS=1 API_URL=http://localhost:8000 \
  PYTHONPATH=deploy/api /tmp/rfd4-api-tests/bin/pytest deploy/api/tests/test_gpu.py -m gpu
```

CPU tests use a fake runner without weights. The live test checks generation, downloads, token rejection and recovery;
it does not prove GPU residency, graph reuse or container restart persistence. GPU serving validation
remains [#78](https://github.com/RosettaCommons/RFD4-Proteina/issues/78).
Remove Kubernetes resources with `kubectl delete -f /tmp/rfd4-api.yaml`; deleting the PVC may delete
results depending on reclaim policy. Locally, stop the container then `docker volume rm rfd4-data`.
