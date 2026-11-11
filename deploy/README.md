# Run the protein-design API

A single-replica Docker/Kubernetes service for Linux x86_64 with an NVIDIA GPU.
Use a [tested GPU](../docs/installation.rst#supported-hardware), a compatible driver,
and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Profile memory for your inputs; Docker Desktop on macOS cannot supply the required GPU.

## Docker

From the repository root, with BuildKit available:

```bash
docker build --platform linux/amd64 --target api -t rfd4-api:local .
docker volume create rfd4-data
docker run --rm --name rfd4-api --gpus all --shm-size=2g \
  -p 127.0.0.1:8000:8000 -v rfd4-data:/workspace rfd4-api:local
```

Select `--target api` explicitly; the default `runtime` target retains the CLI.
CUDA 13 is the default; add `--build-arg RFD4_ACCELERATOR=cu12` for CUDA 12.
The CPU profile does not support serving. Dependencies come from `pyproject.toml`.

Startup downloads the model/autoencoder pinned in `configs/inference.yaml` into
`/workspace/cache/huggingface`. Wait for `/health/ready` to return 200. `/docs` and
`/openapi.json` describe the interface; `/v1/info` reports tasks and limits during loading.
Persist `/workspace` for weights, results and compilation caches. `HF_HOME`, `STORE_PATH`
and `WORK_PATH` override cache, artifact and scratch locations. Prepopulate the cache
and set `HF_HUB_OFFLINE=1` for offline use.

## Kubernetes

Provide GPU nodes with the NVIDIA device plugin (or GPU Operator), a registry readable
by those nodes, and a default StorageClass or matching 100 GiB ReadWriteOnce PV.
Customize node selectors, tolerations, RuntimeClass, storage and imagePullSecrets for
your cluster. The base creates a namespace, Deployment, internal Service and PVC.

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

Prefer an immutable image digest. Keep **one replica and one Uvicorn worker**: the
queue and disk store have one owner. Recreate rollouts release the GPU and volume
before replacement; volume topology may restrict scheduling. Scaling needs a different
queue/storage design. There is no built-in authentication: keep access internal or
provide authenticated HTTPS at ingress.

## Submit and download

The included CIF comes from the repository's short-monomer benchmark:

```bash
curl -fsS http://localhost:8000/health/ready
python3 - <<'PY' > /tmp/rfd4-request.json
import json
from pathlib import Path
print(json.dumps({
    "task": "generation",
    "input": {
        "format": "annotated_cif",
        "example_id": "monomer",
        "file": {"content": Path("deploy/api/examples/monomer-short.cif").read_text()},
    },
    "sampling": {"seed": 42, "num_replicates": 1},
}))
PY
curl -fsS http://localhost:8000/v1/designs \
  -H 'Content-Type: application/json' --data-binary @/tmp/rfd4-request.json
export REQUEST_ID=RETURNED_ID
curl -fsS "http://localhost:8000/v1/designs/$REQUEST_ID"
curl -fsS "http://localhost:8000/v1/designs/$REQUEST_ID/results"
```

Submission returns 202. Poll until `succeeded` or `failed`; successful `/results`
listings contain artifact `download` paths. Completed results survive restarts on
persisted storage; interrupted jobs become failures and can be resubmitted. Recovery
scans history, logs/skips corrupt or mismatched statuses, and retains them for inspection.
Operators manage retention; startup time grows with request history.

## Contract and limits

This accepts a strict subset of the [NIM JSON request](https://docs.nvidia.com/nim/bionemo/rfd4-proteina/api-reference/generate-rfd-4-proteina-designs):
`task`, annotated-CIF `input`, and `sampling.seed` / `sampling.num_replicates`.
`input.file.content` contains UTF-8 CIF; optional `filename` is a label, never a path.
`example_id` defaults to `design` and names the staged CIF/upstream outputs; its allowed
characters and length are specified in OpenAPI. Unknown fields are rejected at every
level, including `partial`, `search`, `metrics`, `output` and extra sampling options.
Earlier multipart clients must migrate to JSON.

Supported tasks are `generation`, `antibody_design`, `enzyme_design`, `motif_scaffolding`
and `small_molecule_binder`; conditioning in the CIF determines what is designed.
`partial_diffusion` and `inverse_folding` return 400 because their sampler arguments
are not exposed. Unsupported fields fail schema validation first with 422.
Omitted/null seeds are randomized and recorded in job `settings`; explicit seeds span
0–2147483647. Replay the whole request with the same software, checkpoint and hardware.

**This is not a drop-in NIM endpoint.** NIM returns designs, sequences, diagnostics,
metadata and the resolved schedule synchronously; this service returns jobs and artifact
links. It uses the upstream medium preset and checkpoint without promising identical
outputs or NIM's per-design seed/batching semantics. Condition specs, search, metrics,
output/batching controls, partial diffusion and inverse folding are not exposed.

| Environment variable | Default |
|---|---|
| `RFD4_MAX_REQUEST_BYTES` | 32 MiB, entire JSON body including chunked transfers |
| `RFD4_MAX_TOKENS` | 1024 transformed tokens, matching the largest compile bucket |
| `RFD4_MAX_REPLICATES` | 4 samples |
| `RFD4_QUEUE_DEPTH` | 8 waiting requests |

These are service policies, not model limits or memory guarantees. Admission checks CIF
syntax/coordinate columns; semantic validation and token limits run asynchronously
before DataLoader construction. Such failures can follow an accepted 202 response.

HTTP errors use `{"error": {"code": "…", "message": "…", "details": []}}`.
Failed jobs poll with HTTP 200, `status: failed`, and the same object under `error`.

| Code | HTTP status |
|---|---|
| `schema_violation`, `invalid_input` | 422 |
| `invalid_request` | 400 |
| `request_too_large`, `out_of_memory` | 413 |
| `model_unavailable`, `service_busy` | 503 |
| `not_found`, `conflict` | 404, 409 |
| `internal_error` | 500 |

A full queue is unavailable capacity (503); OOM asks the caller to shrink the job (413).
Interrupted jobs use `model_unavailable`; unexpected diagnostics stay in server logs.
Historical string errors are read as `internal_error` without rewriting stored files.

## Validation and operation

Readiness means weights are loaded, not compilation is warmed. Upstream compiles buckets
on demand; allow at least 1800 seconds for a cold design, longer on slower hardware.
The Kubernetes startup budget covers loading only. The image sets
`TORCH_COMPILE_CACHE_BLOB=/workspace/cache/torch/compile` for upstream cache persistence;
reuse depends on hardware/runtime compatibility. Checkpoint-load counts do not prove
GPU residency or graph reuse. This is a deployment sample, not a production template.

CPU tests need no weights or model dependencies:

```bash
python3 -m venv /tmp/rfd4-api-tests
/tmp/rfd4-api-tests/bin/pip install fastapi uvicorn 'biotite==1.6.0' hydra-core pytest httpx jsonschema
PYTHONPATH=deploy/api /tmp/rfd4-api-tests/bin/pytest deploy/api/tests -q
```

Against a running GPU server, the following checks two generations, downloads, token
rejection and recovery. It does not test restart persistence, residency or graph reuse;
real GPU serving validation remains tracked in [#78](https://github.com/RosettaCommons/RFD4-Proteina/issues/78).

```bash
RUN_GPU_API_TESTS=1 API_URL=http://localhost:8000 \
  PYTHONPATH=deploy/api /tmp/rfd4-api-tests/bin/pytest deploy/api/tests/test_gpu.py -m gpu
```

Artifacts/status JSON are saved under pytest's `tmp_path`. Inspect deployment failures
with `kubectl logs -n rfd4-proteina deployment/api` and
`kubectl describe pod -n rfd4-proteina -l app=rfd4-api`.
Remove Kubernetes resources with `kubectl delete -f /tmp/rfd4-api.yaml`; deleting the
claim may delete results, depending on PV reclaim policy. Locally,
`docker volume rm rfd4-data` deletes persisted data after stopping the container.
