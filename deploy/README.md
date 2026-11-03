# Run the protein-design API

These assets run on Linux x86_64 with an NVIDIA GPU, using Docker or Kubernetes.
The included plain CIF example is decompressed from this repository's short-monomer benchmark.
Use the repository's [tested GPUs](../docs/installation.rst#supported-hardware).
Profile GPU memory for your inputs. The image follows this checkout's PyTorch/CUDA dependencies.

## Build

From the repository root, with Docker BuildKit available:

```bash
git clone https://github.com/RosettaCommons/RFD4-Proteina.git
cd RFD4-Proteina
docker build --platform linux/amd64 --target api -t rfd4-api:local .
```

The root Dockerfile has two targets: `runtime` provides the existing model CLI,
and `api` adds FastAPI/Uvicorn on port 8000. The default target is `runtime`; select `api` explicitly.
Dependencies come from `pyproject.toml`. At first startup, the upstream inference loader downloads
the model and autoencoder from the Hugging Face repository and revision selected in
`configs/inference.yaml`. Weights are cached on the mounted volume rather than embedded in the image.

## Run locally

Use a Linux NVIDIA GPU host with a driver compatible with the selected CUDA profile and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Docker Desktop on macOS does not provide the required NVIDIA GPU access.

```bash
docker volume create rfd4-data
docker run --rm --name rfd4-api --gpus all --shm-size=2g \
  -p 127.0.0.1:8000:8000 -v rfd4-data:/workspace rfd4-api:local
```

The first startup downloads the pinned Hugging Face bundle into
`/workspace/cache/huggingface`. Wait for `/health/ready` to return 200;
`/docs` provides interactive API documentation. Later startups reuse the cache.
For offline operation, pre-populate this cache and set `HF_HUB_OFFLINE=1`.
`HF_HOME`, `STORE_PATH`, and `WORK_PATH` override the cache, artifact store,
and temporary request directories; keep them on persisted storage.

The image uses the default CUDA 13 profile. CUDA 12 is available with
`--build-arg RFD4_ACCELERATOR=cu12`; the CPU profile does not support the GPU API.

## Run on Kubernetes

Prerequisites: GPU nodes with a compatible driver and the NVIDIA device plugin
(or GPU Operator) advertising `nvidia.com/gpu`; a default StorageClass or an
administrator-provisioned matching 100 GiB ReadWriteOnce PV; and a registry
readable by the nodes. GPU model selection, runtime classes, storage provisioning,
private-registry credentials, and ingress depend on the cluster.

Push the image to your registry:

```bash
export IMAGE=YOUR_REGISTRY/rfd4-api:YOUR_TAG
docker tag rfd4-api:local "$IMAGE"
docker push "$IMAGE"
```

Render the base with the image you pushed (an immutable digest is preferred):

```bash
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

The manifests create namespace `rfd4-proteina`, one GPU-backed Deployment, an
internal ClusterIP Service, and a persistent volume claim. The PVC uses the
cluster's default storage class. Customize the manifests or a Kustomize overlay
for your StorageClass, GPU node selector, tolerations, RuntimeClass, and imagePullSecrets.
A local cluster can instead load `rfd4-api:local` into its node image store and apply
the base directly. Without a storage provisioner, bind a suitable PV first.

Keep **one replica and one Uvicorn worker**: the in-process queue and disk store
have a single owner. Recreate rollouts release the GPU and volume before starting
the next pod. Scaling this manifest above one replica requires a different queue
and storage design. Volume topology can restrict which GPU nodes can host the pod.

## Submit a design

In another terminal, from the repository root:

```bash
curl -fsS http://localhost:8000/health/ready
curl -fsS -X POST http://localhost:8000/v1/designs \
  -F file=@deploy/api/examples/monomer-short.cif -F seed=42 -F num_replicates=1
export REQUEST_ID=RETURNED_ID
curl -fsS "http://localhost:8000/v1/designs/$REQUEST_ID"
curl -fsS "http://localhost:8000/v1/designs/$REQUEST_ID/results"
```

Poll until `status` is `succeeded`, then download an artifact using its returned
`download` path. Failed inputs report `status: failed` with an error.
To exercise two real designs, downloads, token limits, and recovery:

```bash
python3 -m pip install httpx pytest
RUN_GPU_API_TESTS=1 API_URL=http://localhost:8000 \
  PYTHONPATH=deploy/api pytest deploy/api/tests/test_gpu.py -m gpu
```

Set `API_URL` to test another endpoint. Downloaded artifacts and status JSON land in
pytest's `tmp_path` for the run. This check needs a live GPU server and complements the
fake-runner HTTP tests, which need neither weights nor a GPU.
The server loads checkpoints once per startup and runs one design request at a time.
Limits are service policy rather than model limits, so they are configurable and
reported to clients by `GET /v1/info` instead of being hard-coded: `RFD4_QUEUE_DEPTH`
(8 waiting requests), `RFD4_MAX_REPLICATES` (4), `RFD4_MAX_REQUEST_BYTES` (32 MiB) and
`RFD4_MAX_TOKENS` (1024). The token default is the repository's own ceiling —
`configs/inference.yaml` compiles length buckets up to 1024, so a higher value would
leave the compiled shapes behind. Upload admission checks CIF syntax and
required coordinate columns; it does not guarantee valid design annotations. Upstream
preprocessing and token checks run asynchronously, before DataLoader construction;
semantic errors and oversized designs are accepted with 202 and later report `failed`. It uses the upstream default medium
recycling preset
without external refolding or reward-guided search. This token cap is an API
policy, not a model limit or a guarantee that every design fits the selected GPU.

This is a single-replica deployment sample, not a production service template.
It uses the upstream compilation buckets and cache handling. The first request for a
new shape can take minutes to compile; readiness means weights are loaded, not that
all shapes are warmed. Buckets compile on demand as shapes are encountered, rather
than all configured buckets compiling on the first request. Polling clients should allow
at least 1800 seconds for a cold design (longer on slower hardware); the Kubernetes
startup-probe budget covers model loading, not first-request compilation.
Compiled graphs are reused while the process remains alive.
The image also sets
`TORCH_COMPILE_CACHE_BLOB=/workspace/cache/torch/compile` so upstream can save and
restore compatible artifacts across restarts; reuse depends on hardware and runtime compatibility.
Checkpoint-load counts do not measure host/device transfers or prove graph reuse.

Completed results survive container/pod restarts when `/workspace` is persisted.
Interrupted requests are marked failed after restart and may be resubmitted.
Restart recovery scans historical status files, so startup time grows with request history.
Malformed statuses and mismatched request IDs are logged and skipped; they do not prevent
healthy requests from being recovered. Skipped files are retained for operator inspection.
Manage disk retention yourself. The service has no built-in authentication;
keep it internal or add authenticated HTTPS at your ingress before remote access.

## Request vocabulary and errors

The task names, error codes and status mapping match the downstream RFD4-Proteina
service, so a client written against one handles the other without a translation
table. `GET /v1/info` reports the vocabulary and the limits, and answers while the
model is still loading.

`task` is optional and may be any of `antibody_design`, `enzyme_design`,
`generation`, `motif_scaffolding` or `small_molecule_binder`. It records the sampler
mode; what is designed comes from the conditioning in the CIF either way.
`partial_diffusion` and `inverse_folding` are **refused** with 400: both change the
sampler call rather than the conditioning, and this sample passes neither argument,
so accepting them would run an ordinary trajectory and return 200 for a design that
was not requested.

Every failure returns the same body:

```json
{"error": {"code": "service_busy", "message": "No free worker; route elsewhere or retry later.", "details": []}}
```

| code | status | |
|---|---|---|
| `schema_violation`, `invalid_input` | 422 | malformed request or unreadable CIF |
| `invalid_request` | 400 | valid name, unsupported here |
| `request_too_large`, `out_of_memory` | 413 | shrink the input or the replicate count |
| `model_unavailable`, `service_busy` | 503 | not serving yet, or no free worker |
| `not_found`, `conflict` | 404, 409 | unknown request, or results not ready |
| `internal_error` | 500 | |

Two of those look wrong and are not. `service_busy` is 503 rather than 429 because a
full queue means this replica has no free worker, not that the caller is sending too
fast — a load balancer should route elsewhere rather than ask the client to back off.
`out_of_memory` is a 4xx because retrying the same job exhausts the same GPU.

**One deliberate divergence:** submission here is asynchronous (202, poll, download),
while the downstream service is a single synchronous POST with no job id. That service
sits behind a gateway that supplies long-poll headers; this sample does not, and its
first request for a new shape can take minutes to compile. `GET /v1/info` reports
`"submission": "asynchronous"` so a client can tell the two apart.

## Test and cleanup

API tests use a fake runner and need no GPU, checkpoints, or model dependencies:

```bash
python3 -m venv /tmp/rfd4-api-tests
/tmp/rfd4-api-tests/bin/pip install fastapi uvicorn python-multipart biotite hydra-core pytest httpx
PYTHONPATH=deploy/api /tmp/rfd4-api-tests/bin/pytest deploy/api/tests -q
```

Inspect startup failures with `kubectl logs -n rfd4-proteina deployment/api` and
`kubectl describe pod -n rfd4-proteina -l app=rfd4-api`.
Remove the deployment and its claim with `kubectl delete -f /tmp/rfd4-api.yaml`.
Deleting the claim may delete stored results, depending on your PV reclaim policy.
For local Docker, `docker volume rm rfd4-data` deletes the persisted data.
