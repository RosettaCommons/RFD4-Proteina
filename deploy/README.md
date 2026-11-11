# GPU API

Linux x86_64, a [supported NVIDIA GPU](../docs/installation.rst#supported-hardware), compatible driver and NVIDIA Container Toolkit are required.
Use one replica/worker. Persist `/workspace`; keep access internal or add authenticated HTTPS.

```bash
docker build --target api -t rfd4-api:local .
docker run --rm --gpus all --shm-size=2g -p 127.0.0.1:8000:8000 -v rfd4-data:/workspace rfd4-api:local
```

Add `--build-arg RFD4_ACCELERATOR=cu12` for CUDA 12; the default is CUDA 13. The default Docker target remains the CLI.
For Kubernetes, push the image, set it in `kubernetes/deployment.yaml`, then run `kubectl apply -k deploy/kubernetes`.
Provide the NVIDIA device plugin and a default StorageClass (or matching 100 GiB ReadWriteOnce PV); customize registry credentials and GPU scheduling for your cluster.

Wait for `/health/ready`. Startup downloads the pinned Hugging Face weights; prepopulate `HF_HOME` and set `HF_HUB_OFFLINE=1` for offline use.
`/docs` and `/openapi.json` describe requests/responses; `/v1/info` reports tasks and limits, including during loading.
POST `/v1/designs` accepts `task`, `input: {format: annotated_cif, file: {content: "…"}}` and `sampling: {seed: 42, num_replicates: 1}`.
Use the existing `data/benchmarks/monomer/monomer-short.cif.zst` example after decompression. Unknown fields are rejected; omitted/null seeds are randomized and recorded.
Poll `/v1/designs/{id}` after submission returns 202, then use `/results` and its artifact download links.

This is a strict subset of the [NIM request](https://docs.nvidia.com/nim/bionemo/rfd4-proteina/api-reference/generate-rfd-4-proteina-designs), with asynchronous jobs rather than NIM's synchronous designs/metadata.
Partial diffusion, inverse folding, condition specs, search, metrics and output/batching controls are not exposed. Generation uses the upstream medium preset and checkpoint.
Limits are `RFD4_MAX_REQUEST_BYTES=33554432`, `RFD4_MAX_TOKENS=1024`, `RFD4_MAX_REPLICATES=4`, `RFD4_QUEUE_DEPTH=8`; they are service policies, not GPU memory guarantees.
HTTP failures and failed jobs carry `{code, message, details}` errors. Semantic/token checks run asynchronously. Allow at least 1800 seconds for cold compilation; readiness covers weight loading only.

Results persist under `STORE_PATH` (default `/workspace/store`); interrupted jobs fail on restart. Recovery skips corrupt statuses. Operators manage retention, including partial outputs of failed jobs.
`TORCH_COMPILE_CACHE_BLOB` persists upstream compile artifacts; reuse depends on hardware/runtime compatibility. GPU serving/restart validation remains [#78](https://github.com/RosettaCommons/RFD4-Proteina/issues/78).
Run `pytest tests/api` in the normal development environment.
Remove Kubernetes resources with `kubectl delete -k deploy/kubernetes`; deleting the PVC or Docker volume may delete persisted results.
