# Run the protein-design API

These assets run on Linux x86_64 with an NVIDIA GPU, using Docker or Kubernetes.
Use the repository's [tested GPUs](../docs/installation.rst#supported-hardware).
An A100 40 GB was validated for 100-residue examples with the earlier runtime;
profile memory for larger inputs. The image follows this checkout's PyTorch/CUDA dependencies.

## Build

From the repository root, with Docker BuildKit available:

```bash
git clone https://github.com/RosettaCommons/RFD4-Proteina.git
cd RFD4-Proteina
docker build --platform linux/amd64 --target api -t rfd4-api:local .
```

The root Dockerfile has two targets: `runtime` provides the existing model CLI,
and `api` adds FastAPI/Uvicorn on port 8000. The default target is `api`.
Building requires internet access, substantial disk space, and no GPU.
Dependencies come from `pyproject.toml`; checkpoints are downloaded at first startup,
not embedded in the image. Follow the repository's access and license terms.

## Run locally

Use a Linux NVIDIA GPU host with a CUDA 13.2-compatible driver and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Docker Desktop on macOS does not provide the required NVIDIA GPU access.

```bash
docker volume create rfd4-data
docker run --rm --name rfd4-api --gpus all --shm-size=2g \
  -p 127.0.0.1:8000:8000 -v rfd4-data:/workspace rfd4-api:local
```

The first startup downloads about 8 GB of checkpoints into `/workspace/models`.
Wait for `/health/ready` to return 200; `/docs` provides interactive API documentation.

For an offline runtime, pre-stage the checkpoint pair referenced by
`configs/inference.yaml` under the mounted `models/flow_checkpoints/` and
`models/ae_checkpoints/` directories, then set `RFD4_DOWNLOAD_WEIGHTS=false`.
Set `RFPROTEINA_DATA_URL` to use your own HTTPS mirror with the same paths.
`DATA_PATH`, `STORE_PATH`, and `WORK_PATH` can override the models, artifact store,
and temporary request directories; defaults all live under `/workspace`.

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
  -F file=@data/benchmarks/monomer/monomer-short.cif -F seed=42 -F num_replicates=1
export REQUEST_ID=RETURNED_ID
curl -fsS "http://localhost:8000/v1/designs/$REQUEST_ID"
curl -fsS "http://localhost:8000/v1/designs/$REQUEST_ID/results"
```

Poll until `status` is `succeeded`, then download an artifact using its returned
`download` path. Failed inputs report `status: failed` with an error.
To exercise two real designs, downloads, invalid input, token limits, and recovery:

```bash
python3 deploy/api/smoke_test.py
```

This writes validation outputs to `results/`; set `API_URL` to test another endpoint.
The server loads checkpoints once per startup and runs one design request at a time.
It accepts eight waiting requests, 1–4 replicates, annotated mmCIF uploads up to
4 MiB, and at most 512 transformed tokens. It uses the medium generation preset
without external refolding or reward-guided search. This token cap is an API
policy, not a model limit or a guarantee that every design fits the selected GPU.

Completed results survive container/pod restarts when `/workspace` is persisted.
Interrupted requests are marked failed after restart and may be resubmitted.
Manage disk retention yourself. The service has no built-in authentication;
keep it internal or add authenticated HTTPS at your ingress before remote access.

## Test and cleanup

API tests use a fake runner and need no GPU, checkpoints, or model dependencies:

```bash
python3 -m venv /tmp/rfd4-api-tests
/tmp/rfd4-api-tests/bin/pip install -r deploy/api/requirements.txt pytest httpx
PYTHONPATH=deploy/api /tmp/rfd4-api-tests/bin/pytest deploy/api/tests -q
```

Inspect startup failures with `kubectl logs -n rfd4-proteina deployment/api` and
`kubectl describe pod -n rfd4-proteina -l app=rfd4-api`.
Remove the deployment and its claim with `kubectl delete -f /tmp/rfd4-api.yaml`.
Deleting the claim may delete stored results, depending on your PV reclaim policy.
For local Docker, `docker volume rm rfd4-data` deletes the persisted data.
