# Cluster CI runners (maintainers only)

`.github/workflows/cluster-ci.yaml` runs the CPU and GPU test lanes on self-hosted runners at CoreWeave and DIGS.
It is triggered only by `workflow_dispatch` and by pushes to `main`. Pull requests never run on these machines.

## Who may run it

1. The `authorize` job (GitHub-hosted, no cluster access) checks the triggering user against the repository variable
   `CLUSTER_CI_MAINTAINERS`, a comma-separated list of GitHub logins. An empty variable authorizes nobody.
   Set it under Settings → Secrets and variables → Actions → Variables.
2. Every cluster job also uses the `cluster-ci` environment. In Settings → Environments → `cluster-ci`, add the
   maintainers as required reviewers and restrict deployment branches to `main`.
3. Settings → Actions → General → "Fork pull request workflows" should require approval for all outside collaborators.

## Registering a runner

Register repo-level runners (Settings → Actions → Runners → New self-hosted runner), one per label below, on a node that
can submit Slurm jobs, under a dedicated low-privilege account with no access to other projects' data.

| Runner label | Cluster | Lane |
|---|---|---|
| `coreweave-cpu` | CoreWeave | CPU tests (srun, `interactive` partition) |
| `coreweave-gpu` | CoreWeave | GPU tests (srun, 6 GPUs) |
| `digs` | DIGS | CPU tests (srun, `cpu` partition) |

Prefer ephemeral runners (`./config.sh --ephemeral`) started per job, so one run cannot leave state for the next.

## Cluster profile (kept on the runner, not in the repository)

The lanes read site-specific paths from a shell profile whose location comes from `RFD4_CI_PROFILE` in the runner's
environment (put `RFD4_CI_PROFILE=/path/to/profile.env` in the runner's `.env` file). It exports at least:

- `LATENT_APPTAINER_SIF`: the container image with the RFD4-Proteina dependencies (including AtomWorks),
- `DATA_PATH`, `CACHE_DIR`, `PDB_MIRROR_PATH`, `CCD_MIRROR_PATH`,
- `AWS_ENDPOINT_URL` (GPU lane), and optionally `AF2_*`, `AF3_*`, `OF3_MODEL_DIR`, `SQUASHFS_MOUNTS` / `USE_SQUASHFS_MOUNTS`,
  `APPTAINER_BIND_PATHS`.

Repository secrets used: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (GPU lane), `CODECOV_TOKEN` (coverage upload).
Variable used: `SOTA_CKPT_PATH` (optional, enables the checkpoint-gated tests).
