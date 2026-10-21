# Cluster CI runners (maintainers only)

`.github/workflows/cluster-ci.yaml` runs the CPU and GPU test lanes on self-hosted runners at CoreWeave and DIGS.
It is triggered by a `/ci` comment from a listed maintainer on a pull request into `main`, by a push to `main` (a merged
pull request, which also refreshes the Codecov baseline) or by a manual run (`workflow_dispatch`). Plain `pull_request`
events never run on these machines.

## Who may run it

1. The `authorize` job (GitHub-hosted, no cluster access) checks the triggering user against the repository variable
   `CLUSTER_CI_MAINTAINERS`, a comma-separated list of GitHub logins. An empty variable authorizes nobody.
   Set it under Settings → Secrets and variables → Actions → Variables.
2. Cluster jobs run only from `main` (the `authorize` job refuses any other ref). GitHub Environments are not
   available to this organization, so there is no reviewer-approval step.
3. Settings → Actions → General → "Fork pull request workflows" should require approval for all outside collaborators.

## Testing a pull request into main

Comment `/ci` on an open pull request into `main` (or `/ci coreweave` / `/ci digs` for one cluster). Only logins in
`CLUSTER_CI_MAINTAINERS` are honored; comments from anyone else do nothing. The runner reacts with 🚀 and posts the result
on the pull request when it finishes. It tests the PR head at the time of the comment, so read the whole diff and make sure
no new commits landed since you reviewed it, because that code (including `.github/ci/*.sh` and the tests) runs on the cluster
as the runner user.

From the Actions tab you can instead start the workflow from `main` with `pr` and `reviewed_sha`; that run refuses if the PR
head has moved since you reviewed it.

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

The GPU lane also needs the object-store credentials: export `APPTAINERENV_AWS_ACCESS_KEY_ID` and `APPTAINERENV_AWS_SECRET_ACCESS_KEY` in the
profile (file mode 600). Do not store them as repository secrets: any user with write access could read them from a workflow on a
branch. Repository secret used: `CODECOV_TOKEN` (coverage upload only).
Variable used: `SOTA_CKPT_PATH` (optional, enables the checkpoint-gated tests).
