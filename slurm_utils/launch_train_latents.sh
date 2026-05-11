#!/bin/bash
# =============================================================================
# launch_train_latents.sh - Direct cluster launch for latent diffusion training
# =============================================================================
# Run this script directly on the cluster (not from local machine)
#
# Usage:
#   ./launch_train_latents.sh <cluster> [options] [extra_args...]
#
# Options:
#   --no-copy      Skip copying code to run directory (use current repo)
#   --follow, -f   Follow log output after job submission (Ctrl+C to exit)
#   --submissions N  Number of job submissions for auto-requeue (default: 2)
#
# Examples:
#   ./launch_train_latents.sh digs
#   ./launch_train_latents.sh digs --no-copy
#   ./launch_train_latents.sh digs --follow
#   ./launch_train_latents.sh gb200 +experiment=ablation
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Parse arguments
CLUSTER=""
COPY_CODE="true"
FOLLOW_LOGS="false"
NUM_SUBMISSIONS=1  # Submit multiple times with --dependency=singleton for auto-requeue
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-copy)
            COPY_CODE="false"
            shift
            ;;
        --follow|-f)
            FOLLOW_LOGS="true"
            shift
            ;;
        --submissions)
            NUM_SUBMISSIONS="$2"
            shift 2
            ;;
        -*)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        *)
            if [ -z "$CLUSTER" ]; then
                CLUSTER="$1"
            else
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift
            ;;
    esac
done

if [ -z "$CLUSTER" ]; then
    echo "Usage: $0 <cluster> [--no-copy] [extra_args...]"
    echo "Available clusters: digs, ord, gb200"
    exit 1
fi

# Load cluster configuration
load_user_info "$CLUSTER" || exit 1
print_cluster_config

# Find repository root
REPO_ROOT=$(find_repo_root "$SCRIPT_DIR")
if [ -z "$REPO_ROOT" ]; then
    echo "Error: Could not find .project-root"
    exit 1
fi
cd "$REPO_ROOT"
echo "Repository root: $REPO_ROOT"

# Setup PYTHONPATH
source_venv "$REPO_ROOT"
setup_pythonpath "$REPO_ROOT"
setup_env_vars

# Resolve configuration using Hydra (handles composition, interpolations, overrides)
# This properly resolves defaults, experiment configs, and command-line overrides
CONFIG_NAME="training_proteina"
print_header "Resolving Hydra Config"
echo "  Config name: $CONFIG_NAME"
echo "  Extra args: $EXTRA_ARGS"

eval $(resolve_hydra_config "$REPO_ROOT" "$CONFIG_NAME" $EXTRA_ARGS)

# Validate required values
if [ -z "$RUN_NAME" ]; then
    echo "Error: run_name_ not found in resolved config"
    exit 1
fi

if [ -z "$NGPUS_PER_NODE" ] || [ "$NGPUS_PER_NODE" = "0" ]; then
    echo "Error: ngpus_per_node_ must be specified in config"
    exit 1
fi

# Rename for consistency with rest of script
GPUS_PER_NODE="$NGPUS_PER_NODE"

# Create run directory structure
RUN_DIR="$ROOT_REMOTE/train_latents/$RUN_NAME"
mkdir -p "$RUN_DIR/slurm_run_outputs"

# Clear old slurm logs to start fresh
rm -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err"

# Create symlink in active directory for easy browsing
mkdir -p "$ROOT_REMOTE/active"
ln -sfn "$RUN_DIR" "$ROOT_REMOTE/active/$RUN_NAME"

# Create symlink in repo for easy local browsing
mkdir -p "$REPO_ROOT/runs"
ln -sfn "$RUN_DIR" "$REPO_ROOT/runs/$RUN_NAME"
echo "  Local symlink: $REPO_ROOT/runs/$RUN_NAME"

# Copy code to run directory (or use current repo)
if [ "$COPY_CODE" == "true" ]; then
    print_header "Copying Code"
    echo "  $REPO_ROOT -> $RUN_DIR"

    copy_code_to_dir "$REPO_ROOT" "$RUN_DIR" "$CLUSTER"
    CODE_DIR="$RUN_DIR"
else
    print_header "Using Code In-Place"
    echo "  $REPO_ROOT"
    CODE_DIR="$REPO_ROOT"
fi

# Create .env file from ENV_VARS array (defined in .user_info)
write_env_file "$RUN_DIR/.env"

print_header "Job Configuration"
echo "  Run name: $RUN_NAME"
echo "  Run directory: $RUN_DIR"
echo "  Nodes: $NNODES"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  CPUs per task: $CPUS_PER_TASK"
echo "  Submissions: $NUM_SUBMISSIONS (with --dependency=singleton)"
print_mount_summary
echo "  Extra args: $EXTRA_ARGS"
echo ""

# Build SBATCH script (uses JOB_NAME, NODES, GPUS_PER_NODE, SIGNAL_TIME from user_info)
SBATCH_FILE="$RUN_DIR/slurm_batch_script.sh"
JOB_NAME="$RUN_NAME"
NODES="$NNODES"
SIGNAL_TIME="${SIGNAL_TIME:-600}"  # Default 600s if not set in user_info
build_sbatch_header > "$SBATCH_FILE"

cat >> "$SBATCH_FILE" << EOF
#SBATCH --output ./slurm_run_outputs/slurm.out
#SBATCH --error ./slurm_run_outputs/slurm.err

# Environment setup (all clusters)
export PYTHONPATH=$CODE_DIR/src:$CODE_DIR/models/rf3/src:$CODE_DIR/projects/proteinfoundation:$CODE_DIR/lib/atomworks/src:\$PYTHONPATH
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# (GPU/driver logging moved inside container to ensure consistent runtime context)

# Check if scontrol works (required for Lightning auto-requeue)
scontrol show job \$SLURM_JOB_ID > /dev/null 2>&1 && echo '[slurm] scontrol works - auto-requeue enabled' || echo '[slurm] WARNING: scontrol not working - auto-requeue disabled'
EOF

# Apptainer clusters: pass cluster identity to exec_core.sh (via shebang) and
# compute distributed training endpoints dynamically at job time.
# Static env vars (NCCL, PyTorch) come from .ipd/.env.${CLUSTER} via exec_core.sh.
if [ "$CONTAINER_TYPE" == "apptainer" ]; then
    cat >> "$SBATCH_FILE" << EOF

# Cluster identity — read by exec_core.sh (shebang) to source .ipd/.env.\${CLUSTER}
export CLUSTER=$CLUSTER
export CONTAINER_TYPE=$CONTAINER_TYPE

# Distributed training coordination (unique port per job, master = first node)
export MASTER_PORT=\$((1024 + RANDOM % 64512))
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
EOF
fi

# Add cuEquivariance control (DISABLE_CUEQUIVARIANCE=true to disable Triton-based attention)
if [ -n "$DISABLE_CUEQUIVARIANCE" ] && [ "$DISABLE_CUEQUIVARIANCE" == "true" ]; then
    echo "export DISABLE_CUEQUIVARIANCE=true" >> "$SBATCH_FILE"
    echo "# cuEquivariance disabled - using vanilla PyTorch attention" >> "$SBATCH_FILE"
fi

# Add container execution command
SCRIPT_TRAIN="$CODE_DIR/projects/proteinfoundation/proteinfoundation/train.py"

if [ "$CONTAINER_TYPE" == "apptainer" ]; then
    # Apptainer: shebang in train.py handles container execution via exec_core.sh
    if [ "$AF3_NEEDED" == "true" ]; then
        echo "  AF3Oracle detected in resolved config — emitting AF3 daemon pre-spawn block"
        emit_af3_daemon_block "$CODE_DIR" "$GPUS_PER_NODE" >> "$SBATCH_FILE"
    else
        echo "  No AF3Oracle in resolved config — skipping AF3 daemon pre-spawn (saves ~10 GB/node)"
    fi

    cat >> "$SBATCH_FILE" << EOF

# Training. --overlap so it can coexist with the AF3 daemon step on the same nodes.
srun --cpu-bind=none --kill-on-bad-exit --export=ALL --overlap $SCRIPT_TRAIN $EXTRA_ARGS
EOF
else
    CONTAINER_CMD=$(get_srun_cmd)
    
    # Build container mounts string from CONTAINER_MOUNTS and SQUASHFS_MOUNTS arrays
    CONTAINER_MOUNTS_STR=$(build_container_mounts_string)
    SQFS_MOUNT_CMDS=$(build_squashfs_mount_cmds)
    CONTAINER_ENV_CMDS=$(build_container_env_cmds)
    
    # Check if squashfs mounting is enabled (from user_info)
    if [ "${USE_SQUASHFS_MOUNTS:-false}" == "true" ] && [ -n "$SQFS_MOUNT_CMDS" ]; then
        # Squashfs mounting enabled - mount .sqfs files and use squashfuse inside container
        cat >> "$SBATCH_FILE" << EOFSCRIPT
# Container mounts (auto-generated from CONTAINER_MOUNTS and SQUASHFS_MOUNTS arrays)
CONTAINER_MOUNTS="$CONTAINER_MOUNTS_STR"

# Run training with container
# - Mount paths from CONTAINER_MOUNTS array (includes /lustre, slurm/munge, etc.)
# - Mount squashfs files and /dev/fuse for dataset mounting
# - Run squashfuse inside container before training
srun --cpu-bind=none --kill-on-bad-exit $CONTAINER_CMD \\
    --container-mounts="\${CONTAINER_MOUNTS}" \\
    --container-workdir=$RUN_DIR \\
    bash -c "\\
        $CONTAINER_ENV_CMDS && \\
        $SQFS_MOUNT_CMDS && \\
        scontrol show job \\\$SLURM_JOB_ID > /dev/null 2>&1 && echo '[slurm] scontrol works inside container' || echo '[slurm] WARNING: scontrol not working - auto-requeue disabled' && \\
        python -c \"import torch; print('torch=' + torch.__version__); print('torch_cuda=' + str(torch.version.cuda)); print('cuda_available=' + str(torch.cuda.is_available()))\" || true && \\
        exec python $SCRIPT_TRAIN $EXTRA_ARGS\\
    "
EOFSCRIPT
    else
        # Squashfs mounting disabled - just use CONTAINER_MOUNTS
        cat >> "$SBATCH_FILE" << EOFSCRIPT
# Container mounts (auto-generated from CONTAINER_MOUNTS array)
CONTAINER_MOUNTS="$CONTAINER_MOUNTS_STR"

# Run training with container (no squashfs mounting)
srun --cpu-bind=none --kill-on-bad-exit $CONTAINER_CMD \\
    --container-mounts="\${CONTAINER_MOUNTS}" \\
    --container-workdir=$RUN_DIR \\
    bash -c "\\
        $CONTAINER_ENV_CMDS && \\
        scontrol show job \\\$SLURM_JOB_ID > /dev/null 2>&1 && echo '[slurm] scontrol works inside container' || echo '[slurm] WARNING: scontrol not working - auto-requeue disabled' && \\
        python -c \"import torch; print('torch=' + torch.__version__); print('torch_cuda=' + str(torch.version.cuda)); print('cuda_available=' + str(torch.cuda.is_available()))\" || true && \\
        exec python $SCRIPT_TRAIN $EXTRA_ARGS\\
    "
EOFSCRIPT
    fi
fi

print_header "Submitting Job"
echo "SBATCH script: $SBATCH_FILE"
cd "$RUN_DIR"

# Check for already running job
EXISTING_RUNNING=$(squeue -u "$USER" --name="$RUN_NAME" -h -o "%i %T" 2>/dev/null | grep "RUNNING" | head -1)
if [ -n "$EXISTING_RUNNING" ]; then
    EXISTING_ID=$(echo "$EXISTING_RUNNING" | awk '{print $1}')
    echo "Warning: Job '$RUN_NAME' already RUNNING (ID: $EXISTING_ID)"
    echo "  Cancel: scancel $EXISTING_ID | Monitor: tail -f $RUN_DIR/slurm_run_outputs/slurm.err"
    read -p "Continue with submission? [y/N] " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

[ -n "$WANDBKEY" ] && export WANDB_API_KEY="$WANDBKEY"

# Submit with --dependency=singleton for auto-continuation chain
echo "Submitting $NUM_SUBMISSIONS job(s) with --dependency=singleton..."
for i in $(seq 1 $NUM_SUBMISSIONS); do
    SBATCH_OUTPUT=$(sbatch --dependency=singleton slurm_batch_script.sh)
    echo "  [$i/$NUM_SUBMISSIONS] $SBATCH_OUTPUT"
done

echo ""
echo "Monitor: squeue -u $USER --name=$RUN_NAME"
echo "Logs:    tail -f $RUN_DIR/slurm_run_outputs/slurm.err"

# Follow logs if requested
if [ "$FOLLOW_LOGS" == "true" ]; then
    echo -e "\nFollowing logs (Ctrl+C to exit)...\n"
    while [ ! -f "$RUN_DIR/slurm_run_outputs/slurm.err" ]; do
        squeue -u "$USER" --name="$RUN_NAME" -h 2>/dev/null | grep -q . || { echo "Job not in queue"; break; }
        sleep 2
    done
    tail -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err" 2>/dev/null
fi
